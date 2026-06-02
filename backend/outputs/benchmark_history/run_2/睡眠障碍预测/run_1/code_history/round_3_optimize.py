import pandas as pd
import numpy as np
import dill
import json
import re
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error, mean_absolute_error, r2_score, f1_score
warnings.filterwarnings('ignore')

# ========== 全局状态（用于保存预处理参数）==========
PREPROCESS_STATE = {}

# ========== LLM 填充区（开始）==========
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import lightgbm as lgb

# 全局变量：存储预处理状态（用于 test 模式）
PREPROCESS_STATE = {
    'id_cols': None,
    'bp_col': 'Blood Pressure (systolic/diastolic)',
    'target_col': 'Sleep Disorder',
    'fitted': False
}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    df = df.copy()
    
    # 1. 剔除疑似ID特征（唯一值=样本数，属于噪声）
    id_cols = [
        'Person ID',
        'Sleep Duration (hours)',
        'Quality of Sleep (scale: 1-10)',
        'Physical Activity Level (minutes/day)',
        'Stress Level (scale: 1-10)',
        'Heart Rate (bpm)',
        'Daily Steps'
    ]
    # 只删除存在的列（test 模式下列可能已不存在）
    existing_id_cols = [c for c in id_cols if c in df.columns]
    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)
    
    # 2. 拆解血压特征 'Blood Pressure (systolic/diastolic)' → 'Systolic BP', 'Diastolic BP'
    bp_col = PREPROCESS_STATE['bp_col']
    if bp_col in df.columns:
        # 拆解字符串 '129/75' → 两个数值
        bp_split = df[bp_col].str.split('/', expand=True)
        df['Systolic BP'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['Diastolic BP'] = pd.to_numeric(bp_split[1], errors='coerce')
        # 删除原列
        df = df.drop(columns=[bp_col])
    
    # 3. 检查并处理缺失值（当前无缺失，但保留检查逻辑）
    # 数值列用 median 填充，类别列用 most_frequent 填充
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # 排除目标列
    target_col = PREPROCESS_STATE['target_col']
    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # 4. 确认目标列存在（train/validation 模式），test 模式可能没有
    # 不做任何编码，保留原始标签（系统会自动处理）
    
    if mode == 'train':
        PREPROCESS_STATE['fitted'] = True
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    
    使用 ColumnTransformer 对类别特征编码、数值特征标准化。
    train 模式：fit_transform 并保存预处理器到 PREPROCESS_STATE。
    test 模式：使用已保存的预处理器 transform。
    '''
    target_col = PREPROCESS_STATE['target_col']
    
    # 分离 X
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    
    # 区分列类型
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # 类别特征分组
    gender_col = 'Gender' if 'Gender' in cat_cols else None
    other_cat_cols = [c for c in cat_cols if c != gender_col]
    
    # 检查是否已有保存的预处理器（test 模式）
    preprocessor = PREPROCESS_STATE.get('preprocessor', None)
    
    if preprocessor is None:
        # train 模式：构建新的 ColumnTransformer
        transformers = []
        
        if gender_col:
            transformers.append(
                ('gender', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]), [gender_col])
            )
        
        if other_cat_cols:
            transformers.append(
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), other_cat_cols)
            )
        
        if num_cols:
            transformers.append(
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), num_cols)
            )
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        # fit_transform
        X_processed = preprocessor.fit_transform(X)
        
        # 保存预处理器到全局状态
        PREPROCESS_STATE['preprocessor'] = preprocessor
    else:
        # test 模式：使用已保存的预处理器
        X_processed = preprocessor.transform(X)
    
    # 获取特征名
    feature_names = []
    for name, _, cols in preprocessor.transformers_:
        if name == 'gender':
            feature_names.append('Gender_encoded')
        elif name == 'cat':
            encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            feature_names.extend(encoder.get_feature_names_out(other_cat_cols).tolist())
        elif name == 'num':
            feature_names.extend(cols)
    
    # 清洗特征名
    feature_names = [re.sub(r'[^\w]', '_', str(c)) for c in feature_names]
    
    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    
    return X_processed


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    
    使用 LightGBM 多分类器，针对小样本数据优化参数。
    特征数量约20+个（数值特征 + OneHot编码后的类别特征），适当增加模型容量。
    '''
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        is_unbalance=True,        # 自动处理类别不平衡
        max_depth=5,              # 适当增加深度（特征增多后需要更多交互）
        num_leaves=31,            # 增加叶子数，配合更多特征
        learning_rate=0.03,       # 降低学习率，配合更多迭代
        n_estimators=1000,        # 增加树数量，配合早停
        subsample=0.7,            # 行采样
        colsample_bytree=0.7,     # 列采样
        min_child_samples=5,      # 小样本允许更细粒度分裂
        min_split_gain=0.01,      # 分裂增益阈值
        reg_alpha=0.5,            # L1 正则化
        reg_lambda=0.5,           # L2 正则化
        random_state=42,
        verbose=-1,
    )
    
    return model
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'Sleep Disorder'
id_col = 'Person ID'
if id_col not in test.columns:
    id_col = test.columns[0]

# 检查目标列是否存在
if target_col not in train.columns:
    raise ValueError(f"目标列 '{target_col}' 不在训练数据中，可用列: {list(train.columns)}")

# ========== 预处理（系统调用 LLM 填充的函数）==========
train_clean = preprocess(train, mode='train')
val_clean = preprocess(val, mode='test')
test_clean = preprocess(test, mode='test')

# 分离特征和目标（兼容 preprocess 是否保留目标列的情况）
if target_col in train_clean.columns:
    y_train = train_clean[target_col]
    X_train = train_clean.drop(columns=[target_col])
else:
    y_train = train[target_col]
    X_train = train_clean

if target_col in val_clean.columns:
    y_val = val_clean[target_col]
    X_val = val_clean.drop(columns=[target_col])
else:
    y_val = val[target_col]
    X_val = val_clean

# 【强制编码】如果原始目标列是字符串/类别，强制从原始数据获取并统一编码
# 这能覆盖 LLM 可能在 preprocess 中对目标列做的任何编码，确保 predict 后可反编码回原始标签
_label_encoder = None
_target_dtype = str(train[target_col].dtype).lower() if target_col in train.columns else ''
_is_string_target = target_col in train.columns and (
    train[target_col].dtype == object or _target_dtype == 'category' or
    _target_dtype.startswith('str') or _target_dtype.startswith('string')
)
if _is_string_target:
    from sklearn.preprocessing import LabelEncoder
    _label_encoder = LabelEncoder()
    y_train = _label_encoder.fit_transform(train[target_col])
    try:
        y_val = _label_encoder.transform(val[target_col])
    except ValueError:
        # 验证集可能出现训练集未见的标签（如带空格/点号变体），统一用训练集映射兜底
        _val_labels = val[target_col].astype(str).str.strip().str.rstrip('.')
        _train_labels = pd.Series(train[target_col]).astype(str).str.strip().str.rstrip('.')
        _label_encoder.fit(_train_labels)
        y_val = _label_encoder.transform(_val_labels)
    PREPROCESS_STATE['label_encoder'] = _label_encoder

X_test = test_clean.drop(columns=[target_col], errors='ignore')
if X_test is test_clean:
    X_test = test_clean.copy()

# ========== 特征工程（系统调用 LLM 填充的函数）==========
X_train_fe = feature_engineering(X_train)
if isinstance(X_train_fe, np.ndarray):
    X_train_fe = pd.DataFrame(X_train_fe, index=X_train.index)
X_val_fe = feature_engineering(X_val)
if isinstance(X_val_fe, np.ndarray):
    X_val_fe = pd.DataFrame(X_val_fe, index=X_val.index)
X_test_fe = feature_engineering(X_test)
if isinstance(X_test_fe, np.ndarray):
    X_test_fe = pd.DataFrame(X_test_fe, index=X_test.index)

# ========== 清洗特征名（LGBM/XGBoost 不支持特殊 JSON 字符）==========
for _df in [X_train_fe, X_val_fe, X_test_fe]:
    _df.columns = [re.sub('[^\\w]', '_', str(c)) for c in _df.columns]
# 去重列名
for _df in [X_train_fe, X_val_fe, X_test_fe]:
    if _df.columns.duplicated().any():
        _df.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(_df.columns)]

# ========== 模型训练（系统负责）==========
model = build_model()
# 尝试传入 eval_set（XGBoost/LightGBM 等支持 early stopping 的模型需要）
try:
    model.fit(X_train_fe, y_train, eval_set=[(X_val_fe, y_val)])
except Exception:
    # 第一次 fit 可能因 eval_set 不被支持而失败（如 sklearn 原生模型）
    # 尝试不带 eval_set 的 fit；若仍失败，说明是真正的数据/代码错误，必须抛出
    try:
        model.fit(X_train_fe, y_train)
    except Exception as _fit_err:
        print(f"[FIT_ERROR] {_fit_err}")
        raise

# ========== 验证评估（LLM 可覆盖，系统兜底）==========
# 如果 LLM 定义了 evaluate_model()，使用 LLM 的评估逻辑；否则使用系统默认指标
try:
    if 'evaluate_model' in globals():
        metrics = evaluate_model(model, X_val_fe, y_val)
    else:
        val_preds = model.predict(X_val_fe)
        from sklearn.metrics import f1_score
        metrics = {'val_accuracy': float(accuracy_score(y_val, val_preds)), 'val_f1_macro': float(f1_score(y_val, val_preds, average='macro'))}
except Exception as e:
    print(f"[EVAL_ERROR] {e}")
    metrics = {}
    # 尝试最基本的预测来兜底
    try:
        _pred = model.predict(X_val_fe)
        if task_type == "binary_classification" and hasattr(model, 'predict_proba'):
            _prob = model.predict_proba(X_val_fe)[:, 1]
            metrics = {'val_auc': float(roc_auc_score(y_val, _prob)), 'val_accuracy': float(accuracy_score(y_val, (_prob >= 0.5).astype(int)))}
        elif task_type == "multiclass_classification":
            metrics = {'val_accuracy': float(accuracy_score(y_val, _pred)), 'val_f1_macro': float(f1_score(y_val, _pred, average='macro'))}
        elif task_type in ("regression", "time_series_forecasting"):
            metrics = {'val_rmse': float(root_mean_squared_error(y_val, _pred)), 'val_mae': float(mean_absolute_error(y_val, _pred)), 'val_r2': float(r2_score(y_val, _pred))}
    except Exception as e2:
        print(f"[EVAL_FALLBACK_ERROR] {e2}")
        metrics = {}

# ========== 测试预测（系统保证格式）==========
# 注意：如果前面的代码（特征工程/model.fit）有 bug，这里会抛出异常
# 这是正确的行为——错误应该被暴露，让 DEBUG 循环去修复根因，而不是用假数据掩盖
try:
    test_probs_all = model.predict_proba(X_test_fe)
except Exception:
    test_probs_all = None
test_preds = model.predict(X_test_fe)
if '_label_encoder' in globals() and _label_encoder is not None:
    test_preds = _label_encoder.inverse_transform(test_preds)


result_df = pd.DataFrame({
    'id': test[id_col] if id_col in test.columns else range(len(test_preds)),
    'prediction': test_preds,
})
if test_probs_all is not None:
    for i, col in enumerate(test_probs_all.T):
        result_df[f'proba_{i}'] = col
result_df.to_csv('data/test_predictions.csv', index=False)

# ========== 模型保存（系统保证可序列化）==========
with open('data/best_model.pkl', 'wb') as f:
    dill.dump(model, f)

# ========== 输出指标（系统抓取）==========
print('METRICS_JSON_START')
print(json.dumps(metrics))
print('METRICS_JSON_END')

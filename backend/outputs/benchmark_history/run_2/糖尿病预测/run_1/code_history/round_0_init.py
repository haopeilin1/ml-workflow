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
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
import lightgbm as lgb

# 全局状态变量，用于存储训练时拟合的预处理参数
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    """
    df = df.copy()
    
    # 丢弃 id 列（如果存在）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 将医学上不可能的 0 值视为缺失值（Glucose, BloodPressure, SkinThickness, Insulin, BMI）
    zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_as_missing_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    # 分离特征和目标列（仅用于训练时的拟合）
    if mode == 'train':
        # 保存目标列
        if 'Outcome' in df.columns:
            y = df['Outcome'].copy()
        else:
            y = None
        
        # 选择数值特征列
        feature_cols = [col for col in df.columns if col != 'Outcome']
        
        # 拟合 imputer（中位数填充）
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df[feature_cols])
        PREPROCESS_STATE['imputer'] = imputer
        
        # 拟合 scaler
        scaler = StandardScaler()
        scaler.fit(imputer.transform(df[feature_cols]))
        PREPROCESS_STATE['scaler'] = scaler
        
        # 保存特征列顺序
        PREPROCESS_STATE['feature_cols'] = feature_cols
        
        # 应用预处理
        df_processed = pd.DataFrame(
            scaler.transform(imputer.transform(df[feature_cols])),
            columns=feature_cols,
            index=df.index
        )
        
        # 恢复目标列
        if y is not None:
            df_processed['Outcome'] = y.values
        
        return df_processed
    
    elif mode == 'test':
        # 从状态中读取已拟合的参数
        imputer = PREPROCESS_STATE.get('imputer')
        scaler = PREPROCESS_STATE.get('scaler')
        feature_cols = PREPROCESS_STATE.get('feature_cols', [col for col in df.columns if col != 'Outcome'])
        
        if imputer is None or scaler is None:
            raise ValueError("PREPROCESS_STATE 未初始化，请先运行 mode='train'")
        
        # 应用预处理
        df_processed = pd.DataFrame(
            scaler.transform(imputer.transform(df[feature_cols])),
            columns=feature_cols,
            index=df.index
        )
        
        # 恢复目标列（如果存在）
        if 'Outcome' in df.columns:
            df_processed['Outcome'] = df['Outcome'].values
        
        return df_processed
    
    return df


def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    df = df.copy()
    
    # 分离目标列
    if 'Outcome' in df.columns:
        y = df['Outcome'].copy()
        df = df.drop(columns=['Outcome'])
    else:
        y = None
    
    # 创建新特征
    # 1. BMI 分类特征（数值化）
    if 'BMI' in df.columns:
        # 注意：BMI 已经过标准化，但原始值在 preprocess 前已丢失
        # 由于 preprocess 已经标准化了所有特征，我们基于标准化后的值创建特征
        # 实际上更好的做法是在 preprocess 之前做特征工程，但这里我们基于标准化后的值
        pass  # 标准化后的 BMI 无法直接做分类，跳过
    
    # 2. 交互特征
    if 'Glucose' in df.columns and 'BMI' in df.columns:
        df['Glucose_BMI_interaction'] = df['Glucose'] * df['BMI']
    
    if 'Age' in df.columns and 'Pregnancies' in df.columns:
        df['Age_Pregnancies_interaction'] = df['Age'] * df['Pregnancies']
    
    if 'Glucose' in df.columns and 'Age' in df.columns:
        df['Glucose_Age_interaction'] = df['Glucose'] * df['Age']
    
    # 3. 多项式特征
    if 'Glucose' in df.columns:
        df['Glucose_squared'] = df['Glucose'] ** 2
    
    if 'BMI' in df.columns:
        df['BMI_squared'] = df['BMI'] ** 2
    
    if 'Insulin' in df.columns:
        df['Insulin_squared'] = df['Insulin'] ** 2
    
    # 4. 比率特征
    if 'Glucose' in df.columns and 'BMI' in df.columns:
        df['Glucose_BMI_ratio'] = df['Glucose'] / (df['BMI'] + 1e-8)
    
    if 'Insulin' in df.columns and 'Glucose' in df.columns:
        df['Insulin_Glucose_ratio'] = df['Insulin'] / (df['Glucose'] + 1e-8)
    
    # 5. 聚合统计特征（基于原始特征）
    # 计算每行的统计量
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        df['row_mean'] = df[numeric_cols].mean(axis=1)
        df['row_std'] = df[numeric_cols].std(axis=1)
        df['row_max'] = df[numeric_cols].max(axis=1)
        df['row_min'] = df[numeric_cols].min(axis=1)
    
    # 确保所有列都是数值类型
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = pd.factorize(df[col])[0]
    
    # 特征选择：使用 SelectKBest 选择 top-15 特征
    if y is not None:
        selector = SelectKBest(score_func=f_classif, k=min(15, df.shape[1]))
        X_selected = selector.fit_transform(df, y)
        selected_mask = selector.get_support()
        selected_cols = df.columns[selected_mask].tolist()
        df = df[selected_cols]
    
    return df


def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    # 计算类别权重（根据数据画像，正类比例约 0.338，负类约 0.662）
    scale_pos_weight = (1 - 0.3381) / 0.3381  # ≈ 1.96
    
    model = LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )
    
    return model
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'Outcome'
id_col = 'id'
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
        if hasattr(model, 'predict_proba'):
            val_probs = model.predict_proba(X_val_fe)[:, 1]
        else:
            val_probs = model.predict(X_val_fe).astype(float)
        val_preds = (val_probs >= 0.5).astype(int)
        metrics = {
            'val_auc': float(roc_auc_score(y_val, val_probs)),
            'val_accuracy': float(accuracy_score(y_val, val_preds))
        }
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
if hasattr(model, 'predict_proba'):
    test_probs = model.predict_proba(X_test_fe)[:, 1]
else:
    test_probs = model.predict(X_test_fe).astype(float)
test_preds = (test_probs >= 0.5).astype(int)


result_df = pd.DataFrame({
    'id': test[id_col] if id_col in test.columns else range(len(test_preds)),
    'prediction': test_preds,
})
result_df['probability'] = test_probs
result_df.to_csv('data/test_predictions.csv', index=False)

# ========== 模型保存（系统保证可序列化）==========
with open('data/best_model.pkl', 'wb') as f:
    dill.dump(model, f)

# ========== 输出指标（系统抓取）==========
print('METRICS_JSON_START')
print(json.dumps(metrics))
print('METRICS_JSON_END')

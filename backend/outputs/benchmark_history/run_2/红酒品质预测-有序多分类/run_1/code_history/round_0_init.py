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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# 全局状态存储（用于 preprocess 的 fit/transform 模式）
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    global PREPROCESS_STATE
    df = df.copy()
    
    # 1. 丢弃 id 列
    df = df.drop(columns=['id'], errors='ignore')
    
    # 2. 对 wine_type 进行数值编码（red=0, white=1）
    if 'wine_type' in df.columns:
        if mode == 'train':
            # 保存映射关系
            PREPROCESS_STATE['wine_type_mapping'] = {'red': 0, 'white': 1}
        df['wine_type'] = df['wine_type'].map(PREPROCESS_STATE['wine_type_mapping'])
        # 确保 wine_type 是数值类型
        df['wine_type'] = pd.to_numeric(df['wine_type'], errors='coerce')
    
    # 3. 对 chlorides 进行 log1p 变换以降低偏度
    if 'chlorides' in df.columns:
        # 确保 chlorides 是数值类型
        df['chlorides'] = pd.to_numeric(df['chlorides'], errors='coerce')
        # 处理可能的缺失值
        if df['chlorides'].isna().any():
            if mode == 'train':
                PREPROCESS_STATE['chlorides_median'] = df['chlorides'].median()
            df['chlorides'] = df['chlorides'].fillna(PREPROCESS_STATE.get('chlorides_median', 0.05))
        # 确保所有值为正（log1p 要求）
        df['chlorides'] = df['chlorides'].clip(lower=0)
        df['chlorides'] = np.log1p(df['chlorides'])
    
    # 4. 确保所有数值列类型正确（防止 object 类型混入）
    for col in df.columns:
        if col == 'quality':
            # quality 保持原始数值，不做变换
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif df[col].dtype == 'object':
            # 尝试转换为数值
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    '''
    df = df.copy()
    
    # 分离目标列（如果存在）
    target_col = 'quality'
    y = None
    if target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])
    
    # 1. 创建交互特征：alcohol 与 volatile acidity 的比值（酒体平衡度指标）
    if 'alcohol' in df.columns and 'volatile acidity' in df.columns:
        # 避免除零
        df['alcohol_volatile_acidity_ratio'] = df['alcohol'] / (df['volatile acidity'].clip(lower=0.001))
    
    # 2. 创建聚合特征：free sulfur dioxide 与 total sulfur dioxide 的比例
    if 'free sulfur dioxide' in df.columns and 'total sulfur dioxide' in df.columns:
        df['free_total_sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'].clip(lower=1))
        # 限制在合理范围 [0, 1]
        df['free_total_sulfur_ratio'] = df['free_total_sulfur_ratio'].clip(0, 1)
    
    # 3. wine_type 与关键特征的交互项
    if 'wine_type' in df.columns:
        # wine_type * residual sugar（红白葡萄酒残糖量差异显著）
        if 'residual sugar' in df.columns:
            df['wine_type_residual_sugar'] = df['wine_type'] * df['residual sugar']
        
        # wine_type * total sulfur dioxide
        if 'total sulfur dioxide' in df.columns:
            df['wine_type_total_sulfur'] = df['wine_type'] * df['total sulfur dioxide']
        
        # wine_type * alcohol
        if 'alcohol' in df.columns:
            df['wine_type_alcohol'] = df['wine_type'] * df['alcohol']
    
    # 4. 对偏态特征进行 log1p 变换（可选，提升特征分布）
    skewed_cols = ['fixed acidity', 'volatile acidity', 'sulphates', 'residual sugar', 
                   'free sulfur dioxide', 'total sulfur dioxide']
    for col in skewed_cols:
        if col in df.columns:
            # 确保数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 只对正值做 log1p
            if (df[col] >= 0).all():
                df[col + '_log'] = np.log1p(df[col])
    
    # 5. 特征名清洗：替换特殊字符（防止 LightGBM 报 JSON 特殊字符错误）
    import re
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    # 6. 确保所有列都是数值类型
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].dtype in ['float64', 'int64'] else 0)
    
    # 如果目标列存在，加回去（但 feature_engineering 应该返回不含目标列的 X）
    # 根据系统约定，feature_engineering 返回不含目标列的 DataFrame
    return df


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    '''
    import lightgbm as lgb
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.preprocessing import StandardScaler
    
    # LightGBM 多分类模型
    # quality 取值 3-9，共 7 个类别
    # 使用 class_weight='balanced' 处理类别不平衡
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=7,
        class_weight='balanced',
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    # 使用 imblearn Pipeline 嵌入 SMOTE（只在训练时应用）
    # SMOTE 对少数类进行过采样
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('model', model)
    ])
    
    return pipeline


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    返回一个 dict，键名以 val_ 开头。
    '''
    # 获取预测结果
    val_preds = model.predict(X_val)
    
    # 计算 F1-macro（主要指标）
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    
    # 计算辅助指标
    val_accuracy = accuracy_score(y_val, val_preds)
    val_precision_macro = precision_score(y_val, val_preds, average='macro', zero_division=0)
    val_recall_macro = recall_score(y_val, val_preds, average='macro', zero_division=0)
    
    # 计算每个类别的 F1-score
    val_f1_per_class = f1_score(y_val, val_preds, average=None, zero_division=0)
    
    # 构建返回字典
    result = {
        'val_f1_macro': float(val_f1_macro),
        'val_accuracy': float(val_accuracy),
        'val_precision_macro': float(val_precision_macro),
        'val_recall_macro': float(val_recall_macro),
    }
    
    # 添加每个类别的 F1-score
    unique_classes = sorted(np.unique(y_val))
    for i, cls in enumerate(unique_classes):
        if i < len(val_f1_per_class):
            result[f'val_f1_class_{int(cls)}'] = float(val_f1_per_class[i])
    
    return result
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'quality'
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

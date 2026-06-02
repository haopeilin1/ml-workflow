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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

# 全局状态：用于在 preprocess 和 feature_engineering 之间传递信息
PREPROCESS_STATE = {
    'scaler': None,
    'target_col': 'species',
    'id_col': 'id',
    'feature_cols': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
}


def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    # 1. 丢弃 id 列（must_do #1）
    if PREPROCESS_STATE['id_col'] in df.columns:
        df = df.drop(columns=[PREPROCESS_STATE['id_col']])
    
    # 2. 确保数值列类型正确（预防 LIGHTGBM_STRING_DTYPE 和 ONEHOTENCODER_OBJECT_OUTPUT）
    for col in PREPROCESS_STATE['feature_cols']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. 检查缺失值（数据画像显示无缺失，但做防御性处理）
    # 数值列用 median 填充（极小概率出现）
    for col in PREPROCESS_STATE['feature_cols']:
        if col in df.columns and df[col].isnull().any():
            if mode == 'train':
                df[col] = df[col].fillna(df[col].median())
            else:
                # test 模式：用训练集的 median 填充（但这里没有保存，因为数据无缺失）
                # 安全起见，用当前列的 median
                df[col] = df[col].fillna(df[col].median())
    
    # 4. 目标列保留原始格式（分类目标列保护 - 绝对红线）
    # 不做任何编码，系统会自动处理
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    
    包含：
    1. StandardScaler 标准化四个数值特征
    2. 创建交互特征 petal_area = petal_length * petal_width
    '''
    target_col = PREPROCESS_STATE['target_col']
    feature_cols = PREPROCESS_STATE['feature_cols']
    
    # 分离目标列（如果存在）
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col], errors='ignore')
    else:
        X = df.copy()
    
    # 确保特征列存在
    available_features = [c for c in feature_cols if c in X.columns]
    X = X[available_features].copy()
    
    # 创建交互特征：花瓣面积（must_do 中建议的特征工程）
    if 'petal_length' in X.columns and 'petal_width' in X.columns:
        X['petal_area'] = X['petal_length'] * X['petal_width']
    
    # StandardScaler 标准化（must_do #2）
    # 在训练时 fit，在测试时 transform
    if PREPROCESS_STATE['scaler'] is None:
        # 训练模式：fit 并保存 scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        PREPROCESS_STATE['scaler'] = scaler
    else:
        # 测试模式：使用已保存的 scaler
        X_scaled = PREPROCESS_STATE['scaler'].transform(X)
    
    # 重建 DataFrame，保持列名（Pipeline 列名保护）
    X_result = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_result


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    
    使用 LogisticRegression（基线模型）：
    - solver='lbfgs'（sklearn 1.6+ 兼容，支持多分类）
    - max_iter=1000（确保收敛）
    - 不使用 multi_class 参数（sklearn 1.6+ 已移除）
    - 复杂度低，适合 120 条极小样本
    '''
    model = LogisticRegression(
        solver='lbfgs',       # sklearn 1.6+ 兼容，支持多分类
        max_iter=1000,        # 确保收敛
        random_state=42,      # 可复现性
        C=1.0                 # 默认正则化强度，防止过拟合
    )
    return model


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    返回一个 dict，键名以 val_ 开头。
    计算 Accuracy、F1-macro、F1-weighted 和混淆矩阵。
    '''
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    val_preds = model.predict(X_val)
    
    # 计算指标
    val_accuracy = accuracy_score(y_val, val_preds)
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    val_f1_weighted = f1_score(y_val, val_preds, average='weighted')
    val_cm = confusion_matrix(y_val, val_preds)
    
    # 打印混淆矩阵（must_do #5）
    print(f'[INFO] Confusion Matrix:\n{val_cm}')
    print(f'[INFO] Validation Accuracy: {val_accuracy:.4f}')
    print(f'[INFO] Validation F1-macro: {val_f1_macro:.4f}')
    print(f'[INFO] Validation F1-weighted: {val_f1_weighted:.4f}')
    
    return {
        'val_accuracy': float(val_accuracy),
        'val_f1_macro': float(val_f1_macro),
        'val_f1_weighted': float(val_f1_weighted)
    }
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'species'
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

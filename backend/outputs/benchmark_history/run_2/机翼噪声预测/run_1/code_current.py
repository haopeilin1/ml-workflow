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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb

# 全局状态存储
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    """
    global PREPROCESS_STATE
    
    # 复制数据避免修改原始数据
    df = df.copy()
    
    # 丢弃 id 列（如果存在）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 分离特征列和目标列
    feature_cols = ['f', 'alpha', 'c', 'U_infinity', 'delta']
    target_col = 'SSPL'
    
    # 确保特征列都存在
    available_features = [col for col in feature_cols if col in df.columns]
    
    if mode == 'train':
        # 拟合 StandardScaler
        scaler = StandardScaler()
        scaler.fit(df[available_features])
        PREPROCESS_STATE['scaler'] = scaler
    else:
        # 应用已拟合的 scaler
        scaler = PREPROCESS_STATE.get('scaler')
        if scaler is None:
            raise ValueError("PREPROCESS_STATE 中未找到 scaler，请先运行 mode='train'")
    
    # 应用标准化
    df[available_features] = scaler.transform(df[available_features])
    
    return df

def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    # 复制数据
    df = df.copy()
    
    # 特征列列表
    feature_cols = ['f', 'alpha', 'c', 'U_infinity', 'delta']
    target_col = 'SSPL'
    
    # 确保特征列都存在
    available_features = [col for col in feature_cols if col in df.columns]
    
    # 创建交叉特征（物理意义相关）
    df['f_delta'] = df['f'] * df['delta']
    df['alpha_U_infinity'] = df['alpha'] * df['U_infinity']
    df['f_c'] = df['f'] * df['c']  # 与 Strouhal 数相关
    df['alpha_delta'] = df['alpha'] * df['delta']  # 攻角与位移厚度交互
    df['U_infinity_delta'] = df['U_infinity'] * df['delta']  # 速度与位移厚度交互
    
    # 创建比值特征
    df['U_infinity_over_c'] = df['U_infinity'] / (df['c'] + 1e-8)  # 避免除零
    df['f_over_U_infinity'] = df['f'] / (df['U_infinity'] + 1e-8)  # 频率/速度，与波长相关
    df['c_over_delta'] = df['c'] / (df['delta'] + 1e-8)  # 弦长/位移厚度，与相对厚度相关
    
    # 创建多项式特征
    df['f_squared'] = df['f'] ** 2
    df['delta_squared'] = df['delta'] ** 2
    df['alpha_squared'] = df['alpha'] ** 2
    df['U_infinity_squared'] = df['U_infinity'] ** 2
    
    # 创建对数变换特征（声压级常与对数相关）
    df['log_f'] = np.log1p(df['f'])
    df['log_delta'] = np.log1p(df['delta'] * 1000)  # 放大后取对数，避免负值
    df['log_U_infinity'] = np.log1p(df['U_infinity'])
    
    # 选择所有特征列（包括新创建的）
    all_feature_cols = available_features + [
        'f_delta', 'alpha_U_infinity', 'f_c', 'alpha_delta', 'U_infinity_delta',
        'U_infinity_over_c', 'f_over_U_infinity', 'c_over_delta',
        'f_squared', 'delta_squared', 'alpha_squared', 'U_infinity_squared',
        'log_f', 'log_delta', 'log_U_infinity'
    ]
    
    # 分离目标列
    if target_col in df.columns:
        X = df[all_feature_cols]
    else:
        X = df[all_feature_cols]
    
    return X

def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    model = LGBMRegressor(
        objective='regression',
        num_leaves=31,
        max_depth=8,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=0.8,
        subsample_freq=1,
        feature_fraction=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,  # 抑制训练日志
        min_split_gain=0.0
    )
    
    return model

def evaluate_model(model, X_val, y_val):
    """
    自定义验证集评估指标。
    返回一个 dict，键名以 val_ 开头。
    """
    val_preds = model.predict(X_val)
    
    rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    mae = float(mean_absolute_error(y_val, val_preds))
    r2 = float(r2_score(y_val, val_preds))
    
    return {
        'val_rmse': rmse,
        'val_mae': mae,
        'val_r2': r2
    }
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'SSPL'
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
        metrics = {
            'val_rmse': float(root_mean_squared_error(y_val, val_preds)),
            'val_mae': float(mean_absolute_error(y_val, val_preds)),
            'val_r2': float(r2_score(y_val, val_preds))
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
test_preds = model.predict(X_test_fe)
test_probs = test_preds

# 【系统】目标变换逆变换（如 LLM 在 PREPROCESS_STATE 中声明了 target_transform）
# 兼容多种常见键名: target_transform, target_log_transformer
_test_tform = PREPROCESS_STATE.get('target_transform') or PREPROCESS_STATE.get('target_log_transformer')
if _test_tform == 'log1p':
    test_preds = np.expm1(test_preds)
elif _test_tform == 'log':
    test_preds = np.exp(test_preds)
elif _test_tform == 'sqrt':
    test_preds = np.square(test_preds)


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

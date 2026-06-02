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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# 全局状态变量
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    - 丢弃 id 列
    - 对偏度大的特征做对数变换（f, delta）
    - 数值缩放（StandardScaler）在特征工程后统一进行
    - 训练集拟合 scaler，验证/测试集应用
    """
    df = df.copy()
    
    # 丢弃 id 列（如果存在）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 定义数值特征列
    feature_cols = ['f', 'alpha', 'c', 'U_infinity', 'delta']
    
    # 对偏度大的特征做对数变换（在标准化之前）
    # f 的偏度=2.21，delta 的偏度=1.72，对数变换可缓解右偏
    if mode == 'train':
        # 训练集：拟合对数变换参数
        f_log_offset = 0  # f 最小值为200，无需偏移
        delta_log_offset = 0  # delta 最小值为0.0004，无需偏移
        
        # 保存偏移量
        PREPROCESS_STATE['f_log_offset'] = f_log_offset
        PREPROCESS_STATE['delta_log_offset'] = delta_log_offset
        
        # 应用对数变换
        df['f_log'] = np.log1p(df['f'] - f_log_offset) if f_log_offset > 0 else np.log(df['f'])
        df['delta_log'] = np.log1p(df['delta'] - delta_log_offset) if delta_log_offset > 0 else np.log(df['delta'])
        
        # 拟合 scaler（在特征工程后使用，这里先保存原始数据）
        # 注意：标准化将在 feature_engineering 之后进行
        # 这里只做对数变换，不做标准化
        scaler = StandardScaler()
        PREPROCESS_STATE['scaler'] = scaler
    else:
        # 验证集或测试集：应用已拟合的对数变换参数
        f_log_offset = PREPROCESS_STATE.get('f_log_offset', 0)
        delta_log_offset = PREPROCESS_STATE.get('delta_log_offset', 0)
        
        # 应用对数变换
        if 'f' in df.columns:
            df['f_log'] = np.log1p(df['f'] - f_log_offset) if f_log_offset > 0 else np.log(df['f'])
        if 'delta' in df.columns:
            df['delta_log'] = np.log1p(df['delta'] - delta_log_offset) if delta_log_offset > 0 else np.log(df['delta'])
    
    return df


def feature_engineering(df):
    """
    特征工程。
    - 创建有物理意义的交叉特征
    - 标准化所有特征
    - 丢弃目标列 SSPL
    - 返回纯数值特征矩阵
    """
    df = df.copy()
    
    # 创建交叉特征（物理意义明确）
    # 频率与速度的交互（与涡脱落频率相关）
    df['f_x_U'] = df['f'] * df['U_infinity']
    # 攻角与位移厚度的交互（与分离泡相关）
    df['alpha_x_delta'] = df['alpha'] * df['delta']
    # 速度与弦长的比值（与雷诺数相关）
    df['U_div_c'] = df['U_infinity'] / (df['c'] + 1e-8)
    # 频率与位移厚度的交互（与边界层特征相关）
    df['f_x_delta'] = df['f'] * df['delta']
    # 频率与弦长的比值（与斯特劳哈尔数相关）
    df['f_div_c'] = df['f'] / (df['c'] + 1e-8)
    # 攻角与速度的交互
    df['alpha_x_U'] = df['alpha'] * df['U_infinity']
    # 位移厚度与弦长的比值
    df['delta_div_c'] = df['delta'] / (df['c'] + 1e-8)
    
    # 使用对数变换后的特征创建交叉特征
    if 'f_log' in df.columns and 'delta_log' in df.columns:
        df['f_log_x_delta_log'] = df['f_log'] * df['delta_log']
        df['f_log_x_U'] = df['f_log'] * df['U_infinity']
        df['delta_log_x_alpha'] = df['delta_log'] * df['alpha']
    
    # 丢弃目标列（如果存在）
    if 'SSPL' in df.columns:
        df = df.drop(columns=['SSPL'])
    
    # 确保所有列为数值类型
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 填充可能出现的 NaN（如除零）
    df = df.fillna(0)
    
    # 标准化所有特征
    scaler = PREPROCESS_STATE.get('scaler')
    if scaler is not None:
        # 检查 scaler 是否已拟合
        if hasattr(scaler, 'mean_'):
            # 已拟合，应用变换
            df[df.columns] = scaler.transform(df[df.columns])
        else:
            # 未拟合，拟合并变换
            df[df.columns] = scaler.fit_transform(df[df.columns])
    
    return df


def build_model():
    """
    模型构建和超参数设置。
    使用 LightGBM Regressor，配合早停防止过拟合。
    """
    model = lgb.LGBMRegressor(
        objective='regression',
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
        verbose=-1  # 控制日志输出
    )
    return model


def evaluate_model(model, X_val, y_val):
    """
    自定义验证集评估指标。
    返回 RMSE, MAE, R² 等指标。
    """
    y_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))
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

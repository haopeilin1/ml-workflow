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
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score

# 全局状态存储（用于 preprocess 的 fit/transform 模式）
PREPROCESS_STATE = {
    'fitted': False,
    'amount_median': None,
    'hourly_avg_amount': None,
    'time_min': None,
    'time_max': None,
}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    - 丢弃 id 列，保留 Time、Transaction_Amount、feat1-feat28 及目标列
    - 对 Transaction_Amount 应用 log1p 变换
    - 将 Time 列按秒转换为小时（Time/3600）
    - 处理可能的缺失值
    '''
    target_col = 'IsFraud'
    
    # 丢弃 id 列（如果存在）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    df = df.copy()
    
    # 处理缺失值：数值列用中位数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == target_col:
            continue
        # 确保数值类型
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if mode == 'train':
            PREPROCESS_STATE[f'{col}_median'] = df[col].median()
        median_val = PREPROCESS_STATE.get(f'{col}_median', df[col].median())
        df[col] = df[col].fillna(median_val)
    
    # 对 Transaction_Amount 应用 log1p 变换（压缩极端值）
    if 'Transaction_Amount' in df.columns:
        df['Transaction_Amount'] = df['Transaction_Amount'].clip(lower=0)
        df['Transaction_Amount'] = np.log1p(df['Transaction_Amount'])
    
    # Time 转换为小时
    if 'Time' in df.columns:
        df['Time'] = df['Time'] / 3600.0
    
    PREPROCESS_STATE['fitted'] = True
    return df


def feature_engineering(df):
    '''
    特征工程。
    - 基于 Time 创建时间周期特征（小时、日周期正弦/余弦）
    - 创建 Transaction_Amount 的统计特征
    - 保留所有原始 PCA 特征
    返回特征矩阵（不含目标列）。
    '''
    target_col = 'IsFraud'
    df = df.copy()
    
    # 时间特征工程
    if 'Time' in df.columns:
        # 小时特征（0-23）
        df['Time_hour'] = (df['Time'].astype(int) % 24).astype(int)
        # 日周期正弦/余弦变换（捕获周期性，周期=24小时）
        df['Time_sin'] = np.sin(2 * np.pi * df['Time'] / 24.0)
        df['Time_cos'] = np.cos(2 * np.pi * df['Time'] / 24.0)
        # 周周期正弦/余弦变换（周期=168小时=7天）
        df['Time_sin_week'] = np.sin(2 * np.pi * df['Time'] / 168.0)
        df['Time_cos_week'] = np.cos(2 * np.pi * df['Time'] / 168.0)
    
    # 交易金额的统计特征（按小时分组）
    if 'Transaction_Amount' in df.columns and 'Time_hour' in df.columns:
        if PREPROCESS_STATE.get('hourly_avg_amount') is None:
            # 训练模式：计算并保存统计量
            hourly_avg = df.groupby('Time_hour')['Transaction_Amount'].mean().to_dict()
            PREPROCESS_STATE['hourly_avg_amount'] = hourly_avg
        hourly_avg = PREPROCESS_STATE.get('hourly_avg_amount', {})
        
        # 映射每小时的平均交易金额
        df['hourly_avg_amount'] = df['Time_hour'].map(hourly_avg)
        # 填充可能缺失的小时
        if df['hourly_avg_amount'].isna().any():
            global_avg = df['Transaction_Amount'].mean()
            df['hourly_avg_amount'] = df['hourly_avg_amount'].fillna(global_avg)
        
        # 交易金额与小时均值的偏差
        df['amount_deviation'] = df['Transaction_Amount'] - df['hourly_avg_amount']
    
    # 移除目标列（如果存在）
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    return df


def build_model():
    '''
    模型构建和超参数设置。
    使用 LightGBM，设置 scale_pos_weight 处理极度类别不平衡。
    返回 sklearn 兼容的模型对象。
    '''
    import lightgbm as lgb
    
    # 计算 scale_pos_weight：负类数/正类数 ≈ 554.56
    scale_pos_weight = 19964.0 / 36.0
    
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        scale_pos_weight=scale_pos_weight,
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=63,
        min_child_samples=50,
        min_child_weight=1e-3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    
    return model


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    在验证集上搜索最优分类阈值（以 F1-Score 最大化为目标），
    同时计算 ROC-AUC、AP、Recall、Precision、F1-Score。
    '''
    # 获取预测概率
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # 计算 ROC-AUC
    val_auc = roc_auc_score(y_val, y_proba)
    
    # 计算 Average Precision
    val_ap = average_precision_score(y_val, y_proba)
    
    # 搜索最优阈值（以 F1-Score 最大化为目标）
    thresholds = np.arange(0.001, 0.5, 0.001)
    best_f1 = 0.0
    best_threshold = 0.5
    best_recall = 0.0
    best_precision = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_recall = recall_score(y_val, y_pred, zero_division=0)
            best_precision = precision_score(y_val, y_pred, zero_division=0)
    
    # 使用最优阈值计算最终指标
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    val_f1 = f1_score(y_val, y_pred_optimal, zero_division=0)
    val_recall = recall_score(y_val, y_pred_optimal, zero_division=0)
    val_precision = precision_score(y_val, y_pred_optimal, zero_division=0)
    
    # 存储最优阈值供预测使用
    PREPROCESS_STATE['best_threshold'] = best_threshold
    
    return {
        'val_roc_auc': float(val_auc),
        'val_ap': float(val_ap),
        'val_f1': float(val_f1),
        'val_recall': float(val_recall),
        'val_precision': float(val_precision),
        'val_best_threshold': float(best_threshold),
    }
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'IsFraud'
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

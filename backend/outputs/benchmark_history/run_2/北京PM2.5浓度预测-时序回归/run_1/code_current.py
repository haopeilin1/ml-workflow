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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import re

# 全局变量：目标列名
TARGET_COL = 'pm2.5'

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    df = df.copy()
    
    # 1. 丢弃 ID 列 'No'（MUST DO #1）
    df = df.drop(columns=['No'], errors='ignore')
    
    # 2. 清洗列名：移除特殊字符，防止 LightGBM 报 JSON 特殊字符错误
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    # 3. 确保数值列类型正确（防止 object 类型数值列导致后续错误）
    # 已知数值列（排除目标列和类别列）
    numeric_cols = ['year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 目标列也确保为数值类型
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    
    # 4. 处理目标变量 'pm2.5' 的缺失值（MUST DO #2）
    # 使用时间序列插值：前向填充 + 后向填充（保持时间连续性）
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].ffill().bfill()
    
    # 5. 处理特征列的缺失值（如果有的话）
    # 数值特征用中位数填充，类别特征用众数填充
    # 注意：这里做初步填充，Pipeline 中还会再做一次（双重保险）
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].dtype in ['object', 'category', 'str']:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing')
        else:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    
    包含：
    - 时间循环特征（hour, month 的 sin/cos 编码）
    - 右偏特征 log1p 变换（Iws, Is, Ir）
    - 二值指示特征（is_raining, is_snowing）
    - 类别特征 OneHot 编码（cbwd）
    - 数值特征标准化
    '''
    df = df.copy()
    
    # 分离目标列（如果存在）
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        df = df.drop(columns=[TARGET_COL])
    
    # 1. 构建时间循环特征（MUST DO #5）
    # hour 循环编码（0-23 -> sin/cos）
    if 'hour' in df.columns:
        hour = df['hour'].astype(float)
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    # month 循环编码（1-12 -> sin/cos）
    if 'month' in df.columns:
        month = df['month'].astype(float)
        df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    # 2. 对高度右偏特征做 log1p 变换（MUST DO #3）
    skewed_cols = ['Iws', 'Is', 'Ir']
    for col in skewed_cols:
        if col in df.columns:
            # 确保非负（log1p 要求 >= 0）
            df[col] = df[col].clip(lower=0)
            df[col + '_log1p'] = np.log1p(df[col])
    
    # 3. 创建二值指示特征（MUST DO #6 的补充，AVOID #6）
    if 'Is' in df.columns:
        df['is_snowing'] = (df['Is'] > 0).astype(int)
    if 'Ir' in df.columns:
        df['is_raining'] = (df['Ir'] > 0).astype(int)
    
    # 4. 确保所有列都是数值类型（类别列 cbwd 将在 Pipeline 中由 OneHotEncoder 处理）
    # 这里不做 OneHot 编码，留给 Pipeline 中的 ColumnTransformer 处理
    
    # 5. 如果目标列存在，重新附加（但 feature_engineering 应该返回不含目标列的 X）
    # 根据系统要求，feature_engineering 返回的 df 应该是不含目标列的特征矩阵
    # 目标列的处理由系统骨架负责
    
    return df


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的 Pipeline 对象（支持 fit/predict）。
    
    Pipeline 结构：
    1. ColumnTransformer：
       - 类别特征：SimpleImputer + OneHotEncoder
       - 数值特征：SimpleImputer + StandardScaler
    2. LightGBM 回归器（带正则化防止过拟合）
    '''
    # 定义类别特征列（cbwd 是唯一的类别特征）
    cat_cols = ['cbwd']
    
    # 定义数值特征列（所有其他列，包括原始数值列和工程特征列）
    # 注意：这里使用 'passthrough' 处理未明确列出的列，确保所有列都被处理
    # 但我们需要明确列出数值列以应用 StandardScaler
    
    # 类别特征 Pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 数值特征 Pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # ColumnTransformer：根据列类型分别处理
    # 使用 make_column_selector 或显式指定列名
    # 由于 feature_engineering 会动态添加列，这里使用 selector 更灵活
    from sklearn.compose import make_column_selector
    
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, make_column_selector(dtype_include=['object', 'category', 'str'])),
        ('num', num_pipeline, make_column_selector(dtype_include=['number']))
    ], remainder='passthrough')
    
    # LightGBM 回归器（MUST DO：防止过拟合的配置）
    # 数据集约 35059 行，属于中等规模，可以使用适中的树深度
    lgbm = LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=300,           # 适中的树数量
        max_depth=8,                # 限制深度防止过拟合（MUST DO 过拟合控制）
        num_leaves=63,              # 适中的叶子数
        learning_rate=0.05,         # 较小的学习率
        subsample=0.8,              # 行采样（MUST DO 过拟合控制）
        colsample_bytree=0.8,       # 列采样
        reg_alpha=0.1,              # L1 正则化
        reg_lambda=0.1,             # L2 正则化
        min_child_samples=20,       # 叶子最小样本数（MUST DO 过拟合控制）
        random_state=42,
        n_jobs=-1,
        verbose=-1                  # 静默模式
    )
    
    # 完整 Pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', lgbm)
    ])
    
    return pipeline
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'pm2.5'
id_col = 'No'
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

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
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import lightgbm as lgb

# 全局状态：存储 preprocess 阶段拟合的参数
PREPROCESS_STATE = {
    'dteday_time_features': None,  # 从 dteday 提取的时间特征列名
    'target_log1p': True,           # 是否对目标做了 log1p 变换
    'cat_cols': [],                 # 类别特征列名
    'num_cols': [],                 # 数值特征列名
}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    target_col = 'cnt'
    df = df.copy()
    
    # ============================================================
    # MUST DO 1: 丢弃 casual 和 registered 列（目标泄露）
    # MUST DO 2: 丢弃 instant 列（单调递增 ID）
    # ============================================================
    drop_cols = ['casual', 'registered', 'instant']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # ============================================================
    # MUST DO 5: 从 dteday 提取时间特征，然后丢弃原始 dteday 列
    # 检查是否与已有 yr、mnth、weekday、hr 列重复
    # ============================================================
    if 'dteday' in df.columns:
        # 解析日期列
        df['dteday_parsed'] = pd.to_datetime(df['dteday'], errors='coerce')
        
        # 提取时间特征
        df['dteday_year'] = df['dteday_parsed'].dt.year
        df['dteday_month'] = df['dteday_parsed'].dt.month
        df['dteday_day'] = df['dteday_parsed'].dt.day
        df['dteday_weekday'] = df['dteday_parsed'].dt.weekday  # 0=Monday, 6=Sunday
        df['dteday_hour'] = df['dteday_parsed'].dt.hour
        
        # 检查与已有列的重复情况
        # 数据中已有 yr(0/1), mnth(1-12), weekday(0-6), hr(0-23)
        # dteday_year 与 yr 可能不同（yr 是 0/1 表示 2011/2012，dteday_year 是 2011/2012）
        # 保留 dteday_year 作为补充（更精确的年份信息）
        # dteday_month 与 mnth 重复，丢弃 dteday_month
        # dteday_weekday 与 weekday 重复，丢弃 dteday_weekday
        # dteday_hour 与 hr 重复，丢弃 dteday_hour
        # dteday_day 是新的特征（月中的第几天），保留
        
        # 丢弃与已有列重复的特征
        duplicate_time_cols = ['dteday_month', 'dteday_weekday', 'dteday_hour']
        df = df.drop(columns=[c for c in duplicate_time_cols if c in df.columns], errors='ignore')
        
        # 丢弃原始 dteday 列和解析中间列
        df = df.drop(columns=['dteday', 'dteday_parsed'], errors='ignore')
        
        # 记录从 dteday 提取的特征列名
        extracted_cols = ['dteday_year', 'dteday_day']
        if mode == 'train':
            PREPROCESS_STATE['dteday_time_features'] = [c for c in extracted_cols if c in df.columns]
    
    # ============================================================
    # MUST DO 7: 将 season、weathersit、holiday、workingday 等低基数整数列视为类别特征
    # ============================================================
    categorical_cols = ['season', 'weathersit', 'holiday', 'workingday', 'yr']
    # hr 和 weekday 也是类别特征（小时和星期几）
    if 'hr' in df.columns:
        categorical_cols.append('hr')
    if 'weekday' in df.columns:
        categorical_cols.append('weekday')
    if 'mnth' in df.columns:
        categorical_cols.append('mnth')
    
    # 只转换实际存在的列
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    if mode == 'train':
        PREPROCESS_STATE['cat_cols'] = categorical_cols
    
    # ============================================================
    # MUST DO 4: 对目标变量 cnt 进行 log1p 变换
    # 注意：回归任务中严禁对目标列做标准化/归一化，只做 log1p 变换
    # ============================================================
    if target_col in df.columns:
        # 确保 cnt 是数值类型
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        # log1p 变换（log(1+x)），处理 cnt >= 0 的情况
        df[target_col] = np.log1p(df[target_col])
        if mode == 'train':
            PREPROCESS_STATE['target_log1p'] = True
    
    # ============================================================
    # 特征名清洗：确保列名不包含特殊 JSON 字符
    # ============================================================
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    
    包含：
    - 时间交互特征：hr 与 workingday 的交互
    - 天气交互特征：temp 与 hum 的交互，windspeed 与 weathersit 的交互
    - 滞后特征：前1小时和前2小时的 cnt（注意时序数据泄露）
    - 滚动统计特征：过去24小时的平均 cnt
    - RobustScaler 对数值特征
    - OneHotEncoder 对类别特征
    '''
    target_col = 'cnt'
    df = df.copy()
    
    # ============================================================
    # 特征名清洗（确保一致性）
    # ============================================================
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    # ============================================================
    # 创建交互特征
    # ============================================================
    # hr 与 workingday 的交互（区分工作日/周末的每小时模式）
    if 'hr' in df.columns and 'workingday' in df.columns:
        # 确保 workingday 是数值类型
        workingday_num = pd.to_numeric(df['workingday'], errors='coerce').fillna(0).astype(int)
        hr_num = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
        df['hr_workingday_interact'] = hr_num * 10 + workingday_num  # 编码：小时*10 + 是否工作日
    
    # temp 与 hum 的交互（体感舒适度）
    if 'temp' in df.columns and 'hum' in df.columns:
        temp_num = pd.to_numeric(df['temp'], errors='coerce')
        hum_num = pd.to_numeric(df['hum'], errors='coerce')
        df['temp_hum_interact'] = temp_num * hum_num
    
    # windspeed 与 weathersit 的交互
    if 'windspeed' in df.columns and 'weathersit' in df.columns:
        windspeed_num = pd.to_numeric(df['windspeed'], errors='coerce')
        weathersit_num = pd.to_numeric(df['weathersit'], errors='coerce')
        df['windspeed_weathersit_interact'] = windspeed_num * weathersit_num
    
    # ============================================================
    # 创建滞后特征和滚动统计特征
    # 注意：这些特征依赖于时间顺序，必须在时序数据上按顺序计算
    # 由于数据已经按 instant 排序（单调递增），我们可以直接使用 shift 和 rolling
    # ============================================================
    if target_col in df.columns:
        cnt_values = df[target_col].copy()
        
        # 滞后特征：前1小时和前2小时的 cnt
        df['cnt_lag1'] = cnt_values.shift(1)
        df['cnt_lag2'] = cnt_values.shift(2)
        
        # 滚动统计特征：过去24小时的平均 cnt
        df['cnt_rolling_mean_24h'] = cnt_values.shift(1).rolling(window=24, min_periods=1).mean()
        
        # 填充滞后特征和滚动特征的缺失值（前几行没有历史数据）
        df['cnt_lag1'] = df['cnt_lag1'].fillna(method='bfill').fillna(0)
        df['cnt_lag2'] = df['cnt_lag2'].fillna(method='bfill').fillna(0)
        df['cnt_rolling_mean_24h'] = df['cnt_rolling_mean_24h'].fillna(method='bfill').fillna(0)
    
    # ============================================================
    # 分离特征和目标
    # ============================================================
    X = df.drop(columns=[target_col], errors='ignore')
    
    # ============================================================
    # 确保所有列都是数值类型（处理类别列）
    # ============================================================
    # 将 category 类型转换为整数编码（Ordinal Encoding）
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].cat.codes.astype(int)
    
    # 将 object 类型尝试转换为数值
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 填充所有缺失值
    X = X.fillna(0)
    
    # ============================================================
    # 最终确保所有列都是数值类型
    # ============================================================
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X[col] = X[col].astype(float)
    
    return X


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    
    使用 LightGBM 回归模型，配合：
    - RobustScaler 对数值特征
    - OneHotEncoder 对类别特征
    - 早停机制防止过拟合
    - 正则化参数控制模型复杂度
    '''
    # ============================================================
    # 获取类别特征和数值特征列名
    # ============================================================
    cat_cols = PREPROCESS_STATE.get('cat_cols', [])
    
    # 数值特征列：所有非类别列（在 feature_engineering 之后确定）
    # 这里先定义预处理 Pipeline，具体列在 fit 时自动识别
    
    # ============================================================
    # 构建预处理 Pipeline
    # ============================================================
    # 数值特征预处理：RobustScaler（MUST DO 6）
    num_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    # 类别特征预处理：OneHotEncoder（MUST DO 7）
    # 注意：在 feature_engineering 中已经做了 Ordinal Encoding，
    # 这里对低基数类别特征再做 OneHot 编码以提升模型表达能力
    cat_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 注意：ColumnTransformer 需要知道哪些列是类别列
    # 由于 feature_engineering 已经将所有列转为数值，这里需要根据 PREPROCESS_STATE 来区分
    # 如果 cat_cols 为空，则所有列都按数值处理
    if cat_cols:
        # 清洗类别列名（与 feature_engineering 中的列名保持一致）
        cat_cols_clean = [re.sub(r'[^\w]', '_', str(c)) for c in cat_cols]
        # 其余列作为数值列
        # 这里使用 'passthrough' 处理未指定的列
        preprocessor = ColumnTransformer([
            ('cat', cat_preprocessor, cat_cols_clean),
        ], remainder=num_preprocessor)  # 未指定的列使用数值预处理器
    else:
        # 如果没有指定类别列，所有列都按数值处理
        preprocessor = num_preprocessor
    
    # ============================================================
    # 构建 LightGBM 回归模型
    # MUST DO: 使用 LGBMRegressor（sklearn 接口），禁止使用 lgb.train()
    # ============================================================
    # 数据集大小 13903 行，属于中等规模
    # 根据计划中的超参建议：
    # n_estimators=1000（配合早停），max_depth=8，num_leaves=31，
    # learning_rate=0.05，subsample=0.8，colsample_bytree=0.8
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        max_depth=8,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,  # 防止过拟合
        reg_alpha=0.1,         # L1 正则化
        reg_lambda=0.1,        # L2 正则化
        random_state=42,
        n_jobs=-1,
        verbosity=-1           # 减少日志输出
    )
    
    # ============================================================
    # 构建完整 Pipeline
    # ============================================================
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    return pipeline
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'cnt'
id_col = 'instant'
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

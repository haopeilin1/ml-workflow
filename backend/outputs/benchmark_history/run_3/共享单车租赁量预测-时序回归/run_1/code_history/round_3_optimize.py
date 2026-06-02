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
    # ============================================================
    if 'dteday' in df.columns:
        df['dteday_parsed'] = pd.to_datetime(df['dteday'], errors='coerce')
        
        df['dteday_year'] = df['dteday_parsed'].dt.year
        df['dteday_month'] = df['dteday_parsed'].dt.month
        df['dteday_day'] = df['dteday_parsed'].dt.day
        df['dteday_weekday'] = df['dteday_parsed'].dt.weekday
        df['dteday_hour'] = df['dteday_parsed'].dt.hour
        
        # 丢弃与已有列重复的特征
        duplicate_time_cols = ['dteday_month', 'dteday_weekday', 'dteday_hour']
        df = df.drop(columns=[c for c in duplicate_time_cols if c in df.columns], errors='ignore')
        
        # 丢弃原始 dteday 列和解析中间列
        df = df.drop(columns=['dteday', 'dteday_parsed'], errors='ignore')
        
        extracted_cols = ['dteday_year', 'dteday_day']
        if mode == 'train':
            PREPROCESS_STATE['dteday_time_features'] = [c for c in extracted_cols if c in df.columns]
    
    # ============================================================
    # MUST DO 7: 将低基数整数列标记为类别特征（但不在此处转换类型）
    # 记录类别列名到 PREPROCESS_STATE，供 build_model 使用
    # ============================================================
    categorical_cols = ['season', 'weathersit', 'holiday', 'workingday', 'yr']
    if 'hr' in df.columns:
        categorical_cols.append('hr')
    if 'weekday' in df.columns:
        categorical_cols.append('weekday')
    if 'mnth' in df.columns:
        categorical_cols.append('mnth')
    
    # 只保留实际存在的列
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    # 将类别列转换为整数类型（保留原始数值，不做 OneHot）
    for col in categorical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    if mode == 'train':
        PREPROCESS_STATE['cat_cols'] = categorical_cols
    
    # ============================================================
    # MUST DO 4: 对目标变量 cnt 进行 log1p 变换
    # ============================================================
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
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
    - 时间交互特征：hr 与 workingday、season、yr 的交互
    - 天气交互特征：temp 与 hum 的交互，windspeed 与 weathersit 的交互
    - 高峰时段特征：是否为通勤高峰（7-9点、17-19点）
    - 周末特征：是否为周末
    - 时间周期性编码：小时的正弦/余弦变换
    - 多项式特征：温度的平方
    - 滞后特征：前1小时和前2小时的 cnt（基于 log1p 变换后的值）
    - 滚动统计特征：过去24小时的平均 cnt 和标准差
    - 所有特征保持数值类型，类别编码由 build_model 的 Pipeline 处理
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
    if 'hr' in df.columns and 'workingday' in df.columns:
        workingday_num = pd.to_numeric(df['workingday'], errors='coerce').fillna(0).astype(int)
        hr_num = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
        df['hr_workingday_interact'] = hr_num * 10 + workingday_num
    
    if 'hr' in df.columns and 'season' in df.columns:
        season_num = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
        hr_num = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
        df['hr_season_interact'] = hr_num * 10 + season_num
    
    if 'hr' in df.columns and 'yr' in df.columns:
        yr_num = pd.to_numeric(df['yr'], errors='coerce').fillna(0).astype(int)
        hr_num = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
        df['hr_yr_interact'] = hr_num * 10 + yr_num
    
    if 'temp' in df.columns and 'hum' in df.columns:
        temp_num = pd.to_numeric(df['temp'], errors='coerce')
        hum_num = pd.to_numeric(df['hum'], errors='coerce')
        df['temp_hum_interact'] = temp_num * hum_num
    
    if 'temp' in df.columns and 'season' in df.columns:
        temp_num = pd.to_numeric(df['temp'], errors='coerce')
        season_num = pd.to_numeric(df['season'], errors='coerce')
        df['temp_season_interact'] = temp_num * season_num
    
    if 'windspeed' in df.columns and 'weathersit' in df.columns:
        windspeed_num = pd.to_numeric(df['windspeed'], errors='coerce')
        weathersit_num = pd.to_numeric(df['weathersit'], errors='coerce')
        df['windspeed_weathersit_interact'] = windspeed_num * weathersit_num
    
    # ============================================================
    # 创建时间相关特征
    # ============================================================
    if 'hr' in df.columns:
        hr_num = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
        # 是否为通勤高峰时段（7-9点或17-19点）
        df['is_rush_hour'] = hr_num.isin([7, 8, 9, 17, 18, 19]).astype(int)
        # 是否为白天（6-18点）
        df['is_daytime'] = ((hr_num >= 6) & (hr_num <= 18)).astype(int)
        # 是否为深夜（0-5点）
        df['is_night'] = (hr_num <= 5).astype(int)
        # 小时的正弦/余弦周期性编码
        df['hr_sin'] = np.sin(2 * np.pi * hr_num / 24)
        df['hr_cos'] = np.cos(2 * np.pi * hr_num / 24)
    
    if 'weekday' in df.columns:
        weekday_num = pd.to_numeric(df['weekday'], errors='coerce').fillna(0).astype(int)
        # 是否为周末（weekday 5=周六, 6=周日）
        df['is_weekend'] = weekday_num.isin([5, 6]).astype(int)
    
    if 'mnth' in df.columns:
        mnth_num = pd.to_numeric(df['mnth'], errors='coerce').fillna(0).astype(int)
        # 月份的正弦/余弦周期性编码
        df['mnth_sin'] = np.sin(2 * np.pi * mnth_num / 12)
        df['mnth_cos'] = np.cos(2 * np.pi * mnth_num / 12)
    
    # ============================================================
    # 多项式特征：温度的平方
    # ============================================================
    if 'temp' in df.columns:
        temp_num = pd.to_numeric(df['temp'], errors='coerce')
        df['temp_squared'] = temp_num ** 2
    
    if 'hum' in df.columns:
        hum_num = pd.to_numeric(df['hum'], errors='coerce')
        df['hum_squared'] = hum_num ** 2
    
    # ============================================================
    # 创建滞后特征和滚动统计特征
    # 基于 log1p 变换后的 cnt 值（已在 preprocess 中完成变换）
    # ============================================================
    if target_col in df.columns:
        cnt_values = df[target_col].copy()
        
        # 滞后特征：前1小时、前2小时、前24小时（昨天同一时间）的 cnt
        df['cnt_lag1'] = cnt_values.shift(1)
        df['cnt_lag2'] = cnt_values.shift(2)
        df['cnt_lag24'] = cnt_values.shift(24)
        
        # 滚动统计特征：过去24小时的平均 cnt 和标准差
        df['cnt_rolling_mean_24h'] = cnt_values.shift(1).rolling(window=24, min_periods=1).mean()
        df['cnt_rolling_std_24h'] = cnt_values.shift(1).rolling(window=24, min_periods=1).std()
        
        # 滚动统计特征：过去7天的平均 cnt（168小时）
        df['cnt_rolling_mean_168h'] = cnt_values.shift(1).rolling(window=168, min_periods=1).mean()
        
        # 填充缺失值（使用 bfill 和 fillna 0）
        for lag_col in ['cnt_lag1', 'cnt_lag2', 'cnt_lag24', 'cnt_rolling_mean_24h', 'cnt_rolling_std_24h', 'cnt_rolling_mean_168h']:
            if lag_col in df.columns:
                df[lag_col] = df[lag_col].bfill().fillna(0)
    
    # ============================================================
    # 分离特征和目标
    # ============================================================
    X = df.drop(columns=[target_col], errors='ignore')
    
    # ============================================================
    # 确保所有列都是数值类型
    # 类别列已在 preprocess 中转为整数，这里只需处理 object 类型
    # ============================================================
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 填充所有缺失值
    X = X.fillna(0)
    
    # 最终确保所有列都是数值类型
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
    # 获取类别特征列名（原始列名，需要清洗以匹配 feature_engineering 输出）
    # ============================================================
    cat_cols_original = PREPROCESS_STATE.get('cat_cols', [])
    cat_cols_clean = [re.sub(r'[^\w]', '_', str(c)) for c in cat_cols_original]
    
    # ============================================================
    # 构建预处理 Pipeline
    # ============================================================
    # 数值特征预处理：RobustScaler（MUST DO 6）
    num_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    # 类别特征预处理：OneHotEncoder（MUST DO 7）
    cat_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 构建 ColumnTransformer
    # 如果指定了类别列，则分别处理；否则所有列按数值处理
    if cat_cols_clean:
        preprocessor = ColumnTransformer([
            ('cat', cat_preprocessor, cat_cols_clean),
        ], remainder=num_preprocessor)
    else:
        preprocessor = num_preprocessor
    
    # ============================================================
    # 构建 LightGBM 回归模型（优化后的超参数）
    # ============================================================
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=3000,
        max_depth=8,
        num_leaves=127,
        learning_rate=0.02,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.3,
        reg_lambda=0.3,
        min_split_gain=0.01,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
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

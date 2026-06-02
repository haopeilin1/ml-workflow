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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import re

# 全局状态：保存预处理参数
PREPROCESS_STATE = {
    'cbwd_encoder': None,
    'feature_cols': None,
    'target_col': 'pm2.5',
    'drop_cols': ['No'],
    'skew_cols': ['Iws', 'Is', 'Ir'],
    'num_cols': None,
    'scaler': None,
}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    target_col = PREPROCESS_STATE['target_col']
    drop_cols = PREPROCESS_STATE['drop_cols']
    
    # 1. 丢弃 'No' 列（以及其他需要丢弃的列）
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # 2. 清洗列名：移除特殊字符，防止 LightGBM 报 JSON 特殊字符错误
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    # 更新 target_col 和 drop_cols 的引用（如果列名被修改）
    target_col = re.sub(r'[^\w]', '_', str(target_col))
    PREPROCESS_STATE['target_col'] = target_col
    
    # 3. 对目标变量 pm2.5 的缺失值进行时序感知填充（前向填充 + 后向填充）
    if target_col in df.columns:
        df[target_col] = df[target_col].ffill().bfill()
    
    # 4. 对特征列缺失值进行填充
    # 数值列：使用前向填充（时序感知），然后中位数填充剩余
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    
    for col in num_cols:
        if df[col].isnull().any():
            # 先尝试前向填充（时序感知）
            df[col] = df[col].ffill().bfill()
            # 如果仍有缺失（如开头全为 NaN），用中位数填充
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
    
    # 5. 将类别特征 'cbwd' 编码为整数标签
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if 'cbwd' in cat_cols or 'cbwd' in df.columns:
        cbwd_col = 'cbwd'
        if cbwd_col in df.columns:
            if mode == 'train':
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df[cbwd_col] = encoder.fit_transform(df[[cbwd_col]]).astype(int)
                PREPROCESS_STATE['cbwd_encoder'] = encoder
            else:
                encoder = PREPROCESS_STATE.get('cbwd_encoder')
                if encoder is not None:
                    df[cbwd_col] = encoder.transform(df[[cbwd_col]]).astype(int)
    
    # 6. 对 'hour' 列进行正弦/余弦变换，生成 'hour_sin' 和 'hour_cos'
    if 'hour' in df.columns:
        hour = df['hour'].values
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        # 保留原始 hour 列，但后续特征工程中会处理
    
    # 7. 对 'month' 进行循环编码
    if 'month' in df.columns:
        month = df['month'].values
        df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    # 8. 对 'day' 进行循环编码
    if 'day' in df.columns:
        day = df['day'].values
        df['day_sin'] = np.sin(2 * np.pi * day / 31.0)
        df['day_cos'] = np.cos(2 * np.pi * day / 31.0)
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    '''
    target_col = PREPROCESS_STATE['target_col']
    skew_cols = PREPROCESS_STATE['skew_cols']
    
    # 确保列名已清洗
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    target_col = re.sub(r'[^\w]', '_', str(target_col))
    skew_cols = [re.sub(r'[^\w]', '_', str(c)) for c in skew_cols]
    PREPROCESS_STATE['target_col'] = target_col
    PREPROCESS_STATE['skew_cols'] = skew_cols
    
    # 1. 对高度偏斜特征（Iws, Is, Ir）应用对数变换（log1p）
    for col in skew_cols:
        if col in df.columns:
            # 确保非负
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val  # 平移使最小值为0
            df[col] = np.log1p(df[col])
    
    # 2. 对目标变量 pm2.5 应用对数变换（log1p）
    if target_col in df.columns:
        # 确保非负
        min_val = df[target_col].min()
        if min_val < 0:
            df[target_col] = df[target_col] - min_val
        df[target_col] = np.log1p(df[target_col])
    
    # 3. 构造滞后特征：前 1-3 小时的 pm2.5 值
    if target_col in df.columns:
        for lag in [1, 2, 3]:
            lag_col = f'{target_col}_lag_{lag}'
            df[lag_col] = df[target_col].shift(lag)
    
    # 4. 构造滚动窗口统计量：过去 6 小时和 24 小时的 pm2.5 均值、标准差
    if target_col in df.columns:
        for window in [6, 24]:
            roll = df[target_col].rolling(window=window, min_periods=1)
            df[f'{target_col}_rolling_mean_{window}h'] = roll.mean()
            df[f'{target_col}_rolling_std_{window}h'] = roll.std()
    
    # 5. 添加时间特征：是否为周末
    if 'day' in df.columns and 'month' in df.columns and 'year' in df.columns:
        # 构造日期用于判断周末
        try:
            date_str = df['year'].astype(int).astype(str) + '-' + \
                       df['month'].astype(int).astype(str).str.zfill(2) + '-' + \
                       df['day'].astype(int).astype(str).str.zfill(2)
            dates = pd.to_datetime(date_str, errors='coerce')
            df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        except Exception:
            df['is_weekend'] = 0
    
    # 6. 添加季节特征
    if 'month' in df.columns:
        month = df['month'].values
        # 季节：春(3-5)=0, 夏(6-8)=1, 秋(9-11)=2, 冬(12,1,2)=3
        season = np.zeros(len(df), dtype=int)
        season[(month >= 3) & (month <= 5)] = 0  # 春
        season[(month >= 6) & (month <= 8)] = 1  # 夏
        season[(month >= 9) & (month <= 11)] = 2  # 秋
        season[(month == 12) | (month <= 2)] = 3  # 冬
        df['season'] = season
    
    # 7. 填充滞后特征和滚动特征的缺失值（前几行）
    # 滞后特征缺失用前向填充
    lag_cols = [c for c in df.columns if '_lag_' in c]
    for col in lag_cols:
        df[col] = df[col].ffill().bfill().fillna(0)
    
    roll_cols = [c for c in df.columns if '_rolling_' in c]
    for col in roll_cols:
        df[col] = df[col].ffill().bfill().fillna(0)
    
    # 8. 分离 X 和 y
    y = df[target_col] if target_col in df.columns else None
    
    # 特征列：排除目标列
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    
    # 9. 确保所有列都是数值类型
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # 10. 标准化数值特征（在 fit 时学习参数，在 transform 时应用）
    # 注意：这里在 feature_engineering 中做标准化，因为 preprocess 已经处理了编码
    if PREPROCESS_STATE.get('scaler') is None:
        # 训练模式：fit scaler
        scaler = StandardScaler()
        # 排除循环编码特征和二值特征，只标准化原始数值特征
        exclude_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                        'day_sin', 'day_cos', 'is_weekend', 'season', 'cbwd']
        scale_cols = [c for c in X.columns if c not in exclude_cols]
        if scale_cols:
            X_scaled = X.copy()
            X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols].values)
            PREPROCESS_STATE['scaler'] = scaler
            PREPROCESS_STATE['scale_cols'] = scale_cols
            X = X_scaled
    else:
        # 测试模式：应用已保存的 scaler
        scaler = PREPROCESS_STATE['scaler']
        scale_cols = PREPROCESS_STATE.get('scale_cols', [])
        if scale_cols:
            X_scaled = X.copy()
            X_scaled[scale_cols] = scaler.transform(X[scale_cols].values)
            X = X_scaled
    
    # 保存特征列名
    PREPROCESS_STATE['feature_cols'] = X.columns.tolist()
    
    # 返回 X（如果目标列存在，也返回 y 用于训练）
    if y is not None:
        # 将 y 作为额外列附加，供外部使用
        X[target_col] = y.values
    
    return X


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    '''
    # 使用 LightGBM 回归模型
    # 设置合理的超参数防止过拟合
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=31,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    
    return model
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

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# 全局状态变量
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    """
    global PREPROCESS_STATE
    
    # 复制数据，避免修改原始数据
    df = df.copy()
    
    # 丢弃 id 列（标识列，会导致严重过拟合）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 分离特征和目标列（目标列在 mode='test' 时可能不存在）
    target_col = 'median_house_value'
    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()
    
    # 定义数值列和类别列
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if mode == 'train':
        # 构建预处理器
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ], remainder='passthrough')
        
        # 拟合预处理器
        preprocessor.fit(X)
        
        # 保存状态
        PREPROCESS_STATE['preprocessor'] = preprocessor
        PREPROCESS_STATE['num_cols'] = num_cols
        PREPROCESS_STATE['cat_cols'] = cat_cols
        
        # 转换数据
        X_processed = preprocessor.transform(X)
        
        # 获取特征名
        feature_names = []
        # 数值列名
        feature_names.extend(num_cols)
        # 类别编码后的列名
        if cat_cols:
            encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(cat_cols)
            feature_names.extend(cat_feature_names)
        
        # 转换为 DataFrame
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
    elif mode == 'test':
        # 检查是否已拟合
        if 'preprocessor' not in PREPROCESS_STATE:
            raise ValueError("preprocess() must be called with mode='train' before mode='test'")
        
        preprocessor = PREPROCESS_STATE['preprocessor']
        
        # 应用预处理器
        X_processed = preprocessor.transform(X)
        
        # 获取特征名
        num_cols = PREPROCESS_STATE['num_cols']
        cat_cols = PREPROCESS_STATE['cat_cols']
        feature_names = []
        feature_names.extend(num_cols)
        if cat_cols:
            encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(cat_cols)
            feature_names.extend(cat_feature_names)
        
        # 转换为 DataFrame
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    
    # 重新添加目标列
    if y is not None:
        X_processed[target_col] = y.values
    
    return X_processed


def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    # 复制数据
    df = df.copy()
    
    # 分离目标列
    target_col = 'median_house_value'
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    
    # 创建衍生特征（基于原始数值列）
    # 注意：所有列已经是数值类型（类别已编码）
    
    # 1. 基础比例特征
    if 'total_rooms' in X.columns and 'households' in X.columns:
        X['rooms_per_household'] = X['total_rooms'] / (X['households'] + 1e-5)
    
    if 'total_bedrooms' in X.columns and 'total_rooms' in X.columns:
        X['bedrooms_per_room'] = X['total_bedrooms'] / (X['total_rooms'] + 1e-5)
    
    if 'population' in X.columns and 'households' in X.columns:
        X['population_per_household'] = X['population'] / (X['households'] + 1e-5)
    
    if 'median_income' in X.columns and 'total_rooms' in X.columns:
        X['income_per_room'] = X['median_income'] / (X['total_rooms'] + 1e-5)
    
    if 'total_bedrooms' in X.columns and 'households' in X.columns:
        X['bedrooms_per_household'] = X['total_bedrooms'] / (X['households'] + 1e-5)
    
    if 'total_rooms' in X.columns and 'population' in X.columns:
        X['rooms_per_population'] = X['total_rooms'] / (X['population'] + 1e-5)
    
    if 'median_income' in X.columns and 'households' in X.columns:
        X['income_per_household'] = X['median_income'] / (X['households'] + 1e-5)
    
    # 2. 对数变换（对偏态分布的特征）
    skewed_cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
    for col in skewed_cols:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(X[col])
    
    # 3. 多项式特征（median_income 是最重要的特征之一）
    if 'median_income' in X.columns:
        X['median_income_sq'] = X['median_income'] ** 2
        X['median_income_cub'] = X['median_income'] ** 3
    
    # 4. 交互特征：经纬度组合（捕捉地理位置交互）
    if 'longitude' in X.columns and 'latitude' in X.columns:
        X['longitude_latitude_interaction'] = X['longitude'] * X['latitude']
        X['longitude_sq'] = X['longitude'] ** 2
        X['latitude_sq'] = X['latitude'] ** 2
    
    # 5. 区域密度特征
    if 'population' in X.columns and 'total_rooms' in X.columns:
        X['density'] = X['population'] / (X['total_rooms'] + 1e-5)
    
    # 6. 房龄相关特征
    if 'housing_median_age' in X.columns:
        X['age_squared'] = X['housing_median_age'] ** 2
    
    # 确保所有列是数值类型
    X = X.select_dtypes(include=[np.number])
    
    # 处理无穷值和 NaN（防止模型报错）
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    return X


def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=1000,
        max_depth=12,
        num_leaves=80,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,  # 抑制训练日志
        min_split_gain=0.0,
        min_child_weight=1e-3,
        subsample_freq=1
    )
    
    return model


def evaluate_model(model, X_val, y_val):
    """
    自定义验证集评估指标。
    返回一个 dict，键名以 val_ 开头。
    """
    val_preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    return {'val_rmse': rmse}
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'median_house_value'
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

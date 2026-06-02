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
from lightgbm import LGBMClassifier
import lightgbm as lgb

# 全局状态变量，用于存储预处理参数
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    """
    df = df.copy()
    
    # 1. 丢弃ID列（uniqueCount ≈ rowCount，会导致严重过拟合）
    id_cols = ['product_price', 'discount_percent', 'product_rating', 
               'past_return_rate', 'delivery_delay_days', 'session_length_minutes', 'order_id']
    df = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')
    
    # 2. 分离特征和目标列（目标列保留，不做处理）
    target_col = 'returned'
    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()
    
    # 3. 定义数值列和类别列
    cat_cols = ['device_type', 'product_category', 'shipping_method', 'payment_method']
    # 只保留X中实际存在的列
    cat_cols = [col for col in cat_cols if col in X.columns]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # 确保类别列不在数值列中
    num_cols = [col for col in num_cols if col not in cat_cols]
    
    if mode == 'train':
        # 4. 构建预处理器
        preprocessor = ColumnTransformer([
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols)
        ], remainder='passthrough')
        
        # 5. 拟合预处理器
        preprocessor.fit(X)
        
        # 6. 保存状态
        PREPROCESS_STATE['preprocessor'] = preprocessor
        PREPROCESS_STATE['cat_cols'] = cat_cols
        PREPROCESS_STATE['num_cols'] = num_cols
        
        # 7. 转换数据
        X_processed = preprocessor.transform(X)
        
        # 8. 获取特征名
        cat_feature_names = []
        if cat_cols:
            encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
        num_feature_names = num_cols
        # remainder列（如果有）
        remainder_cols = [col for col in X.columns if col not in cat_cols and col not in num_cols]
        all_feature_names = cat_feature_names + num_feature_names + remainder_cols
        
        # 9. 转换为DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)
        
        # 10. 添加目标列
        if y is not None:
            X_processed_df[target_col] = y.values
        
        return X_processed_df
    
    elif mode == 'test':
        # 应用已保存的预处理器
        preprocessor = PREPROCESS_STATE['preprocessor']
        cat_cols_saved = PREPROCESS_STATE['cat_cols']
        num_cols_saved = PREPROCESS_STATE['num_cols']
        
        # 转换数据
        X_processed = preprocessor.transform(X)
        
        # 获取特征名
        cat_feature_names = []
        if cat_cols_saved:
            encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(cat_cols_saved).tolist()
        num_feature_names = num_cols_saved
        remainder_cols = [col for col in X.columns if col not in cat_cols_saved and col not in num_cols_saved]
        all_feature_names = cat_feature_names + num_feature_names + remainder_cols
        
        # 转换为DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)
        
        # 添加目标列（如果存在）
        if y is not None:
            X_processed_df[target_col] = y.values
        
        return X_processed_df
    
    return df


def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    df = df.copy()
    
    # 1. 丢弃目标列
    target_col = 'returned'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    
    # 2. 确保所有列是数值类型
    # 检查是否有非数值列（理论上preprocess后应该都是数值，但做防御性检查）
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        # 如果还有非数值列，用0填充（不应发生）
        for col in non_numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # 3. 创建新特征
    # 注意：product_price, discount_percent等ID列已在preprocess中被丢弃
    # 所以这里只能使用保留的数值列
    
    # 获取当前数值列
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建交互特征（如果相关列存在）
    if 'num_product_views' in num_cols and 'customer_age' in num_cols:
        X['views_per_age'] = X['num_product_views'] / (X['customer_age'] + 1)
    
    if 'past_purchase_count' in num_cols and 'num_product_views' in num_cols:
        X['purchase_to_view_ratio'] = X['past_purchase_count'] / (X['num_product_views'] + 1)
    
    if 'customer_age' in num_cols:
        X['age_squared'] = X['customer_age'] ** 2
        X['age_group'] = (X['customer_age'] // 10).astype(int)
    
    if 'num_product_views' in num_cols:
        X['log_views'] = np.log1p(X['num_product_views'].clip(lower=0))
    
    if 'past_purchase_count' in num_cols:
        X['log_purchases'] = np.log1p(X['past_purchase_count'])
    
    if 'used_coupon' in num_cols:
        X['used_coupon'] = X['used_coupon'].astype(int)
    
    # 4. 确保所有列是数值类型（再次检查）
    X = X.select_dtypes(include=[np.number])
    
    # 5. 处理无穷值和NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    return X


def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    # 注意：scale_pos_weight 需要在训练时根据实际数据计算
    # 这里返回模型配置，fit时传入 eval_set 和 callbacks
    model = LGBMClassifier(
        objective='binary',
        metric='auc',
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
        verbosity=-1
    )
    
    return model
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'returned'
id_col = 'order_id'
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

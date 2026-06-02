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
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import lightgbm as lgb

# 全局状态变量
PREPROCESS_STATE = {}
FEATURE_ENG_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    """
    df = df.copy()
    
    # 丢弃ID列（uniqueCount ≈ rowCount）
    id_cols = ['product_price', 'discount_percent', 'product_rating', 
               'past_return_rate', 'delivery_delay_days', 'session_length_minutes', 'order_id']
    df = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')
    
    # 处理异常值：将负值替换为中位数（仅对数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除目标列
    if 'returned' in numeric_cols:
        numeric_cols.remove('returned')
    
    if mode == 'train':
        # 拟合异常值处理参数
        outlier_thresholds = {}
        for col in numeric_cols:
            median_val = df[col].median()
            outlier_thresholds[col] = median_val
        PREPROCESS_STATE['outlier_thresholds'] = outlier_thresholds
    else:
        outlier_thresholds = PREPROCESS_STATE.get('outlier_thresholds', {})
    
    # 应用异常值处理
    for col in numeric_cols:
        if col in outlier_thresholds:
            median_val = outlier_thresholds[col]
            df[col] = np.where(df[col] < 0, median_val, df[col])
    
    # 类别列和数值列定义
    cat_cols = ['device_type', 'product_category', 'shipping_method', 'payment_method']
    # 只保留数据中存在的类别列
    cat_cols = [col for col in cat_cols if col in df.columns]
    num_cols = [col for col in numeric_cols if col in df.columns]
    
    # 构建预处理器
    if mode == 'train':
        # 类别编码器
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 数值缩放器
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer([
            ('cat', cat_transformer, cat_cols),
            ('num', num_transformer, num_cols)
        ], remainder='passthrough')
        
        # 拟合预处理器
        # 注意：这里只拟合特征部分，不包含目标列
        X = df.drop(columns=['returned'], errors='ignore')
        preprocessor.fit(X)
        PREPROCESS_STATE['preprocessor'] = preprocessor
        PREPROCESS_STATE['cat_cols'] = cat_cols
        PREPROCESS_STATE['num_cols'] = num_cols
    else:
        preprocessor = PREPROCESS_STATE.get('preprocessor')
        if preprocessor is None:
            raise ValueError("PREPROCESS_STATE中没有找到预处理器，请先运行mode='train'")
    
    return df


def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    df = df.copy()
    
    # 分离目标列（如果存在）
    y = None
    if 'returned' in df.columns:
        y = df['returned']
        df = df.drop(columns=['returned'])
    
    # 创建新特征
    # 注意：product_price, discount_percent 等ID列已在preprocess中被丢弃
    # 所以这里只对剩余数值列做特征工程
    
    # 获取数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. 年龄分组
    if 'customer_age' in df.columns:
        df['age_group'] = (df['customer_age'] // 10).astype(int)
    
    # 2. 购买次数与退货率的交互
    if 'past_purchase_count' in df.columns and 'past_return_rate' in df.columns:
        df['return_rate_purchase'] = df['past_return_rate'] * df['past_purchase_count']
    
    # 3. 延迟与浏览次数的交互
    if 'delivery_delay_days' in df.columns and 'num_product_views' in df.columns:
        df['delay_views_interaction'] = df['delivery_delay_days'] * df['num_product_views']
    
    # 4. 浏览效率
    if 'num_product_views' in df.columns and 'session_length_minutes' in df.columns:
        df['session_efficiency'] = df['num_product_views'] / (df['session_length_minutes'] + 1)
    
    # 5. 价格与评分的比率（如果价格列存在）
    if 'product_price' in df.columns and 'product_rating' in df.columns:
        df['price_rating_ratio'] = df['product_price'] / (df['product_rating'] + 1)
    
    # 6. 价格与折扣的交互（如果价格列存在）
    if 'product_price' in df.columns and 'discount_percent' in df.columns:
        df['price_discount_interaction'] = df['product_price'] * df['discount_percent']
    
    # 应用预处理器（将类别列编码为数值）
    preprocessor = PREPROCESS_STATE.get('preprocessor')
    if preprocessor is not None:
        # 获取特征列名（预处理器拟合时使用的列）
        cat_cols = PREPROCESS_STATE.get('cat_cols', [])
        num_cols = PREPROCESS_STATE.get('num_cols', [])
        
        # 确保所有需要的列都存在
        missing_cat = [c for c in cat_cols if c not in df.columns]
        missing_num = [c for c in num_cols if c not in df.columns]
        
        if missing_cat or missing_num:
            # 如果缺少列，填充默认值
            for col in missing_cat:
                df[col] = 'unknown'
            for col in missing_num:
                df[col] = 0
        
        # 转换数据
        X_transformed = preprocessor.transform(df)
        
        # 获取特征名称
        feature_names = []
        # 类别特征名称
        for i, col in enumerate(cat_cols):
            encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            if hasattr(encoder, 'get_feature_names_out'):
                cat_features = encoder.get_feature_names_out([col])
                feature_names.extend(cat_features)
            else:
                feature_names.extend([f"{col}_{j}" for j in range(encoder.categories_[0].shape[0])])
        
        # 数值特征名称
        feature_names.extend(num_cols)
        
        # 创建DataFrame
        X = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
        
        # 添加新创建的特征（这些特征不在预处理器中，需要单独处理）
        new_features = ['age_group', 'return_rate_purchase', 'delay_views_interaction',
                       'session_efficiency', 'price_rating_ratio', 'price_discount_interaction']
        for feat in new_features:
            if feat in df.columns:
                X[feat] = df[feat].values
        
        # 处理无穷值和NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X
    else:
        # 如果没有预处理器（理论上不应该发生），直接返回数值列
        # 只保留数值列
        X = df.select_dtypes(include=[np.number])
        X = X.fillna(0)
        return X


def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    model = LGBMClassifier(
        objective='binary',
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
        n_jobs=-1,
        verbose=-1
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

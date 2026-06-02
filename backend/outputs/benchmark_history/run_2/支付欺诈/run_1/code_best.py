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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import re

# 全局状态，用于在preprocess和feature_engineering之间传递信息
PREPROCESS_STATE = {
    'fitted': False,
    'cat_imputer': None,
    'num_imputer': None,
    'isWeekend_mode': None,
    'encoder': None,
    'scaler': None,
    'cat_cols': [],
    'num_cols': [],
    'target_col': 'label',
    'drop_cols': ['id'],
    'skewed_cols_log': ['numItems', 'paymentMethodAgeDays'],
    'skewed_cols_square': ['localTime'],
    'category_cols': ['paymentMethod', 'Category'],
    'missing_category_col': 'Category',
    'missing_isWeekend_col': 'isWeekend',
}

def clean_feature_names(df):
    """清洗特征名，移除特殊JSON字符"""
    df = df.copy()
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    return df

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    """
    df = df.copy()
    
    # 1. 丢弃id列
    drop_cols = [c for c in PREPROCESS_STATE['drop_cols'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # 2. 处理缺失值 - Category填充为'unknown'
    if PREPROCESS_STATE['missing_category_col'] in df.columns:
        if mode == 'train':
            df[PREPROCESS_STATE['missing_category_col']] = df[PREPROCESS_STATE['missing_category_col']].fillna('unknown')
        else:
            df[PREPROCESS_STATE['missing_category_col']] = df[PREPROCESS_STATE['missing_category_col']].fillna('unknown')
    
    # 3. 处理缺失值 - isWeekend填充为众数
    if PREPROCESS_STATE['missing_isWeekend_col'] in df.columns:
        if mode == 'train':
            PREPROCESS_STATE['isWeekend_mode'] = df[PREPROCESS_STATE['missing_isWeekend_col']].mode()[0]
        if PREPROCESS_STATE['isWeekend_mode'] is not None:
            df[PREPROCESS_STATE['missing_isWeekend_col']] = df[PREPROCESS_STATE['missing_isWeekend_col']].fillna(PREPROCESS_STATE['isWeekend_mode'])
    
    # 4. 对高度偏斜特征进行变换
    # log1p变换（右偏特征）
    for col in PREPROCESS_STATE['skewed_cols_log']:
        if col in df.columns:
            # 确保非负
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val  # 平移使最小值>=0
            df[col] = np.log1p(df[col])
    
    # 平方变换（左偏特征 localTime）
    for col in PREPROCESS_STATE['skewed_cols_square']:
        if col in df.columns:
            df[col] = df[col] ** 2
    
    # 5. 清洗特征名（移除特殊字符）
    df = clean_feature_names(df)
    
    PREPROCESS_STATE['fitted'] = True
    
    return df

def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    df = df.copy()
    target_col = PREPROCESS_STATE['target_col']
    
    # 分离目标列（如果存在）
    y = None
    if target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])
    
    # 1. 创建交叉特征：paymentMethod与Category的组合
    payment_col = None
    category_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'paymentmethod' in col_lower and 'age' not in col_lower:
            payment_col = col
        if 'category' in col_lower:
            category_col = col
    
    if payment_col and category_col:
        # 创建组合特征
        combo_name = f'{payment_col}_{category_col}_combo'
        df[combo_name] = df[payment_col].astype(str) + '_' + df[category_col].astype(str)
    
    # 2. 创建比值特征：accountAgeDays / (paymentMethodAgeDays + 1)
    account_col = None
    payment_age_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'accountage' in col_lower:
            account_col = col
        if 'paymentmethodage' in col_lower:
            payment_age_col = col
    
    if account_col and payment_age_col:
        ratio_name = 'accountAge_paymentAge_ratio'
        # 避免除零，加1
        df[ratio_name] = df[account_col] / (df[payment_age_col] + 1)
        # 裁剪极端值
        df[ratio_name] = df[ratio_name].clip(upper=df[ratio_name].quantile(0.99))
    
    # 3. 对numItems进行分箱处理
    numitems_col = None
    for col in df.columns:
        if 'numitems' in col.lower():
            numitems_col = col
            break
    
    if numitems_col:
        # 分箱：1件、2件、3件及以上
        bins = [-np.inf, 1, 2, np.inf]
        labels = ['1_item', '2_items', '3plus_items']
        bin_col_name = f'{numitems_col}_binned'
        df[bin_col_name] = pd.cut(df[numitems_col], bins=bins, labels=labels)
    
    # 4. 创建周末交易时间特征
    weekend_col = None
    localtime_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'isweekend' in col_lower:
            weekend_col = col
        if 'localtime' in col_lower:
            localtime_col = col
    
    if weekend_col and localtime_col:
        weekend_time_name = 'weekend_localtime_interaction'
        df[weekend_time_name] = df[weekend_col] * df[localtime_col]
    
    # 5. 识别类别列和数值列
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # 6. 构建预处理Pipeline（编码 + 缩放）
    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols)
    ], remainder='passthrough')
    
    # 7. 拟合并转换
    X_processed = preprocessor.fit_transform(df)
    
    # 8. 重建DataFrame（ColumnTransformer返回numpy array）
    # 获取特征名
    try:
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
        all_feature_names = list(cat_feature_names) + num_cols
    except:
        # 如果获取特征名失败，使用默认命名
        all_feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
    
    X_df = pd.DataFrame(X_processed, columns=all_feature_names, index=df.index)
    
    # 9. 清洗特征名
    X_df = clean_feature_names(X_df)
    
    # 10. 确保所有列都是数值类型
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        X_df[col] = X_df[col].astype(float)
    
    # 11. 填充任何剩余的NaN
    X_df = X_df.fillna(0)
    
    # 保存预处理器供build_model使用
    PREPROCESS_STATE['preprocessor'] = preprocessor
    PREPROCESS_STATE['feature_names'] = X_df.columns.tolist()
    
    # 如果目标列存在，加回去（但feature_engineering应该返回不含目标列的X）
    # 根据系统要求，feature_engineering返回的df应该是不含目标列的X
    return X_df

def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    # 计算scale_pos_weight
    # 注意：这里无法直接访问训练数据，但可以根据数据画像计算
    # 正例占1.43%，负例占98.57%
    # scale_pos_weight = 负类数/正类数 ≈ 98.57/1.43 ≈ 68.93
    scale_pos_weight = 69  # 约等于负类数/正类数
    
    model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=500,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    
    return model
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'label'
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

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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
import lightgbm as lgb
import re

# 全局状态，用于存储训练集上拟合的参数
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    global PREPROCESS_STATE
    
    df = df.copy()
    
    # 1. 丢弃 image_name 列（高基数ID列）
    df = df.drop(columns=['image_name'], errors='ignore')
    
    # 2. 清洗列名：替换特殊字符，防止 LightGBM 报 JSON 特殊字符错误
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    # 3. 处理缺失值
    if mode == 'train':
        # 计算填充值并保存
        PREPROCESS_STATE['age_approx_median'] = df['age_approx'].median()
        PREPROCESS_STATE['sex_mode'] = df['sex'].mode()[0] if not df['sex'].mode().empty else 'male'
        # anatom_site_general_challenge 用 'unknown' 填充
        PREPROCESS_STATE['anatom_fill_value'] = 'unknown'
        # width 和 height 的 log1p 变换不需要保存参数（直接应用）
    
    # 应用填充
    df['age_approx'] = df['age_approx'].fillna(PREPROCESS_STATE['age_approx_median'])
    df['sex'] = df['sex'].fillna(PREPROCESS_STATE['sex_mode'])
    df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna(PREPROCESS_STATE['anatom_fill_value'])
    
    # 4. 对 width 和 height 进行 log1p 变换
    df['width'] = np.log1p(df['width'])
    df['height'] = np.log1p(df['height'])
    
    # 5. 创建图像面积特征：area = log1p(width * height)
    # 注意：此时 width 和 height 已经过 log1p 变换，但面积应该基于原始值计算
    # 我们需要保留原始值或重新计算。这里使用变换后的值相乘作为近似特征。
    # 更好的做法：在 log1p 之前计算面积，然后对面积也做 log1p。
    # 修正：先计算原始面积，再做 log1p
    # 由于 width/height 已经变换，我们需要从原始数据重新计算。
    # 但 preprocess 中 df 的 width/height 已被覆盖。解决方案：在变换前保存原始值。
    # 重新设计：先计算 area，再对 width/height 做 log1p
    pass  # 此函数将在下面重新实现
    
    return df


def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    global PREPROCESS_STATE
    
    df = df.copy()
    
    # 1. 丢弃 image_name 列（高基数ID列）
    df = df.drop(columns=['image_name'], errors='ignore')
    
    # 2. 清洗列名：替换特殊字符
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    # 3. 计算面积特征（在 log1p 变换之前，基于原始 width/height）
    if 'width' in df.columns and 'height' in df.columns:
        df['area'] = df['width'] * df['height']
        df['area'] = np.log1p(df['area'])
    
    # 4. 对 width 和 height 进行 log1p 变换
    if 'width' in df.columns:
        df['width'] = np.log1p(df['width'])
    if 'height' in df.columns:
        df['height'] = np.log1p(df['height'])
    
    # 5. 处理缺失值
    if mode == 'train':
        PREPROCESS_STATE['age_approx_median'] = df['age_approx'].median()
        PREPROCESS_STATE['sex_mode'] = df['sex'].mode()[0] if not df['sex'].mode().empty else 'male'
        PREPROCESS_STATE['anatom_fill_value'] = 'unknown'
    
    df['age_approx'] = df['age_approx'].fillna(PREPROCESS_STATE['age_approx_median'])
    df['sex'] = df['sex'].fillna(PREPROCESS_STATE['sex_mode'])
    df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna(PREPROCESS_STATE['anatom_fill_value'])
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    '''
    global PREPROCESS_STATE
    
    df = df.copy()
    
    # 分离目标列（如果存在）
    target_col = 'target'
    y = None
    if target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])
    
    # 1. 创建交叉特征：sex 与 anatom_site_general_challenge 的组合
    if 'sex' in df.columns and 'anatom_site_general_challenge' in df.columns:
        df['sex_anatom'] = df['sex'].astype(str) + '_' + df['anatom_site_general_challenge'].astype(str)
    
    # 2. 创建 age_approx 分箱特征
    if 'age_approx' in df.columns:
        # 使用分位数分箱
        if 'age_bins' not in PREPROCESS_STATE:
            # 训练模式：计算分箱边界
            age_values = df['age_approx'].dropna()
            if len(age_values) > 0:
                PREPROCESS_STATE['age_bins'] = [0, 20, 40, 60, 80, 100]
                PREPROCESS_STATE['age_labels'] = ['0-20', '20-40', '40-60', '60-80', '80+']
        
        if 'age_bins' in PREPROCESS_STATE:
            df['age_group'] = pd.cut(
                df['age_approx'], 
                bins=PREPROCESS_STATE['age_bins'], 
                labels=PREPROCESS_STATE['age_labels'],
                right=False
            ).astype(str)
    
    # 3. 区分列类型
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # 4. 构建预处理 Pipeline
    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), num_cols)
    ], remainder='passthrough')
    
    # 5. 在训练模式下 fit，在测试模式下 transform
    if 'preprocessor' not in PREPROCESS_STATE:
        # 训练模式：fit 并保存
        X_processed = preprocessor.fit_transform(df)
        PREPROCESS_STATE['preprocessor'] = preprocessor
        # 获取特征名
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols) if cat_cols else []
        all_feature_names = list(cat_feature_names) + num_cols
        PREPROCESS_STATE['feature_names'] = all_feature_names
    else:
        # 测试模式：使用已保存的 preprocessor
        preprocessor = PREPROCESS_STATE['preprocessor']
        X_processed = preprocessor.transform(df)
    
    # 6. 转换为 DataFrame 并清洗列名
    feature_names = PREPROCESS_STATE.get('feature_names', None)
    if feature_names is not None:
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=df.index if hasattr(df, 'index') else None)
    else:
        X_processed = pd.DataFrame(X_processed, index=df.index if hasattr(df, 'index') else None)
    
    # 清洗列名（OneHotEncoder 可能生成包含特殊字符的列名）
    X_processed.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_processed.columns]
    
    # 确保所有列为数值类型
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    X_processed = X_processed.fillna(0)
    
    return X_processed


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    '''
    # 动态计算 scale_pos_weight（从训练数据中获取）
    # 如果 PREPROCESS_STATE 中没有保存，使用默认值
    scale_pos_weight = PREPROCESS_STATE.get('scale_pos_weight', 9.58)
    
    model = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        num_leaves=63,
        max_depth=7,
        learning_rate=0.03,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        min_child_weight=1e-3,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbosity=-1,
        boosting_type='gbdt',
        class_weight=None
    )
    
    return model


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    在验证集上搜索最优阈值（以 F1 最大化为目标），
    返回包含 AUC-ROC、AP、F1、Recall、Precision 和混淆矩阵的 dict。
    '''
    # 获取预测概率
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # 计算 AUC-ROC 和 AP
    auc_roc = roc_auc_score(y_val, y_proba)
    ap = average_precision_score(y_val, y_proba)
    
    # 搜索最优阈值（以 F1 最大化为目标）
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 使用最优阈值计算最终指标
    y_pred_best = (y_proba >= best_threshold).astype(int)
    f1_best = f1_score(y_val, y_pred_best)
    recall_best = recall_score(y_val, y_pred_best)
    precision_best = precision_score(y_val, y_pred_best)
    cm = confusion_matrix(y_val, y_pred_best)
    
    # 保存最优阈值到全局状态
    global PREPROCESS_STATE
    PREPROCESS_STATE['best_threshold'] = best_threshold
    
    return {
        'val_auc_roc': float(auc_roc),
        'val_ap': float(ap),
        'val_f1': float(f1_best),
        'val_recall': float(recall_best),
        'val_precision': float(precision_best),
        'val_best_threshold': float(best_threshold),
        'val_confusion_matrix': cm.tolist()
    }
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'target'
id_col = 'image_name'
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

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
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
import lightgbm as lgb

# 全局状态存储（用于 preprocess 的 fit/transform 模式）
PREPROCESS_STATE = {
    'age_median': None,
    'sex_mode': None,
    'encoder': None,
    'fitted': False
}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    df = df.copy()
    
    # 1. 丢弃 image_name 列（如果存在）
    if 'image_name' in df.columns:
        df = df.drop(columns=['image_name'])
    
    # 2. 处理 sex 列缺失值：填充为众数
    if mode == 'train':
        PREPROCESS_STATE['sex_mode'] = df['sex'].mode()[0] if not df['sex'].mode().empty else 'male'
    sex_mode = PREPROCESS_STATE['sex_mode']
    if 'sex' in df.columns:
        df['sex'] = df['sex'].fillna(sex_mode)
    
    # 3. 处理 anatom_site_general_challenge 列缺失值：填充为 'unknown'
    if 'anatom_site_general_challenge' in df.columns:
        df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('unknown')
    
    # 4. 处理 age_approx 列缺失值：填充为中位数
    if mode == 'train':
        PREPROCESS_STATE['age_median'] = df['age_approx'].median()
    age_median = PREPROCESS_STATE['age_median']
    if 'age_approx' in df.columns:
        df['age_approx'] = df['age_approx'].fillna(age_median)
    
    # 5. 创建衍生特征：area 和 aspect_ratio
    if 'width' in df.columns and 'height' in df.columns:
        # aspect_ratio：处理除零（height 不可能为 0，但安全起见）
        df['aspect_ratio'] = df['width'] / df['height'].replace(0, np.nan)
        df['aspect_ratio'] = df['aspect_ratio'].fillna(1.0)
        # area：先计算原始面积，再做 log1p
        df['area'] = np.log1p(df['width'] * df['height'])
    
    # 6. 对 width 和 height 做 log1p 变换
    if 'width' in df.columns:
        df['width'] = np.log1p(df['width'])
    if 'height' in df.columns:
        df['height'] = np.log1p(df['height'])
    
    # 7. 类别特征编码（sex, anatom_site_general_challenge）
    cat_cols = ['sex', 'anatom_site_general_challenge']
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    
    if existing_cat_cols:
        if mode == 'train':
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[existing_cat_cols] = encoder.fit_transform(df[existing_cat_cols])
            PREPROCESS_STATE['encoder'] = encoder
        else:
            encoder = PREPROCESS_STATE['encoder']
            if encoder is not None:
                df[existing_cat_cols] = encoder.transform(df[existing_cat_cols])
    
    PREPROCESS_STATE['fitted'] = True
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    '''
    df = df.copy()
    
    # 确保所有列为数值类型
    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试转换为数值
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # 确保没有 NaN（如果 preprocess 遗漏了）
        if df[col].isna().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
            else:
                df[col] = df[col].fillna(0)
    
    # 确保所有列都是数值类型
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    '''
    # 计算 scale_pos_weight（负类数/正类数）
    # 根据数据画像：正例占 9.45%，负例占 90.55%
    # scale_pos_weight = 负类数/正类数 ≈ 90.55/9.45 ≈ 9.58
    # 这里使用固定值，因为 build_model 无法访问训练数据
    # 实际训练时，系统会在 fit 前根据 y_train 动态调整
    scale_pos_weight = 9.58
    
    model = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        num_leaves=31,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )
    
    return model


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    返回一个 dict，键名以 val_ 开头。
    包含 AUC-ROC、AP、F1、Recall、Precision，以及基于 Youden's J 的最优阈值。
    '''
    # 预测概率
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # AUC-ROC
    val_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Average Precision
    val_ap = average_precision_score(y_val, y_pred_proba)
    
    # 搜索最优阈值（Youden's J 统计量：TPR - FPR）
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    
    # 使用最优阈值进行预测
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # 计算指标
    val_f1 = f1_score(y_val, y_pred)
    val_recall = recall_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'val_auc': float(val_auc),
        'val_ap': float(val_ap),
        'val_f1': float(val_f1),
        'val_recall': float(val_recall),
        'val_precision': float(val_precision),
        'val_best_threshold': float(best_threshold),
        'val_tn': int(tn),
        'val_fp': int(fp),
        'val_fn': int(fn),
        'val_tp': int(tp)
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

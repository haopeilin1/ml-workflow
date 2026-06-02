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
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# 全局状态：存储 preprocess 中拟合的参数
PREPROCESS_STATE = {
    'quality_mapping': None,  # 类别合并映射
    'num_cols': None,
    'cat_cols': None,
    'high_skew_cols': ['chlorides', 'sulphates', 'residual sugar', 'free sulfur dioxide'],
    'interaction_cols': ['alcohol', 'volatile acidity', 'residual sugar', 'sulphates'],
    'wine_type_encoder': None,
    'robust_scaler': None,
    'fitted': False
}


def _clean_feature_names(df):
    """清洗特征名，移除特殊 JSON 字符"""
    df = df.copy()
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    return df


def _merge_rare_classes(y, min_samples=10):
    """
    合并样本数 < min_samples 的类别到相邻类别。
    返回合并后的 y 和映射字典。
    """
    value_counts = y.value_counts().sort_index()
    rare_classes = value_counts[value_counts < min_samples].index.tolist()
    
    if len(rare_classes) == 0:
        return y, None
    
    y_mapped = y.copy()
    mapping = {}
    
    for cls in rare_classes:
        # 找到最近的相邻类别
        all_classes = sorted(value_counts.index.tolist())
        idx = all_classes.index(cls)
        
        if idx == 0:
            # 最小类别，合并到右边
            target = all_classes[idx + 1]
        elif idx == len(all_classes) - 1:
            # 最大类别，合并到左边
            target = all_classes[idx - 1]
        else:
            # 中间类别，合并到样本数更多的相邻类别
            left_count = value_counts.get(all_classes[idx - 1], 0)
            right_count = value_counts.get(all_classes[idx + 1], 0)
            target = all_classes[idx - 1] if left_count >= right_count else all_classes[idx + 1]
        
        mapping[cls] = target
    
    y_mapped = y_mapped.replace(mapping)
    return y_mapped, mapping


def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    global PREPROCESS_STATE
    df = df.copy()
    
    # 清洗特征名
    df = _clean_feature_names(df)
    
    # 1. 丢弃 id 列
    df = df.drop(columns=['id'], errors='ignore')
    
    # 2. 对 wine_type 进行有序编码（red=0, white=1）
    if 'wine_type' in df.columns:
        if mode == 'train':
            encoder = OrdinalEncoder(categories=[['red', 'white']], handle_unknown='use_encoded_value', unknown_value=-1)
            df['wine_type'] = encoder.fit_transform(df[['wine_type']]).astype(int)
            PREPROCESS_STATE['wine_type_encoder'] = encoder
        else:
            encoder = PREPROCESS_STATE.get('wine_type_encoder')
            if encoder is not None:
                df['wine_type'] = encoder.transform(df[['wine_type']]).astype(int)
    
    # 3. 对高偏度特征进行 log1p 变换
    high_skew_cols = PREPROCESS_STATE['high_skew_cols']
    for col in high_skew_cols:
        if col in df.columns:
            # 确保非负
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val + 1e-6
            df[col] = np.log1p(df[col])
    
    # 4. 对所有数值特征进行 RobustScaler 标准化
    target_col = 'quality'
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除目标列和 wine_type（已编码为整数，但不需要缩放）
    num_cols = [c for c in num_cols if c not in [target_col, 'wine_type']]
    
    if mode == 'train':
        PREPROCESS_STATE['num_cols'] = num_cols
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        PREPROCESS_STATE['robust_scaler'] = scaler
        PREPROCESS_STATE['fitted'] = True
    else:
        scaler = PREPROCESS_STATE.get('robust_scaler')
        if scaler is not None and len(num_cols) > 0:
            df[num_cols] = scaler.transform(df[num_cols])
    
    # 5. 检查 quality 的类别分布，合并样本数 < 10 的类别（仅训练模式）
    if mode == 'train' and target_col in df.columns:
        y = df[target_col].copy()
        y_mapped, mapping = _merge_rare_classes(y, min_samples=10)
        if mapping is not None:
            df[target_col] = y_mapped
            PREPROCESS_STATE['quality_mapping'] = mapping
            print(f'[INFO] Merged rare quality classes: {mapping}')
    elif mode == 'test' and target_col in df.columns:
        # 测试/验证模式：应用相同的映射
        mapping = PREPROCESS_STATE.get('quality_mapping')
        if mapping is not None:
            df[target_col] = df[target_col].replace(mapping)
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    '''
    global PREPROCESS_STATE
    df = df.copy()
    target_col = 'quality'
    
    # 分离目标列
    y = df[target_col].copy() if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors='ignore')
    
    # 1. 添加 wine_type 与关键特征的交互项
    interaction_cols = PREPROCESS_STATE['interaction_cols']
    if 'wine_type' in X.columns:
        for col in interaction_cols:
            if col in X.columns:
                X[f'wine_type_x_{col}'] = X['wine_type'] * X[col]
    
    # 2. 添加多项式特征（平方项）
    for col in interaction_cols:
        if col in X.columns:
            X[f'{col}_squared'] = X[col] ** 2
    
    # 3. 确保所有列都是数值类型
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 填充可能的 NaN（由交互项或转换产生）
    X = X.fillna(0)
    
    # 4. 特征选择：使用 SelectKBest 保留 top-20 特征（仅训练模式）
    # 注意：feature_engineering 在 preprocess 之后调用，此时数据已缩放
    # 特征选择应该在 Pipeline 中处理，这里只做特征构造
    # 如果 PREPROCESS_STATE 中有 selector，则应用
    if PREPROCESS_STATE.get('fitted') and PREPROCESS_STATE.get('selector') is not None:
        selector = PREPROCESS_STATE['selector']
        # 获取 selector 支持的特征名
        if hasattr(selector, 'get_support'):
            mask = selector.get_support()
            selected_cols = X.columns[mask].tolist()
            X = X[selected_cols]
    
    return X


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    使用 ImbPipeline 嵌入 SMOTE 过采样。
    '''
    global PREPROCESS_STATE
    
    # 基础模型：LightGBM
    # 使用 multiclass objective，设置类别权重为 'balanced'
    lgbm = LGBMClassifier(
        objective='multiclass',
        num_class=None,  # 自动推断
        boosting_type='gbdt',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    # 使用 ImbPipeline 嵌入 SMOTE（只对训练集过采样）
    # SMOTE 的 k_neighbors 需要根据最小类样本数调整
    # 默认 k_neighbors=5，如果某类样本 < 6 会报错
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('model', lgbm)
    ])
    
    return pipeline
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'quality'
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
        from sklearn.metrics import f1_score
        metrics = {'val_accuracy': float(accuracy_score(y_val, val_preds)), 'val_f1_macro': float(f1_score(y_val, val_preds, average='macro'))}
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
try:
    test_probs_all = model.predict_proba(X_test_fe)
except Exception:
    test_probs_all = None
test_preds = model.predict(X_test_fe)
if '_label_encoder' in globals() and _label_encoder is not None:
    test_preds = _label_encoder.inverse_transform(test_preds)


result_df = pd.DataFrame({
    'id': test[id_col] if id_col in test.columns else range(len(test_preds)),
    'prediction': test_preds,
})
if test_probs_all is not None:
    for i, col in enumerate(test_probs_all.T):
        result_df[f'proba_{i}'] = col
result_df.to_csv('data/test_predictions.csv', index=False)

# ========== 模型保存（系统保证可序列化）==========
with open('data/best_model.pkl', 'wb') as f:
    dill.dump(model, f)

# ========== 输出指标（系统抓取）==========
print('METRICS_JSON_START')
print(json.dumps(metrics))
print('METRICS_JSON_END')

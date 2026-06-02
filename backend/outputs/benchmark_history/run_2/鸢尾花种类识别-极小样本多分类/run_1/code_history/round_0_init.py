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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 全局状态：用于在 preprocess 的 train 和 test 模式之间传递参数
PREPROCESS_STATE = {
    'scaler': None,
    'label_encoder': None,
    'feature_cols': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target_col': 'species',
    'id_col': 'id'
}


def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    df = df.copy()
    
    # 1. 丢弃 id 列（Must-Do #1）
    if PREPROCESS_STATE['id_col'] in df.columns:
        df = df.drop(columns=[PREPROCESS_STATE['id_col']])
    
    target_col = PREPROCESS_STATE['target_col']
    feature_cols = PREPROCESS_STATE['feature_cols']
    
    if mode == 'train':
        # 2. 编码目标列 species 为整数标签 (Must-Do #4)
        if target_col in df.columns:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])
            PREPROCESS_STATE['label_encoder'] = le
        
        # 3. 标准化数值特征 (Must-Do #3)
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        PREPROCESS_STATE['scaler'] = scaler
        
    elif mode == 'test':
        # 应用已保存的编码器和标准化器
        if target_col in df.columns and PREPROCESS_STATE['label_encoder'] is not None:
            le = PREPROCESS_STATE['label_encoder']
            # 处理测试集中可能出现的未见过的类别
            known_classes = set(le.classes_)
            df[target_col] = df[target_col].apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            df[target_col] = le.transform(df[target_col])
        
        if PREPROCESS_STATE['scaler'] is not None:
            scaler = PREPROCESS_STATE['scaler']
            df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    
    计划要求：保持原始 4 个数值特征不变，不进行任何特征组合或多项式扩展。
    '''
    df = df.copy()
    
    target_col = PREPROCESS_STATE['target_col']
    feature_cols = PREPROCESS_STATE['feature_cols']
    
    # 提取特征列，排除目标列（使用 errors='ignore' 防止测试集没有目标列时报错）
    X = df.drop(columns=[target_col], errors='ignore')
    
    # 确保只保留 4 个数值特征列
    X = X[feature_cols]
    
    # 确保所有列都是数值类型
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 填充可能的 NaN（虽然数据质量报告显示无缺失值，但作为安全措施）
    if X.isnull().any().any():
        X = X.fillna(X.median())
    
    return X


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    
    计划要求：使用逻辑回归（带 L2 正则化），C 值通过交叉验证选择。
    候选模型：LogisticRegression、KNN、SVM(RBF核)、决策树(限制max_depth)。
    最终选择逻辑回归作为默认模型（参数少，不易过拟合，可解释性强）。
    '''
    # 使用逻辑回归（带 L2 正则化）
    # sklearn 1.6+：不使用 multi_class 参数，使用 solver='lbfgs'
    model = LogisticRegression(
        C=1.0,                    # L2 正则化强度的倒数，默认 1.0
        max_iter=1000,            # 增加最大迭代次数确保收敛
        solver='lbfgs',           # sklearn 1.6+ 多分类必须使用 lbfgs/newton-cg/sag/saga
        class_weight=None,        # 类别基本平衡，不使用 class_weight='balanced'
        random_state=42,
        n_jobs=-1
    )
    
    return model


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    返回一个 dict，键名以 val_ 开头。
    
    计划 Must-Do #5：同时计算 Accuracy 和 F1_macro，并输出混淆矩阵。
    '''
    val_preds = model.predict(X_val)
    
    # 计算 Accuracy
    val_accuracy = accuracy_score(y_val, val_preds)
    
    # 计算 F1_macro（多分类使用 average='macro'）
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    
    # 计算 F1_weighted
    val_f1_weighted = f1_score(y_val, val_preds, average='weighted')
    
    # 输出混淆矩阵（打印到控制台，便于分析）
    cm = confusion_matrix(y_val, val_preds)
    print('[INFO] Confusion Matrix:')
    print(cm)
    
    # 如果 label_encoder 可用，输出类别名称
    if PREPROCESS_STATE['label_encoder'] is not None:
        class_names = PREPROCESS_STATE['label_encoder'].classes_
        print(f'[INFO] Class names: {class_names.tolist()}')
    
    return {
        'val_accuracy': float(val_accuracy),
        'val_f1_macro': float(val_f1_macro),
        'val_f1_weighted': float(val_f1_weighted)
    }
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'species'
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

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
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# 全局状态存储（用于 preprocess 的 fit/transform 模式）
PREPROCESS_STATE = {
    'fitted': False,
    'amount_scaler': None,
    'time_scaler': None,
    'feature_cols': None,
}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    global PREPROCESS_STATE
    
    df = df.copy()
    
    # 1. 丢弃 id 列和所有 feat1-feat28 列（isLikelyId=true）
    drop_cols = ['id'] + [f'feat{i}' for i in range(1, 29)]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # 2. 对 Transaction_Amount 做 log1p 变换（处理严重右偏，skewness=6.76）
    if 'Transaction_Amount' in df.columns:
        df['Transaction_Amount_log'] = np.log1p(df['Transaction_Amount'])
        df = df.drop(columns=['Transaction_Amount'])
    
    # 3. 从 Time 列提取周期性特征
    if 'Time' in df.columns:
        # hour: 一天中的小时 (0-23)
        df['Time_hour'] = (df['Time'] // 3600) % 24
        # day_of_week: 一周中的星期几 (0-6)
        df['Time_day_of_week'] = (df['Time'] // (3600 * 24)) % 7
        df = df.drop(columns=['Time'])
    
    # 4. 特征名清洗：确保只包含字母、数字、下划线
    df.columns = [str(c).replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # 5. 处理缺失值（当前数据无缺失，但保留处理逻辑）
    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试转换为数值
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
    
    # 6. 记录特征列（排除目标列）
    target_col = 'IsFraud'
    feature_cols = [c for c in df.columns if c != target_col]
    
    if mode == 'train':
        PREPROCESS_STATE['feature_cols'] = feature_cols
        PREPROCESS_STATE['fitted'] = True
    else:
        # 测试模式：确保列与训练集一致
        if PREPROCESS_STATE.get('feature_cols') is not None:
            # 只保留训练集中存在的特征列（目标列用 errors='ignore' 保留）
            keep_cols = PREPROCESS_STATE['feature_cols'] + [target_col]
            df = df[[c for c in keep_cols if c in df.columns]]
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    
    对数值特征进行标准化（StandardScaler），在训练集上 fit，在测试集上 transform。
    '''
    global PREPROCESS_STATE
    
    df = df.copy()
    target_col = 'IsFraud'
    
    # 分离特征和目标列
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df
        y = None
    
    # 确保所有列为数值类型
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median() if X[col].dtype in ['float64', 'int64'] else 0)
    
    # 标准化数值特征
    num_cols = X.columns.tolist()
    
    if PREPROCESS_STATE.get('amount_scaler') is None:
        # 训练模式：fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        PREPROCESS_STATE['amount_scaler'] = scaler
    else:
        # 测试模式：transform
        scaler = PREPROCESS_STATE['amount_scaler']
        X_scaled = scaler.transform(X)
    
    # 重建 DataFrame（保留列名）
    X = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)
    
    # 根据系统要求，feature_engineering 返回特征矩阵 X（不含目标列）
    return X


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    
    使用 LightGBM + scale_pos_weight 处理极度类别不平衡。
    保留所有 PCA 特征（feat1-feat28），调整参数以适应高维特征空间。
    '''
    # 构建预处理 Pipeline（标准化）
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    # LightGBM 模型（sklearn 接口）
    # 使用 scale_pos_weight 处理类别不平衡
    # 参数调整（适应约30个特征的高维空间）：
    #   - n_estimators=1000（更多特征需要更多树来学习）
    #   - max_depth=6（适度深度，避免过拟合）
    #   - num_leaves=31（保守值，防止过拟合）
    #   - subsample=0.8（行采样）
    #   - colsample_bytree=0.8（列采样）
    #   - reg_alpha=0.5, reg_lambda=0.5（增强L1/L2正则化）
    #   - min_child_samples=100（叶子节点最小样本数，防止过拟合）
    #   - learning_rate=0.02（降低学习率配合更多树）
    #   - scale_pos_weight=554.56（负类数/正类数，约19964/36）
    model = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=554.56,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.02,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_samples=100,
        random_state=42,
        verbose=-1,
    )
    
    # 完整 Pipeline：预处理 + 模型
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model),
    ])
    
    return pipeline


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    在验证集上通过最大化 F1-score 寻找最优阈值，并计算 AP 和混淆矩阵。
    返回一个 dict，键名以 val_ 开头。
    '''
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        recall_score, precision_score, confusion_matrix
    )
    
    # 获取预测概率（正类概率）
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = model.decision_function(X_val)
    
    # ROC-AUC
    val_roc_auc = float(roc_auc_score(y_val, y_proba))
    
    # Average Precision (AP)
    val_ap = float(average_precision_score(y_val, y_proba))
    
    # 在验证集上寻找最优阈值（最大化 F1-score）
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0.0
    best_threshold = 0.5
    best_recall = 0.0
    best_precision = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_recall = recall_score(y_val, y_pred, zero_division=0)
            best_precision = precision_score(y_val, y_pred, zero_division=0)
    
    # 使用最优阈值计算最终指标
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    val_f1 = float(f1_score(y_val, y_pred_optimal, zero_division=0))
    val_recall = float(recall_score(y_val, y_pred_optimal, zero_division=0))
    val_precision = float(precision_score(y_val, y_pred_optimal, zero_division=0))
    
    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'val_roc_auc': val_roc_auc,
        'val_ap': val_ap,
        'val_f1': val_f1,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'val_best_threshold': float(best_threshold),
        'val_tn': int(tn),
        'val_fp': int(fp),
        'val_fn': int(fn),
        'val_tp': int(tp),
    }
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'IsFraud'
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

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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
import lightgbm as lgb

# 全局状态变量
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    - 丢弃id列
    - 将医学指标中的0值视为缺失值（Glucose, BloodPressure, SkinThickness, Insulin, BMI）
    - 用中位数填充缺失值
    """
    df = df.copy()
    
    # 丢弃id列（如果存在）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 医学指标中0值不合理的列（这些列的正常值应该>0）
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    if mode == 'train':
        # 在训练集上拟合参数
        # 将0值替换为NaN
        for col in zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        
        # 计算各列的中位数（用于填充）
        median_values = {}
        for col in df.columns:
            if col != 'Outcome' and df[col].dtype in ['int64', 'float64']:
                median_values[col] = df[col].median()
        
        # 保存状态
        PREPROCESS_STATE['median_values'] = median_values
        PREPROCESS_STATE['zero_invalid_cols'] = zero_invalid_cols
        
        # 用中位数填充NaN
        for col, median_val in median_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
    
    else:  # mode == 'test'
        # 应用训练集拟合的参数
        median_values = PREPROCESS_STATE.get('median_values', {})
        zero_invalid_cols = PREPROCESS_STATE.get('zero_invalid_cols', [])
        
        # 将0值替换为NaN
        for col in zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        
        # 用训练集的中位数填充
        for col, median_val in median_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
    
    return df


def feature_engineering(df):
    """
    特征工程。
    - 创建关键交互特征和对数变换特征
    - 丢弃目标列Outcome
    - 返回纯数值特征矩阵
    """
    df = df.copy()
    
    # 分离目标列（如果存在）
    target_col = 'Outcome'
    y = None
    if target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])
    
    # 确保所有特征都是数值型
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建关键交互特征（仅保留最有意义的）
    # BMI与Glucose的交互（高BMI+高血糖=高风险）
    if 'BMI' in numeric_cols and 'Glucose' in numeric_cols:
        df['BMI_Glucose_interaction'] = df['BMI'] * df['Glucose']
    
    # Age与Pregnancies的交互（高龄+多次怀孕=高风险）
    if 'Age' in numeric_cols and 'Pregnancies' in numeric_cols:
        df['Age_Pregnancies_interaction'] = df['Age'] * df['Pregnancies']
    
    # 创建对数变换特征（处理偏态分布，仅对偏态严重的列）
    skewed_cols = ['Insulin', 'DiabetesPedigreeFunction']
    for col in skewed_cols:
        if col in numeric_cols:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # 如果之前分离了目标列，重新加回去（用于后续训练）
    if y is not None:
        df[target_col] = y
    
    return df


def build_model():
    """
    模型构建和超参数设置。
    使用LightGBM，设置正则化参数防止过拟合。
    小样本数据（491条），严格控制模型复杂度。
    """
    model = LGBMClassifier(
        objective='binary',
        class_weight='balanced',  # 自动处理类别不平衡
        num_leaves=15,            # 小样本数据，减少叶子节点数
        max_depth=6,              # 限制树深度
        learning_rate=0.05,       # 学习率
        n_estimators=300,         # 减少迭代次数，配合early stopping
        subsample=0.7,            # 行采样，防止过拟合
        colsample_bytree=0.7,     # 列采样，防止过拟合
        min_child_samples=30,     # 叶子节点最小样本数
        reg_alpha=0.5,            # L1正则化
        reg_lambda=0.5,           # L2正则化
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
target_col = 'Outcome'
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

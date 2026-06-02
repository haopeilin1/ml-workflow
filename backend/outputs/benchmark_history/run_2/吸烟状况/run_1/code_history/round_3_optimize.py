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
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
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
    df = df.copy()
    
    # 1. 丢弃 id 列（如果存在）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 2. 清洗列名：将特殊字符替换为下划线
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    # 3. 分离特征和目标列
    target_col = 'smoking'
    # 注意：test.csv 没有目标列，所以需要检查
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    
    # 4. 数值特征列（所有特征都是数值类型）
    numeric_cols = X.columns.tolist()
    
    # 5. 标准化
    if mode == 'train':
        # 拟合 scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        PREPROCESS_STATE['scaler'] = scaler
        PREPROCESS_STATE['numeric_cols'] = numeric_cols
    else:  # mode == 'test'
        # 应用已拟合的 scaler
        scaler = PREPROCESS_STATE['scaler']
        X_scaled = scaler.transform(X)
    
    # 6. 转换回 DataFrame
    X_processed = pd.DataFrame(X_scaled, columns=numeric_cols)
    
    # 7. 重新添加目标列（如果存在）
    if y is not None:
        X_processed[target_col] = y.values
    
    return X_processed


def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    df = df.copy()
    
    # 丢弃目标列
    target_col = 'smoking'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    # 创建新特征
    # BMI = weight(kg) / (height(m))^2
    if 'height_cm_' in df.columns and 'weight_kg_' in df.columns:
        df['bmi'] = df['weight_kg_'] / ((df['height_cm_'] / 100) ** 2)
    
    # 腰围身高比（反映中心性肥胖）
    if 'waist_cm_' in df.columns and 'height_cm_' in df.columns:
        df['waist_height_ratio'] = df['waist_cm_'] / df['height_cm_']
    
    # 血压差
    if 'systolic' in df.columns and 'relaxation' in df.columns:
        df['blood_pressure_diff'] = df['systolic'] - df['relaxation']
    
    # 平均血压
    if 'systolic' in df.columns and 'relaxation' in df.columns:
        df['mean_arterial_pressure'] = (2 * df['relaxation'] + df['systolic']) / 3
    
    # 血脂比值（triglyceride / HDL）
    if 'triglyceride' in df.columns and 'HDL' in df.columns:
        df['tg_hdl_ratio'] = df['triglyceride'] / (df['HDL'] + 1e-6)
    
    # 非高密度脂蛋白胆固醇
    if 'Cholesterol' in df.columns and 'HDL' in df.columns:
        df['non_hdl_cholesterol'] = df['Cholesterol'] - df['HDL']
    
    # 肝功比值（AST / ALT）
    if 'AST' in df.columns and 'ALT' in df.columns:
        df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1e-6)
    
    # 年龄与BMI交互
    if 'age' in df.columns and 'bmi' in df.columns:
        df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    # 血红蛋白与血脂交互（吸烟影响血红蛋白和血脂）
    if 'hemoglobin' in df.columns and 'triglyceride' in df.columns:
        df['hb_tg_interaction'] = df['hemoglobin'] * df['triglyceride']
    
    # 对偏态分布特征做对数变换
    skewed_cols = ['triglyceride', 'Gtp', 'ALT', 'AST', 'fasting_blood_sugar']
    for col in skewed_cols:
        if col in df.columns:
            # 确保所有值 > 0
            min_val = df[col].min()
            if min_val <= 0:
                df[f'log_{col}'] = np.log1p(df[col] - min_val + 1)
            else:
                df[f'log_{col}'] = np.log(df[col])
    
    # 确保所有列都是数值类型
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 填充可能产生的 NaN（除零保护）
    df = df.fillna(0)
    
    # 处理无穷值
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    model = LGBMClassifier(
        objective='binary',
        class_weight='balanced',  # 使用 balanced 替代手动 scale_pos_weight，更稳定
        num_leaves=25,            # 从31降至25，控制复杂度
        max_depth=6,
        learning_rate=0.03,       # 从0.05降至0.03，更精细学习
        n_estimators=800,         # 增加树的数量，配合低学习率
        subsample=0.7,            # 从0.8降至0.7，增加随机性防过拟合
        colsample_bytree=0.7,     # 新增，增加特征随机性
        min_child_samples=30,     # 从20提升至30，控制叶子节点分裂
        min_split_gain=0.01,      # 新增，分裂最小增益
        reg_alpha=0.01,           # 新增，L1正则化
        reg_lambda=0.01,          # 新增，L2正则化
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
target_col = 'smoking'
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

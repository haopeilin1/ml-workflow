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
from lightgbm import LGBMClassifier
import lightgbm as lgb

# 全局状态存储
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    """
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    """
    df = df.copy()
    
    # 丢弃id列（id列是单调递增的，uniqueCount=rowCount，会导致过拟合）
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 处理目标列：将income从字符串转换为0/1
    if 'income' in df.columns:
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    
    # 类别列列表
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                'relationship', 'race', 'sex', 'native_country']
    
    # 数值列列表
    num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    if mode == 'train':
        # 训练模式：拟合编码器和缩放器
        # 类别列：用众数填充 + OneHot编码
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 数值列：用中位数填充 + 标准化
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 保存列名信息
        PREPROCESS_STATE['cat_cols'] = cat_cols
        PREPROCESS_STATE['num_cols'] = num_cols
        
        # 拟合并转换
        cat_data = cat_transformer.fit_transform(df[cat_cols])
        num_data = num_transformer.fit_transform(df[num_cols])
        
        # 保存转换器
        PREPROCESS_STATE['cat_transformer'] = cat_transformer
        PREPROCESS_STATE['num_transformer'] = num_transformer
        
        # 获取编码后的列名
        cat_feature_names = []
        for i, col in enumerate(cat_cols):
            encoder = cat_transformer.named_steps['encoder']
            if hasattr(encoder, 'categories_'):
                categories = encoder.categories_[i]
                for cat in categories:
                    cat_feature_names.append(f"{col}_{cat}")
        
        # 构建处理后的DataFrame
        processed_df = pd.DataFrame(
            np.hstack([cat_data, num_data]),
            columns=cat_feature_names + num_cols,
            index=df.index
        )
        
        # 添加目标列
        if 'income' in df.columns:
            processed_df['income'] = df['income'].values
        
        PREPROCESS_STATE['feature_names'] = processed_df.columns.tolist()
        
    else:  # mode == 'test'
        # 测试模式：应用已保存的转换器
        cat_transformer = PREPROCESS_STATE['cat_transformer']
        num_transformer = PREPROCESS_STATE['num_transformer']
        cat_cols = PREPROCESS_STATE['cat_cols']
        num_cols = PREPROCESS_STATE['num_cols']
        
        # 转换数据
        cat_data = cat_transformer.transform(df[cat_cols])
        num_data = num_transformer.transform(df[num_cols])
        
        # 获取编码后的列名
        cat_feature_names = []
        for i, col in enumerate(cat_cols):
            encoder = cat_transformer.named_steps['encoder']
            if hasattr(encoder, 'categories_'):
                categories = encoder.categories_[i]
                for cat in categories:
                    cat_feature_names.append(f"{col}_{cat}")
        
        # 构建处理后的DataFrame
        processed_df = pd.DataFrame(
            np.hstack([cat_data, num_data]),
            columns=cat_feature_names + num_cols,
            index=df.index
        )
        
        # 保留目标列（如果有）
        if 'income' in df.columns:
            processed_df['income'] = df['income'].values
    
    return processed_df


def feature_engineering(df):
    """
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    """
    # 分离特征和目标
    if 'income' in df.columns:
        X = df.drop(columns=['income'])
    else:
        X = df
    
    # 确保所有列都是数值类型
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 填充可能出现的NaN
    X = X.fillna(0)
    
    # === 基础特征变换 ===
    # 1. 资本收益对数变换（处理偏态分布）
    if 'capital_gain' in X.columns:
        X['capital_gain_log'] = np.log1p(X['capital_gain'])
    
    # 2. 资本损失对数变换
    if 'capital_loss' in X.columns:
        X['capital_loss_log'] = np.log1p(X['capital_loss'])
    
    # 3. 年龄平方项（年龄与收入可能非线性关系）
    if 'age' in X.columns:
        X['age_squared'] = X['age'] ** 2
    
    # 4. 资本收益与工作时间的交互
    if 'capital_gain' in X.columns and 'hours_per_week' in X.columns:
        X['capital_gain_hours_interaction'] = X['capital_gain'] * X['hours_per_week']
    
    # 5. 教育程度与工作时间的交互
    if 'education_num' in X.columns and 'hours_per_week' in X.columns:
        X['education_hours_interaction'] = X['education_num'] * X['hours_per_week']
    
    # 6. 资本净收益
    if 'capital_gain' in X.columns and 'capital_loss' in X.columns:
        X['capital_net'] = X['capital_gain'] - X['capital_loss']
    
    # === 新增特征 ===
    # 7. 年龄分组（离散化）
    if 'age' in X.columns:
        X['age_group'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 8. 每周工作时间分组
    if 'hours_per_week' in X.columns:
        X['hours_group'] = pd.cut(X['hours_per_week'], bins=[0, 20, 40, 50, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # 9. 是否有资本收益（二值化）
    if 'capital_gain' in X.columns:
        X['has_capital_gain'] = (X['capital_gain'] > 0).astype(int)
    
    # 10. 是否有资本损失（二值化）
    if 'capital_loss' in X.columns:
        X['has_capital_loss'] = (X['capital_loss'] > 0).astype(int)
    
    # 11. 教育程度分组
    if 'education_num' in X.columns:
        X['education_group'] = pd.cut(X['education_num'], bins=[0, 9, 12, 16, 20], labels=[0, 1, 2, 3]).astype(int)
    
    # 12. 年龄与教育程度的交互
    if 'age' in X.columns and 'education_num' in X.columns:
        X['age_education_interaction'] = X['age'] * X['education_num']
    
    # 13. 每周工作时间与教育程度的交互（另一种形式）
    if 'hours_per_week' in X.columns and 'education_num' in X.columns:
        X['hours_education_ratio'] = X['hours_per_week'] / (X['education_num'] + 1)
    
    # 14. fnlwgt的对数变换（该列偏态严重）
    if 'fnlwgt' in X.columns:
        X['fnlwgt_log'] = np.log1p(X['fnlwgt'])
    
    # 再次填充新特征可能产生的NaN
    X = X.fillna(0)
    
    return X


def build_model():
    """
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    """
    # 计算类别权重：income >50K 占比约24%，所以scale_pos_weight ≈ 76/24 ≈ 3.17
    # 略微提高权重以更关注少数类
    scale_pos_weight = 3.5
    
    model = LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        num_leaves=15,              # 从24降低到15，大幅减少过拟合
        max_depth=6,                # 保持6层
        learning_rate=0.02,         # 从0.03降低到0.02，更慢学习
        n_estimators=500,           # 从800降低到500，配合early stopping
        subsample=0.8,              # 保持行采样
        subsample_freq=1,           # 每轮都进行行采样
        colsample_bytree=0.8,       # 列采样，增加随机性
        min_child_samples=50,       # 从30增加到50，防止过拟合
        min_child_weight=10.0,      # 从5.0增加到10.0，更保守
        min_split_gain=0.1,         # 从0.01增加到0.1，减少不必要的分裂
        reg_alpha=0.3,              # L1正则化从0.1增加到0.3
        reg_lambda=0.3,             # L2正则化从0.1增加到0.3
        random_state=42,
        verbosity=-1,
        importance_type='gain'      # 使用gain计算特征重要性
    )
    
    return model
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'income'
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

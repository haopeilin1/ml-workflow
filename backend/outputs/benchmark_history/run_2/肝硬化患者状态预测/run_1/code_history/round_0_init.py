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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, early_stopping
import re

# 全局状态存储，用于在train和test模式间传递拟合好的transformer
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    global PREPROCESS_STATE
    
    target_col = 'Status'
    
    # 1. 丢弃id列
    df = df.drop(columns=['id'], errors='ignore')
    
    # 2. 清洗特征名（移除特殊字符，防止LightGBM报错）
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    # 更新target_col（如果列名被修改）
    if target_col in df.columns:
        pass  # target_col没有被修改
    else:
        # 查找可能的target列名变化
        for col in df.columns:
            if col.lower() == target_col.lower():
                target_col = col
                break
    
    # 3. 分离特征和目标列
    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()
    
    # 4. 识别列类型
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # 5. 识别高缺失率特征（>40%）和低缺失率特征
    high_missing_num = []
    low_missing_num = []
    for col in num_cols:
        missing_rate = X[col].isnull().mean()
        if missing_rate > 0.4:
            high_missing_num.append(col)
        else:
            low_missing_num.append(col)
    
    # 6. 定义需要log变换的右偏特征
    right_skewed_features = ['N_Days', 'Bilirubin', 'Cholesterol', 'Copper', 'Alk_Phos', 'Tryglicerides', 'SGOT']
    # 只保留实际存在于数据中的特征
    right_skewed_features = [f for f in right_skewed_features if f in num_cols]
    
    # 7. 定义类别编码策略
    # Edema: 有序类别 N < S < Y
    edema_col = 'Edema' if 'Edema' in cat_cols else None
    # 二值类别特征（Sex, Ascites, Hepatomegaly, Spiders）
    binary_cat_cols = [c for c in cat_cols if c != edema_col and X[c].nunique() <= 2]
    # 其他类别特征（Drug等）
    other_cat_cols = [c for c in cat_cols if c not in binary_cat_cols and c != edema_col]
    
    if mode == 'train':
        # === 训练模式：构建并拟合所有transformer ===
        
        # 数值管道：IterativeImputer（高缺失） + SimpleImputer（低缺失） + log变换 + StandardScaler
        # 由于IterativeImputer需要先做log变换再填充，我们分步处理
        
        # 先对右偏特征做log变换（在填充之前，因为log变换可以缓解右偏，使填充更合理）
        # 但IterativeImputer需要数值特征，所以我们在Pipeline中处理
        
        # 构建数值预处理管道
        num_pipeline_steps = []
        
        # 对右偏特征做log(1+x)变换
        def log_transform_right_skewed(X):
            X = X.copy()
            for col in right_skewed_features:
                if col in X.columns:
                    # 确保非负，log(1+x)要求x >= -1
                    min_val = X[col].min()
                    if pd.notna(min_val) and min_val < 0:
                        X[col] = X[col] - min_val  # 平移使最小值为0
                    X[col] = np.log1p(X[col].clip(lower=0))
            return X
        
        log_transformer = FunctionTransformer(log_transform_right_skewed, validate=False)
        
        # IterativeImputer用于高缺失率特征
        iterative_imputer = IterativeImputer(random_state=42, max_iter=10, verbose=0)
        
        # StandardScaler
        scaler = StandardScaler()
        
        # 组合数值管道
        num_pipeline = Pipeline([
            ('log_transform', log_transformer),
            ('iterative_imputer', iterative_imputer),
            ('scaler', scaler)
        ])
        
        # 类别管道
        cat_transformers = []
        
        # Edema: OrdinalEncoder，保持顺序 N < S < Y
        if edema_col:
            edema_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(categories=[['N', 'S', 'Y']], handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            cat_transformers.append(('edema', edema_pipeline, [edema_col]))
        
        # 二值类别特征：OrdinalEncoder
        if binary_cat_cols:
            binary_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            cat_transformers.append(('binary_cat', binary_pipeline, binary_cat_cols))
        
        # 其他类别特征（Drug等）：OneHotEncoder
        if other_cat_cols:
            other_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            cat_transformers.append(('other_cat', other_pipeline, other_cat_cols))
        
        # 构建ColumnTransformer
        transformers = []
        if num_cols:
            transformers.append(('num', num_pipeline, num_cols))
        transformers.extend(cat_transformers)
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        # 拟合preprocessor
        X_processed = preprocessor.fit_transform(X)
        
        # 获取特征名
        feature_names = []
        if num_cols:
            feature_names.extend(num_cols)
        if edema_col:
            feature_names.append(edema_col)
        if binary_cat_cols:
            feature_names.extend(binary_cat_cols)
        if other_cat_cols:
            # OneHotEncoder会扩展列
            other_encoder = preprocessor.named_transformers_['other_cat'].named_steps['encoder']
            ohe_features = other_encoder.get_feature_names_out(other_cat_cols).tolist()
            feature_names.extend(ohe_features)
        
        # 保存状态
        PREPROCESS_STATE['preprocessor'] = preprocessor
        PREPROCESS_STATE['feature_names'] = feature_names
        PREPROCESS_STATE['target_col'] = target_col
        PREPROCESS_STATE['num_cols'] = num_cols
        PREPROCESS_STATE['cat_cols'] = cat_cols
        PREPROCESS_STATE['right_skewed_features'] = right_skewed_features
        
        # 构建返回的DataFrame
        X_processed_df = pd.DataFrame(X_processed, index=X.index, columns=feature_names)
        
        if y is not None:
            X_processed_df[target_col] = y.values
        
        return X_processed_df
    
    else:
        # === 测试模式：应用已保存的transformer ===
        preprocessor = PREPROCESS_STATE.get('preprocessor')
        feature_names = PREPROCESS_STATE.get('feature_names', [])
        
        if preprocessor is None:
            raise ValueError("PREPROCESS_STATE['preprocessor'] not found. Run preprocess in 'train' mode first.")
        
        X_processed = preprocessor.transform(X)
        X_processed_df = pd.DataFrame(X_processed, index=X.index, columns=feature_names)
        
        if y is not None:
            X_processed_df[target_col] = y.values
        
        return X_processed_df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    '''
    target_col = PREPROCESS_STATE.get('target_col', 'Status')
    
    # 分离X和y
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    
    # 1. 创建交叉特征：Stage与Bilirubin的交互
    if 'Stage' in X.columns and 'Bilirubin' in X.columns:
        X['Stage_Bilirubin_interact'] = X['Stage'] * X['Bilirubin']
    
    # 2. 创建统计特征：基于N_Days的生存时间分段特征
    if 'N_Days' in X.columns:
        # 分段：短期(<365天)、中期(365-1825天)、长期(>1825天)
        X['N_Days_segment'] = pd.cut(X['N_Days'], 
                                      bins=[-np.inf, 365, 1825, np.inf], 
                                      labels=[0, 1, 2]).astype(float)
        # N_Days与Stage的交互
        if 'Stage' in X.columns:
            X['Stage_N_Days_interact'] = X['Stage'] * X['N_Days']
    
    # 3. 创建Albumin与Bilirubin的比值（肝功能指标）
    if 'Albumin' in X.columns and 'Bilirubin' in X.columns:
        X['Albumin_Bilirubin_ratio'] = X['Albumin'] / (X['Bilirubin'] + 1e-6)
    
    # 4. 创建Age分段特征
    if 'Age' in X.columns:
        # Age在数据中是天数，转换为年
        X['Age_years'] = X['Age'] / 365.25
        # 年龄段
        X['Age_group'] = pd.cut(X['Age_years'], 
                                 bins=[0, 40, 50, 60, 70, 100], 
                                 labels=[0, 1, 2, 3, 4]).astype(float)
    
    # 5. 确保所有列都是数值类型
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 6. 填充可能因特征工程产生的NaN
    X = X.fillna(0)
    
    return X


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    '''
    # 使用LightGBM多分类器
    # 设置class_weight='balanced'处理类别不平衡
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,  # C, CL, D 三个类别
        boosting_type='gbdt',
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,  # 限制深度防止过拟合
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_estimators=500,
        class_weight='balanced',  # 处理类别不平衡
        random_state=42,
        verbose=-1,  # 静默模式
        n_jobs=-1
    )
    
    return model
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'Status'
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

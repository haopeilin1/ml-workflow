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
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, early_stopping
import re

# 全局状态字典，用于在 preprocess 的 train 和 test 模式间传递参数
PREPROCESS_STATE = {}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    target_col = 'Status'
    
    # 1. 丢弃id列
    df = df.drop(columns=['id'], errors='ignore')
    
    # 2. 清洗特征名：移除特殊JSON字符
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    # 更新 target_col 如果列名被修改（Status 不含特殊字符，但保持一致性）
    target_col = re.sub(r'[^\w]', '_', target_col)
    
    # 3. 目标列编码（仅在 train 模式）
    if mode == 'train':
        # 确保目标列存在
        if target_col in df.columns:
            # 编码映射：C->0, CL->1, D->2
            status_mapping = {'C': 0, 'CL': 1, 'D': 2}
            PREPROCESS_STATE['status_mapping'] = status_mapping
            PREPROCESS_STATE['status_reverse_mapping'] = {v: k for k, v in status_mapping.items()}
            df[target_col] = df[target_col].map(status_mapping)
            # 检查是否有未映射的类别
            if df[target_col].isna().any():
                unknown = df[df[target_col].isna()][target_col].unique()
                raise ValueError(f"Unknown categories in target column: {unknown}")
    
    # 4. 识别列类型
    # 排除目标列后的特征列
    feature_cols = [c for c in df.columns if c != target_col]
    
    # 类别特征列表（根据数据画像）
    categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    # 只保留实际存在的列
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    
    # 高缺失率数值特征（>40%缺失）
    high_missing_num_cols = ['Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides']
    high_missing_num_cols = [c for c in high_missing_num_cols if c in feature_cols]
    
    # 需要log1p变换的右偏数值特征
    skewed_cols = ['N_Days', 'Bilirubin', 'Cholesterol', 'Copper', 'Alk_Phos', 'Tryglicerides']
    skewed_cols = [c for c in skewed_cols if c in feature_cols]
    
    # 低缺失率数值特征（<5%缺失，用中位数填充）
    low_missing_num_cols = ['Platelets', 'Prothrombin']
    low_missing_num_cols = [c for c in low_missing_num_cols if c in feature_cols]
    
    # 其他数值特征（Age, Albumin, Stage 等，缺失率为0或很低）
    other_num_cols = [c for c in feature_cols 
                      if c not in categorical_cols 
                      and c not in high_missing_num_cols 
                      and c not in low_missing_num_cols
                      and c not in skewed_cols]
    # 将 skewed_cols 和 high_missing_num_cols 中重叠的列也加入 other_num_cols 的考虑
    all_num_cols = [c for c in feature_cols if c not in categorical_cols]
    
    # 5. 处理类别特征缺失值：将缺失值替换为'Missing'字符串
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Missing')
            df[col] = df[col].astype(str)
    
    # 6. 确保数值列是数值类型
    for col in all_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 7. 对右偏特征进行 log1p 变换
    if mode == 'train':
        PREPROCESS_STATE['skewed_cols'] = skewed_cols
        PREPROCESS_STATE['log1p_min_values'] = {}
        for col in skewed_cols:
            if col in df.columns:
                min_val = df[col].min()
                PREPROCESS_STATE['log1p_min_values'][col] = min_val
                # 如果最小值小于0，先平移再log1p
                if min_val < 0:
                    shift = abs(min_val) + 1
                    PREPROCESS_STATE['log1p_shift'] = PREPROCESS_STATE.get('log1p_shift', {})
                    PREPROCESS_STATE['log1p_shift'][col] = shift
                    df[col] = np.log1p(df[col] + shift)
                else:
                    df[col] = np.log1p(df[col])
    else:
        # test 模式：使用训练时的参数
        for col in skewed_cols:
            if col in df.columns:
                shift = PREPROCESS_STATE.get('log1p_shift', {}).get(col, 0)
                df[col] = np.log1p(df[col] + shift)
    
    # 8. 对低缺失率数值特征用中位数填充
    if mode == 'train':
        PREPROCESS_STATE['low_missing_medians'] = {}
        for col in low_missing_num_cols:
            if col in df.columns:
                median_val = df[col].median()
                PREPROCESS_STATE['low_missing_medians'][col] = median_val
                df[col] = df[col].fillna(median_val)
    else:
        for col in low_missing_num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(PREPROCESS_STATE['low_missing_medians'].get(col, 0))
    
    # 9. 对高缺失率数值特征使用 KNN Imputer
    if mode == 'train':
        # 选择用于KNN插补的数值特征（包括所有数值特征）
        knn_cols = [c for c in all_num_cols if c in df.columns]
        if len(knn_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
            df_knn = df[knn_cols].copy()
            df_knn_imputed = knn_imputer.fit_transform(df_knn)
            PREPROCESS_STATE['knn_imputer'] = knn_imputer
            PREPROCESS_STATE['knn_cols'] = knn_cols
            df[knn_cols] = df_knn_imputed
    else:
        knn_imputer = PREPROCESS_STATE.get('knn_imputer')
        knn_cols = PREPROCESS_STATE.get('knn_cols', [])
        if knn_imputer is not None and len(knn_cols) > 0:
            df_knn = df[knn_cols].copy()
            df_knn_imputed = knn_imputer.transform(df_knn)
            df[knn_cols] = df_knn_imputed
    
    # 10. 对其他数值特征（Age, Albumin, Stage等）用中位数填充（如果有缺失）
    if mode == 'train':
        PREPROCESS_STATE['other_num_medians'] = {}
        for col in other_num_cols:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                PREPROCESS_STATE['other_num_medians'][col] = median_val
                df[col] = df[col].fillna(median_val)
    else:
        for col, median_val in PREPROCESS_STATE.get('other_num_medians', {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
    
    return df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    第一次调用时（训练集）fit 并保存 preprocessor 到 PREPROCESS_STATE，
    后续调用（验证集/测试集）只 transform。
    '''
    target_col = 'Status'
    
    # 分离特征和目标
    X = df.drop(columns=[target_col], errors='ignore')
    
    # 识别列类型
    categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    
    # Edema 是有序类别特征，单独处理
    ordinal_cols = ['Edema'] if 'Edema' in categorical_cols else []
    nominal_cat_cols = [c for c in categorical_cols if c not in ordinal_cols]
    
    # 数值列
    num_cols = [c for c in X.columns if c not in categorical_cols]
    
    # 1. 创建交叉特征（医学领域知识）
    if 'Bilirubin' in X.columns and 'Albumin' in X.columns:
        X['Bilirubin_Albumin_ratio'] = X['Bilirubin'] / (X['Albumin'] + 1e-6)
    
    if 'Stage' in X.columns and 'N_Days' in X.columns:
        X['Stage_N_Days_interact'] = X['Stage'] * X['N_Days']
    
    if 'Bilirubin' in X.columns and 'Stage' in X.columns:
        X['Bilirubin_Stage'] = X['Bilirubin'] * X['Stage']
    
    if 'Albumin' in X.columns and 'Stage' in X.columns:
        X['Albumin_Stage'] = X['Albumin'] * X['Stage']
    
    if 'Age' in X.columns and 'Bilirubin' in X.columns:
        X['Age_Bilirubin'] = X['Age'] * X['Bilirubin'] / 1000.0  # 缩放避免数值过大
    
    if 'Stage' in X.columns:
        X['Stage_squared'] = X['Stage'] ** 2
    
    if 'Prothrombin' in X.columns and 'Bilirubin' in X.columns:
        X['Prothrombin_Bilirubin'] = X['Prothrombin'] * X['Bilirubin']
    
    # 更新 num_cols 以包含新增的交叉特征
    num_cols = [c for c in X.columns if c not in categorical_cols]
    
    # 2. 检查是否已有保存的 preprocessor（判断是训练模式还是测试模式）
    if 'feature_engineering_preprocessor' not in PREPROCESS_STATE:
        # 训练模式：构建并 fit preprocessor
        
        # 有序类别特征管道（Edema: N->0, S->1, Y->2, Missing->-1）
        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoder', OrdinalEncoder(categories=[['Missing', 'N', 'S', 'Y']], handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # 名义类别特征管道
        nominal_cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 数值特征管道
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 构建 ColumnTransformer
        transformers = []
        if len(ordinal_cols) > 0:
            transformers.append(('ordinal', ordinal_pipeline, ordinal_cols))
        if len(nominal_cat_cols) > 0:
            transformers.append(('nominal_cat', nominal_cat_pipeline, nominal_cat_cols))
        transformers.append(('num', num_pipeline, num_cols))
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        X_processed = preprocessor.fit_transform(X)
        
        # 保存 preprocessor 和列名
        PREPROCESS_STATE['feature_engineering_preprocessor'] = preprocessor
        
        # 获取输出列名
        all_feature_names = []
        if len(ordinal_cols) > 0:
            all_feature_names.extend(ordinal_cols)
        if len(nominal_cat_cols) > 0:
            ohe = preprocessor.named_transformers_['nominal_cat'].named_steps['encoder']
            all_feature_names.extend(list(ohe.get_feature_names_out(nominal_cat_cols)))
        all_feature_names.extend(num_cols)
        
        PREPROCESS_STATE['feature_engineering_columns'] = all_feature_names
    else:
        # 测试模式：使用已保存的 preprocessor 只做 transform
        preprocessor = PREPROCESS_STATE['feature_engineering_preprocessor']
        X_processed = preprocessor.transform(X)
        all_feature_names = PREPROCESS_STATE['feature_engineering_columns']
    
    # 3. 创建 DataFrame
    X_processed = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)
    
    # 4. 清洗特征名
    X_processed.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_processed.columns]
    
    # 5. 确保所有列都是数值类型
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    X_processed = X_processed.fillna(0)
    
    return X_processed


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    '''
    # 手动计算类别权重（基于训练集分布：C约68%, CL约5%, D约27%）
    # 使用更精细的权重策略而非简单的 'balanced'
    class_weight_dict = {0: 1.0, 1: 5.0, 2: 1.5}
    
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=7,
        learning_rate=0.03,
        n_estimators=1500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        verbose=-1
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

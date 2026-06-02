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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix, roc_auc_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# 全局状态：存储预处理阶段拟合的参数
PREPROCESS_STATE = {
    'cat_cols': None,
    'num_cols': None,
    'log_cols': None,
    'drop_cols': None,
    'target_col': 'fraud_bool',
    'preprocessor': None,
    'zip_credit_risk_mean': None,  # 存储按zip_count_4w分组的credit_risk_score均值
}

def preprocess(df, mode='train'):
    '''
    数据清洗和预处理。
    mode='train' 时拟合参数，保存到 PREPROCESS_STATE。
    mode='test' 时应用已保存的参数。
    返回处理后的 DataFrame（仍包含目标列）。
    '''
    target_col = PREPROCESS_STATE['target_col']
    
    # 1. 剔除疑似ID特征和常数零值特征
    drop_cols = [
        'id', 'name_email_similarity', 'days_since_request',
        'intended_balcon_amount', 'velocity_6h', 'velocity_24h',
        'velocity_4w', 'session_length_in_minutes',
        'device_fraud_count', 'month'
    ]
    PREPROCESS_STATE['drop_cols'] = drop_cols
    
    # 安全删除：只删除存在的列
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols, errors='ignore')
    
    # 2. 将-1值识别为缺失标记，替换为NaN
    neg_one_cols = [
        'prev_address_months_count', 'current_address_months_count',
        'bank_months_count', 'device_distinct_emails_8w'
    ]
    for col in neg_one_cols:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
    
    # 3. 区分列类型
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # 排除目标列
    cat_cols = [c for c in cat_cols if c != target_col]
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    
    PREPROCESS_STATE['cat_cols'] = cat_cols
    PREPROCESS_STATE['num_cols'] = num_cols
    
    # 4. 确定需要log1p变换的列（高度右偏特征）
    log_cols = ['prev_address_months_count', 'bank_branch_count_8w']
    log_cols = [c for c in log_cols if c in num_cols]
    PREPROCESS_STATE['log_cols'] = log_cols
    
    # 5. 构建预处理Pipeline
    if mode == 'train':
        # 类别特征处理：填充众数 + OrdinalEncoder
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # 数值特征处理：填充中位数 + StandardScaler
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer([
            ('cat', cat_pipeline, cat_cols),
            ('num', num_pipeline, num_cols)
        ], remainder='passthrough')
        
        # 分离特征和目标
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col] if target_col in df.columns else None
        
        # 拟合预处理器
        X_processed = preprocessor.fit_transform(X)
        
        # 重建DataFrame（ColumnTransformer返回numpy array）
        # 获取特征名
        cat_feature_names = cat_cols
        num_feature_names = num_cols
        all_feature_names = cat_feature_names + num_feature_names
        
        X_df = pd.DataFrame(X_processed, index=X.index, columns=all_feature_names)
        
        # 对log_cols做log1p变换（在StandardScaler之后，需要先还原再变换？不，计划要求对原始值做log1p）
        # 实际上log1p应该在StandardScaler之前做。这里我们在Pipeline外单独处理。
        # 重新设计：先做log1p变换，再做StandardScaler
        # 因此需要重新构建Pipeline
        
        # 重新构建：先对log_cols做log1p，再整体做StandardScaler
        def log1p_transform(X):
            X = X.copy()
            for col in log_cols:
                if col in X.columns:
                    # 确保非负（log1p要求输入 >= -1，我们已经将-1替换为NaN并填充了中位数）
                    X[col] = np.log1p(X[col].clip(lower=0))
            return X
        
        log_transformer = FunctionTransformer(log1p_transform, validate=False)
        
        # 数值Pipeline：填充中位数 -> log1p变换 -> StandardScaler
        num_pipeline_v2 = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log1p', log_transformer),
            ('scaler', StandardScaler())
        ])
        
        preprocessor_v2 = ColumnTransformer([
            ('cat', cat_pipeline, cat_cols),
            ('num', num_pipeline_v2, num_cols)
        ], remainder='passthrough')
        
        X_processed_v2 = preprocessor_v2.fit_transform(X)
        X_df = pd.DataFrame(X_processed_v2, index=X.index, columns=all_feature_names)
        
        # 保存预处理器
        PREPROCESS_STATE['preprocessor'] = preprocessor_v2
        
        # 将目标列加回
        if y is not None:
            X_df[target_col] = y.values
        
        return X_df
    
    else:
        # mode='test'：应用已保存的预处理器
        preprocessor = PREPROCESS_STATE.get('preprocessor')
        if preprocessor is None:
            raise ValueError("Preprocessor not fitted. Run preprocess with mode='train' first.")
        
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col] if target_col in df.columns else None
        
        X_processed = preprocessor.transform(X)
        
        cat_cols = PREPROCESS_STATE['cat_cols']
        num_cols = PREPROCESS_STATE['num_cols']
        all_feature_names = cat_cols + num_cols
        
        X_df = pd.DataFrame(X_processed, index=X.index, columns=all_feature_names)
        
        if y is not None:
            X_df[target_col] = y.values
        
        return X_df


def feature_engineering(df):
    '''
    特征工程。
    返回特征矩阵 X（必须不含目标列，所有列必须是数值类型）。
    '''
    target_col = PREPROCESS_STATE['target_col']
    
    # 分离特征和目标
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col] if target_col in df.columns else None
    
    # 确保所有列名是字符串且不含特殊字符
    X.columns = [str(c).replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for c in X.columns]
    
    # 1. 创建交叉特征：device_os × source, housing_status × employment_status
    # 注意：这些列已经被OrdinalEncoder编码为数值，列名保持不变
    cat_cols = PREPROCESS_STATE.get('cat_cols', [])
    
    # device_os 和 source 的交叉
    if 'device_os' in X.columns and 'source' in X.columns:
        X['device_os_x_source'] = X['device_os'].astype(str) + '_' + X['source'].astype(str)
        # 对交叉特征做Label Encoding（简单映射）
        cross_map = {val: idx for idx, val in enumerate(X['device_os_x_source'].unique())}
        X['device_os_x_source'] = X['device_os_x_source'].map(cross_map)
    
    # housing_status 和 employment_status 的交叉
    if 'housing_status' in X.columns and 'employment_status' in X.columns:
        X['housing_x_employment'] = X['housing_status'].astype(str) + '_' + X['employment_status'].astype(str)
        cross_map = {val: idx for idx, val in enumerate(X['housing_x_employment'].unique())}
        X['housing_x_employment'] = X['housing_x_employment'].map(cross_map)
    
    # 2. 创建聚合特征：按zip_count_4w分组计算credit_risk_score的均值
    # 注意：zip_count_4w和credit_risk_score都已被StandardScaler缩放
    # 但聚合特征仍然有意义（相对排序不变）
    if 'zip_count_4w' in X.columns and 'credit_risk_score' in X.columns:
        # 使用分位数分箱来模拟分组（因为zip_count_4w已被缩放）
        # 更好的做法：在preprocess中保留原始值。这里用分位数近似。
        try:
            zip_bins = pd.qcut(X['zip_count_4w'], q=20, labels=False, duplicates='drop')
            zip_mean = X.groupby(zip_bins)['credit_risk_score'].transform('mean')
            X['zip_credit_risk_mean'] = zip_mean
        except Exception:
            # 如果分箱失败（如唯一值太少），跳过
            pass
    
    # 3. 二值特征保持原样（email_is_free, phone_home_valid, phone_mobile_valid, has_other_cards, foreign_request, keep_alive_session）
    # 这些特征已经在preprocess中被StandardScaler处理，无需额外操作
    
    # 确保所有列都是数值类型
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # 填充任何剩余的NaN
    X = X.fillna(0)
    
    return X


def build_model():
    '''
    模型构建和超参数设置。
    返回 sklearn 兼容的模型对象（支持 fit/predict/predict_proba）。
    '''
    # 使用XGBoost，设置scale_pos_weight=86.34
    scale_pos_weight = 86.34
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=500,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False
    )
    
    return model


def evaluate_model(model, X_val, y_val):
    '''
    自定义验证集评估指标。
    在极度不平衡场景下，计算最优阈值（以F1最优为准），
    并返回ROC-AUC、Average Precision、最优阈值下的F1和Recall。
    '''
    # 预测概率
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_proba)
    
    # Average Precision
    ap = average_precision_score(y_val, y_proba)
    
    # 寻找最优阈值（以F1最优为准）
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_threshold = 0.5
    best_recall = 0
    best_precision = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            y_pred_best = y_pred
    
    # 在最优阈值下计算指标
    y_pred_best = (y_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_best).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # 保存最优阈值到全局状态（供测试集预测使用）
    PREPROCESS_STATE['best_threshold'] = best_threshold
    
    return {
        'val_roc_auc': float(roc_auc),
        'val_average_precision': float(ap),
        'val_best_f1': float(best_f1),
        'val_best_threshold': float(best_threshold),
        'val_recall_at_best_threshold': float(recall),
        'val_precision_at_best_threshold': float(precision),
        'val_true_positives': int(tp),
        'val_false_positives': int(fp),
        'val_true_negatives': int(tn),
        'val_false_negatives': int(fn)
    }
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = 'fraud_bool'
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

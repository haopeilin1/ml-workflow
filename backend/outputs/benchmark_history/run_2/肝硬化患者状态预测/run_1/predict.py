import pandas as pd
import numpy as np
import dill
import re
import os

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



# === 加载模型 ===
with open('data/best_model.pkl', 'rb') as f:
    model = dill.load(f)

# === 加载测试集 ===
test = pd.read_csv('data/test.csv')

# === 预处理 ===
test_clean = preprocess(test, mode='test')

# === 分离目标列 ===
target_col = 'Status'
if target_col in test_clean.columns:
    X_test = test_clean.drop(columns=[target_col])
else:
    X_test = test_clean

# === 特征工程 ===
X_test_fe = feature_engineering(X_test)
if isinstance(X_test_fe, np.ndarray):
    X_test_fe = pd.DataFrame(X_test_fe, index=X_test.index)

# === 清洗特征名 ===
X_test_fe.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_test_fe.columns]
if X_test_fe.columns.duplicated().any():
    X_test_fe.columns = [f'{c}_{i}' if i > 0 else str(c) for i, c in enumerate(X_test_fe.columns)]

# === 预测 ===
test_preds = model.predict(X_test_fe)

# === 保存结果 ===
id_col = 'id'
if id_col not in test.columns:
    id_col = test.columns[0]
result_df = pd.DataFrame({
    'id': test[id_col] if id_col in test.columns else range(len(test_preds)),
    'prediction': test_preds,
})
result_df.to_csv('output/eval_predictions.csv', index=False)
print('EVAL_PREDICTIONS_SAVED')
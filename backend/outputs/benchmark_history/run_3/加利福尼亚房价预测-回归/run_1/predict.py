import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

# 全局状态（与训练时一致）
PREPROCESS_STATE = {}

def preprocess(df, mode='test'):
    global PREPROCESS_STATE
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    target_col = 'median_house_value'
    if target_col in df.columns:
        y = df[target_col].copy()
        df_features = df.drop(columns=[target_col])
    else:
        y = None
        df_features = df.copy()
    
    cat_cols = ['ocean_proximity']
    num_cols = [col for col in df_features.columns if col not in cat_cols]
    
    if mode == 'train':
        preprocessor = ColumnTransformer([
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols)
        ], remainder='passthrough')
        preprocessor.fit(df_features)
        PREPROCESS_STATE['preprocessor'] = preprocessor
        PREPROCESS_STATE['num_cols'] = num_cols
        PREPROCESS_STATE['cat_cols'] = cat_cols
        transformed = preprocessor.transform(df_features)
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_feature_names
        df_transformed = pd.DataFrame(transformed, columns=feature_names, index=df_features.index)
    else:
        preprocessor = PREPROCESS_STATE.get('preprocessor')
        if preprocessor is None:
            raise ValueError("PREPROCESS_STATE 中没有预处理器，请先运行 mode='train'")
        transformed = preprocessor.transform(df_features)
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(PREPROCESS_STATE['cat_cols']).tolist()
        feature_names = PREPROCESS_STATE['num_cols'] + cat_feature_names
        df_transformed = pd.DataFrame(transformed, columns=feature_names, index=df_features.index)
    
    if y is not None:
        df_transformed[target_col] = y.values
    return df_transformed


def feature_engineering(df):
    df = df.copy()
    target_col = 'median_house_value'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    if 'total_rooms' in df.columns and 'households' in df.columns:
        df['rooms_per_household'] = df['total_rooms'] / (df['households'] + 1e-5)
    if 'total_bedrooms' in df.columns and 'total_rooms' in df.columns:
        df['bedrooms_per_room'] = df['total_bedrooms'] / (df['total_rooms'] + 1e-5)
    if 'population' in df.columns and 'households' in df.columns:
        df['population_per_household'] = df['population'] / (df['households'] + 1e-5)
    if 'total_bedrooms' in df.columns and 'population' in df.columns:
        df['bedrooms_per_population'] = df['total_bedrooms'] / (df['population'] + 1e-5)
    
    skewed_features = ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    for feat in skewed_features:
        if feat in df.columns:
            df[f'{feat}_log'] = np.log1p(df[feat])
    
    if 'median_income' in df.columns and 'housing_median_age' in df.columns:
        df['income_age_interaction'] = df['median_income'] * df['housing_median_age']
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['lat_lon_combined'] = df['latitude'] * df['longitude']
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    return df


def predict(new_data_path, output_path=None):
    """
    对新数据进行预测。
    
    参数:
        new_data_path: str, 新数据 CSV 文件路径
        output_path: str, 可选，预测结果保存路径
    
    返回:
        pd.DataFrame, 包含原始数据和 prediction 列
    """
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(new_data_path)
    
    # 预处理（需要先拟合预处理器）
    train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        preprocess(train_df, mode='train')
    
    # 对新数据预处理
    df_clean = preprocess(df, mode='test')
    X = df_clean.drop(columns=['median_house_value'], errors='ignore')
    if X is df_clean:
        X = df_clean.copy()
    
    # 特征工程
    X_fe = feature_engineering(X)
    if isinstance(X_fe, np.ndarray):
        X_fe = pd.DataFrame(X_fe, index=X.index)
    X_fe.columns = [re.sub('[^\\w]', '_', str(c)) for c in X_fe.columns]
    if X_fe.columns.duplicated().any():
        X_fe.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X_fe.columns)]
    
    # 预测
    preds = model.predict(X_fe)
    
    # 构建结果
    result = df.copy()
    result['prediction'] = preds
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"预测结果已保存到 {output_path}")
    
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python predict.py <新数据文件路径> [输出文件路径]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    result = predict(input_path, output_path)
    print(result.head())

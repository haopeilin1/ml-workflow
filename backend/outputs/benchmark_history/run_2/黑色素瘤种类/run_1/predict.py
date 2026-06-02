
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 全局状态（与训练时一致）
PREPROCESS_STATE = {}

def preprocess(df, mode='test'):
    global PREPROCESS_STATE
    
    df = df.copy()
    df = df.drop(columns=['image_name'], errors='ignore')
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    if 'width' in df.columns and 'height' in df.columns:
        df['area'] = df['width'] * df['height']
        df['area'] = np.log1p(df['area'])
    
    if 'width' in df.columns:
        df['width'] = np.log1p(df['width'])
    if 'height' in df.columns:
        df['height'] = np.log1p(df['height'])
    
    # 使用默认填充值
    if 'age_approx' in df.columns:
        df['age_approx'] = df['age_approx'].fillna(50)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].fillna('male')
    if 'anatom_site_general_challenge' in df.columns:
        df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('unknown')
    
    return df


def feature_engineering(df):
    global PREPROCESS_STATE
    
    df = df.copy()
    target_col = 'target'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    if 'sex' in df.columns and 'anatom_site_general_challenge' in df.columns:
        df['sex_anatom'] = df['sex'].astype(str) + '_' + df['anatom_site_general_challenge'].astype(str)
    
    if 'age_approx' in df.columns:
        age_bins = [0, 20, 40, 60, 80, 100]
        age_labels = ['0-20', '20-40', '40-60', '60-80', '80+']
        df['age_group'] = pd.cut(df['age_approx'], bins=age_bins, labels=age_labels, right=False).astype(str)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # 注意：这里需要加载训练时保存的 preprocessor
    # 简化版本：使用模型自带的特征名进行对齐
    # 实际使用时，建议将 preprocessor 也保存到 model.pkl 中
    
    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), num_cols)
    ], remainder='passthrough')
    
    X_processed = preprocessor.fit_transform(df)
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols) if cat_cols else []
    all_feature_names = list(cat_feature_names) + num_cols
    
    X_processed = pd.DataFrame(X_processed, columns=all_feature_names)
    X_processed.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_processed.columns]
    
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    X_processed = X_processed.fillna(0)
    
    return X_processed


def predict(new_data_path, output_path='predictions.csv'):
    """
    对新数据进行预测
    
    Parameters:
    -----------
    new_data_path : str
        新数据 CSV 文件路径
    output_path : str
        预测结果输出路径
    """
    # 加载模型
    with open('output/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(new_data_path)
    
    # 预处理
    df_clean = preprocess(df, mode='test')
    
    # 特征工程
    X = feature_engineering(df_clean)
    
    # 对齐特征
    if hasattr(model, 'feature_name_'):
        expected_features = model.feature_name_
    elif hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    else:
        expected_features = X.columns.tolist()
    
    for feat in expected_features:
        if feat not in X.columns:
            X[feat] = 0
    X = X[expected_features]
    
    # 预测
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    
    preds = (probs >= 0.5).astype(int)
    
    # 保存结果
    result = pd.DataFrame({
        'prediction': preds,
        'probability': probs
    })
    
    # 保留原始 ID 列（如果存在）
    id_col = 'image_name'
    if id_col in df.columns:
        result.insert(0, 'id', df[id_col])
    
    result.to_csv(output_path, index=False)
    print(f"预测完成！结果已保存到 {output_path}")
    print(f"总样本数: {len(result)}, 正类比例: {result['prediction'].mean():.4f}")
    
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict(input_file, output_file)
    else:
        print("用法: python predict.py <输入CSV文件> [输出CSV文件]")
        print("示例: python predict.py new_data.csv predictions.csv")

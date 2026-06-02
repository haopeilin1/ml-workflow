
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

# 预处理状态（需与训练时一致）
PREPROCESS_STATE = {}

def preprocess(df, mode='test'):
    target_col = 'Status'
    df = df.drop(columns=['id'], errors='ignore')
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    feature_cols = [c for c in df.columns if c != target_col]
    
    categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    
    skewed_cols = ['N_Days', 'Bilirubin', 'Cholesterol', 'Copper', 'Alk_Phos', 'Tryglicerides']
    skewed_cols = [c for c in skewed_cols if c in feature_cols]
    
    low_missing_num_cols = ['Platelets', 'Prothrombin']
    low_missing_num_cols = [c for c in low_missing_num_cols if c in feature_cols]
    
    all_num_cols = [c for c in feature_cols if c not in categorical_cols]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Missing')
            df[col] = df[col].astype(str)
    
    for col in all_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in skewed_cols:
        if col in df.columns:
            shift = PREPROCESS_STATE.get('log1p_shift', {}).get(col, 0)
            df[col] = np.log1p(df[col] + shift)
    
    for col in low_missing_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(PREPROCESS_STATE.get('low_missing_medians', {}).get(col, 0))
    
    knn_imputer = PREPROCESS_STATE.get('knn_imputer')
    knn_cols = PREPROCESS_STATE.get('knn_cols', [])
    if knn_imputer is not None and len(knn_cols) > 0:
        df_knn = df[[c for c in knn_cols if c in df.columns]].copy()
        if df_knn.shape[1] > 0:
            df_knn_imputed = knn_imputer.transform(df_knn)
            df[[c for c in knn_cols if c in df.columns]] = df_knn_imputed
    
    for col, median_val in PREPROCESS_STATE.get('other_num_medians', {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    
    return df

def feature_engineering(df):
    target_col = 'Status'
    X = df.drop(columns=[target_col], errors='ignore')
    
    categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    num_cols = [c for c in X.columns if c not in categorical_cols]
    
    if 'Bilirubin' in X.columns and 'Albumin' in X.columns:
        X['Bilirubin_Albumin_ratio'] = X['Bilirubin'] / (X['Albumin'] + 1e-6)
    if 'Stage' in X.columns and 'N_Days' in X.columns:
        X['Stage_N_Days_interact'] = X['Stage'] * X['N_Days']
    
    num_cols = [c for c in X.columns if c not in categorical_cols]
    
    preprocessor = PREPROCESS_STATE['feature_engineering_preprocessor']
    X_processed = preprocessor.transform(X)
    all_feature_names = PREPROCESS_STATE['feature_engineering_columns']
    
    X_processed = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)
    X_processed.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_processed.columns]
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    X_processed = X_processed.fillna(0)
    
    return X_processed

def predict(new_data_path, output_path=None):
    """对新数据进行预测"""
    # 加载模型
    with open('output/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 加载预处理状态
    with open('output/preprocess_state.pkl', 'rb') as f:
        global PREPROCESS_STATE
        PREPROCESS_STATE = pickle.load(f)
    
    # 读取数据
    df = pd.read_csv(new_data_path)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    ids = df[id_col].copy() if id_col in df.columns else pd.Series(range(len(df)))
    
    # 预处理和特征工程
    df_clean = preprocess(df, mode='test')
    X = feature_engineering(df_clean)
    
    # 预测
    preds_encoded = model.predict(X)
    probs = model.predict_proba(X)
    
    status_reverse = PREPROCESS_STATE.get('status_reverse_mapping', {0: 'C', 1: 'CL', 2: 'D'})
    preds = [status_reverse.get(p, str(p)) for p in preds_encoded]
    
    result = pd.DataFrame({'id': ids, 'prediction': preds})
    for i in range(probs.shape[1]):
        result[f'proba_{status_reverse.get(i, f"class_{i}")}'] = probs[:, i]
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict(input_file, output_file)
    else:
        print("Usage: python predict.py <input_csv> [output_csv]")

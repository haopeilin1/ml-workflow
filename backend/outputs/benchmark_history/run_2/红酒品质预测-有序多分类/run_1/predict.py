#!/usr/bin/env python3
"""独立的预测脚本 - 加载模型对新数据进行预测"""

import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

# 预处理状态（与训练时保持一致）
PREPROCESS_STATE = {}

def preprocess(df, mode='test'):
    global PREPROCESS_STATE
    df = df.copy()
    
    df = df.drop(columns=['id'], errors='ignore')
    
    if 'wine_type' in df.columns:
        df['wine_type'] = df['wine_type'].map(PREPROCESS_STATE.get('wine_type_mapping', {'red': 0, 'white': 1}))
        df['wine_type'] = pd.to_numeric(df['wine_type'], errors='coerce')
    
    if 'chlorides' in df.columns:
        df['chlorides'] = pd.to_numeric(df['chlorides'], errors='coerce')
        df['chlorides'] = df['chlorides'].fillna(0.05)
        df['chlorides'] = df['chlorides'].clip(lower=0)
        df['chlorides'] = np.log1p(df['chlorides'])
    
    for col in df.columns:
        if col == 'quality':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def feature_engineering(df):
    df = df.copy()
    
    target_col = 'quality'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    if 'alcohol' in df.columns and 'volatile acidity' in df.columns:
        df['alcohol_volatile_acidity_ratio'] = df['alcohol'] / (df['volatile acidity'].clip(lower=0.001))
    
    if 'free sulfur dioxide' in df.columns and 'total sulfur dioxide' in df.columns:
        df['free_total_sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'].clip(lower=1))
        df['free_total_sulfur_ratio'] = df['free_total_sulfur_ratio'].clip(0, 1)
    
    if 'wine_type' in df.columns:
        if 'residual sugar' in df.columns:
            df['wine_type_residual_sugar'] = df['wine_type'] * df['residual sugar']
        if 'total sulfur dioxide' in df.columns:
            df['wine_type_total_sulfur'] = df['wine_type'] * df['total sulfur dioxide']
        if 'alcohol' in df.columns:
            df['wine_type_alcohol'] = df['wine_type'] * df['alcohol']
        if 'density' in df.columns:
            df['wine_type_density'] = df['wine_type'] * df['density']
        if 'pH' in df.columns:
            df['wine_type_pH'] = df['wine_type'] * df['pH']
    
    if 'density' in df.columns and 'alcohol' in df.columns:
        df['density_alcohol_interact'] = df['density'] * df['alcohol']
        df['density_alcohol_ratio'] = df['density'] / (df['alcohol'].clip(lower=0.1))
    
    if 'pH' in df.columns and 'fixed acidity' in df.columns:
        df['pH_fixed_acidity_interact'] = df['pH'] * df['fixed acidity']
        df['pH_fixed_acidity_ratio'] = df['pH'] / (df['fixed acidity'].clip(lower=0.1))
    
    if 'citric acid' in df.columns and 'fixed acidity' in df.columns:
        df['citric_fixed_acidity_ratio'] = df['citric acid'] / (df['fixed acidity'].clip(lower=0.1))
    
    if 'sulphates' in df.columns and 'alcohol' in df.columns:
        df['sulphates_alcohol_interact'] = df['sulphates'] * df['alcohol']
    
    if 'total sulfur dioxide' in df.columns and 'free sulfur dioxide' in df.columns:
        df['bound_sulfur_dioxide'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
        df['bound_sulfur_dioxide'] = df['bound_sulfur_dioxide'].clip(lower=0)
    
    skewed_cols = ['fixed acidity', 'volatile acidity', 'sulphates', 'residual sugar', 
                   'free sulfur dioxide', 'total sulfur dioxide']
    if 'chlorides' in df.columns:
        skewed_cols.append('chlorides')
    
    for col in skewed_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            col_min = df[col].min()
            if col_min >= 0:
                df[col + '_log'] = np.log1p(df[col])
            else:
                shift = abs(col_min) + 1
                df[col + '_log'] = np.log1p(df[col] + shift)
    
    poly_cols = ['alcohol', 'volatile acidity', 'density', 'pH', 'sulphates']
    for col in poly_cols:
        if col in df.columns:
            df[col + '_squared'] = df[col] ** 2
    
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        if df[col].isna().any():
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    return df

def predict(input_path, output_path=None):
    """对输入数据进行预测"""
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 获取训练时的特征名
    if hasattr(model, 'feature_name_'):
        train_feature_names = model.feature_name_
    elif hasattr(model, 'feature_names_in_'):
        train_feature_names = list(model.feature_names_in_)
    else:
        train_feature_names = model.booster_.feature_name()
    
    # 加载数据
    df = pd.read_csv(input_path)
    original_df = df.copy()
    
    # 预处理和特征工程
    df_clean = preprocess(df, mode='test')
    X = df_clean.drop(columns=['quality'], errors='ignore')
    X_fe = feature_engineering(X)
    
    if isinstance(X_fe, np.ndarray):
        X_fe = pd.DataFrame(X_fe, index=X.index)
    
    X_fe.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_fe.columns]
    
    # 对齐特征
    missing_cols = set(train_feature_names) - set(X_fe.columns)
    extra_cols = set(X_fe.columns) - set(train_feature_names)
    
    for col in missing_cols:
        X_fe[col] = 0
    X_fe = X_fe.drop(columns=list(extra_cols))
    X_fe = X_fe[train_feature_names]
    
    # 预测
    preds = model.predict(X_fe)
    try:
        probs = model.predict_proba(X_fe)
    except Exception:
        probs = None
    
    # 构建结果
    result = original_df.copy()
    result['prediction'] = preds
    
    if probs is not None:
        for i in range(probs.shape[1]):
            result[f'proba_{i}'] = probs[:, i]
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"预测结果已保存到: {output_path}")
    
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python predict.py <输入文件路径> [输出文件路径]")
        print("示例: python predict.py data/test.csv output/predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output/predictions.csv'
    
    result = predict(input_file, output_file)
    print(f"预测完成，共 {len(result)} 条记录")
    print(result.head())

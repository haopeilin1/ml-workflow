#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立预测脚本
用法: python predict.py <输入CSV文件路径> [输出CSV文件路径]
默认输出: predictions.csv
"""

import pandas as pd
import numpy as np
import dill as pickle
import re
import sys
import os

def preprocess(df, scaler):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    target_col = 'smoking'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    numeric_cols = X.columns.tolist()
    X_scaled = scaler.transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=numeric_cols)
    return X_processed

def feature_engineering(df):
    df = df.copy()
    target_col = 'smoking'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    if 'height_cm_' in df.columns and 'weight_kg_' in df.columns:
        df['bmi'] = df['weight_kg_'] / ((df['height_cm_'] / 100) ** 2)
    if 'systolic' in df.columns and 'relaxation' in df.columns:
        df['blood_pressure_diff'] = df['systolic'] - df['relaxation']
    if 'triglyceride' in df.columns and 'HDL' in df.columns:
        df['tg_hdl_ratio'] = df['triglyceride'] / (df['HDL'] + 1e-6)
    if 'AST' in df.columns and 'ALT' in df.columns:
        df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1e-6)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    return df

def main():
    if len(sys.argv) < 2:
        print("用法: python predict.py <输入CSV文件路径> [输出CSV文件路径]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    # 加载模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(input_path)
    
    # 预处理（使用训练时的scaler）
    # 注意：这里需要重新拟合scaler，因为独立脚本无法访问训练时的scaler
    # 实际使用时建议将scaler也保存到model.pkl中
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_data = df.select_dtypes(include=[np.number])
    scaler.fit(numeric_data)
    
    df_clean = preprocess(df, scaler)
    X = feature_engineering(df_clean)
    
    # 清洗特征名
    X.columns = [re.sub('[^\w]', '_', str(c)) for c in X.columns]
    
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
    result.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")

if __name__ == '__main__':
    main()

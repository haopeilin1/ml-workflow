import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

def preprocess(df, scaler):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    target_col = 'smoking'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    num_cols = df.columns.tolist()
    X_scaled = scaler.transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=num_cols, index=df.index)
    return X_scaled_df

def feature_engineering(df):
    df = df.copy()
    if 'weight_kg_' in df.columns and 'height_cm_' in df.columns:
        df['bmi'] = df['weight_kg_'] / ((df['height_cm_'] / 100) ** 2)
    if 'systolic' in df.columns and 'relaxation' in df.columns:
        df['blood_pressure_diff'] = df['systolic'] - df['relaxation']
    if 'triglyceride' in df.columns and 'HDL' in df.columns:
        df['triglyceride_hdl_ratio'] = df['triglyceride'] / (df['HDL'] + 1e-6)
    if 'AST' in df.columns and 'ALT' in df.columns:
        df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1e-6)
    if 'eyesight_left_' in df.columns and 'eyesight_right_' in df.columns:
        df['avg_eyesight'] = (df['eyesight_left_'] + df['eyesight_right_']) / 2
    skewed_cols = ['triglyceride', 'Gtp', 'ALT', 'fasting_blood_sugar']
    for col in skewed_cols:
        if col in df.columns:
            min_val = df[col].min()
            shift = 0 if min_val > 0 else abs(min_val) + 1
            df[f'log_{col}'] = np.log1p(df[col] + shift)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    return df

def predict(input_path, output_path, model_path='output/model.pkl', scaler_path='output/scaler.pkl'):
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 读取数据
    df = pd.read_csv(input_path)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    ids = df[id_col].values if 'id' in df.columns else range(len(df))
    
    # 预处理
    df_clean = preprocess(df, scaler)
    
    # 特征工程
    X = feature_engineering(df_clean)
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    if X.columns.duplicated().any():
        X.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X.columns)]
    
    # 预测
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    # 保存结果
    result = pd.DataFrame({'id': ids, 'prediction': preds, 'probability': probs})
    result.to_csv(output_path, index=False)
    print(f"预测完成，结果保存至 {output_path}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python predict.py <输入CSV路径> <输出CSV路径>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])


import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

def preprocess(df):
    """预处理函数"""
    df = df.copy()
    drop_cols = ['instant', 'casual', 'registered']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')
        df['year'] = df['dteday'].dt.year
        df['month'] = df['dteday'].dt.month
        df['day'] = df['dteday'].dt.day
        df['dayofweek'] = df['dteday'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df = df.drop(columns=['dteday'])
    return df

def feature_engineering(df):
    """特征工程函数"""
    X = df.copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    if 'hr' in X.columns:
        X['hr_sin'] = np.sin(X['hr'] * 2 * np.pi / 24)
        X['hr_cos'] = np.cos(X['hr'] * 2 * np.pi / 24)
    if 'temp' in X.columns and 'hum' in X.columns:
        X['temp_hum'] = X['temp'] * X['hum']
    if 'weathersit' in X.columns and 'windspeed' in X.columns:
        X['weathersit_windspeed'] = X['weathersit'] * X['windspeed']
    X['cnt_lag1'] = 0
    X['cnt_lag2'] = 0
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    return X

def predict(input_path, output_path=None):
    """
    对新数据进行预测
    input_path: 输入 CSV 文件路径
    output_path: 输出 CSV 文件路径（可选，默认在输入文件同目录下生成 predictions.csv）
    """
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(input_path)
    
    # 预处理
    df_clean = preprocess(df)
    X = feature_engineering(df_clean)
    
    # 清洗特征名
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    if X.columns.duplicated().any():
        X.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X.columns)]
    
    # 预测
    preds = model.predict(X)
    if preds.max() < 20:
        preds = np.expm1(preds)
    
    # 保存结果
    result = df.copy()
    result['prediction'] = preds
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_path), 'predictions.csv')
    result.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python predict.py <input_csv_path> [output_csv_path]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    predict(input_path, output_path)

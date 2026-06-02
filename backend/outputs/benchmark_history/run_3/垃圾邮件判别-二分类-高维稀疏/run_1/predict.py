import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

# 预处理状态（从训练时保存的参数）
PREPROCESS_STATE = {
    'log1p_cols': None,
    'scaler': None,
    'label_encoder': None,
}

def preprocess(df, mode='test'):
    df = df.drop(columns=['id'], errors='ignore')
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    target_col = 'spam'
    feature_cols = [c for c in df.columns if c != target_col]
    
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    log1p_cols = PREPROCESS_STATE.get('log1p_cols')
    scaler = PREPROCESS_STATE.get('scaler')
    
    if log1p_cols is None:
        log1p_cols = feature_cols
    
    for col in log1p_cols:
        if col in df.columns:
            min_val = df[col].min()
            if min_val < 0:
                df[col] = np.log1p(df[col] - min_val)
            else:
                df[col] = np.log1p(df[col])
    
    if scaler is not None:
        cols_to_scale = [c for c in log1p_cols if c in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    return df

def feature_engineering(df):
    target_col = 'spam'
    X = df.drop(columns=[target_col], errors='ignore')
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    return X

def predict(input_path, output_path='predictions.csv'):
    """对新数据进行预测"""
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(input_path)
    
    # 预处理
    df_clean = preprocess(df, mode='test')
    X = feature_engineering(df_clean)
    
    # 清洗特征名
    X.columns = [re.sub('[^\w]', '_', str(c)) for c in X.columns]
    
    # 预测
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        preds = model.predict(X)
        probs = preds.astype(float)
    
    # 保存结果
    result = pd.DataFrame({
        'prediction': preds,
        'probability': probs,
    })
    result.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict(input_file, output_file)
    else:
        print("用法: python predict.py <输入CSV文件> [输出CSV文件]")

import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

def load_model(model_path='output/model.pkl'):
    """加载已训练的模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess(df):
    """预处理函数 - 与训练时保持一致"""
    df = df.copy()
    df = df.drop(columns=['id'], errors='ignore')
    feature_cols = [c for c in df.columns if c.startswith('f') and c != 'spam']
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'spam']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    df.columns = [str(c).replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_').replace('\\', '_').replace('<', '_').replace('>', '_').replace(',', '_').replace('.', '_').replace(':', '_').replace(';', '_').replace('{', '_').replace('}', '_').replace('[', '_').replace(']', '_').replace('"', '_').replace("'", '_') for c in df.columns]
    return df

def feature_engineering(df):
    """特征工程 - 与训练时保持一致"""
    X = df.drop(columns=['spam'], errors='ignore')
    feature_cols = [c for c in X.columns if c.startswith('f')]
    X = X.copy()
    for col in feature_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))
    return X

def predict(model, df):
    """对新数据进行预测"""
    df_clean = preprocess(df)
    X = feature_engineering(df_clean)
    X.columns = [re.sub('[^\\w]', '_', str(c)) for c in X.columns]
    if X.columns.duplicated().any():
        X.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X.columns)]
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    
    preds = (probs >= 0.5).astype(int)
    
    result = df.copy()
    result['prediction'] = preds
    result['probability'] = probs
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv> [output_csv]")
        print("Example: python predict.py data/test.csv output/predictions.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'output/predictions.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    model = load_model()
    df = pd.read_csv(input_path)
    result = predict(model, df)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Total samples: {len(result)}")
    print(f"Predicted positive rate: {result['prediction'].mean():.4f}")

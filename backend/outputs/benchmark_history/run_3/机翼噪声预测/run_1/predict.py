import pandas as pd
import numpy as np
import dill as pickle
import re
import os
from sklearn.preprocessing import StandardScaler

# 全局预处理状态（需与训练时一致）
PREPROCESS_STATE = {}

def preprocess(df, mode='test'):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    if mode == 'train':
        f_log_offset = 0
        delta_log_offset = 0
        PREPROCESS_STATE['f_log_offset'] = f_log_offset
        PREPROCESS_STATE['delta_log_offset'] = delta_log_offset
        df['f_log'] = np.log1p(df['f'] - f_log_offset) if f_log_offset > 0 else np.log(df['f'])
        df['delta_log'] = np.log1p(df['delta'] - delta_log_offset) if delta_log_offset > 0 else np.log(df['delta'])
        scaler = StandardScaler()
        PREPROCESS_STATE['scaler'] = scaler
    else:
        f_log_offset = PREPROCESS_STATE.get('f_log_offset', 0)
        delta_log_offset = PREPROCESS_STATE.get('delta_log_offset', 0)
        if 'f' in df.columns:
            df['f_log'] = np.log1p(df['f'] - f_log_offset) if f_log_offset > 0 else np.log(df['f'])
        if 'delta' in df.columns:
            df['delta_log'] = np.log1p(df['delta'] - delta_log_offset) if delta_log_offset > 0 else np.log(df['delta'])
    return df

def feature_engineering(df):
    df = df.copy()
    df['f_x_U'] = df['f'] * df['U_infinity']
    df['alpha_x_delta'] = df['alpha'] * df['delta']
    df['U_div_c'] = df['U_infinity'] / (df['c'] + 1e-8)
    df['f_x_delta'] = df['f'] * df['delta']
    df['f_div_c'] = df['f'] / (df['c'] + 1e-8)
    df['alpha_x_U'] = df['alpha'] * df['U_infinity']
    df['delta_div_c'] = df['delta'] / (df['c'] + 1e-8)
    if 'f_log' in df.columns and 'delta_log' in df.columns:
        df['f_log_x_delta_log'] = df['f_log'] * df['delta_log']
        df['f_log_x_U'] = df['f_log'] * df['U_infinity']
        df['delta_log_x_alpha'] = df['delta_log'] * df['alpha']
    if 'SSPL' in df.columns:
        df = df.drop(columns=['SSPL'])
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    scaler = PREPROCESS_STATE.get('scaler')
    if scaler is not None:
        if hasattr(scaler, 'mean_'):
            df[df.columns] = scaler.transform(df[df.columns])
        else:
            df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

def predict(new_data_path, model_path='output/model.pkl'):
    """
    Load model and predict on new data.
    
    Parameters:
    - new_data_path: path to CSV file with same columns as training data
    - model_path: path to saved model pickle file
    
    Returns:
    - DataFrame with original data + prediction column
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load and preprocess data
    df = pd.read_csv(new_data_path)
    df_clean = preprocess(df, mode='test')
    X = df_clean.drop(columns=['SSPL'], errors='ignore')
    X_fe = feature_engineering(X)
    
    if isinstance(X_fe, np.ndarray):
        X_fe = pd.DataFrame(X_fe, index=X.index)
    
    # Clean column names
    X_fe.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_fe.columns]
    
    # Predict
    predictions = model.predict(X_fe)
    
    # Add predictions to original data
    result = df.copy()
    result['prediction'] = predictions
    
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <new_data.csv>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    result = predict(data_path)
    output_path = data_path.replace('.csv', '_predictions.csv')
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

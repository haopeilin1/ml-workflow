
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

# 预处理状态（与训练时一致）
PREPROCESS_STATE = {
    'fitted': False,
}

def preprocess(df, mode='test'):
    target_col = 'IsFraud'
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = PREPROCESS_STATE.get(f'{col}_median', df[col].median())
            df[col] = df[col].fillna(median_val)
    if 'Transaction_Amount' in df.columns:
        df['Transaction_Amount'] = df['Transaction_Amount'].clip(lower=0)
        df['Transaction_Amount'] = np.log1p(df['Transaction_Amount'])
    if 'Time' in df.columns:
        df['Time'] = df['Time'] / 3600.0
    return df

def feature_engineering(df):
    target_col = 'IsFraud'
    df = df.copy()
    if 'Time' in df.columns:
        df['Time_hour_sin'] = np.sin(2 * np.pi * df['Time'] / 24.0)
        df['Time_hour_cos'] = np.cos(2 * np.pi * df['Time'] / 24.0)
        df['Time_hour'] = (df['Time'].astype(int) % 24).astype(int)
    if 'Transaction_Amount' in df.columns:
        df['Amount_squared'] = df['Transaction_Amount'] ** 2
        df['Amount_sqrt'] = np.sqrt(df['Transaction_Amount'].clip(lower=0))
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    return df

def predict(input_file, output_file, model_path='output/model.pkl', threshold=0.5):
    """
    Load model and make predictions on new data.
    
    Parameters:
    - input_file: path to CSV file with features
    - output_file: path to save predictions
    - model_path: path to the saved model pickle file
    - threshold: classification threshold (default 0.5)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv(input_file)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    
    # Preprocess
    df_clean = preprocess(df, mode='test')
    X = df_clean.drop(columns=['IsFraud'], errors='ignore')
    if X is df_clean:
        X = df_clean.copy()
    
    # Feature engineering
    X_fe = feature_engineering(X)
    if isinstance(X_fe, np.ndarray):
        X_fe = pd.DataFrame(X_fe, index=X.index)
    X_fe.columns = [re.sub('[^\\w]', '_', str(c)) for c in X_fe.columns]
    
    # Predict
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_fe)[:, 1]
    else:
        probs = model.predict(X_fe).astype(float)
    
    preds = (probs >= threshold).astype(int)
    
    # Save results
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(preds)),
        'prediction': preds,
        'probability': probs
    })
    result.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py <input_csv> <output_csv> [threshold]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    predict(input_file, output_file, threshold=threshold)

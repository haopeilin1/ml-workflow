#!/usr/bin/env python3
"""
Independent prediction script.
Usage: python predict.py <input_csv_path> [output_csv_path]
Loads output/model.pkl and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import dill as pickle
import sys
import os
import re
import warnings
warnings.filterwarnings('ignore')

def preprocess(df):
    """Preprocess input data (same as training preprocessing)."""
    df = df.copy()
    
    # Drop id and feat columns
    drop_cols = ['id'] + [f'feat{i}' for i in range(1, 29)]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Transaction_Amount log1p transform
    if 'Transaction_Amount' in df.columns:
        df['Transaction_Amount_log'] = np.log1p(df['Transaction_Amount'])
        df = df.drop(columns=['Transaction_Amount'])
    
    # Time features
    if 'Time' in df.columns:
        df['Time_hour'] = (df['Time'] // 3600) % 24
        df['Time_day_of_week'] = (df['Time'] // (3600 * 24)) % 7
        df = df.drop(columns=['Time'])
    
    # Clean column names
    df.columns = [str(c).replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    # Remove target column if present
    if 'IsFraud' in df.columns:
        df = df.drop(columns=['IsFraud'])
    
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv_path> [output_csv_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load and preprocess data
    df = pd.read_csv(input_path)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    ids = df[id_col] if id_col in df.columns else range(len(df))
    
    X = preprocess(df)
    
    # Clean feature names
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    if X.columns.duplicated().any():
        X.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X.columns)]
    
    # Predict
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    
    preds = (probs >= 0.5).astype(int)
    
    # Save results
    result = pd.DataFrame({
        'id': ids,
        'prediction': preds,
        'probability': probs
    })
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Total samples: {len(result)}")
    print(f"Predicted fraud: {preds.sum()} ({preds.mean()*100:.2f}%)")

if __name__ == '__main__':
    main()

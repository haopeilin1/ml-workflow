#!/usr/bin/env python3
"""
Standalone prediction script for bike sharing demand prediction.
Usage: python predict.py <input_csv> <output_csv>
"""

import pandas as pd
import numpy as np
import dill as pickle
import re
import sys
import os

def preprocess(df):
    """Preprocess input data (same as training)."""
    df = df.copy()
    
    drop_cols = ['casual', 'registered', 'instant']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    if 'dteday' in df.columns:
        df['dteday_parsed'] = pd.to_datetime(df['dteday'], errors='coerce')
        df['dteday_year'] = df['dteday_parsed'].dt.year
        df['dteday_month'] = df['dteday_parsed'].dt.month
        df['dteday_day'] = df['dteday_parsed'].dt.day
        df['dteday_weekday'] = df['dteday_parsed'].dt.weekday
        df['dteday_hour'] = df['dteday_parsed'].dt.hour
        
        duplicate_time_cols = ['dteday_month', 'dteday_weekday', 'dteday_hour']
        df = df.drop(columns=[c for c in duplicate_time_cols if c in df.columns], errors='ignore')
        df = df.drop(columns=['dteday', 'dteday_parsed'], errors='ignore')
    
    categorical_cols = ['season', 'weathersit', 'holiday', 'workingday', 'yr']
    if 'hr' in df.columns:
        categorical_cols.append('hr')
    if 'weekday' in df.columns:
        categorical_cols.append('weekday')
    if 'mnth' in df.columns:
        categorical_cols.append('mnth')
    
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    for col in categorical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    return df


def feature_engineering(df):
    """Feature engineering (same as training)."""
    target_col = 'cnt'
    df = df.copy()
    
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    if 'hr' in df.columns and 'workingday' in df.columns:
        workingday_num = pd.to_numeric(df['workingday'], errors='coerce').fillna(0).astype(int)
        hr_num = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
        df['hr_workingday_interact'] = hr_num * 10 + workingday_num
    
    if 'temp' in df.columns and 'hum' in df.columns:
        temp_num = pd.to_numeric(df['temp'], errors='coerce')
        hum_num = pd.to_numeric(df['hum'], errors='coerce')
        df['temp_hum_interact'] = temp_num * hum_num
    
    if 'windspeed' in df.columns and 'weathersit' in df.columns:
        windspeed_num = pd.to_numeric(df['windspeed'], errors='coerce')
        weathersit_num = pd.to_numeric(df['weathersit'], errors='coerce')
        df['windspeed_weathersit_interact'] = windspeed_num * weathersit_num
    
    if 'hr' in df.columns:
        hr_num = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
        df['is_rush_hour'] = hr_num.isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['is_daytime'] = ((hr_num >= 6) & (hr_num <= 18)).astype(int)
    
    if 'weekday' in df.columns:
        weekday_num = pd.to_numeric(df['weekday'], errors='coerce').fillna(0).astype(int)
        df['is_weekend'] = weekday_num.isin([5, 6]).astype(int)
    
    X = df.drop(columns=[target_col], errors='ignore')
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0)
    
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X[col] = X[col].astype(float)
    
    return X


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv> [output_csv]")
        print("  input_csv: Path to input CSV file with same columns as training data")
        print("  output_csv: Path to save predictions (default: predictions.csv)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    
    # Preprocess
    df_clean = preprocess(df)
    X = feature_engineering(df_clean)
    
    # Clean column names
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    
    # Predict
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    
    # Save
    result = pd.DataFrame({'prediction': preds})
    if 'instant' in df.columns:
        result.insert(0, 'id', df['instant'])
    else:
        result.insert(0, 'id', range(len(preds)))
    
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Prediction summary: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")


if __name__ == '__main__':
    main()

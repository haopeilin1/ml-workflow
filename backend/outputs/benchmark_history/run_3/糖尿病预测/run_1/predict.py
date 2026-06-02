import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

# 预处理参数（从训练阶段继承）
PREPROCESS_STATE = {}

def preprocess(df, mode='test'):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    median_values = PREPROCESS_STATE.get('median_values', {})
    
    for col in zero_invalid_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    for col, median_val in median_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    
    return df

def feature_engineering(df):
    df = df.copy()
    target_col = 'Outcome'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'BMI' in numeric_cols and 'Glucose' in numeric_cols:
        df['BMI_Glucose_interaction'] = df['BMI'] * df['Glucose']
    if 'Age' in numeric_cols and 'Pregnancies' in numeric_cols:
        df['Age_Pregnancies_interaction'] = df['Age'] * df['Pregnancies']
    if 'Glucose' in numeric_cols and 'BMI' in numeric_cols:
        df['Glucose_per_BMI'] = df['Glucose'] / (df['BMI'] + 1e-6)
    if 'Insulin' in numeric_cols and 'Glucose' in numeric_cols:
        df['Insulin_per_Glucose'] = df['Insulin'] / (df['Glucose'] + 1e-6)
    if 'BMI' in numeric_cols:
        df['BMI_squared'] = df['BMI'] ** 2
    if 'Glucose' in numeric_cols:
        df['Glucose_squared'] = df['Glucose'] ** 2
    if 'Age' in numeric_cols:
        df['Age_squared'] = df['Age'] ** 2
    
    skewed_cols = ['Insulin', 'DiabetesPedigreeFunction']
    for col in skewed_cols:
        if col in numeric_cols:
            df[f'{col}_log'] = np.log1p(df[col])
    
    return df

def predict(input_path, output_path):
    """Load model and predict on new data."""
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv(input_path)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    
    # Preprocess
    df_clean = preprocess(df, mode='test')
    X = df_clean.drop(columns=['Outcome'], errors='ignore')
    X_fe = feature_engineering(X)
    if isinstance(X_fe, np.ndarray):
        X_fe = pd.DataFrame(X_fe, index=X.index)
    
    # Clean column names
    X_fe.columns = [re.sub('[^\\w]', '_', str(c)) for c in X_fe.columns]
    if X_fe.columns.duplicated().any():
        X_fe.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X_fe.columns)]
    
    # Predict
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_fe)[:, 1]
    else:
        probs = model.predict(X_fe).astype(float)
    preds = (probs >= 0.5).astype(int)
    
    # Save results
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(preds)),
        'prediction': preds,
        'probability': probs
    })
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])

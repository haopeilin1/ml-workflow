#!/usr/bin/env python3
"""
Standalone prediction script for binary classification model.
Usage: python predict.py <input_csv> <output_csv>
"""

import pandas as pd
import numpy as np
import dill as pickle
import re
import sys
import os

def preprocess(df):
    """Apply the same preprocessing as training."""
    df = df.copy()
    
    # Drop id column if exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Replace 0 with NaN for medical columns
    zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_as_missing_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    # Drop target column if exists
    if 'Outcome' in df.columns:
        df = df.drop(columns=['Outcome'])
    
    # Impute missing values with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    feature_cols = df.columns.tolist()
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=feature_cols,
        index=df.index
    )
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_imputed),
        columns=feature_cols,
        index=df.index
    )
    
    # Feature engineering
    df_fe = df_scaled.copy()
    
    if 'Glucose' in df_fe.columns and 'BMI' in df_fe.columns:
        df_fe['Glucose_BMI_interaction'] = df_fe['Glucose'] * df_fe['BMI']
    if 'Glucose' in df_fe.columns and 'Age' in df_fe.columns:
        df_fe['Glucose_Age_interaction'] = df_fe['Glucose'] * df_fe['Age']
    if 'Age' in df_fe.columns and 'Pregnancies' in df_fe.columns:
        df_fe['Age_Pregnancies_interaction'] = df_fe['Age'] * df_fe['Pregnancies']
    if 'BMI' in df_fe.columns and 'Age' in df_fe.columns:
        df_fe['BMI_Age_interaction'] = df_fe['BMI'] * df_fe['Age']
    
    if 'Glucose' in df_fe.columns:
        df_fe['Glucose_squared'] = df_fe['Glucose'] ** 2
    if 'BMI' in df_fe.columns:
        df_fe['BMI_squared'] = df_fe['BMI'] ** 2
    if 'Insulin' in df_fe.columns:
        df_fe['Insulin_squared'] = df_fe['Insulin'] ** 2
    
    if 'Glucose' in df_fe.columns and 'BMI' in df_fe.columns:
        df_fe['Glucose_BMI_ratio'] = df_fe['Glucose'] / (df_fe['BMI'] + 1e-8)
    if 'Insulin' in df_fe.columns and 'Glucose' in df_fe.columns:
        df_fe['Insulin_Glucose_ratio'] = df_fe['Insulin'] / (df_fe['Glucose'] + 1e-8)
    
    # Ensure numeric types
    for col in df_fe.columns:
        if df_fe[col].dtype == 'object' or df_fe[col].dtype.name == 'category':
            df_fe[col] = pd.factorize(df_fe[col])[0]
    
    return df_fe


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv(input_path)
    original_df = df.copy()
    
    # Preprocess
    X = preprocess(df)
    
    # Clean feature names
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    
    # Align features with model
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    elif hasattr(model, 'feature_name_'):
        expected_features = model.feature_name_
    else:
        expected_features = X.columns.tolist()
    
    # Add missing columns
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    
    # Keep only expected columns in order
    X = X[[col for col in expected_features if col in X.columns]]
    
    # Predict
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        preds = model.predict(X)
        probs = preds.astype(float)
    
    # Save results
    result = original_df.copy()
    result['prediction'] = preds
    result['probability'] = probs
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == '__main__':
    main()

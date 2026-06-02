import pandas as pd
import numpy as np
import dill as pickle
import re
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def load_model(model_path='output/model.pkl'):
    """Load the trained model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess(df):
    """Preprocess input data (same as training)."""
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    target_col = 'median_house_value'
    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ], remainder='passthrough')
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)
    feature_names = list(num_cols)
    if cat_cols:
        encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        feature_names.extend(encoder.get_feature_names_out(cat_cols))
    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    if y is not None:
        X_processed[target_col] = y.values
    return X_processed

def feature_engineering(df):
    """Apply feature engineering (same as training)."""
    df = df.copy()
    target_col = 'median_house_value'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    if 'total_rooms' in X.columns and 'households' in X.columns:
        X['rooms_per_household'] = X['total_rooms'] / (X['households'] + 1e-5)
    if 'total_bedrooms' in X.columns and 'total_rooms' in X.columns:
        X['bedrooms_per_room'] = X['total_bedrooms'] / (X['total_rooms'] + 1e-5)
    if 'population' in X.columns and 'households' in X.columns:
        X['population_per_household'] = X['population'] / (X['households'] + 1e-5)
    if 'median_income' in X.columns and 'total_rooms' in X.columns:
        X['income_per_room'] = X['median_income'] / (X['total_rooms'] + 1e-5)
    if 'total_bedrooms' in X.columns and 'households' in X.columns:
        X['bedrooms_per_household'] = X['total_bedrooms'] / (X['households'] + 1e-5)
    if 'total_rooms' in X.columns and 'population' in X.columns:
        X['rooms_per_population'] = X['total_rooms'] / (X['population'] + 1e-5)
    if 'median_income' in X.columns and 'households' in X.columns:
        X['income_per_household'] = X['median_income'] / (X['households'] + 1e-5)
    skewed_cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
    for col in skewed_cols:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(X[col])
    if 'median_income' in X.columns:
        X['median_income_sq'] = X['median_income'] ** 2
        X['median_income_cub'] = X['median_income'] ** 3
    if 'longitude' in X.columns and 'latitude' in X.columns:
        X['longitude_latitude_interaction'] = X['longitude'] * X['latitude']
        X['longitude_sq'] = X['longitude'] ** 2
        X['latitude_sq'] = X['latitude'] ** 2
    if 'population' in X.columns and 'total_rooms' in X.columns:
        X['density'] = X['population'] / (X['total_rooms'] + 1e-5)
    if 'housing_median_age' in X.columns:
        X['age_squared'] = X['housing_median_age'] ** 2
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X

def predict(new_data_path, model_path='output/model.pkl'):
    """Load model and make predictions on new data."""
    model = load_model(model_path)
    df = pd.read_csv(new_data_path)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    df_clean = preprocess(df)
    target_col = 'median_house_value'
    X = df_clean.drop(columns=[target_col], errors='ignore')
    X_fe = feature_engineering(X)
    if isinstance(X_fe, np.ndarray):
        X_fe = pd.DataFrame(X_fe, index=X.index)
    X_fe.columns = [re.sub('[^\\w]', '_', str(c)) for c in X_fe.columns]
    if X_fe.columns.duplicated().any():
        X_fe.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X_fe.columns)]
    predictions = model.predict(X_fe)
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(predictions)),
        'prediction': predictions
    })
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv_path> [output_csv_path]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    result = predict(input_path)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

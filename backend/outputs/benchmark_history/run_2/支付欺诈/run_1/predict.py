
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

def clean_feature_names(df):
    df = df.copy()
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    return df

def preprocess(df):
    df = df.copy()
    drop_cols = ['id']
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if 'Category' in df.columns:
        df['Category'] = df['Category'].fillna('unknown')
    if 'isWeekend' in df.columns:
        df['isWeekend'] = df['isWeekend'].fillna(0)
    for col in ['numItems', 'paymentMethodAgeDays']:
        if col in df.columns:
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val
            df[col] = np.log1p(df[col])
    if 'localTime' in df.columns:
        df['localTime'] = df['localTime'] ** 2
    df = clean_feature_names(df)
    return df

def feature_engineering(df):
    df = df.copy()
    target_col = 'label'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    payment_col = None
    category_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'paymentmethod' in col_lower and 'age' not in col_lower:
            payment_col = col
        if 'category' in col_lower:
            category_col = col
    if payment_col and category_col:
        combo_name = f'{payment_col}_{category_col}_combo'
        df[combo_name] = df[payment_col].astype(str) + '_' + df[category_col].astype(str)

    account_col = None
    payment_age_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'accountage' in col_lower:
            account_col = col
        if 'paymentmethodage' in col_lower:
            payment_age_col = col
    if account_col and payment_age_col:
        ratio_name = 'accountAge_paymentAge_ratio'
        df[ratio_name] = df[account_col] / (df[payment_age_col] + 1)
        df[ratio_name] = df[ratio_name].clip(upper=df[ratio_name].quantile(0.99))

    numitems_col = None
    for col in df.columns:
        if 'numitems' in col.lower():
            numitems_col = col
            break
    if numitems_col:
        bins = [-np.inf, 1, 2, np.inf]
        labels = ['1_item', '2_items', '3plus_items']
        bin_col_name = f'{numitems_col}_binned'
        df[bin_col_name] = pd.cut(df[numitems_col], bins=bins, labels=labels)

    weekend_col = None
    localtime_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'isweekend' in col_lower:
            weekend_col = col
        if 'localtime' in col_lower:
            localtime_col = col
    if weekend_col and localtime_col:
        weekend_time_name = 'weekend_localtime_interaction'
        df[weekend_time_name] = df[weekend_col] * df[localtime_col]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols)
    ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(df)
    try:
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
        all_feature_names = list(cat_feature_names) + num_cols
    except:
        all_feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

    X_df = pd.DataFrame(X_processed, columns=all_feature_names, index=df.index)
    X_df = clean_feature_names(X_df)
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        X_df[col] = X_df[col].astype(float)
    X_df = X_df.fillna(0)

    return X_df

def predict(input_path, output_path=None):
    """
    Load model and make predictions on new data.

    Parameters:
    - input_path: str, path to input CSV file
    - output_path: str, optional, path to save predictions

    Returns:
    - DataFrame with predictions
    """
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data
    df = pd.read_csv(input_path)

    # Preprocess
    df_clean = preprocess(df)
    X = feature_engineering(df_clean)

    # Predict
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    preds = (probs >= 0.5).astype(int)

    # Build result
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(preds)),
        'prediction': preds,
        'probability': probs
    })

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv> [output_csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    result = predict(input_file, output_file)
    print(result.head())

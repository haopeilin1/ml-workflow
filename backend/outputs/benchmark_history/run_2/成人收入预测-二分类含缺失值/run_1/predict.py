
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

def load_model(model_path='model.pkl'):
    """Load the trained model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_data(df, state_path='preprocess_state.pkl'):
    """Preprocess new data using saved state."""
    # Try to load preprocessing state
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                'relationship', 'race', 'sex', 'native_country']
    num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    df = df.copy()
    if 'id' in df.columns:
        ids = df['id']
        df = df.drop(columns=['id'])
    else:
        ids = pd.Series(range(len(df)))
    
    # Load state if available
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        cat_transformer = state['cat_transformer']
        num_transformer = state['num_transformer']
    else:
        # Fallback: fit on the fly (not recommended for production)
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        # This requires training data - in production, always use saved state
        raise FileNotFoundError(f"Preprocessing state not found at {state_path}")
    
    cat_data = cat_transformer.transform(df[cat_cols])
    num_data = num_transformer.transform(df[num_cols])
    
    cat_feature_names = []
    for i, col in enumerate(cat_cols):
        encoder = cat_transformer.named_steps['encoder']
        if hasattr(encoder, 'categories_'):
            categories = encoder.categories_[i]
            for cat in categories:
                cat_feature_names.append(f"{col}_{cat}")
    
    processed = pd.DataFrame(
        np.hstack([cat_data, num_data]),
        columns=cat_feature_names + num_cols,
        index=df.index
    )
    
    # Feature engineering
    X = processed.copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    if 'capital_gain' in X.columns:
        X['capital_gain_log'] = np.log1p(X['capital_gain'])
    if 'capital_loss' in X.columns:
        X['capital_loss_log'] = np.log1p(X['capital_loss'])
    if 'age' in X.columns:
        X['age_squared'] = X['age'] ** 2
    if 'capital_gain' in X.columns and 'hours_per_week' in X.columns:
        X['capital_gain_hours_interaction'] = X['capital_gain'] * X['hours_per_week']
    if 'education_num' in X.columns and 'hours_per_week' in X.columns:
        X['education_hours_interaction'] = X['education_num'] * X['hours_per_week']
    if 'capital_gain' in X.columns and 'capital_loss' in X.columns:
        X['capital_net'] = X['capital_gain'] - X['capital_loss']
    
    X = X.fillna(0)
    X.columns = [re.sub('[^\\w]', '_', str(c)) for c in X.columns]
    
    return X, ids

def predict(data_path, model_path='model.pkl', output_path='predictions.csv'):
    """Make predictions on new data."""
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    X, ids = preprocess_data(df)
    
    # Align features with model expectations
    if hasattr(model, 'feature_name_'):
        expected_features = model.feature_name_
    elif hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    else:
        expected_features = X.columns.tolist()
    
    missing_cols = set(expected_features) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[list(expected_features)]
    
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
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv_path> [output_csv_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    predict(input_path, output_path=output_path)

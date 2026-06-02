
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
from sklearn.preprocessing import OneHotEncoder

# 加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess(df):
    """预处理函数 - 与训练时一致"""
    id_cols = ['id', 'name_email_similarity', 'days_since_request', 
               'intended_balcon_amount', 'velocity_6h', 'velocity_24h', 
               'velocity_4w', 'session_length_in_minutes']
    constant_cols = ['device_fraud_count', 'month']
    drop_cols = id_cols + constant_cols
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    cols_with_minus_one = ['prev_address_months_count', 'current_address_months_count', 
                           'bank_months_count', 'device_distinct_emails_8w']
    if 'session_length_in_minutes' in df.columns:
        cols_with_minus_one.append('session_length_in_minutes')
    
    for col in cols_with_minus_one:
        if col in df.columns:
            missing_col_name = f'{col}_missing'
            df[missing_col_name] = (df[col] == -1).astype(int)
            df.loc[df[col] == -1, col] = np.nan
            df[col] = df[col].fillna(df[col].median())
    
    log_cols = ['prev_address_months_count', 'bank_branch_count_8w']
    if 'session_length_in_minutes' in df.columns:
        log_cols.append('session_length_in_minutes')
    for col in log_cols:
        if col in df.columns:
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val
            df[col] = np.log1p(df[col])
    
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    return df

def feature_engineering(df):
    """特征工程函数 - 与训练时一致"""
    X = df.copy()
    
    if 'income' in X.columns and 'credit_risk_score' in X.columns:
        X['income_credit_risk_interaction'] = X['income'] * X['credit_risk_score']
    
    if 'prev_address_months_count' in X.columns and 'current_address_months_count' in X.columns:
        X['address_change_ratio'] = np.where(
            X['current_address_months_count'] > 0,
            X['prev_address_months_count'] / (X['current_address_months_count'] + 1),
            0
        )
    
    if 'device_distinct_emails_8w' in X.columns and 'date_of_birth_distinct_emails_4w' in X.columns:
        X['email_diff_device_dob'] = X['device_distinct_emails_8w'] - X['date_of_birth_distinct_emails_4w']
    
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category', 'string']).columns.tolist()
    
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat_encoded = encoder.fit_transform(X[cat_cols])
        encoded_feature_names = encoder.get_feature_names_out(cat_cols)
        encoded_feature_names = [re.sub(r'[^\w]', '_', str(c)) for c in encoded_feature_names]
        X_cat_df = pd.DataFrame(X_cat_encoded, columns=encoded_feature_names, index=X.index)
        X_num_df = X[num_cols].reset_index(drop=True)
        X_cat_df = X_cat_df.reset_index(drop=True)
        X = pd.concat([X_num_df, X_cat_df], axis=1)
    
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'string':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    X = X.fillna(0)
    return X

def predict(new_data_path, output_path='predictions.csv'):
    """
    对新数据进行预测
    Args:
        new_data_path: 新数据 CSV 文件路径
        output_path: 输出预测结果路径
    """
    df = pd.read_csv(new_data_path)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    
    df_clean = preprocess(df)
    X = feature_engineering(df_clean)
    
    # 确保特征列与训练时一致
    # 模型训练时的特征名
    expected_features = model.feature_name_ if hasattr(model, 'feature_name_') else None
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    
    preds = (probs >= 0.5).astype(int)
    
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(preds)),
        'prediction': preds,
        'probability': probs
    })
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <new_data.csv>")

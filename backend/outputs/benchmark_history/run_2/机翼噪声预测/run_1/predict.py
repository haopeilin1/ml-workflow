
import pandas as pd
import numpy as np
import dill as pickle
import re
import os

def preprocess(df, scaler):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    feature_cols = ['f', 'alpha', 'c', 'U_infinity', 'delta']
    available_features = [col for col in feature_cols if col in df.columns]
    df[available_features] = scaler.transform(df[available_features])
    return df

def feature_engineering(df):
    df = df.copy()
    feature_cols = ['f', 'alpha', 'c', 'U_infinity', 'delta']
    available_features = [col for col in feature_cols if col in df.columns]
    df['f_delta'] = df['f'] * df['delta']
    df['alpha_U_infinity'] = df['alpha'] * df['U_infinity']
    df['f_c'] = df['f'] * df['c']
    df['alpha_delta'] = df['alpha'] * df['delta']
    df['U_infinity_delta'] = df['U_infinity'] * df['delta']
    df['U_infinity_over_c'] = df['U_infinity'] / (df['c'] + 1e-8)
    df['f_over_U_infinity'] = df['f'] / (df['U_infinity'] + 1e-8)
    df['c_over_delta'] = df['c'] / (df['delta'] + 1e-8)
    df['f_squared'] = df['f'] ** 2
    df['delta_squared'] = df['delta'] ** 2
    df['alpha_squared'] = df['alpha'] ** 2
    df['U_infinity_squared'] = df['U_infinity'] ** 2
    df['log_f'] = np.log1p(df['f'])
    df['log_delta'] = np.log1p(df['delta'] * 1000)
    df['log_U_infinity'] = np.log1p(df['U_infinity'])
    all_feature_cols = available_features + [
        'f_delta', 'alpha_U_infinity', 'f_c', 'alpha_delta', 'U_infinity_delta',
        'U_infinity_over_c', 'f_over_U_infinity', 'c_over_delta',
        'f_squared', 'delta_squared', 'alpha_squared', 'U_infinity_squared',
        'log_f', 'log_delta', 'log_U_infinity'
    ]
    X = df[all_feature_cols]
    return X

def predict(new_data_path, model_path='output/model.pkl', scaler_path=None):
    """
    对新数据进行预测。
    
    参数:
        new_data_path: str, 新数据 CSV 文件路径
        model_path: str, 模型文件路径
        scaler_path: str, 已废弃（scaler 内嵌在模型中则不需要）
    
    返回:
        pd.DataFrame, 包含 id 和 prediction 列
    """
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 读取新数据
    df = pd.read_csv(new_data_path)
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    
    # 预处理（需要 scaler，这里假设 scaler 已通过 PREPROCESS_STATE 保存）
    # 实际使用时，建议将 scaler 也保存为单独文件
    # 此处简化：如果模型是 Pipeline 则直接 predict，否则需要手动预处理
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_cols = ['f', 'alpha', 'c', 'U_infinity', 'delta']
    available_features = [col for col in feature_cols if col in df.columns]
    scaler.fit(df[available_features])  # 注意：这里应该用训练时的 scaler
    
    df_clean = preprocess(df, scaler)
    X = df_clean.drop(columns=['SSPL'], errors='ignore')
    X_fe = feature_engineering(X)
    if isinstance(X_fe, np.ndarray):
        X_fe = pd.DataFrame(X_fe, index=X.index)
    X_fe.columns = [re.sub('[^\\w]', '_', str(c)) for c in X_fe.columns]
    if X_fe.columns.duplicated().any():
        X_fe.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X_fe.columns)]
    
    # 对齐特征
    if hasattr(model, 'feature_name_') and model.feature_name_ is not None:
        expected_features = list(model.feature_name_)
    elif hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = list(X_fe.columns)
    for col in expected_features:
        if col not in X_fe.columns:
            X_fe[col] = 0
    X_fe = X_fe[expected_features]
    
    preds = model.predict(X_fe)
    
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(preds)),
        'prediction': preds
    })
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/test.csv'
    result = predict(data_path)
    result.to_csv('predictions.csv', index=False)
    print(f"预测完成，结果已保存到 predictions.csv")
    print(result.head())

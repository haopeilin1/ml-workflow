import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

# 预处理函数（与训练时一致）
def preprocess(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    for col in ['StreamingTV', 'StreamingMovies']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
    return df

def feature_engineering(df):
    df = df.copy()
    if 'Churn' in df.columns:
        y = df['Churn']
        X = df.drop(columns=['Churn'])
    else:
        X = df
    
    tenure_cols = [c for c in X.columns if 'tenure' in c.lower()]
    monthly_charges_cols = [c for c in X.columns if 'MonthlyCharges' in c.lower()]
    total_charges_cols = [c for c in X.columns if 'TotalCharges' in c.lower()]
    
    if tenure_cols and monthly_charges_cols:
        X['tenure_monthly_ratio'] = np.where(X[tenure_cols[0]] > 0, X[monthly_charges_cols[0]] / X[tenure_cols[0]], 0)
    if total_charges_cols and tenure_cols:
        X['avg_monthly_total'] = np.where(X[tenure_cols[0]] > 0, X[total_charges_cols[0]] / X[tenure_cols[0]], 0)
    if tenure_cols:
        X['tenure_group'] = pd.cut(X[tenure_cols[0]], bins=[-np.inf, 12, 24, 48, np.inf], labels=[0, 1, 2, 3]).astype(float)
    
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    X = X.fillna(0)
    
    if 'y' in locals():
        X['Churn'] = y.values
    return X

def predict(new_data_path, output_path=None):
    """
    对新数据进行预测。
    
    参数:
        new_data_path: str, 新数据 CSV 文件路径
        output_path: str, 输出 CSV 文件路径（可选，默认返回 DataFrame）
    
    返回:
        pd.DataFrame: 包含 id, prediction, probability 列
    """
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(new_data_path)
    
    # 保存 id 列
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    ids = df[id_col] if id_col in df.columns else range(len(df))
    
    # 预处理
    df_clean = preprocess(df)
    
    # 特征工程
    X = feature_engineering(df_clean)
    if 'Churn' in X.columns:
        X = X.drop(columns=['Churn'])
    
    # 清洗特征名
    X.columns = [re.sub('[^\\w]', '_', str(c)) for c in X.columns]
    
    # 预测
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    preds = (probs >= 0.5).astype(int)
    
    result = pd.DataFrame({
        'id': ids,
        'prediction': preds,
        'probability': probs
    })
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"预测结果已保存到 {output_path}")
    
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict(input_file, output_file)
    else:
        print("用法: python predict.py <输入CSV文件> [输出CSV文件]")

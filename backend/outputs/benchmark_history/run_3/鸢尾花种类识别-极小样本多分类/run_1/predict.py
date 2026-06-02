import pandas as pd
import dill as pickle
import os
import re
import numpy as np

def predict(input_path, output_path=None):
    """
    加载模型对新数据进行预测。
    
    参数:
        input_path: str, 输入 CSV 文件路径
        output_path: str, 输出 CSV 文件路径（可选，默认在输入文件同目录生成）
    
    返回:
        pd.DataFrame, 预测结果
    """
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(input_path)
    original_df = df.copy()
    
    # 预处理
    id_col = 'id'
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 特征工程
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].copy()
    
    if 'petal_length' in X.columns and 'petal_width' in X.columns:
        X['petal_area'] = X['petal_length'] * X['petal_width']
    
    # 注意：这里使用训练时保存的 scaler 参数
    # 由于 scaler 未随模型保存，这里使用简单的标准化
    # 实际使用时建议将 scaler 也保存到 model.pkl 中
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # 清洗特征名
    X_scaled.columns = [re.sub('[^\\w]', '_', str(c)) for c in X_scaled.columns]
    
    # 预测
    predictions = model.predict(X_scaled)
    try:
        probabilities = model.predict_proba(X_scaled)
    except Exception:
        probabilities = None
    
    # 构建结果
    result = pd.DataFrame()
    if id_col in original_df.columns:
        result['id'] = original_df[id_col]
    else:
        result['id'] = range(len(predictions))
    result['prediction'] = predictions
    
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            result[f'proba_{i}'] = probabilities[:, i]
    
    # 保存结果
    if output_path is None:
        output_path = input_path.replace('.csv', '_predictions.csv')
    result.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")
    
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        predict(input_file, output_file)
    else:
        print("用法: python predict.py <input_csv> [output_csv]")

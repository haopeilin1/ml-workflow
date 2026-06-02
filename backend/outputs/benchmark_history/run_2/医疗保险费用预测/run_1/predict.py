
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

# 加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(new_data_path, output_path='predictions.csv'):
    """
    对新数据进行预测。
    
    Parameters:
    - new_data_path: str, 新数据 CSV 文件路径
    - output_path: str, 输出预测结果 CSV 文件路径
    
    Returns:
    - pd.DataFrame, 包含 id 和 prediction 列
    """
    df = pd.read_csv(new_data_path)
    
    # 注意：此预测脚本假设输入数据已经过预处理
    # 如果数据是原始格式，需要在此处添加预处理步骤
    
    # 获取特征列（排除 id 和目标列）
    feature_cols = [c for c in df.columns if c not in ['id', 'charges']]
    X = df[feature_cols]
    
    # 预测
    predictions = model.predict(X)
    
    # 构建结果
    id_col = 'id' if 'id' in df.columns else df.columns[0]
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(predictions)),
        'prediction': predictions
    })
    
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict(input_file, output_file)
    else:
        print("Usage: python predict.py <input_csv> [output_csv]")


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
    对新数据进行预测
    new_data_path: CSV 文件路径，包含与训练数据相同的列（可以没有目标列）
    output_path: 输出预测结果的 CSV 文件路径
    """
    df = pd.read_csv(new_data_path)
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    
    # 构建结果
    result = df.copy()
    result['prediction'] = predictions
    
    # 添加概率列
    for i in range(probabilities.shape[1]):
        result[f'proba_class_{i}'] = probabilities[:, i]
    
    result.to_csv(output_path, index=False)
    print(f"预测完成，结果已保存到 {output_path}")
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict(input_file, output_file)
    else:
        print("用法: python predict.py <输入CSV文件> [输出CSV文件]")

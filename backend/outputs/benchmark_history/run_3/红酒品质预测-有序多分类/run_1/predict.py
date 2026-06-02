
import pandas as pd
import dill as pickle
import re
import numpy as np
import os

def predict(new_data_path, output_path='predictions.csv'):
    """加载模型并对新数据进行预测"""
    # 加载模型
    with open('output/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(new_data_path)
    
    # 预处理（简化版，实际使用时需要完整的预处理流程）
    # 这里假设输入数据已经过预处理
    # 请根据实际情况调整
    
    # 预测
    preds = model.predict(df)
    probs = model.predict_proba(df)
    
    # 保存结果
    result = pd.DataFrame({'prediction': preds})
    for i in range(probs.shape[1]):
        result[f'proba_{i}'] = probs[:, i]
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <input_csv_path>")

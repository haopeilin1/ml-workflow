#!/usr/bin/env python3
"""
独立预测脚本
用法: python predict.py <输入CSV文件路径> [输出CSV文件路径]
示例: python predict.py new_data.csv predictions.csv
"""
import pandas as pd
import dill as pickle
import sys
import os

def load_model(model_path='output/model.pkl'):
    """加载保存的模型"""
    if not os.path.exists(model_path):
        # 尝试从当前目录的output子目录加载
        model_path = os.path.join(os.path.dirname(__file__), 'output', 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, data_path, output_path=None):
    """对数据进行预测"""
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"加载数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 保存ID列（如果存在）
    id_col = None
    if 'id' in df.columns:
        id_col = df['id'].copy()
    
    # 预测
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
    else:
        predictions = model.predict(df)
        probabilities = predictions.astype(float)
    
    # 构建结果
    result = pd.DataFrame()
    if id_col is not None:
        result['id'] = id_col
    else:
        result['id'] = range(len(predictions))
    
    result['prediction'] = predictions
    result['probability'] = probabilities
    
    # 保存结果
    if output_path is None:
        output_path = 'predictions.csv'
    
    result.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")
    print(f"预测统计: 正类={predictions.sum()}, 负类={(1-predictions).sum()}")
    
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python predict.py <输入CSV文件路径> [输出CSV文件路径]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    model = load_model()
    predict(model, input_path, output_path)

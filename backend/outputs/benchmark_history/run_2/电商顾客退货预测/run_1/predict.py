#!/usr/bin/env python3
"""
独立预测脚本
用法: python predict.py <输入CSV文件路径> [输出CSV文件路径]
示例: python predict.py new_data.csv predictions.csv
"""
import pandas as pd
import numpy as np
import dill as pickle
import sys
import os
import re
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path='output/model.pkl'):
    """加载已保存的模型"""
    if not os.path.exists(model_path):
        # 尝试从当前目录查找
        if os.path.exists('model.pkl'):
            model_path = 'model.pkl'
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"模型已加载: {model_path}")
    return model

def predict(model, input_path, output_path=None):
    """对输入数据进行预测"""
    # 读取数据
    df = pd.read_csv(input_path)
    print(f"读取数据: {len(df)} 行, {len(df.columns)} 列")
    
    # 保存ID列（如果存在）
    id_col = 'order_id'
    if id_col in df.columns:
        ids = df[id_col].copy()
    else:
        ids = pd.Series(range(len(df)), name='id')
    
    # 预测
    try:
        probabilities = model.predict_proba(df)[:, 1]
    except Exception as e:
        print(f"predict_proba失败: {e}, 尝试predict...")
        probabilities = model.predict(df).astype(float)
    
    predictions = (probabilities >= 0.5).astype(int)
    
    # 构建结果
    result = pd.DataFrame({
        'id': ids.values,
        'prediction': predictions,
        'probability': probabilities
    })
    
    # 保存结果
    if output_path is None:
        output_path = 'predictions.csv'
    
    result.to_csv(output_path, index=False)
    print(f"预测结果已保存: {output_path}")
    print(f"预测统计: 正类={predictions.sum()}, 负类={len(predictions) - predictions.sum()}, 正类比例={predictions.mean()*100:.2f}%")
    
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python predict.py <输入CSV文件路径> [输出CSV文件路径]")
        print("示例: python predict.py new_data.csv predictions.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    model = load_model()
    result = predict(model, input_path, output_path)
    print("\n预测完成!")

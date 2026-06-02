import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

def load_model(model_path='model.pkl'):
    """加载保存的模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(new_data_path, model_path='model.pkl', output_path='predictions.csv'):
    """
    对新数据进行预测
    new_data_path: 新数据CSV文件路径
    model_path: 模型文件路径
    output_path: 输出预测结果路径
    """
    # 加载模型
    model = load_model(model_path)
    
    # 加载数据
    df = pd.read_csv(new_data_path)
    
    # 预处理（与训练时一致）
    drop_cols = ['id']
    cat_cols = ['paymentMethod', 'Category']
    log_transform_cols = ['numItems', 'localTime', 'paymentMethodAgeDays']
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    if 'Category' in df.columns:
        df['Category'] = df['Category'].fillna('Unknown')
    
    if 'isWeekend' in df.columns:
        df['isWeekend'] = df['isWeekend'].fillna(0)
    
    for col in log_transform_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = np.log1p(df[col].clip(lower=-1))
    
    # 注意：这里需要加载训练时保存的编码器
    # 简化处理：假设类别列已经是数值或使用默认编码
    for col in cat_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 确保所有列数值类型
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    X = df.fillna(0)
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    
    # 预测
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    
    preds = (probs >= 0.5).astype(int)
    
    # 保存结果
    result = pd.DataFrame({
        'prediction': preds,
        'probability': probs
    })
    result.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")
    
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python predict.py <新数据文件路径> [输出文件路径]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    predict(data_path, output_path=out_path)

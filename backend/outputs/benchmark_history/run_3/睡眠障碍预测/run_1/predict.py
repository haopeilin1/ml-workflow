
import pandas as pd
import dill as pickle
import dill
import os
import re
import warnings
warnings.filterwarnings('ignore')

def predict(new_data_path, output_path='predictions.csv'):
    """
    加载模型并对新数据进行预测。
    
    Parameters:
    - new_data_path: str, 新数据 CSV 文件路径
    - output_path: str, 输出预测结果 CSV 文件路径
    
    Returns:
    - pd.DataFrame, 预测结果
    """
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        pipeline = dill.load(f)
    
    # 加载数据
    df = pd.read_csv(new_data_path)
    
    # 预处理（与训练时一致）
    if 'Person ID' in df.columns:
        person_ids = df['Person ID'].copy()
    else:
        person_ids = pd.Series(range(len(df)))
    
    # 删除不需要的列
    drop_cols = [
        'Sleep Duration (hours)',
        'Quality of Sleep (scale: 1-10)',
        'Physical Activity Level (minutes/day)',
        'Stress Level (scale: 1-10)',
        'Heart Rate (bpm)',
        'Daily Steps'
    ]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 拆分血压列
    bp_col = 'Blood Pressure (systolic/diastolic)'
    if bp_col in df.columns:
        bp_split = df[bp_col].str.split('/', expand=True)
        if bp_split.shape[1] == 2:
            df['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
            df['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
        df = df.drop(columns=[bp_col])
    
    # 删除目标列（如果存在）
    if 'Sleep Disorder' in df.columns:
        df = df.drop(columns=['Sleep Disorder'])
    
    # 预测
    preds_encoded = pipeline.predict(df)
    probs = pipeline.predict_proba(df)
    
    # 反编码标签
    label_map = {0: 'No Disorder', 1: 'Insomnia', 2: 'Sleep Apnea'}
    preds = [label_map.get(p, str(p)) for p in preds_encoded]
    
    # 构建结果
    result = pd.DataFrame({'Person ID': person_ids, 'prediction': preds})
    for i, cls in enumerate(['No Disorder', 'Insomnia', 'Sleep Apnea']):
        if i < probs.shape[1]:
            result[f'proba_{cls}'] = probs[:, i]
    
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict(input_path, output_path)
    else:
        print("Usage: python predict.py <input_csv> [output_csv]")

import pandas as pd
import dill as pickle
import os
import sys

def predict(input_path, output_path):
    """
    加载模型并对新数据进行预测。
    
    参数:
        input_path: 输入 CSV 文件路径（必须包含特征列）
        output_path: 输出 CSV 文件路径（包含预测结果）
    """
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(input_path)
    
    # 预测
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)
    
    # 构建输出
    result = df.copy()
    result['prediction'] = predictions
    
    for i in range(probabilities.shape[1]):
        result[f'proba_{i}'] = probabilities[:, i]
    
    # 保存结果
    result.to_csv(output_path, index=False)
    print(f"预测完成，结果已保存到 {output_path}")
    return result

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python predict.py <输入文件路径> <输出文件路径>")
        sys.exit(1)
    
    predict(sys.argv[1], sys.argv[2])

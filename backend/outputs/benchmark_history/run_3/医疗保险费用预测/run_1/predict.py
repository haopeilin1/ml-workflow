import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 预处理参数（与训练时一致）
CAT_COLS = ['sex', 'smoker', 'region']
NUM_COLS = ['age', 'bmi', 'children']
FEATURE_NAMES = None  # 将在加载模型后从特征重要性推断

def load_model(model_path='output/model.pkl'):
    """加载已训练的模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_input(df):
    """对输入数据进行预处理"""
    df = df.copy()
    
    # 丢弃ID列
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 构建预处理器
    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CAT_COLS),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), NUM_COLS)
    ], remainder='passthrough')
    
    # 注意：这里需要拟合预处理器，实际使用时应该加载已拟合的预处理器
    # 简化处理：直接对原始数据做转换
    X = df.copy()
    if 'charges' in X.columns:
        X = X.drop(columns=['charges'])
    
    X_processed = preprocessor.fit_transform(X)
    
    # 生成特征名
    ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_names = ohe.get_feature_names_out(CAT_COLS)
    feature_names = list(cat_names) + NUM_COLS
    
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # 清洗特征名
    X_df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_df.columns]
    
    return X_df

def predict(new_data_path, model_path='output/model.pkl'):
    """对新数据进行预测"""
    model = load_model(model_path)
    df = pd.read_csv(new_data_path)
    
    # 预处理
    X = preprocess_input(df)
    
    # 预测
    predictions = model.predict(X)
    
    # 构建结果
    result = pd.DataFrame({
        'prediction': predictions
    })
    
    if 'id' in df.columns:
        result.insert(0, 'id', df['id'])
    
    return result

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'data/test.csv'
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        sys.exit(1)
    
    result = predict(input_file)
    output_file = 'predictions_output.csv'
    result.to_csv(output_file, index=False)
    print(f"预测完成！结果已保存到 {output_file}")
    print(f"共 {len(result)} 条预测记录")
    print(result.head(10))

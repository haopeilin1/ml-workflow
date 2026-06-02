#!/usr/bin/env python3
"""
Standalone prediction script for binary classification model.
Usage: python predict.py <input_csv> <output_csv>
"""

import pandas as pd
import numpy as np
import dill as pickle
import sys
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder

# ========== 预处理参数（从训练时保存）==========
PREPROCESS_STATE = {
    'age_median': None,
    'sex_mode': None,
    'encoder': None,
    'fitted': False
}


def load_preprocess_state():
    """加载预处理状态（如果存在）"""
    import os
    if os.path.exists('output/preprocess_state.pkl'):
        with open('output/preprocess_state.pkl', 'rb') as f:
            state = pickle.load(f)
            PREPROCESS_STATE.update(state)


def preprocess(df, mode='test'):
    df = df.copy()
    
    if 'image_name' in df.columns:
        df = df.drop(columns=['image_name'])
    
    sex_mode = PREPROCESS_STATE.get('sex_mode', 'male')
    if 'sex' in df.columns:
        df['sex'] = df['sex'].fillna(sex_mode)
    
    if 'anatom_site_general_challenge' in df.columns:
        df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('unknown')
    
    age_median = PREPROCESS_STATE.get('age_median', 50)
    if 'age_approx' in df.columns:
        df['age_approx'] = df['age_approx'].fillna(age_median)
    
    if 'width' in df.columns and 'height' in df.columns:
        df['aspect_ratio'] = df['width'] / df['height'].replace(0, np.nan)
        df['aspect_ratio'] = df['aspect_ratio'].fillna(1.0)
        df['area'] = np.log1p(df['width'] * df['height'])
    
    if 'width' in df.columns:
        df['width'] = np.log1p(df['width'])
    if 'height' in df.columns:
        df['height'] = np.log1p(df['height'])
    
    cat_cols = ['sex', 'anatom_site_general_challenge']
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    
    if existing_cat_cols:
        encoder = PREPROCESS_STATE.get('encoder')
        if encoder is not None:
            df[existing_cat_cols] = encoder.transform(df[existing_cat_cols])
    
    return df


def feature_engineering(df):
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
            else:
                df[col] = df[col].fillna(0)
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 加载模型
    with open('output/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 加载预处理状态
    load_preprocess_state()
    
    # 加载数据
    df = pd.read_csv(input_path)
    original_df = df.copy()
    
    # 预处理
    df_clean = preprocess(df, mode='test')
    X = df_clean.drop(columns=['target'], errors='ignore')
    X_fe = feature_engineering(X)
    
    # 清洗特征名
    X_fe.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_fe.columns]
    
    # 预测
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_fe)[:, 1]
    else:
        probs = model.predict(X_fe).astype(float)
    
    preds = (probs >= 0.5).astype(int)
    
    # 保存结果
    result = original_df.copy()
    result['prediction'] = preds
    result['probability'] = probs
    result.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Total rows: {len(result)}")
    print(f"Positive predictions: {preds.sum()} ({preds.mean()*100:.2f}%)")


if __name__ == '__main__':
    main()

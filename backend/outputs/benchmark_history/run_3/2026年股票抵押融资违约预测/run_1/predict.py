#!/usr/bin/env python3
"""
Independent prediction script.
Usage: python predict.py <input_csv_path> [output_csv_path]
If output_csv_path is not provided, predictions are saved to predictions.csv
"""
import pandas as pd
import numpy as np
import dill as pickle
import re
import sys
import os

# ========== 预处理函数（与训练时一致）==========
def preprocess(df):
    df = df.copy()
    id_cols_to_drop = ['Stock code']
    df = df.drop(columns=id_cols_to_drop, errors='ignore')
    
    if 'P/E ratio' in df.columns:
        df['P/E ratio'] = pd.to_numeric(df['P/E ratio'], errors='coerce')
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    return df


def feature_engineering(df):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    df = df.copy()
    X = df
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    def log1p_transform(X_np, skew_indices=None):
        X_np = X_np.copy()
        if skew_indices is not None and len(skew_indices) > 0:
            for idx in skew_indices:
                col_data = X_np[:, idx]
                col_data = np.maximum(col_data, 0)
                X_np[:, idx] = np.log1p(col_data)
        return X_np
    
    # 使用训练时保存的 skew_cols
    skew_cols = ['P_B_ratio', 'changes_in_net_assets', 'Turnover_rate_of_accounts_receivable__Times_', 'Inventory_turnover_rate__Times_', 'EBITDA_interest_cover', 'Ratio_of_accounts_receivable_to_operating_income', 'Ratio_of_other_receivables_to_total_assets']
    skew_indices = [num_cols.index(c) for c in skew_cols if c in num_cols]
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(
            log1p_transform,
            kw_args={'skew_indices': skew_indices},
            validate=False
        )),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ], remainder='passthrough')
    
    X_processed = preprocessor.fit_transform(X)
    
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    else:
        cat_feature_names = []
    
    num_feature_names = num_cols.copy()
    remainder_cols = [c for c in X.columns if c not in cat_cols and c not in num_cols]
    all_feature_names = cat_feature_names + num_feature_names + remainder_cols
    all_feature_names = [re.sub(r'[^\w]', '_', str(c)) for c in all_feature_names]
    
    X_df = pd.DataFrame(X_processed, index=X.index, columns=all_feature_names)
    
    # 交互特征
    pledge_cols = [c for c in X_df.columns if 'Share_pledge_ratio' in c or 'share_pledge' in c.lower().replace(' ', '_')]
    debt_cols = [c for c in X_df.columns if 'Asset_liability_ratio' in c or 'asset_liability' in c.lower().replace(' ', '_')]
    
    for pc in pledge_cols:
        for dc in debt_cols:
            interact_name = f'interact_{pc}_x_{dc}'
            interact_name = re.sub(r'[^\w]', '_', interact_name)
            X_df[interact_name] = X_df[pc] * X_df[dc]
    
    concentration_cols = [c for c in X_df.columns if 'Equity_concentration' in c or 'equity_concentration' in c.lower().replace(' ', '_')]
    checks_cols = [c for c in X_df.columns if 'Equity_checks' in c or 'equity_checks' in c.lower().replace(' ', '_')]
    
    for cc in concentration_cols:
        for ec in checks_cols:
            ratio_name = f'ratio_{cc}_div_{ec}'
            ratio_name = re.sub(r'[^\w]', '_', ratio_name)
            X_df[ratio_name] = X_df[cc] / (X_df[ec].replace(0, np.nan)).fillna(1e-6)
    
    return X_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv_path> [output_csv_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(input_path)
    original_df = df.copy()
    
    # 预处理
    df_clean = preprocess(df)
    X = feature_engineering(df_clean)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # 清洗特征名
    X.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
    
    # 预测
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    preds = (probs >= 0.5).astype(int)
    
    # 保存结果
    result = original_df.copy()
    result['prediction'] = preds
    result['probability'] = probs
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Total samples: {len(result)}")
    print(f"Positive predictions: {preds.sum()} ({preds.mean()*100:.2f}%)")


if __name__ == '__main__':
    main()

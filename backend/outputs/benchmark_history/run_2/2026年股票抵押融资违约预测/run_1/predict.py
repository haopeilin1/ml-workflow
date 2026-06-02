
import pandas as pd
import numpy as np
import dill as pickle
import re
import os
import sys

# 预处理参数（与训练时一致）
DROP_COLS = [
    'Stock code', 'Z-SCORE', 'Stock Volatility', 'P/B ratio',
    'Stock price rise and fall in the last year', 'Annual turnover rate',
    'changes in operating income', 'Net business cycle (days)',
    'Turnover rate of accounts receivable (Times)', 'Inventory turnover rate (Times)',
    'Cash ratio', 'Ratio of accounts receivable to operating income',
    'Ratio of prepayments to operating income', 'Ratio of other receivables to total assets',
    'Company size (LN)', 'Equity checks and balances (2-5 large/1 large)',
    'Total institutional shareholding ratio'
]

LOG1P_COLS = [
    'Pledge ratio of limited sale shares', 'ST', 'ROE',
    'changes in net assets', 'Total asset turnover rate (Times)',
    'Current ratio', 'Monetary capital/short-term debt',
    'EBITDA/interest bearing debt', 'EBIT interest cover',
    'EBITDA interest cover',
    'Net cash flow from operations has been negative for three consecutive years',
    'Average cash income ratio in recent three years', 'Cash income ratio',
    'Ratio of construction in progress to total assets',
    "Minority shareholders' equity/owners' equity",
    'The proportion of goodwill in total assets exceeds',
    'Downgrade or negative', 'audit opinion ',
    'High deposit and loan of 90p', 'Audit fee'
]

WINSOR_COLS = [
    'Pledge ratio of limited sale shares', 'ST', 'ROE',
    'changes in net assets', 'Total asset turnover rate (Times)',
    'Current ratio', 'Monetary capital/short-term debt',
    'EBITDA/interest bearing debt', 'EBIT interest cover',
    'EBITDA interest cover',
    'Net cash flow from operations has been negative for three consecutive years',
    'Average cash income ratio in recent three years', 'Cash income ratio',
    'Ratio of construction in progress to total assets',
    "Minority shareholders' equity/owners' equity",
    'The proportion of goodwill in total assets exceeds',
    'Downgrade or negative', 'audit opinion ',
    'High deposit and loan of 90p', 'Audit fee',
    'Share pledge ratio of controlling shareholders',
    'Pledge ratio of unlimited shares', 'ROA',
    'Gross profit margin on sales', 'Asset liability ratio',
    'Asset liability ratio (excluding advance receipts)',
    'Asset liability ratio (total liabilities - contract liabilities - advance receipts)/(total assets - goodwill - contract liabilities - advance receipts)',
    'Current liabilities/total liabilities', 'Company nature (state owned assets 0, others 1)',
    'Proportion of independent directors', 'Equity concentration (the first largest shareholder)',
    'Number of research institutions concerned', 'Number of research reports (+1 LN)',
    'Financial cycle m2/gdp', 'Two positions in one (1 for the same, 0 for the different)',
    'Whether there are four major audits'
]

TARGET_COL = 'IsDefault'

def preprocess(df):
    df = df.copy()
    df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
    
    drop_cols_clean = [re.sub(r'[^\w]', '_', str(c)) for c in DROP_COLS]
    df = df.drop(columns=[c for c in drop_cols_clean if c in df.columns], errors='ignore')
    
    pe_col = 'P_E_ratio'
    if pe_col in df.columns:
        df[pe_col] = pd.to_numeric(df[pe_col], errors='coerce')
    
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 使用中位数填充（简化版，实际应使用训练时的值）
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_COL]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    
    return df

def feature_engineering(df):
    df = df.copy()
    
    log1p_cols_clean = [re.sub(r'[^\w]', '_', str(c)) for c in LOG1P_COLS]
    for col in log1p_cols_clean:
        if col in df.columns:
            min_val = df[col].min()
            shift = 0
            if min_val < 0:
                shift = abs(min_val) + 1
            df[col] = np.log1p(df[col] + shift)
    
    pledge_col = 'Share_pledge_ratio_of_controlling_shareholders'
    asset_liability_col = 'Asset_liability_ratio'
    if pledge_col in df.columns and asset_liability_col in df.columns:
        df['pledge_x_asset_liability'] = df[pledge_col] * df[asset_liability_col]
    
    equity_conc_col = 'Equity_concentration__the_first_largest_shareholder_'
    equity_checks_col = None
    for c in df.columns:
        if 'Equity_checks' in c or 'equity_checks' in c.lower():
            equity_checks_col = c
            break
    if equity_conc_col in df.columns and equity_checks_col is not None:
        df['equity_balance_ratio'] = df[equity_checks_col] / (df[equity_conc_col] + 1e-8)
    
    roa_col = 'ROA'
    roe_col = 'ROE'
    if roa_col in df.columns and roe_col in df.columns:
        df['profitability_composite'] = (df[roa_col] + df[roe_col]) / 2
    
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    X = df.drop(columns=[TARGET_COL], errors='ignore')
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X

def predict(input_path, output_path):
    """对新数据进行预测"""
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载数据
    df = pd.read_csv(input_path)
    id_col = 'Stock code' if 'Stock code' in df.columns else df.columns[0]
    
    # 预处理
    df_clean = preprocess(df)
    X = feature_engineering(df_clean)
    
    # 清洗列名
    X.columns = [re.sub('[^\w]', '_', str(c)) for c in X.columns]
    if X.columns.duplicated().any():
        X.columns = [f"{c}_{i}" if i > 0 else str(c) for i, c in enumerate(X.columns)]
    
    # 预测
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    # 保存结果
    result = pd.DataFrame({
        'id': df[id_col] if id_col in df.columns else range(len(preds)),
        'prediction': preds,
        'probability': probs
    })
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])

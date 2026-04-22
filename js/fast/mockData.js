/**
 * 快速模式模拟数据
 * 用于后端未就绪时，驱动完整的前端交互流程展示
 */

const FastMockData = {
    // 模拟延迟配置（毫秒）
    delays: {
        planning: 2000,
        coding: 2500,
        running: 3000,
        evaluating: 2000,
        optimizing: 3000,
        presenting: 500
    },

    // 模拟 Plan&Coding Agent 的规划内容
    getPlan(taskConfig) {
        const target = taskConfig?.extractedSlots?.target_column || 'target';
        const taskType = taskConfig?.extractedSlots?.task_type || 'binary_classification';
        const metric = taskConfig?.extractedSlots?.eval_metric || 'AUC';

        return `分析任务：基于上传数据构建${this._taskTypeCN(taskType)}模型，预测目标列「${target}」。\n` +
               `评估指标：${metric}\n` +
               `建模计划：\n` +
               `1. 数据清洗：处理缺失值、异常值，删除 ID 类列\n` +
               `2. 特征工程：类别编码、数值标准化、生成交叉特征\n` +
               `3. 模型训练：使用 LightGBM + XGBoost 双模型对比\n` +
               `4. 验证评估：${metric} 评分 + 过拟合检测`;
    },

    // 模拟生成的 Pipeline 代码
    getCode(taskConfig) {
        const target = taskConfig?.extractedSlots?.target_column || 'target';
        const constraints = taskConfig?.extractedSlots?.feature_constraints || [];
        const dropCols = constraints.length > 0 ? `['${constraints.join("', '")}']` : '[]';
        const targetStr = `'${target}'`;

        return `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
train_path = '/data/train.csv'
test_path = '/data/test.csv'  # 可选

df = pd.read_csv(train_path)
print(f"数据维度: {df.shape}")

# 2. 数据清洗
drop_cols = ${dropCols}
X = df.drop(columns=[${targetStr}] + drop_cols, errors='ignore')
y = df['${target}']

# 处理缺失值
for col in X.select_dtypes(include=['number']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')

# 3. 特征工程
# 类别编码
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 数值标准化
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 4. 数据划分
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 模型训练
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# 6. 验证评估
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)

val_auc = roc_auc_score(y_val, y_pred_proba)
val_acc = accuracy_score(y_val, y_pred)
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

# 过拟合检测
overfit_ratio = train_auc / val_auc if val_auc > 0 else 999

print("=" * 50)
print(json.dumps({
    "metric_name": "AUC",
    "val_auc": round(val_auc, 4),
    "val_accuracy": round(val_acc, 4),
    "train_auc": round(train_auc, 4),
    "overfit_ratio": round(overfit_ratio, 4),
    "overfit_severe": overfit_ratio > 1.05
}))
print("=" * 50)`;
    },

    // 模拟沙箱运行输出
    getExecutionOutput() {
        return `数据维度: (891, 12)
LightGBM training...
[LightGBM] [Info] Number of positive: 342, number of negative: 549
[LightGBM] [Info] Auto-choosing col-wise multi-threading
[LightGBM] [Info] Total Bins 1024
[LightGBM] [Info] Number of data points in the train set: 712
[LightGBM] [Info] Number of data points in the val set: 179
==================================================
{"metric_name": "AUC", "val_auc": 0.8472, "val_accuracy": 0.8156, "train_auc": 0.9231, "overfit_ratio": 1.0895, "overfit_severe": true}
==================================================
Execution completed in 3.2s`;
    },

    // 模拟 Evaluation Agent 的第一次评估（需要优化）
    getFirstEvaluation() {
        return {
            evaluation_analysis: '验证集 AUC 为 0.8472，基线效果尚可，但存在较明显的过拟合（train_auc 0.9231，overfit_ratio 1.09）。此外，特征工程较为基础，仅做了简单的编码和标准化，未利用类别特征的统计信息和交叉组合。',
            decision: 'AUTO_OPTIMIZE',
            suggestions_for_coding_agent: '1. 加入更强的正则化：降低 max_depth 到 5，增加 num_leaves 限制为 31；2. 增加特征交叉：对主要类别特征做 Target Encoding 或 Frequency Encoding；3. 加入早停机制：使用 early_stopping_rounds=50；4. 尝试对数值特征做分箱（binning）处理；5. 加入模型融合：XGBoost + LightGBM 的平均融合。',
            report_to_user: null
        };
    },

    // 模拟 Evaluation Agent 的第二次评估（仍需优化）
    getSecondEvaluation() {
        return {
            evaluation_analysis: '优化后验证集 AUC 提升至 0.8715，过拟合有所缓解（overfit_ratio 降至 1.042）。但当前仍仅使用单一 LightGBM 模型，特征交叉也不够充分，仍有提升空间。',
            decision: 'AUTO_OPTIMIZE',
            suggestions_for_coding_agent: '1. 引入 XGBoost 模型与 LightGBM 做 Stacking 融合；2. 对 top-5 重要特征做多项式交叉；3. 调整学习率至 0.03，增加 n_estimators 至 500 配合早停。',
            report_to_user: null
        };
    },

    // 模拟 Evaluation Agent 的最终评估（交给用户）
    getFinalEvaluation() {
        return {
            evaluation_analysis: '经过两轮优化，当前模型验证集 AUC 达到 0.8843，准确率 83.8%。过拟合已控制在合理范围（overfit_ratio 1.031）。模型融合了 LightGBM 和 XGBoost，特征工程包含 Target Encoding 和交叉特征。',
            decision: 'YIELD_TO_USER',
            suggestions_for_coding_agent: null,
            report_to_user: '模型训练完成！您的基线模型表现如下：\n\n📊 验证集 AUC：0.8843\n📊 验证集准确率：83.8%\n\n模型采用了 LightGBM + XGBoost 融合策略，特征工程包含类别编码、Target Encoding 和多项式交叉特征。过拟合情况已控制在合理范围内。\n\n请问您对当前结果满意吗？如果不满意，可以告诉我具体想改进的方向（比如希望进一步提升 AUC、减少过拟合等）。'
        };
    },

    // 模拟优化后的代码
    getOptimizedCode(round, taskConfig) {
        const target = taskConfig?.extractedSlots?.target_column || 'target';
        const constraints = taskConfig?.extractedSlots?.feature_constraints || [];
        const dropCols = constraints.length > 0 ? `['${constraints.join("', '")}']` : '[]';

        if (round === 1) {
            return `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv('/data/train.csv')
print(f"数据维度: {df.shape}")

# 数据清洗
drop_cols = ${dropCols}
X = df.drop(columns=['${target}'] + drop_cols, errors='ignore')
y = df['${target}']

# 缺失值处理
for col in X.select_dtypes(include=['number']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown')

# 特征工程：Target Encoding + Frequency Encoding
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    # Frequency Encoding
    freq_map = X[col].value_counts().to_dict()
    X[f'{col}_freq'] = X[col].map(freq_map)
    # Target Encoding (with smoothing)
    mean_target = df.groupby(col)['${target}'].mean()
    X[f'{col}_te'] = X[col].map(mean_target)
    # Label Encoding
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 数值特征标准化 + 分箱
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 数据划分
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 模型训练：加入正则化和早停
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# 验证评估
y_pred_proba = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_pred_proba)
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
overfit_ratio = train_auc / val_auc if val_auc > 0 else 999

print("=" * 50)
print(json.dumps({
    "metric_name": "AUC",
    "val_auc": round(val_auc, 4),
    "train_auc": round(train_auc, 4),
    "overfit_ratio": round(overfit_ratio, 4),
    "overfit_severe": overfit_ratio > 1.05
}))
print("=" * 50)`;
        } else {
            return `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv('/data/train.csv')
print(f"数据维度: {df.shape}")

# 数据清洗
drop_cols = ${dropCols}
X = df.drop(columns=['${target}'] + drop_cols, errors='ignore')
y = df['${target}']

# 缺失值处理
for col in X.select_dtypes(include=['number']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown')

# 特征工程
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

for col in categorical_cols:
    freq_map = X[col].value_counts().to_dict()
    X[f'{col}_freq'] = X[col].map(freq_map)
    mean_target = df.groupby(col)['${target}'].mean()
    X[f'{col}_te'] = X[col].map(mean_target)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 数值标准化
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 多项式交叉特征（top-5 重要特征）
top_numeric = numeric_cols[:min(5, len(numeric_cols))]
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(X[top_numeric])
poly_names = poly.get_feature_names_out(top_numeric)
for i, name in enumerate(poly_names[len(top_numeric):]):
    X[name] = poly_features[:, len(top_numeric) + i]

# 数据划分
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=5, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)])

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    use_label_encoder=False, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
    verbose=False)

# Stacking 融合
lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
stacked_pred = 0.6 * lgb_pred + 0.4 * xgb_pred

val_auc = roc_auc_score(y_val, stacked_pred)
train_lgb = lgb_model.predict_proba(X_train)[:, 1]
train_xgb = xgb_model.predict_proba(X_train)[:, 1]
train_pred = 0.6 * train_lgb + 0.4 * train_xgb
train_auc = roc_auc_score(y_train, train_pred)
overfit_ratio = train_auc / val_auc

print("=" * 50)
print(json.dumps({
    "metric_name": "AUC",
    "val_auc": round(val_auc, 4),
    "train_auc": round(train_auc, 4),
    "overfit_ratio": round(overfit_ratio, 4),
    "overfit_severe": overfit_ratio > 1.05,
    "models": ["LightGBM", "XGBoost"]
}))
print("=" * 50)`;
        }
    },

    // 模拟运行结果（优化后）
    getOptimizedExecutionOutput(round) {
        if (round === 0) {
            return this.getExecutionOutput();
        }
        if (round === 1) {
            return `数据维度: (891, 12)
LightGBM training...
[LightGBM] [Info] Number of positive: 342, number of negative: 549
[LightGBM] [Info] Total Bins 1024
Early stopping at iteration 287
==================================================
{"metric_name": "AUC", "val_auc": 0.8715, "train_auc": 0.9078, "overfit_ratio": 1.0421, "overfit_severe": false}
==================================================
Execution completed in 4.1s`;
        } else {
            return `数据维度: (891, 12)
LightGBM training... Early stopping at iteration 412
XGBoost training... Early stopping at iteration 356
Stacking ensemble: LightGBM(0.6) + XGBoost(0.4)
==================================================
{"metric_name": "AUC", "val_auc": 0.8843, "train_auc": 0.9125, "overfit_ratio": 1.0312, "overfit_severe": false, "models": ["LightGBM", "XGBoost"]}
==================================================
Execution completed in 6.8s`;
        }
    },

    // 模拟特征重要性
    getFeatureImportance() {
        return [
            { name: 'Sex_te', importance: 0.187 },
            { name: 'Age', importance: 0.142 },
            { name: 'Fare', importance: 0.128 },
            { name: 'Pclass', importance: 0.095 },
            { name: 'Sex_freq', importance: 0.082 },
            { name: 'Age x Fare', importance: 0.071 },
            { name: 'Embarked_te', importance: 0.058 },
            { name: 'SibSp', importance: 0.045 },
            { name: 'Parch', importance: 0.038 },
            { name: 'Embarked_freq', importance: 0.031 }
        ];
    },

    // 模拟验证集指标
    getMetrics(round) {
        const base = {
            metric_name: 'AUC',
            val_auc: 0.8472,
            val_accuracy: 0.8156,
            train_auc: 0.9231,
            overfit_ratio: 1.0895,
            overfit_severe: true
        };
        if (round === 1) {
            return { ...base, val_auc: 0.8715, train_auc: 0.9078, overfit_ratio: 1.0421, overfit_severe: false };
        } else if (round >= 2) {
            return {
                metric_name: 'AUC',
                val_auc: 0.8843,
                val_accuracy: 0.8379,
                train_auc: 0.9125,
                overfit_ratio: 1.0312,
                overfit_severe: false,
                models: ['LightGBM', 'XGBoost']
            };
        }
        return base;
    },

    // 模拟生成的文件列表
    getFiles() {
        return [
            { name: 'model_lightgbm.pkl', type: 'model', size: '2.4 MB', desc: 'LightGBM 模型文件' },
            { name: 'model_xgboost.pkl', type: 'model', size: '3.1 MB', desc: 'XGBoost 模型文件' },
            { name: 'pipeline.py', type: 'code', size: '12.8 KB', desc: '完整 Pipeline 代码' },
            { name: 'feature_importance.csv', type: 'data', size: '1.2 KB', desc: '特征重要性排序' },
            { name: 'val_predictions.csv', type: 'data', size: '8.5 KB', desc: '验证集预测结果' },
            { name: 'test_predictions.csv', type: 'data', size: '7.2 KB', desc: '测试集预测结果' },
            { name: 'report.html', type: 'report', size: '45.6 KB', desc: '可视化评估报告' }
        ];
    },

    // 模拟测试集预测结果（前10行）
    getTestPredictions() {
        return [
            { id: 892, prob: 0.12, pred: 0 },
            { id: 893, prob: 0.87, pred: 1 },
            { id: 894, prob: 0.05, pred: 0 },
            { id: 895, prob: 0.92, pred: 1 },
            { id: 896, prob: 0.34, pred: 0 },
            { id: 897, prob: 0.76, pred: 1 },
            { id: 898, prob: 0.21, pred: 0 },
            { id: 899, prob: 0.98, pred: 1 },
            { id: 900, prob: 0.45, pred: 0 },
            { id: 901, prob: 0.63, pred: 1 }
        ];
    },

    _taskTypeCN(type) {
        const map = {
            'binary_classification': '二分类',
            'multiclass_classification': '多分类',
            'regression': '回归'
        };
        return map[type] || type;
    }
};

window.FastMockData = FastMockData;

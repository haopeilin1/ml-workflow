/**
 * 深度模式模拟数据
 * 模拟 MCTS 树搜索的节点探索序列、最优代码演化、评估指标
 */

const DepthMockData = {
    // 节点类型配置
    nodeTypeConfig: {
        draft: { label: 'Draft', color: 'bg-blue-100 text-blue-700', icon: 'ph-file-code', desc: '生成基线代码' },
        debug: { label: 'Debug', color: 'bg-red-100 text-red-700', icon: 'ph-bug', desc: '修复执行错误' },
        improve: { label: 'Improve', color: 'bg-purple-100 text-purple-700', icon: 'ph-arrows-clockwise', desc: '策略优化' }
    },

    /**
     * 生成 MCTS 节点探索序列
     * 模拟真实的树搜索过程：draft 建立基线 → improve 优化 → debug 修复 → improve 进一步提升
     */
    getNodeSequence(taskConfig, totalNodes) {
        const target = taskConfig?.extractedSlots?.target_column || 'target';
        const taskType = taskConfig?.extractedSlots?.task_type || 'binary_classification';
        const metric = taskConfig?.extractedSlots?.eval_metric || 'AUC';

        const nodes = [];
        let bestAuc = 0;
        let bestNodeId = null;
        let parentMap = {}; // nodeId -> parentId for tree structure

        // 预定义的探索剧本（可循环）
        const script = [
            { type: 'draft', baseAuc: 0.82, msg: '生成 LightGBM 基线 Pipeline' },
            { type: 'draft', baseAuc: 0.81, msg: '尝试 XGBoost 基线方案' },
            { type: 'improve', baseAuc: 0.85, msg: '基于 Draft #1 增加交叉验证与特征选择', parentOffset: 1 },
            { type: 'draft', baseAuc: 0.83, msg: '尝试 CatBoost 基线方案' },
            { type: 'improve', baseAuc: 0.86, msg: '基于 Improve #3 增加多项式特征与目标编码', parentOffset: 2 },
            { type: 'debug', baseAuc: 0.84, msg: '修复 Draft #4 的类别特征处理错误', parentOffset: 3, wasFail: true },
            { type: 'improve', baseAuc: 0.88, msg: '基于 Improve #5 增加模型集成 Stacking', parentOffset: 3 },
            { type: 'draft', baseAuc: 0.84, msg: '尝试 Neural Network 基线方案' },
            { type: 'improve', baseAuc: 0.89, msg: '基于 Improve #7 进行超参数贝叶斯优化', parentOffset: 4 },
            { type: 'debug', baseAuc: 0.87, msg: '修复 NN 方案的数据泄漏问题', parentOffset: 2, wasFail: true },
            { type: 'improve', baseAuc: 0.90, msg: '基于 Improve #9 增加高级特征工程', parentOffset: 5 },
            { type: 'draft', baseAuc: 0.86, msg: '尝试随机森林基线方案' },
            { type: 'improve', baseAuc: 0.91, msg: '基于 Improve #11 融合全部有效策略', parentOffset: 6 },
            { type: 'debug', baseAuc: 0.90, msg: '修复集成模型的预测维度不匹配', parentOffset: 3, wasFail: true },
            { type: 'improve', baseAuc: 0.92, msg: '最终策略：加权集成 + 后处理校准', parentOffset: 7 },
            { type: 'draft', baseAuc: 0.87, msg: '尝试梯度提升改进方案' },
            { type: 'improve', baseAuc: 0.925, msg: '基于 Final Improve 微调阈值与校准', parentOffset: 8 },
            { type: 'debug', baseAuc: 0.92, msg: '修复校准函数的边界条件', parentOffset: 2, wasFail: true },
            { type: 'improve', baseAuc: 0.93, msg: '终极优化：自适应集成权重', parentOffset: 9 },
            { type: 'draft', baseAuc: 0.88, msg: '尝试新的特征组合策略' },
            { type: 'improve', baseAuc: 0.935, msg: '基于终极策略增加鲁棒性验证', parentOffset: 10 },
            { type: 'debug', baseAuc: 0.93, msg: '修复验证集划分的不一致性', parentOffset: 3, wasFail: true },
            { type: 'improve', baseAuc: 0.94, msg: '最终收敛：最优集成策略确认', parentOffset: 11 },
            { type: 'draft', baseAuc: 0.89, msg: '尝试备选模型架构' },
            { type: 'improve', baseAuc: 0.942, msg: '微调整合权重，收敛至全局最优', parentOffset: 12 }
        ];

        for (let i = 0; i < totalNodes; i++) {
            const s = script[i % script.length];
            const nodeId = i + 1;
            const parentId = s.parentOffset ? Math.max(1, nodeId - s.parentOffset) : null;
            if (parentId) parentMap[nodeId] = parentId;

            // 添加一些随机波动
            const noise = (Math.random() - 0.5) * 0.015;
            let auc = Math.min(0.999, Math.max(0.5, s.baseAuc + noise));

            // 如果是 debug 节点且之前失败，则修复后可能稍低于原目标
            if (s.type === 'debug' && s.wasFail) {
                auc = Math.min(0.999, s.baseAuc + noise);
            }

            const isSuccess = s.type !== 'debug' || !s.wasFail || Math.random() > 0.1;
            const isNewBest = isSuccess && auc > bestAuc;
            if (isNewBest) {
                bestAuc = auc;
                bestNodeId = nodeId;
            }

            // UCT 值模拟（高值表示更值得探索）
            const uct = (Math.random() * 0.5 + 0.3).toFixed(3);

            nodes.push({
                id: nodeId,
                type: s.type,
                parentId: parentId,
                uct: parseFloat(uct),
                status: isSuccess ? 'success' : 'failed',
                message: s.msg,
                metrics: isSuccess ? this._makeMetrics(auc, taskType) : null,
                isNewBest: isNewBest,
                bestAucSoFar: bestAuc,
                codeLevel: this._computeCodeLevel(bestAuc)
            });
        }

        return { nodes, bestNodeId, bestAuc };
    },

    _computeCodeLevel(auc) {
        if (auc >= 0.92) return 3;
        if (auc >= 0.86) return 2;
        return 1;
    },

    _makeMetrics(auc, taskType) {
        const valAcc = Math.min(0.99, auc * 0.95 + Math.random() * 0.02);
        const trainAuc = Math.min(0.999, auc + Math.random() * 0.02);
        const overfit = trainAuc / auc;
        return {
            metric_name: 'AUC',
            val_auc: parseFloat(auc.toFixed(4)),
            val_accuracy: parseFloat(valAcc.toFixed(4)),
            train_auc: parseFloat(trainAuc.toFixed(4)),
            overfit_ratio: parseFloat(overfit.toFixed(3)),
            overfit_severe: overfit > 1.05
        };
    },

    /**
     * 获取指定水平的最优代码
     * level 1-3，越高级越复杂
     */
    getBestCode(taskConfig, level = 1) {
        const target = taskConfig?.extractedSlots?.target_column || 'target';
        const constraints = taskConfig?.extractedSlots?.feature_constraints || [];
        const dropCols = constraints.length > 0 ? `['${constraints.join("', '")}']` : '[]';

        const baseImports = `import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')`;

        let code = '';

        if (level === 1) {
            // 基础版本（类似快速模式）
            code = `${baseImports}

# ===== Level 1: Baseline Pipeline =====
train_path = '/data/train.csv'
df = pd.read_csv(train_path)

drop_cols = ${dropCols}
X = df.drop(columns=['${target}'] + drop_cols, errors='ignore')
y = df['${target}']

# 基础预处理
for col in X.select_dtypes(include=['number']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown')
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
num_cols = X.select_dtypes(include=['number']).columns
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_val)[:, 1]
print(json.dumps({"val_auc": round(roc_auc_score(y_val, y_proba), 4)}))`;
        } else if (level === 2) {
            // 中级版本（交叉验证 + 特征工程）
            code = `${baseImports}
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# ===== Level 2: CV + Feature Engineering =====
train_path = '/data/train.csv'
df = pd.read_csv(train_path)

drop_cols = ${dropCols}
X = df.drop(columns=['${target}'] + drop_cols, errors='ignore')
y = df['${target}']

# 高级缺失值处理
for col in X.select_dtypes(include=['number']).columns:
    X[col + '_missing'] = X[col].isnull().astype(int)
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Missing')
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 多项式特征
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
num_cols = X.select_dtypes(include=['number']).columns.tolist()[:5]
X_poly = poly.fit_transform(X[num_cols])
X = pd.concat([X, pd.DataFrame(X_poly, index=X.index).iloc[:, len(num_cols):]], axis=1)

# 特征选择
selector = SelectKBest(mutual_info_classif, k=min(50, X.shape[1]))
X_selected = selector.fit_transform(X, y)
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask].tolist()
X = X[selected_features]

# 交叉验证训练
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
models = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr)
    oof_preds[val_idx] = m.predict_proba(X_val)[:, 1]
    models.append(m)

val_auc = roc_auc_score(y, oof_preds)
print(json.dumps({"val_auc": round(val_auc, 4), "cv_folds": 5, "n_features": len(selected_features)}))`;
        } else {
            // 高级版本（Stacking 集成 + 高级特征工程）
            code = `${baseImports}
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# ===== Level 3: Stacking Ensemble (Optimal) =====
train_path = '/data/train.csv'
df = pd.read_csv(train_path)

drop_cols = ${dropCols}
X = df.drop(columns=['${target}'] + drop_cols, errors='ignore')
y = df['${target}']

# 高级特征工程
for col in X.select_dtypes(include=['number']).columns:
    X[col + '_missing'] = X[col].isnull().astype(int)
    X[col] = X[col].fillna(X[col].median())
    X[col + '_log'] = np.log1p(np.abs(X[col]) + 1e-6)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Missing')
    freq = X[col].value_counts(normalize=True)
    X[col + '_freq'] = X[col].map(freq)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 交互特征
num_cols = X.select_dtypes(include=['number']).columns.tolist()[:6]
for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        X[f'{num_cols[i]}_x_{num_cols[j]}'] = X[num_cols[i]] * X[num_cols[j]]

# 特征选择
selector = SelectKBest(mutual_info_classif, k=min(80, X.shape[1]))
X_sel = selector.fit_transform(X, y)
selected_mask = selector.get_support()
X = X[X.columns[selected_mask]]

# 基模型定义
lgb_params = {"n_estimators": 800, "learning_rate": 0.02, "max_depth": 8,
              "subsample": 0.85, "colsample_bytree": 0.8, "random_state": 42, "verbose": -1}
xgb_params = {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 6,
              "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "use_label_encoder": False, "eval_metric": "logloss"}
rf_params = {"n_estimators": 300, "max_depth": 12, "min_samples_split": 5, "random_state": 42}

# Stacking: 第一层 OOF 预测
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {"lgb": [], "xgb": [], "rf": []}
oof = {"lgb": np.zeros(len(X)), "xgb": np.zeros(len(X)), "rf": np.zeros(len(X))}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]
    for name, params, clf in [("lgb", lgb_params, lgb.LGBMClassifier),
                               ("xgb", xgb_params, xgb.XGBClassifier),
                               ("rf", rf_params, RandomForestClassifier)]:
        m = clf(**params)
        m.fit(X_tr, y_tr)
        oof[name][val_idx] = m.predict_proba(X_val)[:, 1]
        models[name].append(m)

# 第二层：元模型
meta_X = np.column_stack([oof["lgb"], oof["xgb"], oof["rf"]])
meta_model = LogisticRegression(max_iter=1000, C=0.1)
meta_preds = cross_val_predict(meta_model, meta_X, y, cv=skf, method='predict_proba')[:, 1]
final_auc = roc_auc_score(y, meta_preds)

print(json.dumps({
    "val_auc": round(final_auc, 4),
    "ensemble": "LGBM + XGBoost + RF -> LogisticRegression",
    "cv_folds": 5,
    "n_features": X.shape[1]
}))`;
        }

        return code;
    },

    getFeatureImportance() {
        return [
            { name: 'Age', importance: 0.142 },
            { name: 'Fare', importance: 0.128 },
            { name: 'Sex_male', importance: 0.115 },
            { name: 'Pclass', importance: 0.098 },
            { name: 'Embarked_C', importance: 0.076 },
            { name: 'SibSp', importance: 0.065 },
            { name: 'Parch', importance: 0.058 },
            { name: 'Age_x_Fare', importance: 0.052 }
        ];
    },

    getTestPredictions() {
        return Array.from({ length: 10 }, (_, i) => ({
            id: i + 892,
            prob: 0.15 + Math.random() * 0.7,
            pred: Math.random() > 0.4 ? 1 : 0
        })).map(p => ({ ...p, prob: parseFloat(p.prob.toFixed(4)) }));
    },

    getFiles() {
        return [
            { name: 'optimal_pipeline.py', type: 'code', desc: '最优模型训练 Pipeline', size: '4.2 KB' },
            { name: 'submission.csv', type: 'data', desc: '测试集预测结果', size: '12.8 KB' },
            { name: 'model_ensemble.pkl', type: 'model', desc: '训练好的集成模型', size: '2.1 MB' },
            { name: 'feature_importance.png', type: 'report', desc: '特征重要性可视化', size: '156 KB' },
            { name: 'mcts_exploration_report.html', type: 'report', desc: 'MCTS 探索过程报告', size: '89 KB' },
            { name: 'evaluation_report.pdf', type: 'report', desc: '模型评估详细报告', size: '340 KB' }
        ];
    },

    getFinalReport(bestMetrics, totalNodes, successNodes) {
        const { val_auc, val_accuracy, overfit_ratio } = bestMetrics || {};
        return `深度探索报告
━━━━━━━━━━━━━━━━━━━━━━
总探索节点数: ${totalNodes}
成功执行节点: ${successNodes}
最优验证 AUC: ${val_auc || 'N/A'}
验证准确率: ${val_accuracy || 'N/A'}
过拟合比: ${overfit_ratio || 'N/A'}

探索策略总结:
• MCTS 树搜索通过 UCT 值动态平衡探索与利用
• Draft 节点建立多样化基线策略
• Improve 节点基于成功策略进行深度优化
• Debug 节点修复执行错误保证搜索空间完整
• 最终采用 Stacking 集成策略收敛至最优解`;
    },

    getTerminalLog(node) {
        const logs = {
            draft: [
                `[MCTS] Selecting node #${node.id} (Draft) UCT=${node.uct}`,
                `  Parent: ${node.parentId || 'root'}`,
                `  Policy: generating baseline code...`,
                `  CodeAgent: drafting pipeline with ${node.id <= 5 ? 'LightGBM' : 'ensemble'}...`,
                `  Sandbox: executing...`,
                node.status === 'success'
                    ? `  ✅ Execution succeeded | val_auc=${node.metrics?.val_auc || 'N/A'}`
                    : `  ❌ Execution failed: ${node.message}`
            ],
            improve: [
                `[MCTS] Selecting node #${node.id} (Improve) UCT=${node.uct}`,
                `  Parent: #${node.parentId}`,
                `  Policy: optimizing from parent strategy...`,
                `  StrategyAgent: applying ${node.message.split('增加')[1]?.split('，')[0] || 'optimization'}...`,
                `  Sandbox: executing improved pipeline...`,
                node.status === 'success'
                    ? `  ✅ Improvement successful | val_auc=${node.metrics?.val_auc || 'N/A'} ${node.isNewBest ? '(NEW BEST)' : ''}`
                    : `  ❌ Improvement failed`
            ],
            debug: [
                `[MCTS] Selecting node #${node.id} (Debug) UCT=${node.uct}`,
                `  Parent: #${node.parentId} (failed)`,
                `  Policy: repairing execution errors...`,
                `  DebugAgent: analyzing error traceback...`,
                `  Fix: ${node.message}`,
                node.status === 'success'
                    ? `  ✅ Repair successful | val_auc=${node.metrics?.val_auc || 'N/A'}`
                    : `  ❌ Repair failed, marking node as dead`
            ]
        };
        return logs[node.type] || logs.draft;
    }
};

window.DepthMockData = DepthMockData;

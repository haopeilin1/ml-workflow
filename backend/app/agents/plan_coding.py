"""
Plan & Coding Agent
生成建模计划与可执行 Python Pipeline 代码
"""

import logging
import re
from typing import Optional, List, Dict, Any

from app.agents.base import BaseAgent
from app.models.schemas import CodeOutput, TaskConfig
from app.knowledge_base.loader import KnowledgeBaseLoader
from app.agents.plan_agent import PlanAgent, PlanResult
from app.agents.coding_agent import CodingAgent

logger = logging.getLogger(__name__)

# ========== System Prompt ==========
PLAN_CODING_SYSTEM_PROMPT = """你是一名资深且高效的机器学习工程师。当前你运行在"Fast Engine（快速基线引擎）"模式下。
你的核心任务是：为结构化数据快速构建、修复或优化端到端的机器学习 Python 代码。

Core Constraints (绝对红线，必须遵守)
1. 沙箱隔离：代码将在无网络、无外网权限的 Docker 容器中运行。绝对禁止使用 os.system / os.popen / os.execve / os.fork / os.kill / os.remove 等危险系统调用。允许 import os，但仅限安全操作（如 os.path.join）。
2. 产物要求：【重要】当前流程中，绝对不要将模型保存到 output/ 目录，也绝对不要到处预测结果文件。你只需要在代码的最后一步，将验证集的评估指标（如 AUC、RMSE、是否严重过拟合等）通过 print() 结构化输出到控制台即可，供系统后续抓取。
3. 过拟合控制（关键）：模型必须在训练集和验证集上表现一致。严禁使用会导致严重过拟合的模型配置：
   - 树模型（RandomForest/XGBoost/LightGBM）：必须限制 max_depth（建议 ≤ 8），设置 min_samples_leaf（建议 ≥ 5），使用 subsample（建议 ≤ 0.8）。
   - 线性模型（Ridge/Lasso/LogisticRegression）：必须设置合理的 alpha / C 值（如 Ridge 的 alpha=1.0）。
   - 严禁使用 n_estimators > 500 的极端配置，严禁完全不设正则化参数。
   特别地：如果代码成功执行并得到了满意的模型，请使用 sklearn Pipeline 将预处理（如编码、缩放）和模型包装在一起，保存**完整的 Pipeline** 为 `data/best_model.pkl`（【关键】优先使用 dill 保存，因为它能序列化含自定义函数的 Pipeline；dill 失败才回退到 pickle），以便后续产物生成阶段直接加载使用，避免重复训练。
   **模型接口约束（关键）**：Pipeline 的最后一步必须是 sklearn 兼容的 estimator。具体来说：
   - 如果使用 LightGBM，必须使用 `lightgbm.LGBMClassifier` 或 `LGBMRegressor`（sklearn 兼容接口），**禁止**直接使用 `lgb.train()` 返回的裸 `Booster` 对象塞进 Pipeline。
   - 如果使用 XGBoost，必须使用 `xgboost.XGBClassifier` 或 `XGBRegressor`（sklearn 兼容接口），**禁止**直接使用 `xgb.train()` 返回的裸 `Booster` 对象塞进 Pipeline。
   - 如果使用 sklearn 原生模型（RandomForest、LogisticRegression 等），直接放入 Pipeline 即可。
   **禁止使用 `pd.get_dummies` 手动做 One-Hot Encoding**（这会导致训练和测试集列数不一致）。应使用 sklearn 的 `ColumnTransformer` + `OneHotEncoder` / `OrdinalEncoder` 并通过 Pipeline 包装。
   
   **【数据预处理强制规则 - 绝对红线，违反会导致执行失败】**
   - **所有类别特征列（dtype=object/str/categorical）必须在传入模型前编码为数值**。
   - 正确做法：使用 `ColumnTransformer` + `OneHotEncoder`（低基数类别）或 `OrdinalEncoder`（高基数/有序类别），将编码步骤嵌入 Pipeline。
   - **严禁**直接将包含字符串列的 DataFrame 喂给 LightGBM/XGBoost/LogisticRegression，否则会报 `ValueError: pandas dtypes must be int, float or bool`。
   - 如果数据中有缺失值（>20%），必须在编码前用 `SimpleImputer` 填充：类别列用 `strategy='most_frequent'`，数值列用 `strategy='median'`。
   
   【类别编码正确示例 - 必须照抄】
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
   from sklearn.impute import SimpleImputer
   from sklearn.pipeline import Pipeline
   from lightgbm import LGBMClassifier
   
   cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
   num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
   
   preprocessor = ColumnTransformer([
       ('cat', Pipeline([
           ('imputer', SimpleImputer(strategy='most_frequent')),
           ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
       ]), cat_cols),
       ('num', Pipeline([
           ('imputer', SimpleImputer(strategy='median')),
           ('scaler', StandardScaler())
       ]), num_cols)
   ], remainder='passthrough')
   
   pipeline = Pipeline([
       ('preprocess', preprocessor),
       ('model', LGBMClassifier(random_state=42))
   ])
   pipeline.fit(X_train, y_train)
   ```
   
3. 算法优选：优先使用 Scikit-Learn, LightGBM, XGBoost 等快速且效果好的树模型。
4. 【关键版本兼容性 - 必须严格遵守】沙箱中 LightGBM 版本为 4.6.0，与网上旧教程的 API 完全不同。以下参数在 LGBMClassifier.fit() 中**已被移除**，使用会导致代码执行失败：
   - ❌ early_stopping_rounds（已移除）
   - ❌ verbose（已移除）
   - ❌ eval_at（已移除）
   
   【正确写法 - 必须照抄】
   ```python
   model = lgb.LGBMClassifier(
       objective='binary',
       scale_pos_weight=scale_pos_weight,
       num_leaves=31,
       max_depth=6,
       learning_rate=0.05,
       n_estimators=500,
       subsample=0.8,
       min_child_samples=20
   )
   model.fit(
       X_train, y_train,
       eval_set=[(X_val, y_val)],
       callbacks=[lgb.early_stopping(stopping_rounds=50)]
   )
   ```
   【错误写法 - 绝对禁止】
   ```python
   model.fit(..., early_stopping_rounds=50, verbose=False)  # 错误！会导致执行失败
   ```
5. 环境依赖：当前沙箱已预装以下 Python 包，请优先使用这些库编写代码（无需 pip install）：
   - 数据处理：pandas, numpy
   - 机器学习：scikit-learn (sklearn), xgboost, lightgbm (版本 4.6.0)
   - 类别不平衡处理：imbalanced-learn (SMOTE, RandomOverSampler, RandomUnderSampler, ADASYN)
   - 类别编码：category-encoders (TargetEncoder, LeaveOneOutEncoder, CatBoostEncoder, WOEEncoder)
   - 超参优化：optuna
   - 模型可解释性：shap
   - 特征工程：feature-engine
   - 科学计算：scipy
   - 可视化：matplotlib, seaborn
   - 工具：joblib, threadpoolctl, dill
   - 图像：Pillow (PIL)
   - 统计：statsmodels
   
   **关键库使用规范（必须遵守，否则会导致执行失败）**：
   
   **(a) imbalanced-learn（类别不平衡处理）**
   - `SMOTE` / `RandomOverSampler` / `RandomUnderSampler` 等**只能 fit 训练集，绝对不能 fit 验证集或测试集**。这是数据泄露的红线。
   - 正确做法：由于本系统已提供 `data/validation.csv`，你应该将 `data/train.csv` 作为训练数据，在其内部做重采样（如需要），而 `data/validation.csv` 直接用于评估，不做任何重采样。
   - 在 Pipeline 中使用重采样器时，**必须用 `imblearn.pipeline.Pipeline`**（不是 sklearn.pipeline.Pipeline），否则重采样器会在预测阶段也作用到验证集/测试集。
   
   【imbalanced-learn 正确写法】
   ```python
   from imblearn.over_sampling import SMOTE
   from imblearn.pipeline import Pipeline as ImbPipeline
   from sklearn.preprocessing import StandardScaler
   from lightgbm import LGBMClassifier
   
   pipeline = ImbPipeline([
       ('scaler', StandardScaler()),
       ('smote', SMOTE(random_state=42)),
       ('model', LGBMClassifier(random_state=42))
   ])
   pipeline.fit(X_train, y_train)
   ```
   
   **(b) category-encoders（高基数类别编码）**
   - `TargetEncoder` / `LeaveOneOutEncoder` 等**只能 fit 训练集**，用训练集的统计量 transform 验证集和测试集。
   
   【category-encoders 正确写法】
   ```python
   from category_encoders import TargetEncoder
   encoder = TargetEncoder(cols=['category_col'])
   X_train_enc = encoder.fit_transform(X_train, y_train)
   X_val_enc = encoder.transform(X_val)
   ```
   
   **(c) sklearn 1.6+ 破坏性变更**
   - `LogisticRegression` 已**移除** `multi_class` 参数（传了会报错）
   - `solver='liblinear'` **不支持多分类**（n_classes >= 3 时报错）
   - 多分类请用 `solver='lbfgs'` 或 `'newton-cg'` / `'sag'` / `'saga'`
   
   **(c.1) sklearn FunctionTransformer（常见陷阱）**
   `FunctionTransformer` 的 `func` 接收的参数类型取决于它在 Pipeline 中的位置：
   - 如果直接对原始 DataFrame 使用，接收的是 **pandas DataFrame**（不能用 `X[:, i]` 这种 numpy 切片）
   - 如果在 `ColumnTransformer` 之后使用，接收的是 **numpy array**（列名已丢失）
   
   【FunctionTransformer 正确写法】
   ```python
   from sklearn.preprocessing import FunctionTransformer
   
   # 方法1：使用 pandas 的 apply（推荐，兼容 DataFrame）
   def log_transform_df(X):
       X = X.copy()
       for col in X.columns:
           if X[col].min() >= 0:
               X[col] = np.log1p(X[col])
       return X
   
   log_tf = FunctionTransformer(log_transform_df)
   
   # 方法2：如果确定输入是 numpy array
   def log_transform_np(X):
       return np.log1p(X)
   
   log_tf = FunctionTransformer(log_transform_np)
   ```
   【FunctionTransformer 错误写法 - 绝对禁止】
   ```python
   def log_transform(X):
       col_min = np.min(X[:, i])  # 错误！X 可能是 pandas DataFrame，不能用 numpy 切片
       return np.log1p(X)
   ```
   
   **(d) LightGBM 4.6.0**
   - `fit()` 已移除 `early_stopping_rounds` / `verbose` / `eval_at`
   - 正确写法：`callbacks=[lgb.early_stopping(stopping_rounds=50)]`
   
   如果代码需要 import 未在上述列表中的第三方库，请先检查是否可用已安装库替代，避免执行失败。
5. 数据路径（关键）：
   - 所有上传的数据文件在沙箱中会被**统一转换为 CSV 格式**，文件名固定为：`train.csv`（训练集）、`validation.csv`（验证集）、`test.csv`（测试集，如有）。
   - 沙箱工作目录下有一个 `data/` 子目录，所有数据文件都位于其中。
   - 读取数据时必须使用相对路径。
   
   **【训练阶段 - 绝对红线】当前处于训练/调试阶段，沙箱中只有以下两个文件：**
   - `pd.read_csv('data/train.csv')` — 训练数据，**必须读取**
   - `pd.read_csv('data/validation.csv')` — 验证数据，**必须读取**
   - **`data/test.csv 在训练阶段绝对不存在`**。任何包含 `pd.read_csv('data/test.csv')` 的代码都会立即触发 `FileNotFoundError` 导致执行失败。
   - **训练代码中严禁出现 `test_df`、`X_test`、`y_test` 等测试集相关变量，严禁在训练阶段对测试集做任何操作。**
   
   **【产物阶段】只有系统进入产物生成阶段后，才会恢复 data/test.csv，此时方可读取。**
   
   - **【绝对禁止】不要自己用 train_test_split 或其他方式重新切分验证集。data/validation.csv 已经由系统预先切分好，直接使用即可。**
   - **【时序任务特别约束】如果是时序任务（is_time_series=true），data/validation.csv 已经按时间顺序切分（前80%为训练集，后20%为验证集）。严禁重新切分或打乱顺序，必须保持时间连续性。**
   - 严禁使用绝对路径（如 `D:\...` 或 `/home/...`），不要直接使用文件名（如 `pd.read_csv('train.csv')` 会找不到文件），也不要使用原始 `.xlsx` 文件名（沙箱内不存在 `.xlsx` 文件）。

6. 【编码错误经验知识库 - 常见 TypeError/SyntaxError 预防指南】
   以下错误在历次运行中高频出现，生成代码时必须主动避免：
   
   (a) 字符串与数值运算 TypeError
   - **任何分类列（dtype=object/str）在进行数学运算（加减乘除、对数、平方等）前必须先编码为数值**。
   - 错误示例：`df['Stage'] * df['Edema']`（Stage 和 Edema 是字符串分类列，直接相乘会报 `TypeError: can't multiply sequence by non-int of type 'float'`）
   - 正确做法：先用 `OrdinalEncoder` 或 `OneHotEncoder` 编码，或确保特征交互只在数值列上进行：`num_cols = X.select_dtypes(include=[np.number]).columns`
   
   (b) pandas 2.1+ 破坏性变更
   - `df.fillna(method='ffill')` 已移除 `method` 参数，会报 `TypeError: fillna() got an unexpected keyword argument 'method'`
   - 正确做法：`df.ffill()` 或 `df.bfill()`
   
   (c) sklearn 1.2+ 破坏性变更
   - `OneHotEncoder(sparse=True)` 已改为 `OneHotEncoder(sparse_output=True)`
   - 传 `sparse` 会报 `TypeError: __init__() got an unexpected keyword argument 'sparse'`
   
   (d) XGBoost 新版 API 变更
   - `XGBClassifier.fit(..., early_stopping_rounds=50)` 在某些版本中参数位置已变化
   - 正确做法：`model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)`
   - `eval_metric` 应优先作为 `XGBClassifier(..., eval_metric='logloss')` 的构造参数，而非 `fit()` 参数
   
   (e) JSON 序列化 TypeError
   - `json.dumps()` 不能序列化 `numpy.bool_` / `numpy.int64` 等 numpy 标量，会报 `TypeError: Object of type bool_ is not JSON serializable`
   - 正确做法：输出指标前统一转换：`float(val)`, `bool(val)`, `int(val)`。严禁直接 `print(json.dumps({"ok": np.bool_(True)}))`
   
   (f) 未闭合字符串语法错误预防
   - **严禁在 f-string / print 语句中嵌入过长文本、多行字符串或复杂嵌套表达式**。长文本应赋值给变量，再引用变量。
   - 错误示例：`print(f"Report: {df.describe().to_string()}")`（过长文本极易被 LLM 截断，导致未闭合引号）
   - 正确做法：先计算 `desc = df.describe().to_dict()`，然后 `print(json.dumps({"describe": desc}))`
   - 如果需要在代码中写长字符串（如 HTML 模板），**必须使用三引号 `'''` 或 `"\"\"\"`**，且确保闭合三引号在代码末尾清晰可见。

Input Context
- 【意图澄清与任务配置 Task Config】: 包含目标列、任务类型、核心评估指标、用户建模建议等
- 【数据画像 Data Profile】: 含表头及部分统计特征
- 【当前运行状态 Run State】: INIT / DEBUG / OPTIMIZE
- 【上下文载荷 Context Payload】: 报错日志 / 评估Agent的建议 / 用户人工修改建议
- 【历史代码 Previous Code】: 上一轮代码（如有）
- 【用户建模建议 User Modeling Suggestions】: 用户在任务描述中提出的建模偏好或建议（如算法选择、预处理方法、评估侧重等）。这些建议是重要参考，你应结合数据和任务特点灵活采纳，而非死板执行。如果建议不合理或与数据特点冲突，你有权进行专业调整并说明理由。

Action Guidelines & State Machine
状态 1: INIT (初始无代码及结果)
- 动作要求：
  1. 规划（Plan）：提出包含"数据清洗 -> 特征工程 -> 模型训练 -> 验证集运行"的完整 Pipeline 规划。
  2. 编码（Code）：依据规划编写完整代码。
状态 2: DEBUG (上次运行代码报错)
- 动作要求：
  1. 规划（Plan）：仔细分析 Context Payload 中的 traceback 报错信息，精确定位 bug 原因，提出极简的修复计划。
  2. 编码（Code）：基于历史代码进行修改，修复 bug 使得代码能跑通，尽量不推翻原有的核心 Pipeline。
状态 3: OPTIMIZE (评估Agent或用户提出优化信号)
- 动作要求：
  1. 规划（Plan）：仔细阅读 Context Payload 传来的调优建议（可能是系统评估Agent给出的超参调整/换模型建议，也可能是用户直接用自然语言下达的强干预指令）。将建议转化为具体的代码调整策略。
  2. 编码（Code）：编写修改后的完整代码。

Output Format
你必须严格按以下标签格式输出，务必"先规划，后编码"：
<plan>
在这里用清晰的步骤描述你的完整规划思路或 Bug 修复/调优策略。
</plan>
<code>
```python
import pandas as pd
import json
... 你的机器学习 Pipeline 代码 ...

# ========== 【关键新增】预测准备函数 ==========
# 如果在 Pipeline 外部做了任何特征工程（如添加新列、删除列、变换等），
# 必须定义此函数，确保预测阶段能复现完全相同的预处理。
# 【重要】此函数必须是"自包含"的：所有需要的信息（列名列表、参数等）必须在函数内部定义，
# 不能依赖函数外部的全局变量。因为预测阶段只会注入这个函数本身，不会注入外部变量。
def prepare_for_prediction(df):
    # 对输入 DataFrame 执行与训练阶段完全一致的预处理。
    # 此函数将在预测阶段被直接调用。
    # 【必须自包含】把所有需要的列名、参数都在函数内部重新定义
    # 示例：
    # feature_cols = ['col1', 'col2', 'col3']  # 在函数内部重新定义
    # df = df[feature_cols].copy()
    # df = add_features(df)  # 如果在训练时调用了自定义特征工程
    # df = df.drop(columns=['leakage_col'])  # 如果在训练时删除了某些列
    # ... 所有训练时对 X 做的变换 ...
    return df

# 保存模型（【关键】优先使用 dill，它能序列化含自定义函数的 Pipeline）
try:
    import dill
    with open('data/best_model.pkl', 'wb') as f:
        dill.dump(pipeline, f)
    print("Model saved with dill")
except Exception as e:
    print(f"dill save failed: {e}, falling back to pickle")
    import pickle
    with open('data/best_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved with pickle")

# 输出验证集指标
print(json.dumps({
    "metric_name": "accuracy",
    "val_accuracy": float(valid_acc),
    "train_score": float(train_acc),
    "val_auc": float(val_auc),
    "overfit_ratio": float(train_acc / valid_acc) if valid_acc > 0 else 1.0
}))
```
</code>"""

ARTIFACT_SYSTEM_PROMPT = """你是一名资深机器学习工程师。当前处于"产物生成阶段"，用户已对模型结果表示满意，需要你生成最终交付产物。

核心任务
基于下面提供的【最佳代码】，生成一个完整的产物脚本，该脚本将：
1. 加载已训练好的模型（优先从 data/best_model.pkl 加载，如果存在则无需重新训练；如果不存在才重新训练）
2. 对测试集进行预测（如果存在 data/test.csv）
3. 保存产物文件到 output/ 目录下

产物要求（根据任务类型差异化）

【通用产物 - 所有任务都必须生成】
1. 模型文件：output/model.pkl（使用 pickle 保存最终模型）。**必须是完整的 sklearn Pipeline**（包含 ColumnTransformer 等预处理 + 模型），以便直接对原始数据做预测。
2. 特征重要性：output/feature_importance.csv（name, importance 两列）。**重要：特征重要性必须排除目标列（y），只包含输入特征（X）。**
3. 特征重要性图：output/feature_importance.png（使用 matplotlib，保存为 PNG）。**图中也必须排除目标列。**
4. 可视化报告：output/report.html（包含模型指标摘要、特征重要性表格、测试集预测预览）
5. 配套预测脚本：output/predict.py（独立的预测脚本，加载 model.pkl 对新数据预测）

【任务类型差异化产物 - 根据任务类型选择生成】

**二分类任务 (binary_classification)**：
- output/test_predictions.csv：包含原始测试集 + prediction 列 + probability 列（正类概率）
- output/report_fig.png：PR 曲线（Precision-Recall，极度不平衡时）或 ROC 曲线（标准二分类时），图标题标注 AUC/AP 值
- report.html 中增加：混淆矩阵表格、分类报告（precision/recall/f1 per class）

**多分类任务 (multiclass_classification)**：
- output/test_predictions.csv：包含原始测试集 + prediction 列 + proba_0, proba_1, ... 列（各类概率）
- output/report_fig.png：混淆矩阵热力图（sns.heatmap, annot=True 显示数值）
- report.html 中增加：混淆矩阵表格、每类 precision/recall/f1

**回归任务 (regression / 含时序回归)**：
- output/test_predictions.csv：包含原始测试集 + prediction 列
- output/report_fig.png：预测值 vs 真实值散点图，叠加 y=x 对角线
- report.html 中增加：残差统计、RMSE/MAE/R2 指标

**时序任务 (time_series_forecasting)**：
- output/test_predictions.csv：包含时间列 + prediction 列
- output/report_fig.png：时间序列预测曲线图（真实值 vs 预测值随时间变化）
- report.html 中增加：趋势分析、预测区间（如有）

**聚类任务 (clustering)**：
- output/test_predictions.csv：包含原始数据 + cluster 列（聚类标签）
- output/report_fig.png：聚类结果散点图（PCA 降维到 2D 后绘制，不同颜色代表不同簇）
- report.html 中增加：每簇统计特征、轮廓系数(Silhouette Score)

**大数据集优化（>10万行时）**：
- **禁止生成 feature_importance.png**（matplotlib 大图对大数据集极其耗时）
- **禁止生成 report_fig.png**（差异化可视化对大数据集禁用）
- **禁止计算 SHAP 值**（大数据集 SHAP 计算可能耗时数分钟）
- **禁止生成复杂可视化**（如热力图、散点图矩阵等）
- 只需保留核心文件：output/model.pkl + output/feature_importance.csv + output/report.html（简单文本表格）
- 特征重要性 CSV 可以直接从模型属性获取（如 `model.feature_importances_`），无需额外计算

【代码质量要求 - 绝对红线】
1. **严禁在 print 语句中使用 f-string 嵌套复杂表达式或 JSON**。正确的做法：
   - 先构造好字典变量，然后 `print(json.dumps(summary_dict))`
   - 错误示例：`print(json.dumps({"key": f"{value}"}))`  ← 这会导致语法错误！
2. 所有文件路径使用相对路径：data/train.csv, output/model.pkl 等
3. 测试集（data/test.csv）**没有目标列（y）**，只有特征列（X）。
   - 删除目标列时必须条件判断：`if target_col in df.columns: df = df.drop(columns=[target_col])`
4. 产物代码必须能独立运行，所有 import 在代码顶部声明

数据路径规则（关键）
- 训练集：data/train.csv
- 验证集：data/validation.csv
- 测试集（如有）：data/test.csv
- 产物输出目录：output/（已存在，直接写入即可）
- 严禁使用绝对路径

环境说明
- 沙箱已预装 pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, dill
- 产物生成模式下允许使用 os, pathlib, shutil 等文件操作模块
- 允许保存 .pkl, .csv, .png, .html, .py 等文件

请严格按照以下格式输出：
<plan>
产物生成计划概述
</plan>
<code>
```python
import pandas as pd
import json
import pickle
import os
os.makedirs('output', exist_ok=True)
... 产物生成代码 ...
```
</code>"""

PREDICT_SCRIPT_SYSTEM_PROMPT = """你是一名资深机器学习工程师。你的任务是根据提供的训练代码，生成一个配套预测脚本 predict.py。

核心要求
1. 加载 `data/best_model.pkl`（**必须使用 dill 加载**，因为模型可能包含自定义函数）和 `data/test.csv`。
2. **【关键】如果训练代码中定义了任何自定义类或函数（如特征工程函数、自定义 Transformer），必须将它们完整复制到 predict.py 中**，否则 pickle/dill 加载模型时会报 AttributeError。
3. 输出 `output/eval_predictions.csv`，格式：
   - 第一列：id 列（与测试集一致）
   - prediction 列：预测标签
   - probability 列（二分类/多分类且支持 predict_proba 时）：正类概率
   - 多分类（>2类）时：额外 proba_0, proba_1, ... 列
4. 脚本中必须处理以下边界情况：
   - 如果 best_model.pkl 是 dict（含 preprocessor + model），正确解包
   - 如果 predict_proba 失败，只输出 prediction 列
   - 如果测试集没有 id 列，使用第一列作为 id

复制自定义类/函数的判定方法：
- 查找训练代码中所有 `class Xxx:`（自定义类）和 `def xxx():`（自定义函数）定义
- 如果该 class/function 被 Pipeline 引用或模型依赖，**必须完整复制其代码到 predict.py 中**
- 例如：训练代码定义了 `class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):`，predict.py 中必须有完全相同的类定义

输出格式
<code>
```python
import pandas as pd
import dill
import numpy as np
import os
import json

# 【必须】复制训练代码中的自定义类/函数定义（如有）
... 自定义类/函数定义 ...

# 加载模型
with open('data/best_model.pkl', 'rb') as f:
    model = dill.load(f)

# 加载测试集
test = pd.read_csv('data/test.csv')

# 预测
... 预测代码 ...

# 保存结果
result.to_csv('output/eval_predictions.csv', index=False)
print('EVAL_PREDICTIONS_SAVED')
```
</code>"""


class PlanCodingAgent(BaseAgent):
    """
    Plan & Coding Agent
    
    职责：
    - INIT 状态：从零生成完整的建模计划 + Pipeline 代码
    - DEBUG 状态：基于历史代码和报错信息修复 bug
    - OPTIMIZE 状态：基于评估建议或用户反馈优化代码
    
    路由逻辑（新增）：
    - 简单任务（complexity=simple）：保持原有单步 Plan+Coding 模式
    - 复杂任务（complexity=complex）：Plan 与 Coding 分离
      1. PlanAgent 生成结构化计划（核心挑战、must_do、avoid、pipeline）
      2. CodingAgent 基于结构化计划生成代码，严格遵循 must_do/avoid
    """
    
    def __init__(self, llm_client=None, plan_llm_client=None, coding_llm_client=None, unified_llm_client=None):
        super().__init__(llm_client)
        self._structured_plan: Optional[PlanResult] = None
        self._plan_agent: Optional[PlanAgent] = None
        self._coding_agent: Optional[CodingAgent] = None
        # 【新增】支持 Plan/Coding/Unified 各自使用不同的 LLM
        self._plan_llm = plan_llm_client or llm_client or self.llm
        self._coding_llm = coding_llm_client or llm_client or self.llm
        self._unified_llm = unified_llm_client or llm_client or self.llm
    
    def _get_plan_agent(self) -> PlanAgent:
        if self._plan_agent is None:
            self._plan_agent = PlanAgent(llm_client=self._plan_llm)
        return self._plan_agent
    
    def _get_coding_agent(self) -> CodingAgent:
        if self._coding_agent is None:
            self._coding_agent = CodingAgent(llm_client=self._coding_llm)
        return self._coding_agent
    
    def generate(
        self,
        task_config: TaskConfig,
        run_state: str = "INIT",
        context_payload: str = "",
        previous_code: str = "",
        prebuilt_plan: Optional[str] = None,
        evaluation_history: Optional[List[Dict[str, Any]]] = None
    ) -> CodeOutput:
        """
        生成建模计划与代码
        
        Args:
            task_config: 任务配置（目标列、任务类型、评估指标、文件角色等）
            run_state: 当前运行状态（INIT / DEBUG / OPTIMIZE）
            context_payload: 上下文载荷（报错信息 / 优化建议 / 用户反馈）
            previous_code: 历史代码（DEBUG/OPTIMIZE 状态时传入）
            prebuilt_plan: 【新增】预构建的结构化计划（由 EvaluationAgent.evaluate() 在 AUTO_OPTIMIZE 时返回的 replan_output）。
                          如果传入，复杂任务在 OPTIMIZE 状态下将跳过 PlanAgent，直接使用此计划。
            evaluation_history: 【新增】历史评估记录（用户反馈路径传入 PlanAgent，避免重复犯错）
            
        Returns:
            CodeOutput: 包含 plan（规划文本）和 code（Python 代码）
        """
        is_complex = task_config.extracted_slots.complexity == "complex"
        
        if is_complex:
            result = self._generate_complex(
                task_config, run_state, context_payload, previous_code, prebuilt_plan, evaluation_history
            )
        else:
            result = self._generate_simple(
                task_config, run_state, context_payload, previous_code
            )
        
        # 【统一后处理】对所有生成的训练代码做 sanitize 和语法检查
        if result.code:
            result.code = self._sanitize_training_code(result.code)
            result.code = self._ensure_code_valid(result.code, "训练代码")
        
        return result
    
    def _generate_simple(
        self,
        task_config: TaskConfig,
        run_state: str,
        context_payload: str,
        previous_code: str
    ) -> CodeOutput:
        """简单任务：单步 Plan + Coding（使用 _simple_llm）"""
        user_prompt = self._build_user_prompt(
            task_config, run_state, context_payload, previous_code
        )
        
        logger.info(f"[PlanCodingAgent] 简单任务生成代码, state={run_state}, llm={self._unified_llm.model if self._unified_llm else 'default'}")
        
        # 简单任务使用独立的 unified_llm（如果配置了的话）
        original_llm = self.llm
        if self._unified_llm is not original_llm:
            self.llm = self._unified_llm
        try:
            response = self._call_llm(PLAN_CODING_SYSTEM_PROMPT, user_prompt)
        finally:
            self.llm = original_llm
        
        plan, code = self._parse_response(response)
        
        # 【训练代码清理】删除训练阶段不应该出现的 test.csv 读取
        if code:
            code = self._sanitize_training_code(code)
        
        # 【语法检查】训练代码返回前用 ast.parse 预检
        if code:
            code = self._ensure_code_valid(code, "简单任务训练代码")
        
        logger.info(f"[PlanCodingAgent] 简单任务解析完成, plan长度={len(plan)}, code长度={len(code)}")
        
        return CodeOutput(plan=plan, code=code, raw_response=response)
    
    def _generate_complex(
        self,
        task_config: TaskConfig,
        run_state: str,
        context_payload: str,
        previous_code: str,
        prebuilt_plan: Optional[str] = None,
        evaluation_history: Optional[List[Dict[str, Any]]] = None
    ) -> CodeOutput:
        """复杂任务：Plan + Coding 分离"""
        
        # INIT 状态：先生成结构化计划（首次必须使用 PlanAgent）
        if run_state == "INIT":
            logger.info("[PlanCodingAgent] 复杂任务 INIT：调用 PlanAgent 生成结构化计划...")
            plan_result = self._get_plan_agent().generate(task_config)
            self._structured_plan = plan_result
            
            formatted_plan = self._get_plan_agent().format_plan_for_coding(plan_result)
            logger.info(f"[PlanCodingAgent] PlanAgent 完成:\n{formatted_plan[:500]}...")
            
            # 然后将计划传给 CodingAgent 生成代码
            logger.info("[PlanCodingAgent] 复杂任务 INIT：调用 CodingAgent 生成代码...")
            code_output = self._get_coding_agent().generate(
                task_config=task_config,
                structured_plan=formatted_plan,
                run_state=run_state,
                context_payload=context_payload,
                previous_code=previous_code
            )
            # 合并 plan（结构化计划）和 coding plan
            combined_plan = f"{formatted_plan}\n\n{'='*60}\n【Coding Agent 实现计划】\n{'='*60}\n{code_output.plan}"
            return CodeOutput(plan=combined_plan, code=code_output.code, raw_response=code_output.raw_response)
        
        # DEBUG 状态：复用已有计划，只重新生成代码
        if run_state == "DEBUG":
            if self._structured_plan is None:
                logger.warning("[PlanCodingAgent] 复杂任务 DEBUG 但无缓存计划，重新生成...")
                plan_result = self._get_plan_agent().generate(task_config)
                self._structured_plan = plan_result
            
            formatted_plan = self._get_plan_agent().format_plan_for_coding(self._structured_plan)
            logger.info(f"[PlanCodingAgent] 复杂任务 DEBUG：复用已有计划，调用 CodingAgent...")
            
            code_output = self._get_coding_agent().generate(
                task_config=task_config,
                structured_plan=formatted_plan,
                run_state=run_state,
                context_payload=context_payload,
                previous_code=previous_code
            )
            
            combined_plan = f"{formatted_plan}\n\n{'='*60}\n【Coding Agent DEBUG 计划】\n{'='*60}\n{code_output.plan}"
            return CodeOutput(plan=combined_plan, code=code_output.code, raw_response=code_output.raw_response)
        
        # OPTIMIZE 状态：
        # 【架构变更】系统自动优化时，如果传入了 prebuilt_plan（由 EvaluationAgent.evaluate() 返回的 replan_output），
        # 则跳过 PlanAgent，直接使用此计划。只有当用户反馈优化（prebuilt_plan=None）时才调用 PlanAgent。
        if run_state == "OPTIMIZE":
            if prebuilt_plan:
                # 【新路径】EvaluationAgent 已生成重新规划，直接使用
                logger.info("[PlanCodingAgent] 复杂任务 OPTIMIZE：使用 EvaluationAgent 预构建计划，跳过 PlanAgent...")
                formatted_plan = prebuilt_plan
                logger.info(f"[PlanCodingAgent] 预构建计划:\n{formatted_plan[:500]}...")
                
                code_output = self._get_coding_agent().generate(
                    task_config=task_config,
                    structured_plan=formatted_plan,
                    run_state=run_state,
                    context_payload="",  # 计划已包含所有优化方向
                    previous_code=previous_code
                )
                
                combined_plan = f"{formatted_plan}\n\n{'='*60}\n【Coding Agent OPTIMIZE 计划】\n{'='*60}\n{code_output.plan}"
                return CodeOutput(plan=combined_plan, code=code_output.code, raw_response=code_output.raw_response)
            else:
                # 【旧路径】用户反馈优化，需要 PlanAgent 重新规划
                logger.info("[PlanCodingAgent] 复杂任务 OPTIMIZE（用户反馈）：调用 PlanAgent 重新规划...")
                plan_result = self._get_plan_agent().generate(
                    task_config=task_config,
                    context_payload=context_payload,
                    evaluation_history=evaluation_history
                )
                self._structured_plan = plan_result
                
                formatted_plan = self._get_plan_agent().format_plan_for_coding(plan_result)
                logger.info(f"[PlanCodingAgent] PlanAgent 优化完成，调用 CodingAgent...")
                
                code_output = self._get_coding_agent().generate(
                    task_config=task_config,
                    structured_plan=formatted_plan,
                    run_state=run_state,
                    context_payload="",  # 优化建议已传给 PlanAgent，CodingAgent 只需基于新计划写代码
                    previous_code=previous_code
                )
                
                combined_plan = f"{formatted_plan}\n\n{'='*60}\n【Coding Agent OPTIMIZE 计划】\n{'='*60}\n{code_output.plan}"
                return CodeOutput(plan=combined_plan, code=code_output.code, raw_response=code_output.raw_response)
        
        # 其他状态（fallback）：复用已有计划
        if self._structured_plan is None:
            logger.warning(f"[PlanCodingAgent] 复杂任务 {run_state} 无缓存计划，重新生成...")
            plan_result = self._get_plan_agent().generate(task_config)
            self._structured_plan = plan_result
        
        formatted_plan = self._get_plan_agent().format_plan_for_coding(self._structured_plan)
        code_output = self._get_coding_agent().generate(
            task_config=task_config,
            structured_plan=formatted_plan,
            run_state=run_state,
            context_payload=context_payload,
            previous_code=previous_code
        )
        combined_plan = f"{formatted_plan}\n\n{'='*60}\n【Coding Agent {run_state} 计划】\n{'='*60}\n{code_output.plan}"
        return CodeOutput(plan=combined_plan, code=code_output.code, raw_response=code_output.raw_response)
    
    def generate_artifacts(
        self,
        task_config: TaskConfig,
        best_code: str,
        has_test_set: bool = False,
        error_message: str = "",
        data_dir: Optional[str] = None
    ) -> CodeOutput:
        """
        生成产物代码
        
        Args:
            task_config: 任务配置
            best_code: 最佳代码（作为参考逻辑）
            has_test_set: 是否有测试集
            error_message: 如果传入，则基于错误信息修复产物代码
            
        Returns:
            CodeOutput: 产物生成代码
        """
        slots = task_config.extracted_slots
        
        fix_instruction = ""
        if error_message:
            fix_instruction = f"""
【重要：上一次执行的报错信息】:
```
{error_message}
```
请根据上述报错修复代码，确保所有依赖库均已导入且正确使用，然后重新生成完整的产物代码。
"""
        
        user_suggestions = slots.user_modeling_suggestions or '无'
        
        has_saved_model = False
        model_hint = ""
        is_large_dataset = False
        dataset_size_hint = ""
        if data_dir:
            from pathlib import Path
            import pandas as pd
            model_path = Path(data_dir) / "best_model.pkl"
            if model_path.exists():
                has_saved_model = True
                model_hint = """
【重要提示】：data/best_model.pkl 已经存在（这是之前训练阶段保存的最优模型）。
请直接在产物代码中加载该模型（`pickle.load(open('data/best_model.pkl', 'rb'))`），**无需重新训练**。
加载模型后即可进行测试集预测、生成特征重要性和报告。
如果加载失败，才回退到重新训练。"""
            else:
                model_hint = """
【提示】：data/best_model.pkl 不存在，说明训练阶段未保存模型。请在产物代码中重新训练模型（逻辑与最佳代码一致），训练完成后保存到 output/model.pkl 即可。"""
            
            # 检测数据集大小
            train_path = Path(data_dir) / "train.csv"
            if train_path.exists():
                try:
                    # 只读取一行来快速获取行数（避免加载整个大文件）
                    import subprocess, sys
                    result = subprocess.run(
                        [sys.executable, "-c", f"import pandas as pd; df=pd.read_csv(r'{train_path}'); print(len(df))"],
                        capture_output=True, text=True, timeout=10
                    )
                    n_rows = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
                    if n_rows > 100000:
                        is_large_dataset = True
                        dataset_size_hint = f"""
【数据集大小提示】：训练集包含 {n_rows} 行（大数据集，超过 10 万行）。
**产物代码必须简化**：禁止生成 feature_importance.png、禁止 SHAP、禁止复杂可视化。
只需生成：output/model.pkl + output/feature_importance.csv（直接从模型获取）+ output/report.html（纯文本表格）。"""
                        logger.info(f"[PlanCodingAgent] 检测到大数据集: {n_rows} 行，产物将简化")
                except Exception:
                    pass
        
        user_prompt = f"""【任务配置】:
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}
- 是否有测试集: {'是' if has_test_set else '否'}

【沙箱可用数据文件】（产物代码只能使用以下存在的文件，禁止读取不存在的文件）：
- data/train.csv（训练数据，一定存在）
{'- data/test.csv（测试数据）' if has_test_set else ''}
{'- data/best_model.pkl（已保存的最佳模型）' if has_saved_model else ''}

【用户建模建议】（重要参考，灵活采纳）:
{user_suggestions}

【最佳代码参考】（请基于以下代码的逻辑重新编写产物生成代码，确保模型训练和预处理方式一致）:
```python
{best_code}
```
{fix_instruction}
{model_hint}
{dataset_size_hint}

请生成产物代码，要求：
1. {'加载 data/best_model.pkl（优先）或重新训练最终模型' if has_saved_model else '重新训练最终模型（仅使用 data/train.csv）'}
2. 保存模型到 output/model.pkl
{'3. 【跳过】数据集过大，不生成 feature_importance.png' if is_large_dataset else '3. 生成特征重要性图到 output/feature_importance.png'}
4. 生成特征重要性 CSV 到 output/feature_importance.csv（直接从模型属性获取，禁止 SHAP）
5. 生成 HTML 报告到 output/report.html（{'纯文本表格即可，禁止复杂可视化' if is_large_dataset else '包含模型指标摘要、特征重要性表格、测试集预测预览'}）
{'6. 对测试集预测并保存到 output/test_predictions.csv' if has_test_set else ''}
7. 最后 print(json.dumps({...})) 输出摘要
"""
        
        logger.info(f"[PlanCodingAgent] 生成产物代码, has_test_set={has_test_set}, llm={self._coding_llm.model if self._coding_llm else 'default'}")
        
        # 【修改】产物生成使用 coding_llm（代码能力更强），而非 plan_llm
        # 同时临时增加 max_tokens，产物代码通常很长（10k-15k 字符），需要更多 tokens
        original_llm = self.llm
        if self._coding_llm is not original_llm:
            self.llm = self._coding_llm
        original_max_tokens = self.llm.max_tokens
        self.llm.max_tokens = max(original_max_tokens, 16384)
        try:
            response = self._call_llm(ARTIFACT_SYSTEM_PROMPT, user_prompt)
        finally:
            self.llm.max_tokens = original_max_tokens
            self.llm = original_llm
        
        plan, code = self._parse_response(response)
        
        logger.info(f"[PlanCodingAgent] 产物代码解析完成, code长度={len(code)}")
        
        # 【语法检查】产物代码返回前用 ast.parse 预检，防止未闭合字符串等语法错误
        if code:
            try:
                import ast
                ast.parse(code)
                logger.info("[PlanCodingAgent] 产物代码 AST 语法检查通过")
            except SyntaxError as e:
                logger.warning(f"[PlanCodingAgent] 产物代码 AST 语法错误: {e}，尝试自动修复")
                code = self._fix_unterminated_strings(code)
                try:
                    ast.parse(code)
                    logger.info("[PlanCodingAgent] 自动修复后 AST 语法检查通过")
                except SyntaxError as e2:
                    logger.error(f"[PlanCodingAgent] 自动修复失败，仍有语法错误: {e2}")
        
        return CodeOutput(plan=plan, code=code, raw_response=response)
    
    def generate_predict_script(
        self,
        task_config: TaskConfig,
        best_code: str,
        data_dir: Optional[str] = None
    ) -> Optional[CodeOutput]:
        """
        专门生成配套预测脚本 predict.py
        
        由于训练代码已将所有特征工程放入 Pipeline，predict.py 只需：
        1. 加载 data/best_model.pkl
        2. 加载 data/test.csv
        3. 调用 pipeline.predict() 直接预测
        4. 保存 output/eval_predictions.csv
        """
        slots = task_config.extracted_slots
        
        has_saved_model = False
        model_hint = ""
        if data_dir:
            from pathlib import Path
            model_path = Path(data_dir) / "best_model.pkl"
            if model_path.exists():
                has_saved_model = True
                model_hint = "data/best_model.pkl 已存在，直接加载。"
            else:
                model_hint = "data/best_model.pkl 不存在，predict.py 需要重新训练模型。"
        
        user_prompt = f"""【任务配置】:
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}

【训练代码参考】（请基于以下代码的 Pipeline 结构生成预测脚本）:
```python
{best_code}
```

{model_hint}

请生成 predict.py 的完整代码。注意：
- 由于训练代码已将所有特征工程放入 sklearn Pipeline，predict.py 只需直接调用 pipeline.predict()
- 不需要手动做任何特征工程（如提取时间特征、编码等）
- 脚本必须完全自包含
"""
        
        logger.info(f"[PlanCodingAgent] 生成预测脚本 predict.py, llm={self._coding_llm.model if self._coding_llm else 'default'}")
        
        # 【修改】predict.py 生成使用 coding_llm（代码能力更强）
        original_llm = self.llm
        if self._coding_llm is not original_llm:
            self.llm = self._coding_llm
        original_max_tokens = self.llm.max_tokens
        self.llm.max_tokens = max(original_max_tokens, 8192)
        try:
            response = self._call_llm(PREDICT_SCRIPT_SYSTEM_PROMPT, user_prompt)
            plan, code = self._parse_response(response)
            
            if not code:
                logger.warning(f"[PlanCodingAgent] 预测脚本生成失败，未解析到代码")
                return None
            
            # 【语法检查】预测脚本返回前用 ast.parse 预检
            try:
                import ast
                ast.parse(code)
            except SyntaxError as e:
                logger.warning(f"[PlanCodingAgent] 预测脚本语法错误: {e}，尝试自动修复")
                code = self._fix_unterminated_strings(code)
                try:
                    ast.parse(code)
                except SyntaxError as e2:
                    logger.error(f"[PlanCodingAgent] 预测脚本自动修复失败: {e2}")
            
            logger.info(f"[PlanCodingAgent] 预测脚本解析完成, code长度={len(code)}")
            return CodeOutput(plan=plan, code=code, raw_response=response)
        except Exception as e:
            logger.warning(f"[PlanCodingAgent] 预测脚本生成异常: {e}")
            return None
        finally:
            self.llm.max_tokens = original_max_tokens
            self.llm = original_llm
    
    def _build_user_prompt(
        self,
        task_config: TaskConfig,
        run_state: str,
        context_payload: str,
        previous_code: str
    ) -> str:
        """构建用户提示词"""
        slots = task_config.extracted_slots
        
        # 原始文件信息（保留透明度）
        raw_files_info = "\n".join([
            f"- {f.name} (role={f.role.value})"
            for f in task_config.uploaded_files
        ])
        
        # 沙箱内实际可用的 CSV 文件（数据切分器统一转换后的文件名）
        sandbox_files_info = []
        has_train = any(f.role.value == "train" for f in task_config.uploaded_files)
        has_val = any(f.role.value == "validation" for f in task_config.uploaded_files)
        has_test = any(f.role.value == "test" for f in task_config.uploaded_files)
        
        if has_train:
            sandbox_files_info.append("- data/train.csv （训练集，必用）")
        if has_val or has_train:
            # 无验证集时 train 会被切分，也会生成 validation.csv
            sandbox_files_info.append("- data/validation.csv （验证集，必用）")
        if has_test:
            sandbox_files_info.append("- data/test.csv （测试集，仅用于最终预测）")
        
        files_info = f"""【原始上传文件】:
{raw_files_info}

【沙箱内可用文件（已被统一转换为 CSV）】:
{chr(10).join(sandbox_files_info)}

【重要提醒】代码中读取数据时，请只使用上述【沙箱内可用文件】的路径，不要引用原始文件名。"""
        
        profile_json = json.dumps(task_config.data_profile, ensure_ascii=False, indent=2) if task_config.data_profile else '暂无详细画像'
        code_section = '```python\n' + previous_code + '\n```' if previous_code else '无（当前为首次生成）'
        
        user_suggestions = slots.user_modeling_suggestions or '无'
        
        # 【数据预处理强制检查清单】从数据画像中提取关键信息，以列表形式强制呈现
        preprocessing_checklist = ""
        if task_config.data_profile and "columns" in task_config.data_profile:
            cols = task_config.data_profile["columns"]
            target_col = slots.target_column or ""
            
            # 1. 目标列缺失值
            target_missing = [c for c in cols if c["name"] == target_col and c.get("missingCount", 0) > 0]
            if target_missing:
                preprocessing_checklist += f"\n- 目标列 '{target_col}' 有 {target_missing[0]['missingCount']} 个缺失值，必须在读取数据后立即用 df = df.dropna(subset=['{target_col}']) 删除这些行。"
            
            # 2. 类别特征列（object/string 类型）
            cat_cols = [c for c in cols if c.get("type") == "categorical" and c["name"] != target_col]
            if cat_cols:
                cat_names = [c["name"] for c in cat_cols]
                preprocessing_checklist += f"\n- 类别特征列 {cat_names} 必须纳入 Pipeline 的 ColumnTransformer 中，使用 OneHotEncoder(handle_unknown='ignore') 或 OrdinalEncoder 编码，严禁直接传入模型。"
            
            # 3. 数值特征缺失值（非目标列）
            num_missing_cols = [c for c in cols if c.get("type") == "numeric" and c.get("missingCount", 0) > 0 and c["name"] != target_col]
            if num_missing_cols:
                num_names = [c["name"] for c in num_missing_cols]
                preprocessing_checklist += f"\n- 数值特征列 {num_names} 有缺失值，必须在 ColumnTransformer 中使用 SimpleImputer(strategy='median') 填充。"
            
            # 4. id 列识别与丢弃
            id_cols = [c for c in cols if c.get("isLikelyId") and c["name"] != target_col]
            if id_cols:
                id_names = [c["name"] for c in id_cols]
                preprocessing_checklist += f"\n- id 列 {id_names} 必须丢弃（uniqueCount ≈ rowCount），严禁传入模型，否则会导致严重过拟合。"
            
            # 5. 时序特征工程建议（当存在 year/month/day/hour 等时间列时）
            time_cols = [c["name"] for c in cols if c["name"] in ["year", "month", "day", "hour", "minute", "second", "weekday"]]
            if time_cols:
                preprocessing_checklist += f"\n- 检测到时间列 {time_cols}，建议在 Pipeline 内增加周期性编码（如 sin/cos 变换：np.sin(2*np.pi*month/12)），以提升时序建模效果。"
        
        if not preprocessing_checklist:
            preprocessing_checklist = "（数据画像中未发现明显的预处理问题，但仍请按规范检查）"
        
        # 知识库建议加载（仅 complex 任务）
        kb_refs = ""
        if task_config.extracted_slots.complexity == "complex":
            try:
                from app.agents.intent_recognition import IntentResult
                from app.models.schemas import TaskType
                intent = IntentResult(
                    target_column=task_config.extracted_slots.target_column or "",
                    task_type=task_config.extracted_slots.task_type or TaskType.BINARY_CLASSIFICATION,
                    eval_metric=task_config.extracted_slots.eval_metric,
                    complexity=task_config.extracted_slots.complexity or "simple",
                    complexity_reason=task_config.extracted_slots.complexity_reason
                )
                kb_loader = KnowledgeBaseLoader()
                refs = kb_loader.load_references(intent)
                if refs:
                    kb_refs = "\n".join(f"- {r}" for r in refs)
            except Exception as e:
                logger.warning(f"[PlanCodingAgent] 知识库加载失败: {e}")
        
        kb_section = ""
        if kb_refs:
            kb_section = f"""
【知识库参考 Knowledge Base】（以下为该类任务的历史最佳实践，请灵活参考而非死板执行）：
{kb_refs}

"""
        
        prompt = f"""【当前运行状态 Run State】: {run_state}

【意图澄清与任务配置 Task Config】:
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}
- 复杂度判定: {slots.complexity or 'unknown'}（原因: {slots.complexity_reason or '未说明'}）
- 特征约束（需丢弃的列）: {slots.feature_constraints or []}
- 用户描述: {task_config.user_description or '无'}

【数据预处理强制检查清单 Data Preprocessing Checklist】（以下检查必须无条件执行，否则代码会执行失败）：
{preprocessing_checklist}
{kb_section}
【用户建模建议 User Modeling Suggestions】（重要参考，灵活采纳而非死板执行）：
{user_suggestions}

【上传文件信息】:
{files_info}

【数据画像 Data Profile】:
{profile_json}

【上下文载荷 Context Payload】:
{context_payload or '无'}

【历史代码 Previous Code】:
{code_section}

请根据以上信息，生成对应的 <plan> 和 <code>。
在规划时，请特别关注【数据预处理强制检查清单】、【用户建模建议】和【上下文载荷】中的评估建议，将其作为调优方向的重要参考，但保持专业判断，不合理的建议可调整并说明理由。
"""
        return prompt
    
    def _parse_response(self, response: str) -> tuple:
        """
        解析 LLM 响应，提取 plan 和 code
        兼容多种输出格式：<plan>/<code> 标签、纯 markdown 代码块、混合格式
        """
        # ====== 提取 plan ======
        plan = ""
        # 策略1: 标准 <plan> 标签
        plan_match = re.search(r'<plan>(.*?)</plan>', response, re.DOTALL)
        if plan_match:
            plan = plan_match.group(1).strip()
        
        # 策略2: 从 markdown 标题/段落中提取计划描述（无 <plan> 标签时兜底）
        if not plan:
            plan_fallback = re.search(
                r'(?:计划|策略|方案|思路|步骤|pipeline)[：:]\s*\n?(.*?)(?:\n\n|\n```|\Z)',
                response, re.DOTALL | re.IGNORECASE
            )
            if plan_fallback:
                plan = plan_fallback.group(1).strip()
        
        # 策略3: 提取所有非代码块的文本作为 plan
        if not plan:
            text_parts = re.split(r'```(?:python)?\s*', response)
            if text_parts and len(text_parts[0]) > 50:
                plan = text_parts[0].strip()
        
        # ====== 提取 code ======
        code = ""
        # 策略1: 标准格式 <code>...```python ... ```...</code>
        code_match = re.search(r'<code>.*?(?:```python\s*(.*?)\s*```|`(.*?)`)</code>', response, re.DOTALL)
        if code_match:
            code = code_match.group(1) or code_match.group(2) or ""
        
        # 策略2: 兜底直接提取 ```python ... ```
        if not code:
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
        
        # 策略3: 处理 LLM 忘记闭合 ``` 的情况（长代码常见）
        if not code:
            python_block_start = response.find('```python')
            if python_block_start != -1:
                code_start = python_block_start + len('```python')
                code_end = response.find('```', code_start)
                if code_end == -1:
                    code_end = response.find('</code>', code_start)
                if code_end == -1:
                    code_end = len(response)
                code = response[code_start:code_end].strip()
                if code:
                    logger.info(f"[PlanCodingAgent] 检测到未闭合代码块，已提取 {len(code)} 字符")
                    # 未闭合代码块可能是 LLM 输出被截断，尝试修复未闭合字符串
                    code = self._fix_unterminated_strings(code)
        
        # 策略4: 如果没有 ```python，尝试提取任何 ``` 代码块
        if not code:
            code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
        
        # 再次清理 code 中可能残留的 markdown 标记
        code = code.strip()
        if code.startswith("python"):
            code = code[6:].strip()
        
        if not plan:
            logger.warning("[PlanCodingAgent] 未解析到 plan 内容，使用代码注释/文本兜底")
        if not code:
            logger.warning("[PlanCodingAgent] 未解析到 code 内容")
        
        return plan, code
    
    def _fix_unterminated_strings(self, code: str) -> str:
        """
        修复产物代码中未闭合的字符串（常见于 LLM 输出被截断时）
        
        主要场景：
        - HTML 报告使用 f'''...'''，但闭合在代码末尾被截断
        - 代码中的多行注释/文档字符串未闭合
        - 代码末尾的单行字符串（如 print 参数）被截断
        - f-string 嵌套复杂表达式导致未闭合
        """
        import re
        
        lines = code.split('\n')
        if not lines:
            return code
        
        # ========== 策略1: 修复未闭合的三引号 ==========
        triple_double = 0
        triple_single = 0
        in_triple = None
        last_triple_line = -1
        
        for i, line in enumerate(lines):
            j = 0
            while j < len(line):
                if in_triple is None:
                    if line[j:j+3] == '"""':
                        triple_double += 1
                        in_triple = '"""'
                        last_triple_line = i
                        j += 3
                        continue
                    elif line[j:j+3] == "'''":
                        triple_single += 1
                        in_triple = "'''"
                        last_triple_line = i
                        j += 3
                        continue
                elif in_triple == '"""' and line[j:j+3] == '"""':
                    triple_double += 1
                    in_triple = None
                    last_triple_line = i
                    j += 3
                    continue
                elif in_triple == "'''" and line[j:j+3] == "'''":
                    triple_single += 1
                    in_triple = None
                    last_triple_line = i
                    j += 3
                    continue
                j += 1
        
        if in_triple is not None:
            logger.info(f"[PlanCodingAgent] 检测到未闭合的 {in_triple}，自动添加闭合")
            code = code.rstrip() + '\n' + in_triple
            return code
        
        # ========== 策略2: 修复代码末尾被截断的单行字符串 ==========
        def _count_quotes(line: str) -> tuple:
            """统计一行中单/双引号的数量（考虑转义，但不考虑三引号）"""
            single = 0
            double = 0
            i = 0
            while i < len(line):
                if line[i] == '\\' and i + 1 < len(line):
                    i += 2
                    continue
                if line[i:i+3] in ('"""', "'''"):
                    i += 3
                    continue
                if line[i] == "'":
                    single += 1
                elif line[i] == '"':
                    double += 1
                i += 1
            return single, double
        
        for check_idx in range(max(0, len(lines) - 3), len(lines)):
            line = lines[check_idx]
            single_count, double_count = _count_quotes(line)
            
            if single_count % 2 == 1:
                if not line.rstrip().endswith("'") or check_idx == len(lines) - 1:
                    logger.info(f"[PlanCodingAgent] 第{check_idx+1}行有未闭合单引号，截断修复")
                    lines = lines[:check_idx]
                    code = '\n'.join(lines)
                    return code
            
            if double_count % 2 == 1:
                if not line.rstrip().endswith('"') or check_idx == len(lines) - 1:
                    logger.info(f"[PlanCodingAgent] 第{check_idx+1}行有未闭合双引号，截断修复")
                    lines = lines[:check_idx]
                    code = '\n'.join(lines)
                    return code
        
        # ========== 策略3: 尝试修复 f-string 嵌套导致的未闭合 ==========
        last_line = lines[-1] if lines else ""
        f_match = re.search(r'f["\']', last_line)
        if f_match:
            quote_char = last_line[f_match.start() + 1]
            rest = last_line[f_match.start() + 2:]
            quote_count = 0
            i = 0
            while i < len(rest):
                if rest[i] == '\\':
                    i += 2
                    continue
                if rest[i] == quote_char:
                    quote_count += 1
                i += 1
            if quote_count % 2 == 1 or quote_count == 0:
                logger.info(f"[PlanCodingAgent] 最后一行有未闭合 f-string，截断修复")
                lines = lines[:-1]
                code = '\n'.join(lines)
                return code
        
        return code
    
    def _sanitize_training_code(self, code: str) -> str:
        """
        清理训练代码中训练阶段不应该出现的模式。
        训练阶段 data/test.csv 不存在，任何读取都会导致 FileNotFoundError。
        """
        import re
        
        if not code:
            return code
        
        lines = code.split('\n')
        result = []
        test_csv_patterns = [
            r"pd\.read_csv\(['\"]data/test\.csv['\"]\)",
            r"pd\.read_csv\(['\"]\.\/data\/test\.csv['\"]\)",
        ]
        removed_count = 0
        
        for line in lines:
            stripped = line.strip()
            should_remove = False
            for pattern in test_csv_patterns:
                if re.search(pattern, stripped):
                    logger.warning(f"[PlanCodingAgent] 自动删除训练代码中的 test.csv 读取: {stripped[:80]}")
                    should_remove = True
                    removed_count += 1
                    break
            if not should_remove:
                result.append(line)
        
        if removed_count > 0:
            code = '\n'.join(result)
            if 'test_df' in code and 'test_df = ' not in code and 'test_df=' not in code:
                logger.warning("[PlanCodingAgent] 代码中仍有 test_df 使用但无定义，添加安全占位")
                code = "# test.csv 在训练阶段不存在\ntest_df = None\n\n" + code
        else:
            code = '\n'.join(result)
        
        return code
    
    def _ensure_code_valid(self, code: str, label: str = "代码") -> str:
        """
        确保代码语法有效。先尝试 ast.parse，失败则尝试 _fix_unterminated_strings。
        返回修复后的代码（或原始代码如果无法修复）。
        """
        if not code:
            return code
        
        try:
            import ast
            ast.parse(code)
            return code
        except SyntaxError as e:
            logger.warning(f"[PlanCodingAgent] {label} AST 语法错误: {e}，尝试自动修复")
            fixed = self._fix_unterminated_strings(code)
            try:
                ast.parse(fixed)
                logger.info(f"[PlanCodingAgent] {label} 自动修复后 AST 检查通过")
                return fixed
            except SyntaxError as e2:
                logger.error(f"[PlanCodingAgent] {label} 自动修复失败，仍有语法错误: {e2}")
                return code


import json

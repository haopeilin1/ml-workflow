"""
Coding Agent — 复杂任务专用代码生成 Agent

职责：接收 PlanAgent 的结构化计划，生成严格遵循 must_do/avoid 的 Python Pipeline 代码。

设计哲学：PlanAgent 已经想好了"做什么"，Coding Agent 只需要专注于"怎么做"——把计划翻译成代码。
"""

import logging
import re
from typing import Optional

from app.agents.base import BaseAgent
from app.models.schemas import CodeOutput, TaskConfig

logger = logging.getLogger(__name__)


CODING_AGENT_SYSTEM_PROMPT = """你是一名资深机器学习工程师。当前任务是根据一份**已完成的结构化建模计划**，编写可执行的 Python Pipeline 代码。

你的唯一职责：将计划中的每一项要求翻译成正确的代码。你不是在做架构决策——决策已经在计划中了。

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
   
   # 1. 区分列类型（关键步骤，必须先做）
   cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
   num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
   
   # 2. 构建预处理 Pipeline（缺失值填充 + 编码/缩放）
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
   
   # 3. 完整 Pipeline：预处理 + 模型
   pipeline = Pipeline([
       ('preprocess', preprocessor),
       ('model', LGBMClassifier(random_state=42))
   ])
   pipeline.fit(X_train, y_train)
   ```
   
4. 算法优选：优先使用 Scikit-Learn, LightGBM, XGBoost 等快速且效果好的树模型。
5. 【关键版本兼容性 - 必须严格遵守】沙箱中的库版本较新，与网上旧教程的 API 不同。以下 API 变更**必须使用新写法**，否则代码执行会立即失败：

   **(a) LightGBM 4.6.0**
   以下参数在 LGBMClassifier.fit() / LGBMRegressor.fit() 中**已被移除**：
   - ❌ early_stopping_rounds（已移除）
   - ❌ verbose（已移除）
   - ❌ eval_at（已移除）
   
   【LightGBM 正确写法】
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

   **(b) scikit-learn 1.6+**
   `LogisticRegression` 已发生以下**破坏性变更**：
   - ❌ `multi_class` 参数**已被移除**（不再存在，传了会报 `unexpected keyword argument`）
   - ❌ `solver='liblinear'` **不支持多分类**（n_classes >= 3 时会报 `liblinear solver does not support multiclass classification`）
   - ✅ 多分类任务请使用 `solver='lbfgs'`（默认）或 `solver='newton-cg'`、`solver='sag'`、`solver='saga'`
   
   【sklearn 正确写法 - 多分类】
   ```python
   # 多分类 LogisticRegression（sklearn 1.6+）
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(
       max_iter=1000,
       solver='lbfgs',           # 必须使用 lbfgs/newton-cg/sag/saga，禁止用 liblinear
       class_weight='balanced',  # 处理类别不平衡
       random_state=42
   )
   # 绝对不要传 multi_class 参数！该参数已不存在。
   ```
   【sklearn 错误写法 - 绝对禁止】
   ```python
   LogisticRegression(multi_class='multinomial', solver='liblinear')  # 错误！multi_class已移除，liblinear不支持多分类
   ```
   
   **(c) sklearn FunctionTransformer（常见陷阱）**
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
6. 环境依赖：当前沙箱已预装以下 Python 包，请优先使用这些库编写代码（无需 pip install）：
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
   
   **(c) imbalanced-learn（类别不平衡处理）**
   - `SMOTE` / `RandomOverSampler` / `RandomUnderSampler` 等**只能 fit 训练集，绝对不能 fit 验证集或测试集**。这是数据泄露的红线。
   - 正确做法：先用 `train_test_split` 分出训练/验证（但注意：**本系统已预先切分好 validation.csv，严禁自己重新切分**），然后只对训练集 X_train 做重采样，验证集保持原样。
   - 由于本系统已提供 `data/validation.csv`，你应该将 `data/train.csv` 作为训练数据，在其内部做重采样（如需要），而 `data/validation.csv` 直接用于评估，不做任何重采样。
   - 如果训练集本身需要切分为子训练集和子验证集用于 early stopping，可以在这内部使用 SMOTE，但验证子集必须保持原始分布。
   
   【imbalanced-learn 正确写法】
   ```python
   from imblearn.over_sampling import SMOTE
   from imblearn.under_sampling import RandomUnderSampler
   from imblearn.pipeline import Pipeline as ImbPipeline  # 注意：必须用 imblearn.pipeline.Pipeline，不是 sklearn.pipeline.Pipeline
   
   # 方式1：在 sklearn Pipeline 中嵌入重采样（推荐）
   # 必须用 imblearn.pipeline.Pipeline 才能兼容重采样器作为中间步骤
   from imblearn.pipeline import Pipeline as ImbPipeline
   from sklearn.preprocessing import StandardScaler
   from lightgbm import LGBMClassifier
   
   pipeline = ImbPipeline([
       ('scaler', StandardScaler()),
       ('smote', SMOTE(random_state=42)),  # SMOTE 只能在训练时 fit，ImbPipeline 会自动确保这一点
       ('model', LGBMClassifier(random_state=42))
   ])
   pipeline.fit(X_train, y_train)  # SMOTE 只会在 fit 时作用于训练数据
   ```
   
   **(d) category-encoders（高基数类别编码）**
   - `TargetEncoder` / `LeaveOneOutEncoder` 等**只能 fit 训练集**，用训练集的统计量 transform 验证集和测试集。
   - 严禁在 fit 时传入验证集或全量数据。
   
   【category-encoders 正确写法】
   ```python
   from category_encoders import TargetEncoder
   
   encoder = TargetEncoder(cols=['category_col'])
   X_train_enc = encoder.fit_transform(X_train, y_train)  # 只能 fit 训练集
   X_val_enc = encoder.transform(X_val)                   # 验证集用训练集学到的统计量 transform
   ```
   
   **(e) optuna（超参优化）**
   - 如需使用，请设置 `optuna.logging.set_verbosity(optuna.logging.WARNING)` 减少日志输出。
   - 在极小样本任务（< 200 行）上慎用，容易过拟合验证集。
   
   **(f) shap（模型可解释性）**
   - 树模型可用 `shap.TreeExplainer`，其他模型用 `shap.Explainer`。
   - 样本量大时只取子集计算（如 `shap.sample(X, 100)`），避免内存溢出。
   
   **(g) feature-engine（特征工程）**
   - 该库的 transformer 同样遵循 sklearn fit/transform 规范：fit 训练集，transform 验证/测试集。
   
   如果代码需要 import 未在上述列表中的第三方库，请先检查是否可用已安装库替代，避免执行失败。
6. 数据路径（关键）：
   - 所有上传的数据文件在沙箱中会被**统一转换为 CSV 格式**，文件名固定为：`train.csv`（训练集）、`validation.csv`（验证集）、`test.csv`（测试集，如有）。
   - 沙箱工作目录下有一个 `data/` 子目录，所有数据文件都位于其中。
   - 读取数据时必须使用相对路径：`pd.read_csv('data/train.csv')`、`pd.read_csv('data/validation.csv')`、`pd.read_csv('data/test.csv')`。
   - **【绝对禁止】不要自己用 train_test_split 或其他方式重新切分验证集。data/validation.csv 已经由系统预先切分好，直接使用即可。**
   - **【绝对禁止】代码中严禁 import train_test_split（即使不使用也不允许导入）。**
   - **【时序任务特别约束】如果是时序任务（is_time_series=true），data/validation.csv 已经按时间顺序切分（前80%为训练集，后20%为验证集）。严禁重新切分或打乱顺序，必须保持时间连续性。**
   - 严禁使用绝对路径（如 `D:\...` 或 `/home/...`），不要直接使用文件名（如 `pd.read_csv('train.csv')` 会找不到文件），也不要使用原始 `.xlsx` 文件名（沙箱内不存在 `.xlsx` 文件）。

MUST DO 执行规则（最重要）
1. 【强制实现】计划中的 **must_do（尤其是 critical=true 的项）必须在代码中有明确体现**。如果计划要求 "scale_pos_weight = 负类数/正类数"，代码中必须有 `scale_pos_weight=np.sum(y==0)/np.sum(y==1)` 或等效实现。
2. 【强制避免】计划中的 **avoid 项在代码中绝对不能出现**。如果计划要求避免 "class_weight='balanced'"，代码中任何地方都不能有 `class_weight='balanced'`。
3. 【强制对齐】计划中的 **pipeline_plan 步骤必须按顺序实现**。如果计划说第1步是 "对 Amount 做 log1p 变换"，代码中必须先做这个变换。
4. 【强制定义 prepare_for_prediction】代码中**必须定义一个独立的 `def prepare_for_prediction(df):` 函数**，满足以下要求：
   - 【必须自包含】函数内部必须重新定义所有需要的列名、参数，不能依赖外部全局变量
   - 【必须完整】该函数必须处理训练阶段对 X 做的**所有**预处理（包括列选择、丢弃列、log 变换、缺失值填充、编码等）
   - 【不能只在 Pipeline 内实现】不能认为 "Pipeline 里已经有了 FunctionTransformer 所以不需要 prepare_for_prediction"。预测阶段可能只注入这个函数本身，不会注入外部 Pipeline 对象。
   - 【必须返回 DataFrame】返回与训练阶段预处理完成后完全一致的 DataFrame
5. 【强制验证】代码写完后，请在脑中检查一遍：
   - 每个 critical must_do 是否都有对应的代码行？
   - 每个 avoid 项是否都没有出现在代码中？
   - `prepare_for_prediction` 是否独立定义且自包含？
   - 如果答案为否，修改代码直到满足要求。

Output Format
你必须严格按以下标签格式输出：
<plan>
简要说明：基于结构化计划，我将在代码中实现以下关键点...
（列出你确认的 must_do 实现方式和 avoid 规避方式）
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




DEBUG_SYSTEM_PROMPT = """你是一名资深机器学习工程师。当前处于 **DEBUG 模式**——上一版代码在沙箱中执行失败了，你的任务是**根因分析 + 彻底修复**。

## 你的职责（与 INIT 模式不同）

INIT 模式中你只需翻译计划。DEBUG 模式中你需要：
1. **先分析根因**：不要只看最后一行报错，要理解为什么会报这个错
2. **检查历史错误**：确保本次修复不会重复之前已经犯过的错误
3. **彻底修复**：修复后代码必须能完整执行通过，不要留隐患
4. **有权调整策略**：如果错误证明原计划中的某个技术选型（如 LightGBM）在当前环境下不可行，你有权改用其他模型（如 RandomForest），但必须在 plan 中说明理由

## Debug 核心原则

1. 【根因分析优先】不要只修表面症状。例如：
   - 报错 "TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'verbose'"
   - 根因不是"删掉 verbose"，而是"LightGBM 4.6.0 已移除 verbose 参数，说明我在用旧版 API"
   - 彻底修复：改用新版 API（callbacks=[lgb.early_stopping(...)]），并检查是否还有其他旧版参数

2. 【历史错误清单 - 必须避免重复】
   以下是之前版本已经犯过的错误，本次修复**绝对不能再犯**：
   - ❌ LGBMClassifier.fit() 中使用 early_stopping_rounds 参数（已移除）
   - ❌ LGBMClassifier.fit() 中使用 verbose 参数（已移除）
   - ❌ LGBMClassifier.fit() 中使用 eval_at 参数（已移除）
   - ❌ 使用 lgb.train() 替代 LGBMClassifier（lgb.train() 返回裸 Booster，不能放入 Pipeline）
   - ❌ LogisticRegression 传入 multi_class 参数（sklearn 1.6+ 已移除）
   - ❌ 多分类任务使用 solver='liblinear'（不支持多分类，n_classes>=3 时报错）
   - ❌ SMOTE / RandomOverSampler 等重采样器 fit 了验证集或全量数据（数据泄露）
   - ❌ TargetEncoder / LeaveOneOutEncoder 等编码器 fit 了验证集或全量数据（数据泄露）
   - ❌ import train_test_split（即使不使用也不能导入）
   - ❌ 在训练阶段读取 data/test.csv（训练阶段只有 train.csv 和 validation.csv）
   - ❌ 使用 XGBoost 时出现 feature_names mismatch（Pipeline 预处理后的列名与原始数据不一致）

3. 【关键库正确写法 - 若使用则必须照抄】
   
   **(a) LightGBM 4.6.0**
   ```python
   model = lgb.LGBMClassifier(
       objective='binary',
       scale_pos_weight=scale_pos_weight,
       num_leaves=31, max_depth=6, learning_rate=0.05,
       n_estimators=500, subsample=0.8, min_child_samples=20
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
   
   **(b) sklearn 1.6+ LogisticRegression（多分类）**
   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(
       max_iter=1000,
       solver='lbfgs',           # 多分类必须用 lbfgs/newton-cg/sag/saga
       class_weight='balanced',
       random_state=42
   )
   # 绝对不要传 multi_class！该参数已不存在。
   ```
   
   **(c) imbalanced-learn SMOTE（必须配合 imblearn.pipeline.Pipeline）**
   ```python
   from imblearn.over_sampling import SMOTE
   from imblearn.pipeline import Pipeline as ImbPipeline
   from sklearn.preprocessing import StandardScaler
   from lightgbm import LGBMClassifier
   
   pipeline = ImbPipeline([
       ('scaler', StandardScaler()),
       ('smote', SMOTE(random_state=42)),  # SMOTE 只能在训练时 fit
       ('model', LGBMClassifier(random_state=42))
   ])
   pipeline.fit(X_train, y_train)  # 验证集保持原样，不做重采样
   ```

4. 【历史错误上下文】你收到的 Context Payload 中包含了**所有历史执行错误**（按时间顺序），请逐条分析：
   - 第 1 次错误：根因是什么？是否已经修复？
   - 第 2 次错误：是在修复第 1 次时引入的新错误，还是第 1 次没修干净？
   - 最后一次错误：当前需要重点修复的问题
   
   **绝对禁止**出现 "修复了 A 又引入 B，修复了 B 又变回 A" 的循环。

6. 【验证清单】修复完成后，请在脑中检查：
   - 所有历史错误是否都已避免？
   - 是否有新的潜在错误？
   - 代码是否能从头执行到尾不报错？

Output Format（与 INIT 相同）
<plan>
Debug 分析与修复策略：
1. 根因分析：...
2. 历史错误检查：...
3. 修复方案：...
（如果调整了原计划的技术选型，请在此说明理由）
</plan>
<code>
```python
... 完整修复后的代码 ...
```
</code>"""
class CodingAgent(BaseAgent):
    """
    Coding Agent — 基于结构化计划生成代码
    """

    def generate(
        self,
        task_config: TaskConfig,
        structured_plan: str,
        run_state: str = "INIT",
        context_payload: str = "",
        previous_code: str = ""
    ) -> CodeOutput:
        """
        生成代码

        Args:
            task_config: 任务配置
            structured_plan: PlanAgent 生成的格式化计划文本
            run_state: INIT / DEBUG / OPTIMIZE
            context_payload: 报错信息 / 优化建议
            previous_code: 历史代码
        """
        user_prompt = self._build_user_prompt(
            task_config, structured_plan, run_state, context_payload, previous_code
        )

        logger.info(f"[CodingAgent] 生成代码, state={run_state}")

        # 根据 run_state 选择 system prompt
        if run_state == "DEBUG":
            system_prompt = DEBUG_SYSTEM_PROMPT
            logger.info(f"[CodingAgent] 使用 DEBUG 专用 prompt")
        else:
            system_prompt = CODING_AGENT_SYSTEM_PROMPT
        
        response = self._call_llm(system_prompt, user_prompt)

        plan, code = self._parse_response(response)

        logger.info(f"[CodingAgent] 解析完成, plan长度={len(plan)}, code长度={len(code)}")

        return CodeOutput(plan=plan, code=code, raw_response=response)

    def _build_user_prompt(
        self,
        task_config: TaskConfig,
        structured_plan: str,
        run_state: str,
        context_payload: str,
        previous_code: str
    ) -> str:
        """构建用户提示词"""
        slots = task_config.extracted_slots

        # 文件信息
        raw_files_info = "\n".join([
            f"- {f.name} (role={f.role.value})"
            for f in task_config.uploaded_files
        ])

        sandbox_files_info = []
        has_train = any(f.role.value == "train" for f in task_config.uploaded_files)
        has_val = any(f.role.value == "validation" for f in task_config.uploaded_files)
        has_test = any(f.role.value == "test" for f in task_config.uploaded_files)
        if has_train:
            sandbox_files_info.append("- data/train.csv （训练集，必用）")
        if has_val or has_train:
            sandbox_files_info.append("- data/validation.csv （验证集，必用）")
        if has_test:
            sandbox_files_info.append("- data/test.csv （测试集，仅用于最终预测）")

        profile_json = ""
        if task_config.data_profile:
            import json
            profile_json = json.dumps(task_config.data_profile, ensure_ascii=False, indent=2)

        code_section = '```python\n' + previous_code + '\n```' if previous_code else '无（当前为首次生成）'
        user_suggestions = slots.user_modeling_suggestions or '无'

        # 【新增】加载编码经验知识库（error_patterns.json）
        kb_section = ""
        try:
            import json
            from pathlib import Path
            kb_path = Path(__file__).resolve().parent.parent / "knowledge_base" / "errors" / "error_patterns.json"
            if kb_path.exists():
                with open(kb_path, "r", encoding="utf-8") as f:
                    kb_data = json.load(f)
                patterns = kb_data.get("patterns", [])
                if patterns:
                    kb_lines = ["【编码经验知识库】（历史错误模式及预防措施，编码时必须遵守）：\n"]
                    for i, p in enumerate(patterns, 1):
                        prevention = p.get("prevention_prompt", "")
                        solution_code = p.get("solution_code", "")
                        kb_lines.append(f"{i}. {p.get('error_type', '未知错误')}")
                        if prevention:
                            kb_lines.append(f"   预防: {prevention}")
                        if solution_code:
                            kb_lines.append(f"   参考代码:\n   ```python\n   {solution_code}\n   ```")
                        kb_lines.append("")
                    kb_section = "\n".join(kb_lines)
        except Exception as e:
            logger.warning(f"[CodingAgent] 编码知识库加载失败: {e}")

        prompt = f"""【当前运行状态 Run State】: {run_state}

{structured_plan}

【意图澄清与任务配置 Task Config】:
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}
- 复杂度判定: {slots.complexity or 'unknown'}（原因: {slots.complexity_reason or '未说明'}）
- 是否时序: {slots.is_time_series}
- 特征约束（需丢弃的列）: {slots.feature_constraints or []}
- 用户描述: {task_config.user_description or '无'}

【用户建模建议 User Modeling Suggestions】（重要参考，灵活采纳而非死板执行）：
{user_suggestions}

【文件信息】:
原始上传文件:
{raw_files_info}

沙箱内可用文件:
{chr(10).join(sandbox_files_info)}

【数据画像 Data Profile】:
{profile_json if profile_json else '暂无详细画像'}

【上下文载荷 Context Payload】:
{context_payload or '无'}

{kb_section}

【历史代码 Previous Code】:
{code_section}

请根据上述【结构化建模计划】，编写完整的 Python Pipeline 代码。
特别提醒：
1. 计划中的 must_do（尤其是 critical=true 的项）必须在代码中明确实现，不能遗漏。
2. 计划中的 avoid 项在代码中绝对不能出现。
3. 如果当前是 DEBUG 状态，请在修复 bug 的同时，保持 must_do/avoid 的约束不变。
4. 如果当前是 OPTIMIZE 状态，请在优化性能的同时，保持 must_do/avoid 的约束不变。
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

        # 策略2: 直接提取 ```python ... ```
        if not code:
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()

        # 策略3: 处理 LLM 忘记闭合 ``` 的情况
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
                    logger.info(f"[CodingAgent] 检测到未闭合代码块，已提取 {len(code)} 字符")
        
        # 策略4: 如果没有 ```python，尝试提取任何 ``` 代码块
        if not code:
            code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()

        code = code.strip()
        if code.startswith("python"):
            code = code[6:].strip()

        if not plan:
            logger.warning("[CodingAgent] 未解析到 plan 内容，使用代码注释/文本兜底")
        if not code:
            logger.warning("[CodingAgent] 未解析到 code 内容")

        return plan, code

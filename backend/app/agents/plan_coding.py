"""
Plan & Coding Agent
生成建模计划与可执行 Python Pipeline 代码
"""

import logging
import re
from typing import Optional

from app.agents.base import BaseAgent
from app.models.schemas import CodeOutput, TaskConfig
from app.knowledge_base.loader import KnowledgeBaseLoader

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
3. 算法优选：优先使用 Scikit-Learn, LightGBM, XGBoost 等快速且效果好的树模型。
4. 环境依赖：当前沙箱已预装以下 Python 包，请优先使用这些库编写代码（无需 pip install）：
   - 数据处理：pandas, numpy
   - 机器学习：scikit-learn (sklearn), xgboost, lightgbm
   - 科学计算：scipy
   - 可视化：matplotlib, seaborn
   - 工具：joblib, threadpoolctl
   - 图像：Pillow (PIL)
   如果代码需要 import 未在上述列表中的第三方库，请先检查是否可用 sklearn / pandas / numpy 原生方案替代，避免执行失败。
5. 数据路径（关键）：
   - 所有上传的数据文件在沙箱中会被**统一转换为 CSV 格式**，文件名固定为：`train.csv`（训练集）、`validation.csv`（验证集）、`test.csv`（测试集，如有）。
   - 沙箱工作目录下有一个 `data/` 子目录，所有数据文件都位于其中。
   - 读取数据时必须使用相对路径：`pd.read_csv('data/train.csv')`、`pd.read_csv('data/validation.csv')`、`pd.read_csv('data/test.csv')`。
   - 严禁使用绝对路径（如 `D:\...` 或 `/home/...`），不要直接使用文件名（如 `pd.read_csv('train.csv')` 会找不到文件），也不要使用原始 `.xlsx` 文件名（沙箱内不存在 `.xlsx` 文件）。

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

产物要求
1. 模型文件：output/model.pkl（使用 pickle 保存最终模型）。**必须是完整的 sklearn Pipeline**（包含 ColumnTransformer 等预处理 + 模型），以便直接对原始数据做预测。
2. 测试集预测：output/test_predictions.csv（包含原始测试集 + prediction 列）
3. 特征重要性：output/feature_importance.csv（name, importance 两列）。**重要：特征重要性必须排除目标列（y），只包含输入特征（X）。**
4. 特征重要性图：output/feature_importance.png（使用 matplotlib，保存为 PNG）。**图中也必须排除目标列。**
5. 任务类型关键可视化图：output/report_fig.png（单图，使用 matplotlib 保存为 PNG）。**根据任务类型选择最关键的 1 个图**：
   - **回归**（含时序回归）：画预测值 vs 真实值散点图，叠加 y=x 对角线。一眼看出模型是否系统性高估/低估。
   - **二分类**：画 PR 曲线（Precision-Recall，极度不平衡时）或 ROC 曲线（标准二分类时），并在图标题中标注 AUC/AP 值。
   - **多分类**：画混淆矩阵热力图（sns.heatmap），annot=True 显示数值。
   - **大数据集（>10万行）时**：**禁止生成此图**，只保留特征重要性图。
6. 可视化报告：output/report.html（包含模型指标摘要、特征重要性表格、测试集预测预览。若有 report_fig.png，在报告中用 `<img src="report_fig.png">` 引用）
6. 配套预测脚本：
   - 如果 `data/predict.py` 已经存在（训练阶段已生成），**直接使用它**，无需重新生成。
   - 如果不存在，才需要生成一个简化的预测脚本到 `output/predict.py`。

**大数据集优化（关键）**：
如果训练集行数超过 10 万行（大数据集），产物代码必须遵守以下限制以防止超时：
- **禁止生成 feature_importance.png**（matplotlib 大图对大数据集极其耗时）
- **禁止生成 report_fig.png**（差异化可视化对大数据集禁用）
- **禁止计算 SHAP 值**（大数据集 SHAP 计算可能耗时数分钟）
- **禁止生成复杂可视化**（如热力图、散点图矩阵等）
- 只需保留核心文件：output/model.pkl + output/feature_importance.csv + output/report.html（简单文本表格）
- 特征重要性 CSV 可以直接从模型属性获取（如 `model.feature_importances_`），无需额外计算
7. 最后通过 print 输出一个 JSON 摘要，格式如下：
print(json.dumps({
    "status": "success",
    "has_test_set": true/false,
    "test_rows": 测试集行数,
    "feature_count": 特征数量
}))

数据路径规则（关键）
- 训练集：data/train.csv
- 验证集：data/validation.csv
- 测试集（如有）：data/test.csv
- 产物输出目录：output/（已存在，直接写入即可）
- 严禁使用绝对路径

**测试集数据处理关键约束**：
- 测试集（data/test.csv）**没有目标列（y）**，只有特征列（X）。
- 如果产物代码中需要从 DataFrame 中删除目标列，必须使用条件判断：`if target_col in df.columns: df = df.drop(columns=[target_col])`。
- 严禁直接写 `df = df.drop(columns=[target_col])`，否则在测试集上会报 KeyError。

环境说明
- 沙箱已预装 pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn
- 产物生成模式下允许使用 os, pathlib, shutil 等文件操作模块
- 允许保存 .pkl, .csv, .png, .html 等文件

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
    """
    
    def generate(
        self,
        task_config: TaskConfig,
        run_state: str = "INIT",
        context_payload: str = "",
        previous_code: str = ""
    ) -> CodeOutput:
        """
        生成建模计划与代码
        
        Args:
            task_config: 任务配置（目标列、任务类型、评估指标、文件角色等）
            run_state: 当前运行状态（INIT / DEBUG / OPTIMIZE）
            context_payload: 上下文载荷（报错信息 / 优化建议 / 用户反馈）
            previous_code: 历史代码（DEBUG/OPTIMIZE 状态时传入）
            
        Returns:
            CodeOutput: 包含 plan（规划文本）和 code（Python 代码）
        """
        user_prompt = self._build_user_prompt(
            task_config, run_state, context_payload, previous_code
        )
        
        logger.info(f"[PlanCodingAgent] 生成代码, state={run_state}")
        
        response = self._call_llm(PLAN_CODING_SYSTEM_PROMPT, user_prompt)
        
        plan, code = self._parse_response(response)
        
        logger.info(f"[PlanCodingAgent] 解析完成, plan长度={len(plan)}, code长度={len(code)}")
        
        return CodeOutput(plan=plan, code=code, raw_response=response)
    
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
1. {'加载 data/best_model.pkl（优先）或重新训练最终模型' if has_saved_model else '重新训练最终模型（train.csv）'}
2. 保存模型到 output/model.pkl
{'3. 【跳过】数据集过大，不生成 feature_importance.png' if is_large_dataset else '3. 生成特征重要性图到 output/feature_importance.png'}
4. 生成特征重要性 CSV 到 output/feature_importance.csv（直接从模型属性获取，禁止 SHAP）
5. 生成 HTML 报告到 output/report.html（{'纯文本表格即可，禁止复杂可视化' if is_large_dataset else '包含模型指标摘要、特征重要性表格、测试集预测预览'}）
{'6. 对测试集预测并保存到 output/test_predictions.csv' if has_test_set else ''}
7. 最后 print(json.dumps({...})) 输出摘要
"""
        
        logger.info(f"[PlanCodingAgent] 生成产物代码, has_test_set={has_test_set}")
        
        response = self._call_llm(ARTIFACT_SYSTEM_PROMPT, user_prompt)
        
        plan, code = self._parse_response(response)
        
        logger.info(f"[PlanCodingAgent] 产物代码解析完成, code长度={len(code)}")
        
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
        
        logger.info(f"[PlanCodingAgent] 生成预测脚本 predict.py")
        
        try:
            response = self._call_llm(PREDICT_SCRIPT_SYSTEM_PROMPT, user_prompt)
            plan, code = self._parse_response(response)
            
            if not code:
                logger.warning(f"[PlanCodingAgent] 预测脚本生成失败，未解析到代码")
                return None
            
            logger.info(f"[PlanCodingAgent] 预测脚本解析完成, code长度={len(code)}")
            return CodeOutput(plan=plan, code=code, raw_response=response)
        except Exception as e:
            logger.warning(f"[PlanCodingAgent] 预测脚本生成异常: {e}")
            return None
    
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
                    complexity=task_config.extracted_slots.complexity or "simple"
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
        
        预期格式：
        <plan>
        ...规划内容...
        </plan>
        <code>
        ```python
        ...代码内容...
        ```
        </code>
        """
        # 提取 plan
        plan_match = re.search(r'<plan>(.*?)</plan>', response, re.DOTALL)
        plan = plan_match.group(1).strip() if plan_match else ""
        
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
        # 检测: 存在 ```python 但无闭合 ```，则提取到 </code> 为止
        if not code:
            python_block_start = response.find('```python')
            if python_block_start != -1:
                code_start = python_block_start + len('```python')
                # 先尝试找闭合的 ```
                code_end = response.find('```', code_start)
                if code_end == -1:
                    # 无闭合，尝试找 </code>
                    code_end = response.find('</code>', code_start)
                if code_end == -1:
                    # 连 </code> 都没有，取到字符串末尾
                    code_end = len(response)
                code = response[code_start:code_end].strip()
                if code:
                    logger.info(f"[PlanCodingAgent] 检测到未闭合代码块，已提取 {len(code)} 字符")
        
        # 再次清理 code 中可能残留的 markdown 标记
        code = code.strip()
        if code.startswith("python"):
            code = code[6:].strip()
        
        if not plan:
            logger.warning("[PlanCodingAgent] 未解析到 plan 内容")
        if not code:
            logger.warning("[PlanCodingAgent] 未解析到 code 内容")
        
        return plan, code


import json

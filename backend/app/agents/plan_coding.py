"""
Plan & Coding Agent
生成建模计划与可执行 Python Pipeline 代码
"""

import logging
import re
from typing import Optional

from app.agents.base import BaseAgent
from app.models.schemas import CodeOutput, TaskConfig

logger = logging.getLogger(__name__)

# ========== System Prompt ==========
PLAN_CODING_SYSTEM_PROMPT = """你是一名资深且高效的机器学习工程师。当前你运行在"Fast Engine（快速基线引擎）"模式下。
你的核心任务是：为结构化数据快速构建、修复或优化端到端的机器学习 Python 代码。

Core Constraints (绝对红线，必须遵守)
1. 沙箱隔离：代码将在无网络、无外网权限的 Docker 容器中运行。绝对禁止使用 os.system 等危险系统调用，严禁 import os 模块。
2. 产物要求：【重要】当前流程中，绝对不要将模型保存为离线文件（如 .pkl）或到处预测结果文件。你只需要在代码的最后一步，将验证集的评估指标（如 AUC、RMSE、是否严重过拟合等）通过 print() 结构化输出到控制台即可，供系统后续抓取。
3. 算法优选：优先使用 Scikit-Learn, LightGBM, XGBoost 等快速且效果好的树模型。
4. 环境依赖：当前沙箱已预装 pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn 等常用库。请优先使用这些已安装库。如果代码需要 import 未预装的第三方库，请先检查是否可用 sklearn 原生方案替代，避免执行失败。
5. 数据路径（关键）：
   - 所有上传的数据文件在沙箱中会被**统一转换为 CSV 格式**，文件名固定为：`train.csv`（训练集）、`validation.csv`（验证集）、`test.csv`（测试集，如有）。
   - 沙箱工作目录下有一个 `data/` 子目录，所有数据文件都位于其中。
   - 读取数据时必须使用相对路径：`pd.read_csv('data/train.csv')`、`pd.read_csv('data/validation.csv')`、`pd.read_csv('data/test.csv')`。
   - 严禁使用绝对路径（如 `D:\...` 或 `/home/...`），不要直接使用文件名（如 `pd.read_csv('train.csv')` 会找不到文件），也不要使用原始 `.xlsx` 文件名（沙箱内不存在 `.xlsx` 文件）。

Input Context
- 【意图澄清与任务配置 Task Config】: 包含目标列、任务类型、核心评估指标等
- 【数据画像 Data Profile】: 含表头及部分统计特征
- 【当前运行状态 Run State】: INIT / DEBUG / OPTIMIZE
- 【上下文载荷 Context Payload】: 报错日志 / 评估Agent的建议 / 用户人工修改建议
- 【历史代码 Previous Code】: 上一轮代码（如有）

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
代码最后必须通过 print 输出验证集结果指标字典（字段名必须严格匹配）：
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
1. 重新训练最终模型（使用 data/train.csv，逻辑与最佳代码一致）
2. 对测试集进行预测（如果存在 data/test.csv）
3. 保存产物文件到 output/ 目录下

产物要求
1. 模型文件：output/model.pkl（使用 pickle 保存最终模型）
2. 测试集预测：output/test_predictions.csv（包含原始测试集 + prediction 列）
3. 特征重要性：output/feature_importance.csv（name, importance 两列）
4. 特征重要性图：output/feature_importance.png（使用 matplotlib，保存为 PNG）
5. 可视化报告：output/report.html（包含模型指标摘要、特征重要性表格、测试集预测预览）
6. 最后通过 print 输出一个 JSON 摘要，格式如下：
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
        error_message: str = ""
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
        
        user_prompt = f"""【任务配置】:
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}
- 是否有测试集: {'是' if has_test_set else '否'}

【最佳代码参考】（请基于以下代码的逻辑重新编写产物生成代码，确保模型训练和预处理方式一致）:
```python
{best_code}
```
{fix_instruction}

请生成产物代码，要求：
1. 重新训练最终模型（train.csv）
2. 保存模型到 output/model.pkl
3. 生成特征重要性图到 output/feature_importance.png
4. 生成特征重要性 CSV 到 output/feature_importance.csv
5. 生成 HTML 报告到 output/report.html
{'6. 对测试集预测并保存到 output/test_predictions.csv' if has_test_set else ''}
7. 最后 print(json.dumps({...})) 输出摘要
"""
        
        logger.info(f"[PlanCodingAgent] 生成产物代码, has_test_set={has_test_set}")
        
        response = self._call_llm(ARTIFACT_SYSTEM_PROMPT, user_prompt)
        
        plan, code = self._parse_response(response)
        
        logger.info(f"[PlanCodingAgent] 产物代码解析完成, code长度={len(code)}")
        
        return CodeOutput(plan=plan, code=code, raw_response=response)
    
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
        
        prompt = f"""【当前运行状态 Run State】: {run_state}

【意图澄清与任务配置 Task Config】:
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}
- 特征约束（需丢弃的列）: {slots.feature_constraints or []}
- 用户描述: {task_config.user_description or '无'}

【上传文件信息】:
{files_info}

【数据画像 Data Profile】:
{profile_json}

【上下文载荷 Context Payload】:
{context_payload or '无'}

【历史代码 Previous Code】:
{code_section}

请根据以上信息，生成对应的 <plan> 和 <code>。
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
        
        # 提取 code（支持 ```python ... ``` 或纯代码块）
        code_match = re.search(r'<code>.*?(?:```python\s*(.*?)\s*```|`(.*?)`)</code>', response, re.DOTALL)
        if code_match:
            code = code_match.group(1) or code_match.group(2) or ""
        else:
            # 兜底：尝试直接提取 ```python ... ```
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""
        
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

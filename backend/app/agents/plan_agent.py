"""
Plan Agent — 复杂任务专用计划生成 Agent

职责：根据意图识别结果、数据画像、知识库，生成结构化的建模计划。
输出包含：核心挑战、must_do 清单、avoid 清单、pipeline 步骤、模型选择、风险预判。

设计哲学：让 LLM 在写代码之前，先充分理解任务难点并做出明确承诺（must_do）。
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.models.schemas import TaskConfig
from app.knowledge_base.loader import KnowledgeBaseLoader

logger = logging.getLogger(__name__)


class MustDoItem(BaseModel):
    """必须执行的事项"""
    item: str
    reason: str
    critical: bool = True  # 若为 True，则 Coding Agent 必须实现，否则视为失败


class AvoidItem(BaseModel):
    """必须避免的事项"""
    item: str
    reason: str


class PipelineStep(BaseModel):
    """Pipeline 中的一个步骤"""
    step: str
    actions: List[str]


class PlanResult(BaseModel):
    """Plan Agent 的结构化输出"""
    task_analysis: Dict[str, Any] = Field(default_factory=dict)
    core_challenges: List[str] = Field(default_factory=list)
    must_do: List[MustDoItem] = Field(default_factory=list)
    avoid: List[AvoidItem] = Field(default_factory=list)
    pipeline_plan: List[PipelineStep] = Field(default_factory=list)
    model_choice: str = ""
    model_config_rationale: str = ""  # 为什么选这个模型及配置
    expected_performance: str = ""
    risks: List[str] = Field(default_factory=list)
    # 原始文本，用于日志展示
    raw_plan_text: str = ""


PLAN_AGENT_SYSTEM_PROMPT = """你是一名资深机器学习架构师。

【你的调用场景】你只在以下两种情况下被调用：
1. 【INIT 冷启动】：任务首次启动，从零开始做完整的架构规划。
2. 【用户反馈优化】：用户看了初步结果后不满意，上传了具体的修改建议，需要你结合人类意图重新规划。

系统自动优化（AUTO_OPTIMIZE）时不会调用你——那时由 EvaluationAgent 直接做重新规划。因此你只需专注于：
- INIT 时：基于静态数据画像做完整、稳健的初始架构设计
- 用户反馈时：深入理解人类意图，将其转化为可执行的策略调整

你的角色不是写代码，而是做架构决策和策略规划：
1. 深入分析数据特点和任务难点
2. 识别所有可能导致失败的陷阱
3. 制定明确的 must_do（必须执行）和 avoid（必须避免）清单
4. 设计完整的 Pipeline 步骤

【重要边界】PlanAgent 只负责"做什么"和"为什么做"，不干涉"怎么做"。
- must_do 中只能写策略/目标级别的要求（如"所有类别列必须编码为数值"），**严禁**写具体代码实现。
- 具体代码实现由 Coding Agent 根据策略自行决定。

输出格式（严格遵循以下 YAML 结构）：

```yaml
task_analysis:
  task_type: "binary_classification / multiclass_classification / regression"
  primary_metric: "AUC / accuracy / F1 / RMSE / MAE / R2 / ..."
  secondary_metrics: ["accuracy", "precision", "recall", "F1", ...]
  dataset_size: "大致行数"
  class_balance: "如: 极度不平衡 (正例占0.17%) / 基本平衡"
  time_series: true / false
  key_features: "数据中最有价值的特征描述"
  data_quality_issues: ["缺失值多", "高偏度", ...]

core_challenges:
  - "挑战1的详细描述，包括如果不处理会怎样"
  - "挑战2的详细描述..."

must_do:
  - item: "具体必须执行的操作1"
    reason: "为什么必须这么做，不做的后果"
    critical: true
  - item: "具体必须执行的操作2"
    reason: "..."
    critical: true

avoid:
  - item: "具体必须避免的操作1"
    reason: "为什么不能用，用了会怎样"
  - item: "具体必须避免的操作2"
    reason: "..."

pipeline_plan:
  - step: "数据预处理"
    actions:
      - "具体动作1"
      - "具体动作2"
  - step: "特征工程"
    actions:
      - "..."
  - step: "模型训练"
    actions:
      - "..."
  - step: "验证与调优"
    actions:
      - "..."

model_choice: "LightGBM / XGBoost / RandomForest / Ridge / ..."
model_config_rationale: "为什么选择这个模型，以及关键超参的设定理由"
expected_performance: "如: 验证集 AP > 0.3"
risks:
  - "风险1: 如果不...，可能导致..."
  - "风险2: ..."
```

关键原则：
1. 【必须详尽但不过度】每个 must_do 项必须具体到**策略层面**，严禁写具体代码实现。
2. 【必须量化】核心挑战中必须包含数据层面的量化证据（如 "正例仅占0.17%，共492条"）。
3. 【必须关联】每个 must_do 必须直接对应一个 core_challenge，形成 "问题→解决方案" 的闭环。
4. 【严禁空话】禁止写 "进行特征工程"、"选择合适的模型" 等无法落地的描述。
5. 【严禁越界】禁止在 must_do 中指定具体类名、函数名、API 参数细节。
6. 【优先级】must_do 中 critical=true 的项必须不超过 5 个，且按重要性排序。
7. 【用户反馈场景特别要求】如果输入中包含用户建议，必须在 must_do 中明确体现用户意图的采纳方式，未采纳的部分说明理由。
8. 【指标选择】task_analysis 中必须明确 primary_metric（主要指标）和可选的 secondary_metrics（辅助指标）。
   - 如果用户明确指定了评估指标，primary_metric 必须与用户指定一致。
   - 如果用户未指定，根据任务类型和特点自行推断最合理的 primary_metric（如二分类不平衡选 AUC/F1，回归选 RMSE/MAE）。
   - secondary_metrics 用于提供更全面的模型表现视角，但评估和决策应以 primary_metric 为核心。
   - must_do 中应包含确保 primary_metric 被正确计算的步骤（如类别不平衡时避免仅用 accuracy）。
"""


PLAN_AGENT_OPTIMIZE_SYSTEM_PROMPT = """你是一名资深机器学习架构师，当前处于 **OPTIMIZE 模式**。

你的任务：基于【当前最优代码】和【评估结果】，生成一个**最小改动、针对性极强**的优化计划。

## 与 INIT 模式的核心区别

INIT 模式：从零开始设计完整架构，允许大刀阔斧。
OPTIMIZE 模式：**只修复评估中指出的问题**，其他部分一个字都不要动。

## 绝对红线（违反会导致优化失败，浪费 API token）

1. **最小改动原则**：must_do 不超过 5 项，每项必须直接对应评估中的一个低分维度或具体问题。
2. **严禁重写整个函数**：如果评估只说"max_depth 过浅"，计划只能要求"调大 max_depth"，不能要求"重新设计 feature_engineering"。
3. **PREPROCESS_STATE 致命陷阱**：
   - 全局变量 `PREPROCESS_STATE` 初始是**空字典 `{}`**。
   - 如果优化计划涉及 `preprocess()`，必须在 must_do 中明确要求：
     - `mode='train'` 时必须先 `setdefault` 或赋值，再读取
     - 严禁出现 `drop_cols = PREPROCESS_STATE['drop_cols']` 这种先读后写的模式
   - 正确示范：`PREPROCESS_STATE.setdefault('drop_cols', [])`
4. **历史失败避免**：如果历史记录显示某优化方向已尝试过且 score 未提升，必须在 avoid 中明确列出，不得重复。
5. **验证指标缺失 = 最高优先级**：如果评估指出"验证集指标缺失（val_auc/val_rmse is None）"，must_do 的第一项必须是修复模型训练逻辑，而不是调参。

## 输出格式

与 INIT 模式相同的 YAML 结构，但：
- must_do 更精简（3-5 项）
- 每项必须直接关联评估中的具体问题
- avoid 必须包含历史已失败的方向
"""


class PlanAgent(BaseAgent):
    """
    Plan Agent — 为复杂任务生成结构化建模计划
    """

    def generate(
        self,
        task_config: TaskConfig,
        run_state: str = "INIT",
        context_payload: str = "",
        evaluation_history: Optional[List[Dict[str, Any]]] = None,
        best_code: Optional[str] = None,
        best_evaluation: Optional[Any] = None,
        best_metrics: Optional[Any] = None,
        optimization_history: Optional[List[Dict[str, Any]]] = None
    ) -> PlanResult:
        """
        生成结构化建模计划
        
        Args:
            task_config: 任务配置
            run_state: "INIT" 或 "OPTIMIZE"
            context_payload: 优化建议/评估反馈（用户反馈路径传入）
            evaluation_history: 历史评估记录
            best_code: 当前最优代码（OPTIMIZE 时传入）
            best_evaluation: 当前最优代码的评估结果（OPTIMIZE 时传入）
            best_metrics: 当前最优代码的指标（OPTIMIZE 时传入）
            optimization_history: 优化/调试历史记录（包含失败尝试和代码）
        """
        if run_state == "OPTIMIZE":
            user_prompt = self._build_optimize_user_prompt(
                task_config, best_code, best_evaluation, best_metrics, evaluation_history, optimization_history
            )
            system_prompt = PLAN_AGENT_OPTIMIZE_SYSTEM_PROMPT
            logger.info(f"[PlanAgent] 生成优化计划, task_type={task_config.extracted_slots.task_type}")
        else:
            user_prompt = self._build_user_prompt(task_config, context_payload, evaluation_history)
            system_prompt = PLAN_AGENT_SYSTEM_PROMPT
            logger.info(f"[PlanAgent] 生成初始计划, task_type={task_config.extracted_slots.task_type}")

        response = self._call_llm(system_prompt, user_prompt)

        plan_result = self._parse_yaml_response(response)
        plan_result.raw_plan_text = response

        logger.info(
            f"[PlanAgent] 计划生成完成: "
            f"challenges={len(plan_result.core_challenges)}, "
            f"must_do={len(plan_result.must_do)}, "
            f"avoid={len(plan_result.avoid)}, "
            f"steps={len(plan_result.pipeline_plan)}"
        )
        return plan_result

    def _build_user_prompt(
        self,
        task_config: TaskConfig,
        context_payload: str = "",
        evaluation_history: Optional[List[Dict[str, Any]]] = None
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
            sandbox_files_info.append("- data/train.csv （训练集）")
        if has_val or has_train:
            sandbox_files_info.append("- data/validation.csv （验证集）")
        if has_test:
            sandbox_files_info.append("- data/test.csv （测试集）")

        profile_json = json.dumps(task_config.data_profile, ensure_ascii=False, indent=2) if task_config.data_profile else '暂无详细画像'

        # 知识库建议加载
        kb_refs = ""
        if slots.complexity == "complex":
            try:
                from app.agents.intent_recognition import IntentResult
                from app.models.schemas import TaskType
                intent = IntentResult(
                    target_column=slots.target_column or "",
                    task_type=slots.task_type or TaskType.BINARY_CLASSIFICATION,
                    eval_metric=slots.eval_metric,
                    complexity=slots.complexity or "simple",
                    complexity_reason=slots.complexity_reason
                )
                kb_loader = KnowledgeBaseLoader()
                refs = kb_loader.load_references(intent)
                if refs:
                    kb_refs = "\n".join(f"- {r}" for r in refs)
            except Exception as e:
                logger.warning(f"[PlanAgent] 知识库加载失败: {e}")

        kb_section = f"""
【知识库参考 Knowledge Base】（历史最佳实践，仅供参考，你应根据数据画像自主分析后选择性采纳）：
{kb_refs if kb_refs else "（无知识库条目）"}
""" if kb_refs else ""

        # 数据预处理检查清单（供计划参考）
        preprocessing_notes = ""
        if task_config.data_profile and "columns" in task_config.data_profile:
            cols = task_config.data_profile["columns"]
            target_col = slots.target_column or ""

            notes = []
            # 目标列缺失
            target_missing = [c for c in cols if c["name"] == target_col and c.get("missingCount", 0) > 0]
            if target_missing:
                notes.append(f"目标列 '{target_col}' 有 {target_missing[0]['missingCount']} 个缺失值")

            # 类别特征
            cat_cols = [c for c in cols if c.get("type") == "categorical" and c["name"] != target_col]
            if cat_cols:
                notes.append(f"类别特征列: {[c['name'] for c in cat_cols]}")

            # 数值缺失
            num_missing = [c for c in cols if c.get("type") == "numeric" and c.get("missingCount", 0) > 0 and c["name"] != target_col]
            if num_missing:
                notes.append(f"数值特征缺失: {[c['name'] for c in num_missing]}")

            # ID 列
            id_cols = [c for c in cols if c.get("isLikelyId") and c["name"] != target_col]
            if id_cols:
                notes.append(f"疑似ID列（需丢弃）: {[c['name'] for c in id_cols]}")

            # 时序列
            time_cols = [c["name"] for c in cols if c["name"] in ["year", "month", "day", "hour", "minute", "second", "weekday", "dteday"]]
            if time_cols:
                notes.append(f"时间相关列: {time_cols}")

            # 高偏度数值列
            skewed = [c for c in cols if c.get("type") == "numeric" and abs(c.get("skewness", 0)) > 2 and c["name"] != target_col]
            if skewed:
                notes.append(f"高度偏斜列（建议log变换）: {[c['name'] for c in skewed]}")

            # 不平衡
            if task_config.data_profile.get("classBalance"):
                cb = task_config.data_profile["classBalance"]
                if cb.get("isSeverelyImbalanced"):
                    notes.append(f"目标列极度不平衡: 最小类占比 {cb.get('minClassRatio', 'unknown')}, 不平衡比 {cb.get('imbalanceRatio', 'unknown')}")

            if notes:
                preprocessing_notes = "\n".join(f"- {n}" for n in notes)
            else:
                preprocessing_notes = "（数据画像中未发现明显问题）"

        # 【新增】历史评估记录（用户反馈路径传入，避免重复犯错）
        history_section = ""
        if evaluation_history:
            history_lines = ["【历史评估记录 Evaluation History】（严禁重复历史已失败的优化方向）："]
            for i, h in enumerate(evaluation_history, 1):
                hist_method = h.get('method_summary', '')
                hist_suggestions = h.get('suggestions_for_coding_agent', '') or h.get('suggestions', '')
                history_lines.append(f"  第{i}轮自动优化: score={h.get('score', 'N/A')}, decision={h.get('decision', 'N/A')}")
                if hist_method:
                    history_lines.append(f"    方法总结: {hist_method[:120]}")
                if hist_suggestions:
                    history_lines.append(f"    优化建议: {hist_suggestions[:120]}")
                # 维度评分摘要
                dim_scores = h.get('dimension_scores', [])
                if dim_scores:
                    weak_dims = [f"{d['name']}={d['score']}" for d in dim_scores if d.get('score', 100) < 60]
                    if weak_dims:
                        history_lines.append(f"    低分维度: {', '.join(weak_dims)}")
            history_section = "\n".join(history_lines) + "\n"

        prompt = f"""【任务配置":
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}
- 复杂度判定: {slots.complexity or 'unknown'}（原因: {slots.complexity_reason or '未说明'}）
- 是否时序: {slots.is_time_series}
- 用户描述: {task_config.user_description or '无'}
- 用户建模建议: {slots.user_modeling_suggestions or '无'}

【文件信息】:
原始上传文件:
{raw_files_info}

沙箱内可用文件:
{chr(10).join(sandbox_files_info)}

【数据画像】:
{profile_json}

【数据预处理要点】:
{preprocessing_notes}
{kb_section}
{history_section}
请基于以上信息，严格按照 YAML 格式输出完整的建模计划。
特别要求：
1. must_do 中的每一项必须具体到可以在代码中直接实现的程度
2. 知识库中的建议仅供参考，你应根据具体数据特征自主判断哪些建议适用，哪些不适用
3. 你的 must_do/avoid 必须基于数据画像和任务特点分析得出，而不是简单复制知识库内容
4. 每个 must_do 必须说明 "如果不做会怎样" 的后果
5. 如果传入了【评估优化建议】，请仔细分析这些建议，将其中的合理部分纳入 must_do/pipeline_plan，不合理的部分说明理由后舍弃。
6. 【关键】如果历史评估记录显示某种优化方向已经尝试过且效果不佳（score 未提升或 decision 仍为 AUTO_OPTIMIZE），严禁在 must_do 中重复同样的建议。必须在 avoid 中明确列出已失败的方向，并选择全新的优化策略。
"""
        return prompt

    def _build_optimize_user_prompt(
        self,
        task_config: TaskConfig,
        best_code: Optional[str],
        best_evaluation: Optional[Any],
        best_metrics: Optional[Any],
        evaluation_history: Optional[List[Dict[str, Any]]],
        optimization_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """构建 OPTIMIZE 模式的用户提示词——基于当前最优代码做最小改动计划"""
        slots = task_config.extracted_slots

        # 提取 best_code 中的关键代码片段
        code_summary = ""
        if best_code:
            key_lines = []
            for line in best_code.split('\n'):
                stripped = line.strip()
                if any(kw in stripped for kw in [
                    'def preprocess', 'def feature_engineering', 'def build_model',
                    'LGBMClassifier', 'LGBMRegressor', 'XGBClassifier', 'XGBRegressor',
                    'RandomForest', 'LogisticRegression', 'Ridge', 'Lasso',
                    'scale_pos_weight', 'class_weight', 'SMOTE', 'RandomOverSampler',
                    'OneHotEncoder', 'OrdinalEncoder', 'StandardScaler',
                    'ColumnTransformer', 'Pipeline',
                    'early_stopping', 'eval_set',
                    'PREPROCESS_STATE'
                ]):
                    key_lines.append(line)
            code_summary = '\n'.join(key_lines[:40])
            if len(key_lines) > 40:
                code_summary += "\n...（仅展示关键行）"

        # 当前最优指标
        metrics_lines = []
        if best_metrics:
            if getattr(best_metrics, 'val_auc', None) is not None:
                metrics_lines.append(f"  - 验证集 AUC: {best_metrics.val_auc:.4f}")
            if getattr(best_metrics, 'val_accuracy', None) is not None:
                metrics_lines.append(f"  - 验证集 Accuracy: {best_metrics.val_accuracy:.4f}")
            if getattr(best_metrics, 'val_rmse', None) is not None:
                metrics_lines.append(f"  - 验证集 RMSE: {best_metrics.val_rmse:.4f}")
            if getattr(best_metrics, 'val_score', None) is not None:
                metrics_lines.append(f"  - 验证集 Score: {best_metrics.val_score:.4f}")
        metrics_section = '\n'.join(metrics_lines) if metrics_lines else "  - 无可用指标"

        # 当前最优评估
        eval_section = ""
        if best_evaluation:
            eval_lines = [f"  - 综合评分 score: {best_evaluation.score}"]
            if getattr(best_evaluation, 'method_summary', None):
                eval_lines.append(f"  - 方法总结: {best_evaluation.method_summary[:200]}")
            if getattr(best_evaluation, 'suggestions_for_coding_agent', None):
                eval_lines.append(f"  - 上一轮优化建议: {best_evaluation.suggestions_for_coding_agent[:200]}")
            dim_scores = getattr(best_evaluation, 'dimension_scores', None)
            if dim_scores:
                weak_dims = [f"{d.name}={d.score}" for d in dim_scores if d.score < 65]
                if weak_dims:
                    eval_lines.append(f"  - 低分维度: {', '.join(weak_dims)}")
            eval_section = '\n'.join(eval_lines)
        else:
            eval_section = "  - 无可用评估记录"

        # 历史评估记录（避免重复）
        history_section = ""
        if evaluation_history:
            history_lines = ["【历史评估记录】（严禁重复历史已失败的优化方向）："]
            for i, h in enumerate(evaluation_history, 1):
                hist_score = h.get('score', 'N/A')
                hist_decision = h.get('decision', 'N/A')
                hist_method = h.get('method_summary', '')
                hist_suggestions = h.get('suggestions_for_coding_agent', '') or h.get('suggestions', '')
                history_lines.append(f"  第{i}轮: score={hist_score}, decision={hist_decision}")
                if hist_method:
                    history_lines.append(f"    方法: {hist_method[:100]}")
                if hist_suggestions:
                    history_lines.append(f"    建议: {hist_suggestions[:100]}")
            history_section = '\n'.join(history_lines) + '\n'
        
        # 【框架对齐】优化/调试历史记录（包含失败尝试和代码变更）
        opt_history_section = ""
        if optimization_history:
            opt_lines = ["【优化/调试历史记录】（包含代码执行失败和成功但未超越的尝试）："]
            
            # 展示最近1次失败的完整记录
            recent_failures = [h for h in reversed(optimization_history) if not h.get('success', True)]
            if recent_failures:
                fail = recent_failures[0]
                opt_lines.append("  【最近失败尝试】")
                opt_lines.append(f"    类型: {fail.get('run_type', 'unknown')}, 轮次: {fail.get('round', 'N/A')}")
                if fail.get('error_message'):
                    opt_lines.append(f"    失败原因: {fail['error_message'][:300]}")
                if fail.get('code'):
                    # 提取失败代码的关键片段（前20行）
                    fail_code_lines = fail['code'].split('\n')[:20]
                    opt_lines.append(f"    失败代码片段:")
                    opt_lines.append("    ```python")
                    for line in fail_code_lines:
                        opt_lines.append(f"    {line}")
                    opt_lines.append("    ```")
            
            # 展示所有非最优尝试的摘要（避免重复方向）
            non_best = [h for h in optimization_history if not h.get('is_best', False) and h.get('success', False)]
            if non_best:
                opt_lines.append("  【已尝试但未提升的方向】")
                for h in non_best[-3:]:  # 最近3次
                    opt_lines.append(f"    - {h.get('run_type', 'unknown')} 轮{h.get('round', 'N/A')}: score={h.get('score', 'N/A')}")
                    if h.get('plan'):
                        opt_lines.append(f"      方法: {h['plan'][:100]}")
            
            opt_history_section = '\n'.join(opt_lines) + '\n'

        prompt = f"""【任务配置】
- 任务类型: {slots.task_type or 'unknown'}
- 目标列: {slots.target_column or 'unknown'}
- 评估指标: {slots.eval_metric or 'unknown'}
- 用户描述: {task_config.user_description or '无'}

【当前最优代码关键片段】（只修改计划中明确指出的部分，其余保持不动）：
```python
{code_summary or '（无代码片段）'}
```

【当前最优验证指标】
{metrics_section}

【当前最优评估结果】
{eval_section}

{history_section}
{opt_history_section}
【优化要求】
1. must_do 不超过 5 项，每项必须直接关联评估中的具体问题
2. 严禁重写整个函数——只改计划中明确要求改进的部分
3. avoid 必须包含历史已失败的方向
4. PREPROCESS_STATE 初始是空字典 {{}}，mode='train' 时必须先写入后读取

请基于以上信息，严格按照 YAML 格式输出优化计划。
"""
        return prompt

    def _parse_yaml_response(self, response: str) -> PlanResult:
        """
        解析 LLM 的 YAML 格式响应为 PlanResult
        使用容错策略，尽量提取有效信息
        """
        result = PlanResult(raw_plan_text=response)

        # 尝试提取 YAML 块
        yaml_content = response
        yaml_block_match = re.search(r'```yaml\s*(.*?)\s*```', response, re.DOTALL)
        if yaml_block_match:
            yaml_content = yaml_block_match.group(1).strip()
        else:
            # 尝试找 ``` 不加 yaml 的情况
            code_block_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if code_block_match:
                yaml_content = code_block_match.group(1).strip()

        # 用简单的文本解析提取各个部分（不依赖 yaml 库，减少依赖）
        try:
            import yaml
            data = yaml.safe_load(yaml_content)
            if data and isinstance(data, dict):
                result.task_analysis = data.get("task_analysis", {})
                result.core_challenges = data.get("core_challenges", [])
                result.pipeline_plan = [
                    PipelineStep(step=s.get("step", ""), actions=s.get("actions", []))
                    for s in data.get("pipeline_plan", [])
                ]
                result.model_choice = data.get("model_choice", "")
                result.model_config_rationale = data.get("model_config_rationale", "")
                result.expected_performance = data.get("expected_performance", "")
                result.risks = data.get("risks", [])

                # must_do
                for item in data.get("must_do", []):
                    if isinstance(item, dict):
                        result.must_do.append(MustDoItem(
                            item=str(item.get("item", "")),
                            reason=str(item.get("reason", "")),
                            critical=bool(item.get("critical", True))
                        ))

                # avoid
                for item in data.get("avoid", []):
                    if isinstance(item, dict):
                        result.avoid.append(AvoidItem(
                            item=str(item.get("item", "")),
                            reason=str(item.get("reason", ""))
                        ))
        except ImportError:
            logger.warning("[PlanAgent] yaml 库未安装，使用文本解析 fallback")
            self._parse_text_fallback(response, result)
        except Exception as e:
            logger.warning(f"[PlanAgent] YAML 解析失败: {e}，使用文本解析 fallback")
            self._parse_text_fallback(response, result)

        return result

    def _parse_text_fallback(self, response: str, result: PlanResult):
        """文本级 fallback 解析"""
        # core_challenges
        challenges = re.findall(r'^\s*-\s+"?(.+?)"?\s*$', response, re.MULTILINE)
        if not challenges:
            challenges = re.findall(r'core_challenges:\s*\n((?:\s*-\s*.+\n)+)', response)
        result.core_challenges = [c.strip() for c in challenges if len(c.strip()) > 5][:10]

        # must_do
        must_do_pattern = re.findall(
            r'item:\s*"?(.+?)"?\s*\n\s*reason:\s*"?(.+?)"?\s*\n\s*critical:\s*(true|false)',
            response, re.IGNORECASE | re.DOTALL
        )
        for item, reason, critical in must_do_pattern:
            result.must_do.append(MustDoItem(
                item=item.strip().replace('\n', ' '),
                reason=reason.strip().replace('\n', ' '),
                critical=critical.lower() == 'true'
            ))

        # avoid
        avoid_pattern = re.findall(
            r'item:\s*"?(.+?)"?\s*\n\s*reason:\s*"?(.+?)"?',
            response, re.IGNORECASE | re.DOTALL
        )
        for item, reason in avoid_pattern:
            result.avoid.append(AvoidItem(
                item=item.strip().replace('\n', ' '),
                reason=reason.strip().replace('\n', ' ')
            ))

        # model_choice
        mc = re.search(r'model_choice:\s*"?(.+?)"?\s*$', response, re.MULTILINE)
        if mc:
            result.model_choice = mc.group(1).strip()

        # expected_performance
        ep = re.search(r'expected_performance:\s*"?(.+?)"?\s*$', response, re.MULTILINE)
        if ep:
            result.expected_performance = ep.group(1).strip()

        logger.info(f"[PlanAgent] 文本 fallback 解析完成: challenges={len(result.core_challenges)}, must_do={len(result.must_do)}")

    def format_plan_for_coding(self, plan: PlanResult) -> str:
        """将 PlanResult 格式化为给 Coding Agent 的文本"""
        lines = ["=" * 60, "【结构化建模计划】", "=" * 60]

        lines.append("\n【任务分析】")
        for k, v in plan.task_analysis.items():
            lines.append(f"  {k}: {v}")

        lines.append("\n【核心挑战】")
        for i, c in enumerate(plan.core_challenges, 1):
            lines.append(f"  {i}. {c}")

        lines.append("\n【必须执行 (MUST DO) —— 后续代码必须逐项实现】")
        for i, m in enumerate(plan.must_do, 1):
            marker = "【关键】" if m.critical else ""
            lines.append(f"  {i}. {marker}{m.item}")
            lines.append(f"     原因: {m.reason}")

        lines.append("\n【必须避免 (AVOID) —— 代码中严禁出现】")
        for i, a in enumerate(plan.avoid, 1):
            lines.append(f"  {i}. {a.item}")
            lines.append(f"     原因: {a.reason}")

        lines.append("\n【Pipeline 计划】")
        for s in plan.pipeline_plan:
            lines.append(f"  ▶ {s.step}")
            for a in s.actions:
                lines.append(f"    - {a}")

        lines.append(f"\n【模型选择】{plan.model_choice}")
        if plan.model_config_rationale:
            lines.append(f"【选择理由】{plan.model_config_rationale}")

        if plan.expected_performance:
            lines.append(f"\n【预期表现】{plan.expected_performance}")

        if plan.risks:
            lines.append("\n【风险预判】")
            for r in plan.risks:
                lines.append(f"  ⚠ {r}")

        lines.append("=" * 60)
        return "\n".join(lines)

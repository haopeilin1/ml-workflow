
"""
Evaluation Agent
评估模型效果，决策 AUTO_OPTIMIZE 或 YIELD_TO_USER
输出多维度评分及加权总分

【架构变更 v2】
- evaluate() 在一次 LLM 调用中同时完成：评估 + 重新规划 + 方法总结
- 真正减少调用次数（旧：evaluate + plan_agent + coding = 3次 → 新：evaluate(含replan) + coding = 2次）
- 传入 evaluation_history 避免重复犯错
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any

from app.agents.base import BaseAgent
from app.models.schemas import EvaluationResult, DecisionType, ExecutionMetrics, DimensionScore

logger = logging.getLogger(__name__)

# ========== System Prompt ==========
EVALUATION_SYSTEM_PROMPT = """你是一名资深机器学习评估专家与架构师。
当前正在"Fast Engine（快速基线引擎）"模式下。模型代码已成功在沙箱中运行并没有报错。

你的双重任务：
1. 评估当前模型效果（多维度评分 + 决策）
2. 如果决定 AUTO_OPTIMIZE，同时输出下一轮的结构化重新规划计划和方法总结

【重要】你将在一次输出中完成所有任务，不要分两次调用。

Input Context
- 【任务目标 Task Target】: 例如：预测是否流失，看重 AUC 指标
- 【本次运行沙箱输出 Metrics】: 包含验证集得分、训练集得分、耗时等
- 【内部优化轮数 Optimize Round】: 当前轮数 / 3 (注意：系统限制最多进行 3 次内部自动优化)
- 【历史评估记录 Evaluation History】: 之前所有轮次的评估结果和优化尝试（避免重复犯同样的错误）

Evaluation Rules
请仔细评估当前指标，并从以下两种决策中选择一种：
1. 【DECISION: AUTO_OPTIMIZE】(判断需要调优，打回重构)
  - 触发条件：当前分数极差，或存在极为严重的过拟合/欠拟合问题，并且 内部优化轮数（Optimize Round）< 3。
  - 要求：此时坚决不能把半成品丢给用户。你必须根据当前方案的具体问题，提出有针对性的优化建议。
    【建议类型举例（不要局限于这些，要因问题而异）】
    - 调参类："当前 max_depth=3 过浅导致欠拟合，尝试 6~10"
    - 特征类："某数值特征范围差异过大，应加入标准化"
    - 模型类："当前线性模型无法捕捉非线性关系，建议换用 XGBoost/LightGBM"
    - 数据类："类别特征未编码导致模型报错，应使用 OrdinalEncoder"
    重要：不要每次都说同样的话。仔细分析当前指标和维度评分，找出最致命的短板，优先解决那个问题。
    【关键】如果历史记录显示之前已经尝试过某种优化但效果不佳，不要重复同样的建议。换一个新方向。
2. 【DECISION: YIELD_TO_USER】(判断不需要调优或次数已满，交棒用户)
  - 触发条件：模型基线已达到及格水平，或者 内部自动优化次数已达到 3 次上限。
  - 要求：生成一段面向非专业用户的汇报总结。

Scoring Criteria (多维度评分，每项 0-100)
【重要】每个维度的 reason 字段请控制在 20 个汉字以内。

1. 【指标表现 metric_performance】权重 30%
2. 【过拟合控制 overfit_control】权重 25%
3. 【算法选择 algorithm_choice】权重 20%
4. 【Pipeline 完整性 pipeline_completeness】权重 15%
5. 【任务匹配度 task_alignment】权重 10%

Output Format (Strict JSON)
你必须严格输出如下 JSON 格式（不要包含 markdown 代码块标记）。
【关键】replan_output 是核心字段，必须放在 JSON 前面，确保不被截断：
{
  "decision": "AUTO_OPTIMIZE",
  "score": 79.5,
  "method_summary": "本轮使用了XGBoost(max_depth=3)，存在欠拟合。关键代码特征：使用了ColumnTransformer+OneHotEncoder。主要问题：树深度过浅。",
  "replan_output": "【重新规划】基于评估结果，下一轮优化方向：must_do: 1.增加树深度到6-8 2.添加类别权重处理不平衡 avoid: 1.重复之前的浅树方案",
  "suggestions_for_coding_agent": "具体技术建议...",
  "report_to_user": null,
  "evaluation_analysis": "对本次运行结果的客观专业分析",
  "dimension_scores": [
    {"name": "metric_performance", "score": 78, "weight": 0.30, "reason": "AUC 0.82 超及格线"},
    {"name": "overfit_control", "score": 85, "weight": 0.25, "reason": "差距仅3%"},
    {"name": "algorithm_choice", "score": 80, "weight": 0.20, "reason": "XGBoost适合该任务"},
    {"name": "pipeline_completeness", "score": 75, "weight": 0.15, "reason": "流程完整"},
    {"name": "task_alignment", "score": 70, "weight": 0.10, "reason": "匹配目标"}
  ]
}

【字段要求】
1. decision: 先输出决策，AUTO_OPTIMIZE 或 YIELD_TO_USER
2. score: 加权总分
3. method_summary: 用1-2句话总结模型、关键问题
4. replan_output（仅 AUTO_OPTIMIZE 时）: 
   - 必须简洁，控制在300字以内
   - 格式: 【重新规划】must_do: ... avoid: ...
   - 基于上一轮实际结果，量化问题
   - 历史失败策略必须在 avoid 中列出
5. suggestions_for_coding_agent: 给 CodingAgent 的技术建议
6. report_to_user（仅 YIELD_TO_USER 时）: 面向用户的汇报
7. evaluation_analysis: 专业分析
8. dimension_scores: 5个维度评分，每个 reason 控制在20字以内"""


class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent

    职责：
    - 解析沙箱输出的验证集指标
    - 评估模型质量（多维度打分）
    - 决策：AUTO_OPTIMIZE（继续内部优化）或 YIELD_TO_USER（提交用户确认）
    - 【新增】AUTO_OPTIMIZE 时同时输出重新规划计划和方法总结（一次调用完成）
    """

    # 维度权重定义（与 Prompt 中的权重一致）
    DIMENSION_WEIGHTS = {
        "metric_performance": 0.30,
        "overfit_control": 0.25,
        "algorithm_choice": 0.20,
        "pipeline_completeness": 0.15,
        "task_alignment": 0.10,
    }

    def evaluate(
        self,
        task_target: str,
        metrics: ExecutionMetrics,
        optimize_round: int,
        max_optimize_rounds: int = 3,
        execution_output: str = "",
        user_modeling_suggestions: Optional[str] = None,
        eval_metric: Optional[str] = None,
        evaluation_history: Optional[List[Dict[str, Any]]] = None,
        current_code: str = "",
    ) -> EvaluationResult:
        """
        评估模型效果并决策

        【架构变更】一次 LLM 调用同时完成：
        1. 多维度评估 + 决策
        2. 方法总结 (method_summary)
        3. 重新规划 (replan_output，仅 AUTO_OPTIMIZE 时)

        Args:
            evaluation_history: 历史评估记录列表，每项包含 {round, score, decision, suggestions, method_summary}
            current_code: 本轮实际运行的代码（用于方法总结）
        """
        user_prompt = self._build_user_prompt(
            task_target, metrics, optimize_round, max_optimize_rounds,
            execution_output, user_modeling_suggestions, eval_metric,
            evaluation_history, current_code
        )

        logger.info(f"[EvaluationAgent] 评估模型(含replan), optimize_round={optimize_round}/{max_optimize_rounds}")

        response = self._call_llm(EVALUATION_SYSTEM_PROMPT, user_prompt)

        result = self._parse_response(response)
        result.raw_response = response

        # 校验/修正加权总分
        result = self._normalize_score(result)

        # 【关键兜底】AUTO_OPTIMIZE 时必须提供 replan_output，否则从 suggestions 构建
        if result.decision == DecisionType.AUTO_OPTIMIZE and not result.replan_output:
            if result.suggestions_for_coding_agent:
                result.replan_output = self._build_replan_from_suggestions(result.suggestions_for_coding_agent)
                logger.info(f"[EvaluationAgent] 从 suggestions 自动构建 replan_output, 长度={len(result.replan_output)}")
            else:
                result.replan_output = self._build_replan_from_suggestions("")
                logger.warning("[EvaluationAgent] AUTO_OPTIMIZE 但无 suggestions，使用空 replan")

        logger.info(f"[EvaluationAgent] 决策: {result.decision}, score={result.score}")
        if result.dimension_scores:
            for ds in result.dimension_scores:
                logger.info(f"  [{ds.name}] {ds.score}/100 (weight={ds.weight}) {ds.reason[:60]}...")
        if result.method_summary:
            logger.info(f"[EvaluationAgent] 方法总结: {result.method_summary[:120]}...")
        if result.replan_output:
            logger.info(f"[EvaluationAgent] 重新规划长度: {len(result.replan_output)}")

        return result

    def _build_user_prompt(
        self,
        task_target: str,
        metrics: ExecutionMetrics,
        optimize_round: int,
        max_optimize_rounds: int,
        execution_output: str,
        user_modeling_suggestions: Optional[str] = None,
        eval_metric: Optional[str] = None,
        evaluation_history: Optional[List[Dict[str, Any]]] = None,
        current_code: str = "",
    ) -> str:
        """构建用户提示词"""
        metrics_dict = {}
        if metrics:
            metrics_dict = {
                k: v for k, v in metrics.model_dump().items()
                if v is not None
            }

        suggestions_section = ""
        if user_modeling_suggestions:
            suggestions_section = f"""
【用户建模建议 User Modeling Suggestions】:
{user_modeling_suggestions}

请特别考虑用户的建模偏好在本次运行中的体现情况。如果建议未被采纳且影响了效果，请在优化建议中提出如何结合用户偏好进行改进。
"""

        primary_metric = eval_metric or ""
        metric_section = f"""
【核心评估指标 Core Metric】: {primary_metric or '未明确指定，请根据任务类型和指标情况自行判断'}
【重要】评估时请优先以该指标为核心标准。
""" if primary_metric else ""

        # 【新增】历史评估记录（避免重复犯错）
        history_section = ""
        if evaluation_history:
            history_lines = ["【历史评估记录 Evaluation History】（严禁重复历史已失败的优化方向）："]
            for i, h in enumerate(evaluation_history, 1):
                hist_suggestions = h.get('suggestions_for_coding_agent', '') or h.get('suggestions', '')
                hist_method = h.get('method_summary', '')
                history_lines.append(f"  第{i}轮: score={h.get('score', 'N/A')}, decision={h.get('decision', 'N/A')}")
                if hist_method:
                    history_lines.append(f"    方法: {hist_method[:100]}")
                if hist_suggestions:
                    history_lines.append(f"    建议: {hist_suggestions[:100]}")
            history_section = "\n".join(history_lines) + "\n"

        # 代码摘要（用于方法总结）【精简至500字符以内】
        code_summary = ""
        if current_code:
            code_summary = f"""
【本轮代码摘要】（用于生成 method_summary，已精简至前500字符）：
```python
{current_code[:500]}
```
{"...（代码截断，仅展示前500字符）" if len(current_code) > 500 else ""}
"""

        prompt = f"""【任务目标 Task Target】: {task_target}{metric_section}

【本次运行沙箱输出 Metrics】:
```json
{json.dumps(metrics_dict, ensure_ascii=False, indent=2)}
```

【沙箱完整输出】:
```
{execution_output[:1500] if execution_output else '无'}
```
{suggestions_section}
{history_section}
{code_summary}
【内部优化轮数 Optimize Round】: {optimize_round} / {max_optimize_rounds}
{"【注意】已达到最大优化轮数上限，必须强制 YIELD_TO_USER。" if optimize_round >= max_optimize_rounds else ""}

请根据以上信息，严格按照 JSON 格式输出评估结论、方法总结和重新规划（如需要）。
必须包含：dimension_scores 数组、score、decision、method_summary。
如果 decision=AUTO_OPTIMIZE，必须同时提供详细的 replan_output。
"""
        return prompt

    def _parse_response(self, response: str) -> EvaluationResult:
        """
        解析 LLM 响应，提取 JSON 格式的评估结果

        容错策略：
        1. 先尝试标准 JSON 解析
        2. 如果失败，尝试修复截断的 JSON
        3. 如果还是失败，从文本中尽量提取关键信息
        4. 最后兜底
        """
        # 策略1：提取 markdown JSON 代码块
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
        else:
            # 策略2：直接匹配 JSON 对象
            json_match = re.search(r'\{[\s\S]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
            else:
                json_str = ""

        if json_str:
            # 尝试标准解析
            try:
                data = json.loads(json_str)
                return self._build_result_from_dict(data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[EvaluationAgent] JSON 标准解析失败: {e}")

            # 尝试修复截断的 JSON
            try:
                fixed_json = self._fix_truncated_json(json_str)
                data = json.loads(fixed_json)
                logger.info(f"[EvaluationAgent] JSON 修复成功")
                return self._build_result_from_dict(data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[EvaluationAgent] JSON 修复失败: {e}")

            # 尝试从截断的 JSON 中提取关键信息
            try:
                extracted = self._extract_from_broken_json(json_str)
                if extracted:
                    logger.info(f"[EvaluationAgent] 从截断 JSON 中提取关键信息成功")
                    return extracted
            except Exception as e:
                logger.warning(f"[EvaluationAgent] 提取失败: {e}")

        # 兜底
        logger.warning(f"[EvaluationAgent] 所有解析方法失败，使用兜底解析")
        return self._fallback_parse(response)

    def _build_result_from_dict(self, data: dict) -> EvaluationResult:
        """从字典构建 EvaluationResult"""
        dim_scores = []
        for d in data.get("dimension_scores", []):
            if isinstance(d, dict):
                dim_scores.append(DimensionScore(
                    name=str(d.get("name", "unknown")),
                    score=float(d.get("score", 0)),
                    weight=float(d.get("weight", 0)),
                    reason=str(d.get("reason", ""))
                ))

        return EvaluationResult(
            evaluation_analysis=data.get("evaluation_analysis", "未提供分析"),
            decision=DecisionType(data.get("decision", "YIELD_TO_USER")),
            suggestions_for_coding_agent=data.get("suggestions_for_coding_agent"),
            report_to_user=data.get("report_to_user"),
            score=data.get("score"),
            dimension_scores=dim_scores,
            method_summary=data.get("method_summary"),
            replan_output=data.get("replan_output")
        )

    def _fix_truncated_json(self, json_str: str) -> str:
        """修复截断的 JSON"""
        fixed = json_str.strip()

        quote_count = fixed.count('"')
        if quote_count % 2 != 0:
            last_quote = fixed.rfind('"')
            if last_quote > 0:
                after_quote = fixed[last_quote+1:].strip()
                if not after_quote.startswith((',', '}', ']')):
                    fixed = fixed[:last_quote+1] + '"' + fixed[last_quote+1:]

        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        fixed += '}' * max(0, open_braces)
        fixed += ']' * max(0, open_brackets)
        fixed = fixed.rstrip(',')

        return fixed

    def _extract_from_broken_json(self, json_str: str) -> Optional[EvaluationResult]:
        """从损坏的 JSON 中尽量提取关键字段
        
        增强版：支持多行字符串提取（method_summary / replan_output / suggestions）
        """
        decision_match = re.search(r'"decision"\s*:\s*"(AUTO_OPTIMIZE|YIELD_TO_USER)"', json_str)
        decision = DecisionType(decision_match.group(1)) if decision_match else DecisionType.YIELD_TO_USER

        score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', json_str)
        score = float(score_match.group(1)) if score_match else None

        # 使用增强的多行字符串提取方法
        analysis = self._extract_json_string_field(json_str, "evaluation_analysis") or "JSON截断，部分信息提取"
        suggestions = self._extract_json_string_field(json_str, "suggestions_for_coding_agent")
        report = self._extract_json_string_field(json_str, "report_to_user")
        method_summary = self._extract_json_string_field(json_str, "method_summary")
        replan_output = self._extract_json_string_field(json_str, "replan_output")

        dim_scores = []
        dim_pattern = r'\{\s*"name"\s*:\s*"([^"]*)"\s*,\s*"score"\s*:\s*(\d+\.?\d*)\s*,\s*"weight"\s*:\s*(\d+\.?\d*)\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}'
        for m in re.finditer(dim_pattern, json_str):
            dim_scores.append(DimensionScore(
                name=m.group(1),
                score=float(m.group(2)),
                weight=float(m.group(3)),
                reason=m.group(4)
            ))

        if score is not None or dim_scores:
            return EvaluationResult(
                evaluation_analysis=analysis,
                decision=decision,
                suggestions_for_coding_agent=suggestions,
                report_to_user=report,
                score=score,
                dimension_scores=dim_scores,
                method_summary=method_summary,
                replan_output=replan_output
            )
        return None

    def _extract_json_string_field(self, json_str: str, field_name: str) -> Optional[str]:
        """从可能截断的 JSON 字符串中提取指定字段的字符串值
        
        支持三种情况：
        1. 普通单行字符串: "field": "value"
        2. 包含转义引号的字符串: "field": "val\\"ue"
        3. 截断的字符串（到末尾）: "field": "value...（没有闭合引号）
        """
        # 模式1: 尝试匹配完整字符串（处理转义引号）
        # 使用非贪婪匹配，但跳过 \\"
        pattern_full = rf'"{re.escape(field_name)}"\s*:\s*"((?:[^"\\]|\\.)*?)"'
        match = re.search(pattern_full, json_str)
        if match:
            return match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        
        # 模式2: 匹配截断字符串（到文件末尾或下一个字段开头）
        # 查找字段名后的内容，直到下一个 JSON 键或数组/对象结束
        pattern_trunc = rf'"{re.escape(field_name)}"\s*:\s*"(.*)'
        match = re.search(pattern_trunc, json_str, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            # 截断到合理的结束位置（下一个键、闭合括号等）
            end_markers = ['",\n', '",', '"\n', '\n}"', '\n  }', '\n]', '},\n']
            for marker in end_markers:
                idx = raw.find(marker)
                if idx > 10:  # 至少保留10个字符
                    raw = raw[:idx]
                    break
            return raw.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        
        return None

    def _normalize_score(self, result: EvaluationResult) -> EvaluationResult:
        """校验并修正加权总分"""
        if not result.dimension_scores:
            default_score = result.score or 50.0
            result.dimension_scores = [
                DimensionScore(name="metric_performance", score=default_score, weight=0.30, reason="综合评分兜底"),
                DimensionScore(name="overfit_control", score=default_score, weight=0.25, reason="综合评分兜底"),
                DimensionScore(name="algorithm_choice", score=default_score, weight=0.20, reason="综合评分兜底"),
                DimensionScore(name="pipeline_completeness", score=default_score, weight=0.15, reason="综合评分兜底"),
                DimensionScore(name="task_alignment", score=default_score, weight=0.10, reason="综合评分兜底"),
            ]
            if result.score is None:
                result.score = default_score
            return result

        calculated = 0.0
        for ds in result.dimension_scores:
            w = self.DIMENSION_WEIGHTS.get(ds.name, ds.weight)
            calculated += ds.score * w

        if result.score is None or abs(result.score - calculated) > 5:
            logger.warning(f"[EvaluationAgent] 修正加权总分: {result.score} -> {calculated:.1f}")
            result.score = round(calculated, 1)
        else:
            result.score = round(result.score, 1)

        for ds in result.dimension_scores:
            ds.weight = self.DIMENSION_WEIGHTS.get(ds.name, ds.weight)

        return result

    def _build_replan_from_suggestions(self, suggestions: str) -> str:
        """从 suggestions 构建一个基本的 replan_output（避免回退到 PlanAgent）"""
        if not suggestions:
            return "【重新规划】\n基于评估建议进行优化:\n must_do:\n  - 改进模型方案\n avoid:\n  - 重复之前的错误\n"
        return f"""【重新规划】
基于评估发现的严重问题，下一轮优化方向如下:

【任务分析】
  上一轮问题: {suggestions[:200]}

【must_do】
  - {suggestions[:300]}

【avoid】
  - 重复之前已失败的优化方向
"""

    def _fallback_parse(self, response: str) -> EvaluationResult:
        """兜底解析"""
        response_upper = response.upper()

        if "AUTO_OPTIMIZE" in response_upper:
            decision = DecisionType.AUTO_OPTIMIZE
        elif "YIELD_TO_USER" in response_upper:
            decision = DecisionType.YIELD_TO_USER
        else:
            decision = DecisionType.YIELD_TO_USER

        suggestions = None
        report = None
        replan_output = None

        if decision == DecisionType.AUTO_OPTIMIZE:
            sug_match = re.search(r'(?:建议|suggestions|优化).*?(:|：)\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
            if sug_match:
                suggestions = sug_match.group(2).strip()
            else:
                suggestions = "请尝试更换模型或调整超参数。"
            # 尝试提取 replan_output
            replan_match = re.search(r'【结构化建模计划.*?(?:=){20,}(.*?)(?:=){20,}', response, re.DOTALL)
            if replan_match:
                replan_output = replan_match.group(0)
            else:
                # 兜底：从 suggestions 构建 replan
                replan_output = self._build_replan_from_suggestions(suggestions)
        else:
            rep_match = re.search(r'(?:汇报|report|总结).*?(:|：)\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
            if rep_match:
                report = rep_match.group(2).strip()
            else:
                report = "模型基线已生成，请查看右侧结果面板。"

        logger.warning(f"[EvaluationAgent] 使用兜底解析, decision={decision.value}")

        fallback_score = 75.0 if decision == DecisionType.YIELD_TO_USER else 45.0
        dim_scores = [
            DimensionScore(name="metric_performance", score=fallback_score, weight=0.30, reason="兜底解析"),
            DimensionScore(name="overfit_control", score=fallback_score, weight=0.25, reason="兜底解析"),
            DimensionScore(name="algorithm_choice", score=fallback_score, weight=0.20, reason="兜底解析"),
            DimensionScore(name="pipeline_completeness", score=fallback_score, weight=0.15, reason="兜底解析"),
            DimensionScore(name="task_alignment", score=fallback_score, weight=0.10, reason="兜底解析"),
        ]

        return EvaluationResult(
            evaluation_analysis="LLM 输出格式异常，使用兜底解析。",
            decision=decision,
            suggestions_for_coding_agent=suggestions if decision == DecisionType.AUTO_OPTIMIZE else None,
            report_to_user=report if decision == DecisionType.YIELD_TO_USER else None,
            score=fallback_score,
            dimension_scores=dim_scores,
            method_summary="兜底解析，无法提取方法总结。",
            replan_output=replan_output
        )

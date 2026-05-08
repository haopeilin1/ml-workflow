"""
Evaluation Agent
评估模型效果，决策 AUTO_OPTIMIZE 或 YIELD_TO_USER
输出多维度评分及加权总分
"""

import json
import logging
import re
from typing import Optional

from app.agents.base import BaseAgent
from app.models.schemas import EvaluationResult, DecisionType, ExecutionMetrics, DimensionScore

logger = logging.getLogger(__name__)

# ========== System Prompt ==========
EVALUATION_SYSTEM_PROMPT = """你是一名资深机器学习评估专家与产品交付官。
当前正在"Fast Engine（快速基线引擎）"模式下。模型代码已成功在沙箱中运行并没有报错，你的任务是评估运行产出的验证集指标，并决定是"在系统内部自动打回调优"，还是"向用户汇报结果交由用户定夺"。

Input Context
- 【任务目标 Task Target】: 例如：预测是否流失，看重 AUC 指标
- 【本次运行沙箱输出 Metrics】: 包含验证集得分、训练集得分、耗时等
- 【内部优化轮数 Optimize Round】: 当前轮数 / 3 (注意：系统限制最多进行 3 次内部自动优化)

Evaluation Rules
请仔细评估当前指标，并从以下两种决策中选择一种：
1. 【DECISION: AUTO_OPTIMIZE】(判断需要调优，打回重构)
  - 触发条件：当前分数极差（如分类模型 AUC 在 0.5~0.6 之间，相当于瞎猜），或存在极为严重的过拟合/欠拟合问题，并且 内部优化轮数（Optimize Round）< 3。
  - 要求：此时坚决不能把半成品丢给用户。你必须根据当前方案的具体问题，提出有针对性的优化建议，让代码Agent重新干活。
    【建议类型举例（不要局限于这些，要因问题而异）】
    - 调参类："当前 max_depth=3 过浅导致欠拟合，尝试 6~10"、"减小 learning_rate 并增加 n_estimators"
    - 特征类："目标列与特征 X/Y 的分布显示强相关性，建议构造交叉特征"、"某数值特征范围差异过大，应加入标准化"
    - 模型类："当前线性模型无法捕捉非线性关系，建议换用 XGBoost/LightGBM"、"当前树模型过拟合严重，尝试加入正则化或换用线性模型"
    - 方向类："当前方案完全偏离任务目标（如用回归做分类），建议重新设计 Pipeline"、"特征工程方向错误，应重新分析数据分布后再建模"
    - 数据类："类别特征未编码导致模型报错，应使用 OrdinalEncoder"、"缺失值处理策略不当，建议用中位数填充而非直接删除"
    重要：不要每次都说同样的话。仔细分析当前指标和维度评分，找出最致命的短板，优先解决那个问题。
2. 【DECISION: YIELD_TO_USER】(判断不需要调优或次数已满，交棒用户)
  - 触发条件：模型基线已达到及格水平（效果尚可），或者 内部自动优化次数已达到 3 次上限（必须强制交出控制权，避免死循环和算力空转）。
  - 要求：生成一段面向非专业用户的汇报总结。解释当前的成绩，指出模型的优缺点，并询问用户是否满意当前的基线版本。

特别说明：
- 如果用户在建模建议中提出了特定的评估侧重（如"重点关注召回率"），请在评估时优先考虑该指标的表现，并据此调整优化方向。
- 如果用户建议了特定的算法或方法但当前代码未采用，请在评估中指出这一点，并建议是否应该在下一轮中尝试。

Scoring Criteria (多维度评分，每项 0-100)
请基于以下 5 个维度对本次方案进行评分，并给出每个维度的具体理由：
【重要】每个维度的 reason 字段请控制在 20 个汉字以内，避免输出过长导致 JSON 截断。

1. 【指标表现 metric_performance】权重 30%
   - 以【用户指定的核心评估指标】为准进行判断（如用户要求用 F1，则以 F1 为核心；用户要求用 AUC，则以 AUC 为核心）
   - 不要机械套用固定阈值，应结合数据特点、任务难度和指标本身特性灵活判断
   - 参考标准（仅供参考，非绝对）：二分类 AUC > 0.7 通常及格，> 0.8 良好；多分类 Accuracy > 0.6 通常及格；回归 R² > 0.5 通常及格
   - 如果指标表现接近随机水平（如二分类 AUC 在 0.5~0.6 之间），应判定为极差

2. 【过拟合控制 overfit_control】权重 25%
   - 评估训练集与验证集指标的差距
   - 差距 < 5% 为优秀，5%~10% 为良好，10%~20% 为及格，> 20% 为差
   - 如果过拟合比（train/val）> 1.2，分数应显著降低

3. 【算法选择 algorithm_choice】权重 20%
   - 评估模型选择是否适合当前任务类型和数据特点
   - 是否使用了行业标准的算法（如树模型用于结构化数据）
   - 是否有明显的算法误用（如用线性回归做高度非线性任务）

4. 【Pipeline 完整性 pipeline_completeness】权重 15%
   - 评估代码是否包含完整的数据清洗、特征工程、模型训练、验证流程
   - 是否有明显的缺失环节（如没有处理缺失值、没有划分训练验证等）
   - 代码逻辑是否自洽，能否在沙箱中稳定运行

5. 【任务匹配度 task_alignment】权重 10%
   - 评估模型方案是否与用户设定的任务目标一致
   - 是否针对用户描述中的特殊需求进行了适配
   - 如果用户有建模建议，是否被合理采纳

综合评分计算方式：
score = metric_performance * 0.30 + overfit_control * 0.25 + algorithm_choice * 0.20 + pipeline_completeness * 0.15 + task_alignment * 0.10

Output Format (Strict JSON)
你必须严格输出如下 JSON 格式（不要包含 markdown 代码块标记）：
{
  "evaluation_analysis": "对本次运行结果的客观专业分析：拟合情况如何？分数是否达标？",
  "dimension_scores": [
    {"name": "metric_performance", "score": 78, "weight": 0.30, "reason": "AUC 0.82 超及格线"},
    {"name": "overfit_control", "score": 85, "weight": 0.25, "reason": "差距仅3%"},
    {"name": "algorithm_choice", "score": 80, "weight": 0.20, "reason": "XGBoost适合该任务"},
    {"name": "pipeline_completeness", "score": 75, "weight": 0.15, "reason": "流程完整"},
    {"name": "task_alignment", "score": 70, "weight": 0.10, "reason": "匹配目标"}
  ],
  "score": 79.5,
  "decision": "AUTO_OPTIMIZE" 或 "YIELD_TO_USER",
  "suggestions_for_coding_agent": "如果 decision 为 AUTO_OPTIMIZE，在此写出具体的技术调优或换模型建议。否则填 null。",
  "report_to_user": "如果 decision 为 YIELD_TO_USER，在此写出给用户的自然语言汇报，需通俗易懂并带有引导性。否则填 null。"
}"""


class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent

    职责：
    - 解析沙箱输出的验证集指标
    - 评估模型质量（多维度打分）
    - 决策：AUTO_OPTIMIZE（继续内部优化）或 YIELD_TO_USER（提交用户确认）
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
        eval_metric: Optional[str] = None
    ) -> EvaluationResult:
        """
        评估模型效果并决策
        
        注：针对某些模型（如 deepseek-v4-flash）JSON 输出易被截断的问题，
        已增加 max_tokens 和 JSON 修复机制。
        """
        """
        评估模型效果并决策
        """
        user_prompt = self._build_user_prompt(
            task_target, metrics, optimize_round, max_optimize_rounds, execution_output, user_modeling_suggestions, eval_metric
        )

        logger.info(f"[EvaluationAgent] 评估模型, optimize_round={optimize_round}/{max_optimize_rounds}")

        response = self._call_llm(EVALUATION_SYSTEM_PROMPT, user_prompt)

        result = self._parse_response(response)
        result.raw_response = response

        # 校验/修正加权总分
        result = self._normalize_score(result)

        logger.info(f"[EvaluationAgent] 决策: {result.decision}, score={result.score}")
        if result.dimension_scores:
            for ds in result.dimension_scores:
                logger.info(f"  [{ds.name}] {ds.score}/100 (weight={ds.weight}) {ds.reason[:60]}...")

        return result

    def _build_user_prompt(
        self,
        task_target: str,
        metrics: ExecutionMetrics,
        optimize_round: int,
        max_optimize_rounds: int,
        execution_output: str,
        user_modeling_suggestions: Optional[str] = None,
        eval_metric: Optional[str] = None
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
【重要】评估时请优先以该指标为核心标准。如果该指标表现极差，应优先考虑 AUTO_OPTIMIZE；如果该指标已达及格线但其他辅助指标不佳，可酌情处理。
""" if primary_metric else ""

        prompt = f"""【任务目标 Task Target】: {task_target}{metric_section}

【本次运行沙箱输出 Metrics】:
```json
{json.dumps(metrics_dict, ensure_ascii=False, indent=2)}
```

【沙箱完整输出】:
```
{execution_output[:2000] if execution_output else '无'}
```
{suggestions_section}
【内部优化轮数 Optimize Round】: {optimize_round} / {max_optimize_rounds}
{"【注意】已达到最大优化轮数上限，必须强制 YIELD_TO_USER。" if optimize_round >= max_optimize_rounds else ""}

请根据以上信息，严格按照 JSON 格式输出评估结论。必须包含 dimension_scores 数组和加权后的 score 字段。
"""
        return prompt

    def _parse_response(self, response: str) -> EvaluationResult:
        """
        解析 LLM 响应，提取 JSON 格式的评估结果
        
        容错策略：
        1. 先尝试标准 JSON 解析
        2. 如果失败，尝试修复截断的 JSON（补充缺失的闭合符号）
        3. 如果还是失败，从文本中尽量提取关键信息
        4. 最后兜底：基于关键词判断
        """
        # 策略1：提取 markdown JSON 代码块
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
        else:
            # 策略2：直接匹配 JSON 对象（贪婪匹配以获取完整内容）
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

        # 兜底：基于关键词判断
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
            dimension_scores=dim_scores
        )
    
    def _fix_truncated_json(self, json_str: str) -> str:
        """
        修复截断的 JSON：
        1. 找到最后一个完整的键值对
        2. 补充缺失的闭合符号
        """
        fixed = json_str.strip()
        
        # 如果 JSON 在某个字符串值中被截断，尝试找到最后一个闭合引号
        # 策略：从后向前找，找到最后一个完整的字段，然后补充闭合符号
        
        # 情况1：在某个字符串值中被截断（如 "reason": "未完成的...）
        # 找到最后一个未闭合的引号，补充闭合
        quote_count = fixed.count('"')
        if quote_count % 2 != 0:
            # 奇数个引号，说明有未闭合的字符串
            last_quote = fixed.rfind('"')
            if last_quote > 0:
                # 检查引号后是否已有逗号或闭合括号
                after_quote = fixed[last_quote+1:].strip()
                if not after_quote.startswith((',', '}', ']')):
                    fixed = fixed[:last_quote+1] + '"' + fixed[last_quote+1:]
        
        # 补充缺失的括号
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        fixed += '}' * max(0, open_braces)
        fixed += ']' * max(0, open_brackets)
        
        # 移除末尾的逗号
        fixed = fixed.rstrip(',')
        
        return fixed
    
    def _extract_from_broken_json(self, json_str: str) -> Optional[EvaluationResult]:
        """
        从损坏的 JSON 中尽量提取关键字段
        """
        # 提取 decision
        decision_match = re.search(r'"decision"\s*:\s*"(AUTO_OPTIMIZE|YIELD_TO_USER)"', json_str)
        decision = DecisionType(decision_match.group(1)) if decision_match else DecisionType.YIELD_TO_USER
        
        # 提取 score
        score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', json_str)
        score = float(score_match.group(1)) if score_match else None
        
        # 提取 evaluation_analysis
        analysis_match = re.search(r'"evaluation_analysis"\s*:\s*"([^"]*)"', json_str)
        analysis = analysis_match.group(1) if analysis_match else "JSON截断，部分信息提取"
        
        # 提取 suggestions
        sug_match = re.search(r'"suggestions_for_coding_agent"\s*:\s*"([^"]*)"', json_str)
        suggestions = sug_match.group(1) if sug_match else None
        
        # 提取 report
        rep_match = re.search(r'"report_to_user"\s*:\s*"([^"]*)"', json_str)
        report = rep_match.group(1) if rep_match else None
        
        # 提取维度评分（尽可能）
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
                dimension_scores=dim_scores
            )
        return None

    def _normalize_score(self, result: EvaluationResult) -> EvaluationResult:
        """
        校验并修正加权总分
        - 如果 LLM 未提供 dimension_scores，根据现有 score 生成默认维度
        - 如果提供了 dimension_scores，重新计算加权总分以确保一致
        """
        if not result.dimension_scores:
            # LLM 未返回维度评分，生成默认维度
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

        # 重新计算加权总分（确保与维度评分一致）
        calculated = 0.0
        for ds in result.dimension_scores:
            w = self.DIMENSION_WEIGHTS.get(ds.name, ds.weight)
            calculated += ds.score * w

        # 如果 LLM 提供的 score 与计算值差距 > 5，以计算值为准（防止 LLM 计算错误）
        if result.score is None or abs(result.score - calculated) > 5:
            logger.warning(f"[EvaluationAgent] 修正加权总分: {result.score} -> {calculated:.1f}")
            result.score = round(calculated, 1)
        else:
            # 保留 LLM 的评分（差距不大时）
            result.score = round(result.score, 1)

        # 确保每个维度的 weight 与系统定义一致
        for ds in result.dimension_scores:
            ds.weight = self.DIMENSION_WEIGHTS.get(ds.name, ds.weight)

        return result

    def _fallback_parse(self, response: str) -> EvaluationResult:
        """
        兜底解析：当 JSON 解析失败时，基于关键词判断决策
        """
        response_upper = response.upper()

        if "AUTO_OPTIMIZE" in response_upper:
            decision = DecisionType.AUTO_OPTIMIZE
        elif "YIELD_TO_USER" in response_upper:
            decision = DecisionType.YIELD_TO_USER
        else:
            decision = DecisionType.YIELD_TO_USER

        suggestions = None
        report = None

        if decision == DecisionType.AUTO_OPTIMIZE:
            sug_match = re.search(r'(?:建议|suggestions|优化).*?(:|：)\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
            if sug_match:
                suggestions = sug_match.group(2).strip()
            else:
                suggestions = "请尝试更换模型或调整超参数。"
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
            dimension_scores=dim_scores
        )

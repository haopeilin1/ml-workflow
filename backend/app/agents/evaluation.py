"""
Evaluation Agent
评估模型效果，决策 AUTO_OPTIMIZE 或 YIELD_TO_USER
"""

import json
import logging
import re
from typing import Optional

from app.agents.base import BaseAgent
from app.models.schemas import EvaluationResult, DecisionType, ExecutionMetrics

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
  - 要求：此时坚决不能把半成品丢给用户。你必须提出具体的优化建议（如："更换为 XGBoost"、"增加树深度"、"加入正则化"、"处理某些特征的共线性"等），让代码Agent重新干活。
2. 【DECISION: YIELD_TO_USER】(判断不需要调优或次数已满，交棒用户)
  - 触发条件：模型基线已达到及格水平（效果尚可），或者 内部自动优化次数已达到 3 次上限（必须强制交出控制权，避免死循环和算力空转）。
  - 要求：生成一段面向非专业用户的汇报总结。解释当前的成绩，指出模型的优缺点，并询问用户是否满意当前的基线版本。

Scoring Criteria (综合评分 0-100)
请基于以下维度对本次方案进行综合评分：
- 连贯性 (Coherence)：代码逻辑是否自洽，Pipeline 是否完整
- 可信度 (Credibility)：算法选择是否合理，是否符合该任务类型的最佳实践
- 可验证性 (Verifiability)：是否能在沙箱中成功运行并产出可验证的指标
- 一致性 (Alignment)：方案是否与用户设定的任务目标保持一致
- 过拟合风险：训练集与验证集指标差距是否可控
注意，在进行评估时对指标的分析应该考虑任务背景和行业标准，而不是简单地套用固定的分数线，不能只基于绝对数值进行评估。请结合实际情况进行综合判断。

Output Format (Strict JSON)
你必须严格输出如下 JSON 格式：
{
  "evaluation_analysis": "对本次运行结果的客观专业分析：拟合情况如何？分数是否达标？",
  "score": 85,
  "decision": "AUTO_OPTIMIZE" 或 "YIELD_TO_USER",
  "suggestions_for_coding_agent": "如果 decision 为 AUTO_OPTIMIZE，在此写出具体的技术调优或换模型建议。否则填 null。",
  "report_to_user": "如果 decision 为 YIELD_TO_USER，在此写出给用户的自然语言汇报，需通俗易懂并带有引导性。否则填 null。"
}"""


class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent
    
    职责：
    - 解析沙箱输出的验证集指标
    - 评估模型质量（过拟合检测、分数达标判断）
    - 决策：AUTO_OPTIMIZE（继续内部优化）或 YIELD_TO_USER（提交用户确认）
    """
    
    def evaluate(
        self,
        task_target: str,
        metrics: ExecutionMetrics,
        optimize_round: int,
        max_optimize_rounds: int = 3,
        execution_output: str = ""
    ) -> EvaluationResult:
        """
        评估模型效果并决策
        
        Args:
            task_target: 任务目标描述
            metrics: 沙箱执行返回的指标
            optimize_round: 当前已进行的优化轮数
            max_optimize_rounds: 最大优化轮数上限
            execution_output: 完整的沙箱 stdout（用于辅助分析）
            
        Returns:
            EvaluationResult: 包含评估分析、决策、建议/汇报
        """
        user_prompt = self._build_user_prompt(
            task_target, metrics, optimize_round, max_optimize_rounds, execution_output
        )
        
        logger.info(f"[EvaluationAgent] 评估模型, optimize_round={optimize_round}/{max_optimize_rounds}")
        
        response = self._call_llm(EVALUATION_SYSTEM_PROMPT, user_prompt)
        
        result = self._parse_response(response)
        result.raw_response = response
        
        logger.info(f"[EvaluationAgent] 决策: {result.decision}")
        
        return result
    
    def _build_user_prompt(
        self,
        task_target: str,
        metrics: ExecutionMetrics,
        optimize_round: int,
        max_optimize_rounds: int,
        execution_output: str
    ) -> str:
        """构建用户提示词"""
        metrics_dict = {}
        if metrics:
            metrics_dict = {
                k: v for k, v in metrics.model_dump().items()
                if v is not None
            }
        
        prompt = f"""【任务目标 Task Target】: {task_target}

【本次运行沙箱输出 Metrics】:
```json
{json.dumps(metrics_dict, ensure_ascii=False, indent=2)}
```

【沙箱完整输出】:
```
{execution_output[:2000] if execution_output else '无'}
```

【内部优化轮数 Optimize Round】: {optimize_round} / {max_optimize_rounds}
{"【注意】已达到最大优化轮数上限，必须强制 YIELD_TO_USER。" if optimize_round >= max_optimize_rounds else ""}

请根据以上信息，严格按照 JSON 格式输出评估结论。
"""
        return prompt
    
    def _parse_response(self, response: str) -> EvaluationResult:
        """
        解析 LLM 响应，提取 JSON 格式的评估结果
        
        处理策略：
        1. 先尝试提取 ```json ... ``` 代码块
        2. 再尝试直接匹配 JSON 对象
        3. 兜底：基于规则生成默认结果
        """
        # 策略1：提取 markdown JSON 代码块
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
        else:
            # 策略2：直接匹配 JSON 对象
            json_match = re.search(r'\{[\s\S]*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
            else:
                json_str = ""
        
        if json_str:
            try:
                data = json.loads(json_str)
                return EvaluationResult(
                    evaluation_analysis=data.get("evaluation_analysis", "未提供分析"),
                    decision=DecisionType(data.get("decision", "YIELD_TO_USER")),
                    suggestions_for_coding_agent=data.get("suggestions_for_coding_agent"),
                    report_to_user=data.get("report_to_user"),
                    score=data.get("score")
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[EvaluationAgent] JSON 解析失败: {e}, 原始响应: {response[:500]}")
        
        # 兜底：基于关键词判断
        return self._fallback_parse(response)
    
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
            # 默认交给用户
            decision = DecisionType.YIELD_TO_USER
        
        # 尝试提取建议或报告
        suggestions = None
        report = None
        
        if decision == DecisionType.AUTO_OPTIMIZE:
            # 尝试提取建议部分
            sug_match = re.search(r'(?:建议|suggestions|优化).*?(:|：)\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
            if sug_match:
                suggestions = sug_match.group(2).strip()
            else:
                suggestions = "请尝试更换模型或调整超参数。"
        else:
            # 尝试提取报告部分
            rep_match = re.search(r'(?:汇报|report|总结).*?(:|：)\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
            if rep_match:
                report = rep_match.group(2).strip()
            else:
                report = "模型基线已生成，请查看右侧结果面板。"
        
        logger.warning(f"[EvaluationAgent] 使用兜底解析, decision={decision.value}")
        
        # 兜底评分：基于决策简单估算
        fallback_score = 75.0 if decision == DecisionType.YIELD_TO_USER else 45.0
        
        return EvaluationResult(
            evaluation_analysis="LLM 输出格式异常，使用兜底解析。",
            decision=decision,
            suggestions_for_coding_agent=suggestions if decision == DecisionType.AUTO_OPTIMIZE else None,
            report_to_user=report if decision == DecisionType.YIELD_TO_USER else None,
            score=fallback_score
        )

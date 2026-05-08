"""
LLM as Judge Agent
用于自动化评测系统中，对测试集上的模型表现进行最终评估
"""

import json
import logging
import re
from typing import Optional

from app.agents.base import BaseAgent, LLMClient
from app.models.evaluate_schemas import JudgeResult, TestSetMetrics
from app.models.schemas import ExecutionMetrics, LLMConfig, TaskType

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """你是一名资深机器学习模型评估专家与质量控制官。
你的任务是对模型的最终交付质量进行一次性判定：该模型是否可以被接受为任务完成。

【重要说明】
1. 这是"首次交付"的最终判定，不需要提出迭代优化建议。即使判定为拒绝，也不会重新训练模型。
2. 测试集数据在建模与迭代过程中完全不可见，只在最终交付时进行一次预测。因此测试集指标能真实反映模型的泛化能力。
3. 判断时必须以【建模时使用的主要指标】为核心标准：
   - 如果验证集主要指标是 AUC，则重点看测试集 AUC
   - 如果验证集主要指标是 F1，则重点看测试集 F1
   - 如果验证集主要指标是 Accuracy，则重点看测试集 Accuracy
   - 如果验证集主要指标是 RMSE/R²，则重点看测试集 RMSE/R²
   其他指标仅作为辅助参考。

评估维度：
1. 核心指标表现：模型在测试集上的【主要指标】是否达到该任务类型的合理基线
2. 泛化一致性：测试集表现与验证集是否基本一致（差距 < 10% 为良好）
3. 过拟合风险：训练集与验证集/测试集指标差距是否过大

判断标准（请灵活判断，不要机械套用固定阈值）：
- 以【用户指定的核心评估指标】为唯一核心标准，其他指标仅作辅助参考
- 结合数据特点、任务难度和指标特性进行综合判断，不要死守固定数值
- 判断标准因指标而异（以下为参考，请灵活应用）：
  * 二分类 AUC > 0.7 或 F1 > 0.65 通常可接受；AUC < 0.6 且 F1 < 0.5 通常不可接受
  * 多分类 Accuracy > 0.6 或 F1-macro > 0.55 通常可接受
  * Log Loss（越低越好）：< 1.0 通常可接受，> 2.0 通常不可接受
  * Cohen's Kappa / Matthews MCC：> 0.4 通常可接受
  * 回归 R² > 0.5 或 RMSE 在合理范围内通常可接受
- 核心指标接近随机水平（如 AUC < 0.6、Accuracy < 0.5、Log Loss > 2.5 等）通常不可接受

特别说明：
- 如果测试集核心指标与验证集差距过大（如差距 > 0.15），说明泛化能力差，应拒绝
- 如果模型存在严重过拟合（训练集指标远高于验证集/测试集），应拒绝
- 如果模型未成功运行（所有指标为 None），必须拒绝
- 如果测试集核心指标略低于验证集但在可接受范围内，可以酌情接受

输出格式（严格 JSON，不要包含 markdown 代码块标记）：
{
  "accepted": true/false,
  "analysis": "对模型表现的详细专业分析，必须明确指出使用了哪个核心指标进行判断",
  "reason": "接受或拒绝的具体理由，必须引用测试集核心指标的具体数值"
}"""


class EvaluateJudgeAgent(BaseAgent):
    """
    LLM as Judge Agent

    职责：
    - 根据验证集和测试集指标判断模型是否可接受
    - 给出具体的评估分析和理由
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config:
            llm_client = LLMClient(
                provider=llm_config.provider,
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                extra_body=llm_config.extra_body
            )
            super().__init__(llm_client=llm_client)
            logger.info(f"[EvaluateJudgeAgent] 使用独立 LLM 配置: provider={llm_config.provider}, model={llm_config.model}")
        else:
            super().__init__()
            logger.info("[EvaluateJudgeAgent] 使用全局默认 LLM 配置")

    def judge(
        self,
        task_type: TaskType,
        target_column: str,
        eval_metric: Optional[str],
        val_metrics: Optional[ExecutionMetrics],
        test_metrics: Optional[TestSetMetrics]
    ) -> JudgeResult:
        """
        评估模型并决定是否接受

        Args:
            task_type: 任务类型
            target_column: 目标列名
            eval_metric: 评估指标名称
            val_metrics: 验证集指标
            test_metrics: 测试集指标

        Returns:
            JudgeResult
        """
        user_prompt = self._build_user_prompt(
            task_type, target_column, eval_metric, val_metrics, test_metrics
        )

        try:
            raw_response = self._call_llm(JUDGE_SYSTEM_PROMPT, user_prompt)
            return self._parse_judge_response(raw_response)
        except Exception as e:
            logger.error(f"[EvaluateJudgeAgent] LLM 调用失败: {e}")
            return JudgeResult(
                accepted=False,
                analysis=f"Judge Agent 调用失败: {str(e)}",
                reason="无法完成评估，默认拒绝",
                raw_response=None
            )

    def _build_user_prompt(
        self,
        task_type: TaskType,
        target_column: str,
        eval_metric: Optional[str],
        val_metrics: Optional[ExecutionMetrics],
        test_metrics: Optional[TestSetMetrics]
    ) -> str:
        """构建 Judge 的用户 Prompt"""

        # 推断建模时使用的主要指标
        primary_metric = eval_metric or ""
        if not primary_metric and val_metrics:
            # 从 metric_name 或非空字段推断
            if val_metrics.metric_name:
                primary_metric = val_metrics.metric_name
            elif val_metrics.val_auc is not None:
                primary_metric = "AUC"
            elif val_metrics.val_rmse is not None:
                primary_metric = "RMSE"
            elif val_metrics.val_accuracy is not None:
                primary_metric = "Accuracy"
            elif val_metrics.val_score is not None:
                primary_metric = "Score"

        # 格式化验证集指标
        val_lines = []
        if val_metrics:
            if val_metrics.val_auc is not None:
                val_lines.append(f"  - 验证集 AUC: {val_metrics.val_auc:.4f}")
            if val_metrics.val_accuracy is not None:
                val_lines.append(f"  - 验证集 Accuracy: {val_metrics.val_accuracy:.4f}")
            if val_metrics.val_rmse is not None:
                val_lines.append(f"  - 验证集 RMSE: {val_metrics.val_rmse:.4f}")
            if val_metrics.val_score is not None:
                val_lines.append(f"  - 验证集 Score: {val_metrics.val_score:.4f}")
            if val_metrics.train_auc is not None:
                val_lines.append(f"  - 训练集 AUC: {val_metrics.train_auc:.4f}")
            if val_metrics.train_score is not None:
                val_lines.append(f"  - 训练集 Score: {val_metrics.train_score:.4f}")
            if val_metrics.overfit_ratio is not None:
                val_lines.append(f"  - 过拟合比: {val_metrics.overfit_ratio:.4f}")
        else:
            val_lines.append("  - 无验证集指标（模型可能未成功运行）")

        # 格式化测试集指标
        test_lines = []
        if test_metrics:
            if test_metrics.auc is not None:
                test_lines.append(f"  - 测试集 AUC: {test_metrics.auc:.4f}")
            if test_metrics.accuracy is not None:
                test_lines.append(f"  - 测试集 Accuracy: {test_metrics.accuracy:.4f}")
            if test_metrics.f1 is not None:
                test_lines.append(f"  - 测试集 F1: {test_metrics.f1:.4f}")
            if test_metrics.f1_macro is not None:
                test_lines.append(f"  - 测试集 F1-macro: {test_metrics.f1_macro:.4f}")
            if test_metrics.rmse is not None:
                test_lines.append(f"  - 测试集 RMSE: {test_metrics.rmse:.4f}")
            if test_metrics.mae is not None:
                test_lines.append(f"  - 测试集 MAE: {test_metrics.mae:.4f}")
            if test_metrics.r2 is not None:
                test_lines.append(f"  - 测试集 R²: {test_metrics.r2:.4f}")
            if test_metrics.log_loss is not None:
                test_lines.append(f"  - 测试集 Log Loss: {test_metrics.log_loss:.4f}")
        else:
            test_lines.append("  - 无测试集指标（模型可能未成功运行）")

        return f"""请对以下机器学习模型的最终交付质量进行一次性判定。

【任务信息】
- 任务类型: {task_type.value}
- 目标列: {target_column}
- 【建模时使用的主要指标】: {primary_metric or '未明确指定，请根据指标情况自行判断'}

【验证集指标】
{chr(10).join(val_lines)}

【测试集指标】（测试集在建模过程中完全不可见，仅最终预测一次）
{chr(10).join(test_lines)}

【判定要求】
1. 请以【建模时使用的主要指标：{primary_metric or '核心指标'}】为核心判断标准
2. 重点关注测试集上的该指标表现是否达到可交付水平
3. 同时检查测试集与验证集指标是否一致（差距是否过大）
4. 这是最终判定，不需要提出优化建议

请给出你的评估结论。
"""

    def _parse_judge_response(self, raw_response: str) -> JudgeResult:
        """解析 Judge LLM 的输出"""
        # 去除 markdown 代码块标记
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # 尝试直接解析 JSON
        try:
            data = json.loads(cleaned)
            return JudgeResult(
                accepted=bool(data.get("accepted", False)),
                analysis=str(data.get("analysis", "")),
                reason=str(data.get("reason", "")),
                raw_response=raw_response
            )
        except json.JSONDecodeError:
            pass

        # 尝试从文本中提取 JSON
        json_match = re.search(r'\{[\s\S]*?\}', cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return JudgeResult(
                    accepted=bool(data.get("accepted", False)),
                    analysis=str(data.get("analysis", "")),
                    reason=str(data.get("reason", "")),
                    raw_response=raw_response
                )
            except json.JSONDecodeError:
                pass

        # 解析失败，根据文本关键词判断
        accepted = "接受" in cleaned or "accepted" in cleaned.lower() or "通过" in cleaned
        logger.warning(f"[EvaluateJudgeAgent] JSON 解析失败，基于关键词推断 accepted={accepted}")
        return JudgeResult(
            accepted=accepted,
            analysis=cleaned[:500],
            reason="无法解析标准 JSON 格式，基于关键词推断",
            raw_response=raw_response
        )

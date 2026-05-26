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
2. 测试集特征（test.csv）在建模过程中可用，但真值（ground truth）不在沙箱中，因此代码无法计算测试集指标。测试集指标只能在系统层面通过对比预测结果与真值获得。
3. 判断时必须以【建模时使用的主要指标】为核心标准：
   - 如果验证集主要指标是 AUC，则重点看测试集 AUC
   - 如果验证集主要指标是 F1，则重点看测试集 F1
   - 如果验证集主要指标是 Accuracy，则重点看测试集 Accuracy
   - 如果验证集主要指标是 RMSE/R²，则重点看测试集 RMSE/R²
   其他指标仅作为辅助参考。

评估维度：
1. 核心指标表现：模型在测试集上的【主要指标】是否达到该任务类型的合理基线
2. 泛化一致性：测试集表现与验证集是否基本一致（相对差距 < 10% 为良好，< 20% 为可接受）
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
- 泛化一致性判定应使用【相对差距】而非绝对差距：
  * 对于 AUC/Accuracy/F1/R²（越高越好）：相对差距 = (val - test) / val
  * 对于 RMSE/MAE/Log Loss（越低越好）：相对差距 = (test - val) / val
  * 相对差距 < 10% 为良好，10%-30% 为可接受范围（时序/回归任务可放宽至 50%），> 50% 说明泛化能力差，应拒绝
- 【时序/回归任务特别放宽】对于时间序列回归任务（如预测、销量、租赁量等），测试集数据通常在验证集之后，可能跨越季节边界或遇到分布自然偏移。此时：
  * 若 R² > 0.95 或测试集 MAE/RMSE 绝对值很小（如 MAE < 10% 目标均值），即使相对差距达到 30%-50% 也可接受
  * 应以【测试集绝对指标质量】为主，【验证-测试相对差距】为辅进行综合判断
  * 不应仅凭 RMSE 相对差距 > 20% 就拒绝一个 R² > 0.98 的高质量回归模型
- 如果模型存在严重过拟合（训练集指标远高于验证集/测试集），应拒绝
- 如果模型未成功运行（所有指标为 None），必须拒绝
- 如果验证集核心指标本身已接近随机水平（如 AUC < 0.6、Accuracy < 0.5），即使 test 差距不大也应拒绝
- 【关键】如果测试集预测使用了通用回退模板（generic fallback），说明预测阶段的特征工程与训练阶段很可能不一致，预测结果不可靠。此时即使指标看起来尚可，也应拒绝，并在 reason 中明确说明"使用了通用回退模板，特征工程不一致导致预测不可靠"。

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
        test_metrics: Optional[TestSetMetrics],
        prediction_strategy: Optional[str] = None
    ) -> JudgeResult:
        """
        评估模型并决定是否接受

        Args:
            task_type: 任务类型
            target_column: 目标列名
            eval_metric: 评估指标名称
            val_metrics: 验证集指标
            test_metrics: 测试集指标
            prediction_strategy: 测试集预测策略 (llm_predict / injected / generic_fallback)

        Returns:
            JudgeResult
        """
        user_prompt = self._build_user_prompt(
            task_type, target_column, eval_metric, val_metrics, test_metrics, prediction_strategy
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
        test_metrics: Optional[TestSetMetrics],
        prediction_strategy: Optional[str] = None
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

        # 预测策略说明
        strategy_lines = []
        if prediction_strategy == "llm_predict":
            strategy_lines.append("- 测试集预测策略: LLM 生成的 predict.py（可靠性高）")
        elif prediction_strategy == "injected":
            strategy_lines.append("- 测试集预测策略: 注入式预测脚本（复用训练代码自定义定义，可靠性中高）")
        elif prediction_strategy == "generic_fallback":
            strategy_lines.append("- 测试集预测策略: 通用回退模板（⚠️ 可靠性低：特征工程可能与训练阶段不一致）")
        else:
            strategy_lines.append("- 测试集预测策略: 未知")

        return f"""请对以下机器学习模型的最终交付质量进行一次性判定。

【任务信息】
- 任务类型: {task_type.value}
- 目标列: {target_column}
- 【建模时使用的主要指标】: {primary_metric or '未明确指定，请根据指标情况自行判断'}

【验证集指标】
{chr(10).join(val_lines)}

【测试集指标】（测试集在建模过程中完全不可见，仅最终预测一次）
{chr(10).join(test_lines)}

【预测策略信息】
{chr(10).join(strategy_lines)}

【判定要求】
1. 请以【建模时使用的主要指标：{primary_metric or '核心指标'}】为核心判断标准
2. 重点关注测试集上的该指标表现是否达到可交付水平
3. 同时检查测试集与验证集指标是否一致（差距是否过大）
4. 如果预测策略为"通用回退模板"，必须拒绝，因为特征工程不一致导致预测结果不可靠
5. 这是最终判定，不需要提出优化建议

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

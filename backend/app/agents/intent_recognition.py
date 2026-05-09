"""
意图识别 Agent
基于 LLM 从数据画像 + 任务描述中识别目标列、任务类型、评估指标
复用全局 LLM 配置（与用户真实使用时一致）
"""

import json
import logging
from typing import Optional, List, Dict

from app.agents.base import BaseAgent, LLMClient
from app.config import build_eval_llm_config
from app.models.schemas import TaskType

logger = logging.getLogger(__name__)


class IntentResult:
    """意图识别结果"""
    def __init__(self, target_column: str, task_type: TaskType, eval_metric: Optional[str] = None, complexity: str = "simple", is_time_series: bool = False):
        self.target_column = target_column
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.complexity = complexity  # "simple" 或 "complex"
        self.is_time_series = is_time_series  # 是否为时序任务

    def __repr__(self):
        return f"IntentResult(target={self.target_column}, type={self.task_type.value}, metric={self.eval_metric}, complexity={self.complexity}, ts={self.is_time_series})"


class IntentRecognitionAgent(BaseAgent):
    """
    意图识别 Agent
    默认使用 EVAL_INTENT_* 配置（若未配置则回退到全局 LLM_*）
    """

    def __init__(self):
        cfg = build_eval_llm_config("intent")
        llm_client = LLMClient(
            provider=cfg.get("provider"),
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
            model=cfg.get("model"),
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens"),
            extra_body=cfg.get("extra_body"),
        )
        super().__init__(llm_client=llm_client)

    SYSTEM_PROMPT = """你是一个数据建模任务的意图识别专家。你的任务是从数据画像和任务描述中，准确提取 target_column、task_type 和 eval_metric。

【核心原则：任务描述优先，语义驱动，不要机械判断】

1. target_column（目标列）
   - 必须是数据画像中真实存在的列名
   - 【第一步：排除id列】首先排除唯一值数量 ≈ 数据行数（或 > 行数×0.9）的列，这些是id/序号/索引列（如 user_id、order_id、row_num、index），绝不可能是目标列。数据画像中已用 "isLikelyId": true 标记这类列。
   - 【第二步：任务描述匹配】如果任务描述中明确提到了某个列名（如"预测charges列""判断是否会Churn"），直接选择该列。
   - 【第三步：语义推断】如果任务描述未提及，选择最像"标签/目标/结果"的列。参考信号：
     * 列名关键词：label、target、class、category、status、type、grade、level、result、outcome、quality、score、value、amount、price、rate、churn、fraud、default、spam、outcome、diagnosis、disorder、proximity、species、rent、cnt（count缩写）
     * 结合数据类型：分类任务的target通常是类别型（object/string 或 int但唯一值少），回归任务的target通常是连续数值（float 或 int但范围大）
     * 结合样本值：看该列的实际取值是否有明显语义（如是/否、A/B/C、具体价格数字）

2. task_type（任务类型）
   - 【任务描述绝对优先】如果任务描述明确说了"二分类""多分类""回归""预测连续值"等，直接按描述判断，不要再看数据统计。
   - 分类任务（binary/multiclass）的特征：
     * 目标列是离散类别（如 是/否、A/B/C、高/中/低、 spam/ham）
     * 即使数值编码（0/1/2），如果语义上是离散类别，也是分类
     * 二分类：目标列只有 2 个语义类别
     * 多分类：目标列有 3 个及以上语义类别
   - 回归任务（regression）的特征：
     * 目标列是连续数值（如价格、数量、温度、评分、概率、浓度、租金）
     * 唯一值通常较多，但不要用固定阈值（如">10"）作为绝对标准
   - 时序回归：如果数据包含时间列且任务是"预测未来某指标"，归为 regression
   - 【反例提醒】不要仅根据唯一值数量判断：0-5分的评分（6个值）是分类；0-100分的百分制（可能几十个值）是回归；用户ID有2万个不同值但绝不是回归目标。

3. eval_metric（评估指标）
   - 【任务描述优先提取】仔细阅读任务描述，找出明确指定的评估指标（如"评估指标采用AUC""以F1-macro为主要指标"）。
   - 如果任务描述中未指定，根据任务语义和数据特点推断：
     * 二分类 + 关注排序能力（如违约预测、欺诈检测） → AUC
     * 二分类 + 关注正类识别（如疾病诊断、垃圾邮件检测） → F1 / Recall / Precision（视语义侧重）
     * 多分类 + 类别均衡 → Accuracy
     * 多分类 + 类别明显不平衡 → F1-macro
     * 回归 + 关注预测误差大小（如房价预测、保费预测） → RMSE / MAE
     * 回归 + 关注拟合程度（如物理量预测） → R²

4. complexity（复杂度判定）
   - 请同时判定该任务是"simple"还是"complex"
   - complex 信号（满足任一即判定为 complex）：
     * 存在时间相关列（year/month/day/hour/season/week/datetime 等）
     * 多分类任务（3个及以上类别）
     * 类别极度不平衡（最大类占比 > 90% 或最小类 < 5%）
     * 缺失值严重（任意列缺失率 > 20%）
     * 高维稀疏特征（列数 > 50 且大量零值）
   - 无上述信号则判定为 simple

5. is_time_series（时序任务判定）
   - 如果数据中存在时间相关列（如 year/month/day/hour/dteday/No/instant/date/datetime/season/week 等），且任务是预测未来的某个指标，则判定为 true
   - 否则判定为 false
   - 注意：即使 task_type 是 regression，只要有明显的时间列且任务语义涉及时序预测，也判定为 true

【输出格式】
严格输出 JSON，不要任何额外文字：
{
  "target_column": "列名",
  "task_type": "binary_classification / multiclass_classification / regression",
  "eval_metric": "用户指定的评估指标名称（如AUC/F1/Log Loss/RMSE等，开放取值不限列表）",
  "complexity": "simple / complex",
  "is_time_series": true / false
}"""

    def recognize(
        self,
        columns: List[Dict],
        task_description: str,
        row_count: int = 0,
        col_count: int = 0
    ) -> IntentResult:
        """
        识别任务意图

        Args:
            columns: 列信息列表，每项为 dict(name, type, missing_rate, unique_count)
            task_description: 任务描述文本
            row_count: 数据行数
            col_count: 数据列数

        Returns:
            IntentResult
        """
        data_profile = {
            "row_count": row_count,
            "col_count": col_count,
            "columns": columns
        }

        user_prompt = f"""【任务描述】（优先级最高，其中的明确指令必须严格遵守）：
{task_description}

【数据画像】（供参考和验证，当与任务描述冲突时，以任务描述为准）：
{json.dumps(data_profile, ensure_ascii=False, indent=2)}

【提取要求】
1. 如果任务描述中明确提到了目标列名（如"预测charges列""判断是否会Churn""目标列为fraud_bool"），必须直接选择该列，不要再看数据统计推断。
2. 如果任务描述中明确提到了任务类型（如"二分类""回归""多分类"），必须直接采用，不要被数据的统计特征误导。
3. 如果任务描述中明确提到了评估指标（如"以AUC为评估标准"），必须直接采用。
4. 只有在任务描述未明确说明时，才结合数据画像进行推断。
5. 必须同时判定 complexity（simple 或 complex），参考标准见 system prompt。
6. 必须同时判定 is_time_series（true 或 false），参考标准见 system prompt。

请严格输出 JSON：
{{
  "target_column": "列名",
  "task_type": "binary_classification/multiclass_classification/regression",
  "eval_metric": "用户指定的评估指标名称（开放取值）",
  "complexity": "simple/complex",
  "is_time_series": true/false
}}"""

        try:
            content = self._call_llm(self.SYSTEM_PROMPT, user_prompt)
            result = self._parse_json(content)

            target_column = result.get("target_column", "")
            task_type_str = result.get("task_type", "binary_classification")
            eval_metric = result.get("eval_metric")
            complexity = result.get("complexity", "simple")
            is_time_series = result.get("is_time_series", False)
            # 合法性校验
            if complexity not in ("simple", "complex"):
                complexity = "simple"
            if not isinstance(is_time_series, bool):
                is_time_series = False

            # 验证 target_column 是否在 columns 中
            valid_cols = [c["name"] for c in columns]
            if target_column not in valid_cols:
                logger.warning(
                    f"[IntentRecognition] LLM 返回的 target_column '{target_column}' "
                    f"不在有效列 {valid_cols} 中，尝试 fallback"
                )
                target_column = self._fallback_target_column(columns, task_description)

            task_type = self._map_task_type(task_type_str)

            logger.info(
                f"[IntentRecognition] 识别结果: target={target_column}, "
                f"type={task_type.value}, metric={eval_metric}, complexity={complexity}, ts={is_time_series}"
            )
            return IntentResult(
                target_column=target_column,
                task_type=task_type,
                eval_metric=eval_metric,
                complexity=complexity,
                is_time_series=is_time_series
            )

        except Exception as e:
            logger.error(f"[IntentRecognition] LLM 识别失败: {e}，使用规则 fallback")
            return self._rule_based_recognition(columns, task_description)

    def _parse_json(self, content: str) -> dict:
        """从 LLM 输出中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        import re
        pattern = r'```(?:json)?\s*(.*?)```'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试从花括号中提取
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从 LLM 输出中提取 JSON: {content[:200]}")

    def _map_task_type(self, task_type_str: str) -> TaskType:
        """映射任务类型字符串到枚举"""
        mapping = {
            "binary_classification": TaskType.BINARY_CLASSIFICATION,
            "multiclass_classification": TaskType.MULTICLASS_CLASSIFICATION,
            "regression": TaskType.REGRESSION,
        }
        return mapping.get(task_type_str.lower().strip(), TaskType.BINARY_CLASSIFICATION)

    def _fallback_target_column(self, columns: List[Dict], task_description: str) -> str:
        """target_column 不在有效列中时 fallback"""
        # 优先选择数据画像中标记为目标的列
        for c in columns:
            name = c["name"].lower()
            if any(k in name for k in ["target", "label", "class", "quality", "status", "value"]):
                return c["name"]
        # 否则选最后一列
        return columns[-1]["name"] if columns else "target"

    def _rule_based_recognition(
        self, columns: List[Dict], task_description: str
    ) -> IntentResult:
        """LLM 失败时的规则 fallback"""
        target_column = self._fallback_target_column(columns, task_description)

        # 从列名推断是否为时序任务
        time_keywords = {"year", "month", "day", "hour", "dteday", "date", "time", "datetime", "season", "week", "instant", "no"}
        is_time_series = any(
            any(kw in c["name"].lower() for kw in time_keywords)
            for c in columns
        )

        # 从列的 unique_count 推断任务类型
        for c in columns:
            if c["name"] == target_column:
                unique_count = c.get("unique_count", 0)
                if unique_count == 2:
                    task_type = TaskType.BINARY_CLASSIFICATION
                    eval_metric = "AUC"
                elif 3 <= unique_count <= 10:
                    task_type = TaskType.MULTICLASS_CLASSIFICATION
                    eval_metric = "Accuracy"
                else:
                    task_type = TaskType.REGRESSION
                    eval_metric = "RMSE"
                return IntentResult(
                    target_column=target_column,
                    task_type=task_type,
                    eval_metric=eval_metric,
                    complexity="simple",
                    is_time_series=is_time_series
                )

        # 默认 fallback
        return IntentResult(
            target_column=target_column,
            task_type=TaskType.BINARY_CLASSIFICATION,
            eval_metric="AUC",
            complexity="simple",
            is_time_series=is_time_series
        )

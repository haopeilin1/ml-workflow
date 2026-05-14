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
    def __init__(self, target_column: str, task_type: TaskType, eval_metric: Optional[str] = None, complexity: str = "simple", is_time_series: bool = False, complexity_reason: Optional[str] = None):
        self.target_column = target_column
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.complexity = complexity  # "simple" 或 "complex"
        self.is_time_series = is_time_series  # 是否为时序任务
        self.complexity_reason = complexity_reason  # 复杂度判定原因说明

    def __repr__(self):
        return f"IntentResult(target={self.target_column}, type={self.task_type.value}, metric={self.eval_metric}, complexity={self.complexity}, ts={self.is_time_series}, reason={self.complexity_reason})"
    
    def get_complexity_reason(self) -> str:
        """获取复杂度判定原因的简短描述"""
        return self.complexity_reason or "LLM直接判定"


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

5. is_time_series（时序任务判定）——【请特别谨慎】
   - 时序任务的真正定义：数据按**完整的时间戳**排列（如 year+month+day+hour），目标是基于历史时间序列预测未来值。
   - 【强信号】有 year(2000-2030) + month(1-12) + day(1-31) 组合，或有 dteday/Date 日期列 → 判定为 true
   - 【强信号】任务描述明确提到"时序回归""按时间顺序""预测未来趋势" → 判定为 true
   - 【排除信号】仅有一个 Time/Timestamp 列（如信用卡交易的秒级时间戳），但无 year/month/day → 判定为 false（这是交易时刻，不是时间序列索引）
   - 【排除信号】month 列全是 0 或同一个值 → 判定为 false
   - 【排除信号】仅有 hour 列但无 year/month/day → 判定为 false（孤立的小时段不是时序）
   - 默认判定为 false，除非有明确的时间序列证据

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

            # 【关键修复】先进行时序判定校正，再校正 complexity
            # 原 bug：_correct_complexity 使用 LLM 原始返回的 is_time_series，
            # 导致时序任务无法触发 complexity=complex 强制升级
            is_time_series, ts_reason = self._detect_time_series(
                columns, row_count, task_description, is_time_series
            )
            if ts_reason:
                logger.info(f"[IntentRecognition] 时序判定校正: ts={is_time_series}, reason={ts_reason}")

            # 规则校正 complexity：防止 LLM（尤其是小模型）判定过于保守
            complexity, complexity_reason = self._correct_complexity(
                columns, row_count, col_count, target_column, task_type, complexity, is_time_series
            )

            logger.info(
                f"[IntentRecognition] 识别结果: target={target_column}, "
                f"type={task_type.value}, metric={eval_metric}, complexity={complexity}, ts={is_time_series}, "
                f"reason={complexity_reason}"
            )
            return IntentResult(
                target_column=target_column,
                task_type=task_type,
                eval_metric=eval_metric,
                complexity=complexity,
                is_time_series=is_time_series,
                complexity_reason=complexity_reason
            )

        except Exception as e:
            logger.error(f"[IntentRecognition] LLM 识别失败: {e}，使用规则 fallback")
            return self._rule_based_recognition(columns, task_description)

    def _correct_complexity(
        self,
        columns: List[Dict],
        row_count: int,
        col_count: int,
        target_column: str,
        task_type: TaskType,
        complexity: str,
        is_time_series: bool
    ) -> tuple:
        """
        基于数据画像规则校正 complexity，防止 LLM（尤其是 7b 小模型）判定过于保守。
        只要满足任一 complex 信号，强制升级为 complex。
        
        Returns:
            (complexity, reason): 校正后的复杂度及原因说明
        """
        if complexity == "complex":
            return "complex", "LLM直接判定为complex"

        # 规则1: 时序任务（已有时间列或 is_time_series=True）
        if is_time_series:
            reason = "时序任务强制complex"
            logger.info(f"[IntentRecognition] 规则校正: {reason}")
            return "complex", reason

        # 规则2: 多分类任务
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            reason = "多分类任务强制complex"
            logger.info(f"[IntentRecognition] 规则校正: {reason}")
            return "complex", reason

        # 规则3: 极度不平衡二分类（最大类占比 > 95% 或最小类 < 1%）
        if task_type == TaskType.BINARY_CLASSIFICATION and row_count > 0:
            for c in columns:
                if c.get("name") == target_column:
                    most_common_freq = c.get("mostCommonFreq", 0)
                    if most_common_freq is None:
                        most_common_freq = 0
                    if most_common_freq > 0:
                        imbalance_ratio = most_common_freq / row_count
                        if imbalance_ratio > 0.90:
                            reason = f"极度不平衡(最大类占比{imbalance_ratio:.1%})强制complex"
                            logger.info(f"[IntentRecognition] 规则校正: {reason}")
                            return "complex", reason
                    # 【关键增强】mostCommonFreq 缺失时，用 sampleValues 推断
                    if most_common_freq == 0:
                        samples = c.get("sampleValues", [])
                        if samples and len(samples) >= 2:
                            from collections import Counter
                            sample_counter = Counter(str(v) for v in samples)
                            if sample_counter:
                                most_common_count = sample_counter.most_common(1)[0][1]
                                inferred_ratio = most_common_count / len(samples)
                                if inferred_ratio >= 0.85:
                                    reason = f"极度不平衡(sampleValues推断最大类占比{inferred_ratio:.0%})强制complex"
                                    logger.info(f"[IntentRecognition] 规则校正: {reason}")
                                    return "complex", reason
                    break

        # 规则4: 严重缺失值（任意列缺失率 > 20%）
        if row_count > 0:
            for c in columns:
                missing_count = c.get("missingCount", 0)
                if missing_count / row_count > 0.20:
                    reason = f"列'{c['name']}'缺失率{missing_count/row_count:.1%}强制complex"
                    logger.info(f"[IntentRecognition] 规则校正: {reason}")
                    return "complex", reason

        # 规则5: 高维稀疏（列数 > 50）
        if col_count > 50:
            reason = f"高维数据({col_count}列)强制complex"
            logger.info(f"[IntentRecognition] 规则校正: {reason}")
            return "complex", reason

        return complexity, "LLM判定为simple，无复杂信号触发"

    def _detect_time_series(
        self,
        columns: List[Dict],
        row_count: int,
        task_description: str,
        llm_is_time_series: bool
    ) -> tuple:
        """
        基于数据画像规则检测时序任务，校正 LLM 的时序判断。
        
        核心原则：时序数据的关键特征是【时间维度单调递增】，
        不能仅凭列名含 "time" 就判定（如信用卡欺诈的 Time 是随机交易时刻）。
        
        改进点：
        1. 使用 isMonotonic 特征区分"时间戳"vs"时间序列索引"
        2. 使用 isDateParseable 特征识别日期字符串列
        3. 不强制排除，而是基于证据强度判断
        
        Returns:
            (is_time_series, reason)
        """
        task_desc_lower = task_description.lower() if task_description else ""
        
        # ========== 0. 任务描述强信号 ==========
        ts_keywords = ["时序回归", "时间序列", "按时间顺序", "时序预测", "time series", "temporal", "sequential"]
        if any(kw in task_desc_lower for kw in ts_keywords):
            return True, "任务描述明确涉及时序"
        
        # ========== 1. 提取候选时间列及其特征 ==========
        year_col = None
        month_col = None
        day_col = None
        hour_col = None
        dt_col = None           # 可解析为日期的列
        monotonic_time_col = None   # 单调递增的时间相关列
        monotonic_seq_col = None    # 单调递增的序列索引列
        
        for c in columns:
            name = c.get("name", "").lower()
            most_common = c.get("mostCommon", "")
            is_mono = c.get("isMonotonic", False)
            is_date = c.get("isDateParseable", False)
            unique_count = c.get("uniqueCount", 0)
            
            # 日期字符串列（通过 isDateParseable 或列名识别）
            if is_date or name in ["dteday", "date", "datetime"]:
                dt_col = c
                continue
            
            # year 列：值在 2000-2030 范围内
            if name == "year":
                try:
                    val = int(most_common) if most_common else 0
                    if 2000 <= val <= 2030:
                        year_col = c
                except:
                    pass
            
            # month 列：值在 1-12 范围内
            elif name in ["month", "mnth"]:
                try:
                    val = int(most_common) if most_common else 0
                    if 1 <= val <= 12:
                        month_col = c
                except:
                    pass
            
            # day 列：值在 1-31 范围内
            elif name == "day":
                try:
                    val = int(most_common) if most_common else 0
                    if 1 <= val <= 31:
                        day_col = c
                except:
                    pass
            
            # hour 列：值在 0-23 范围内
            elif name in ["hour", "hr"]:
                try:
                    val = int(most_common) if most_common else 0
                    if 0 <= val <= 23:
                        hour_col = c
                except:
                    pass
            
            # 【关键改进】单调递增的时间相关列
            # 包括：timestamp, time（如果是单调的，说明是时间序列索引而非随机交易时刻）
            elif name in ["timestamp", "time", "epoch", "unix_time"]:
                if is_mono and unique_count == row_count:
                    monotonic_time_col = c
            
            # 单调递增的序列索引（如 instant, No）
            elif name in ["instant", "no"]:
                if is_mono and unique_count == row_count and row_count > 100:
                    monotonic_seq_col = c
        
        # ========== 2. 强信号：完整时间戳或日期列 ==========
        if year_col and month_col:
            return True, f"有year+month{'+day' if day_col else ''}{'+hour' if hour_col else ''}构成完整时间戳"
        
        if dt_col:
            return True, f"有日期列{dt_col['name']}"
        
        # ========== 3. 【关键改进】单调递增时间戳 ==========
        # 如果 Time/Timestamp 列是单调递增的，说明是时间序列索引（如传感器数据）
        if monotonic_time_col:
            return True, f"有单调递增时间列{monotonic_time_col['name']}"
        
        # ========== 4. 弱信号：单调递增索引 + 其他时间特征 ==========
        has_season = any(c.get("name", "").lower() == "season" for c in columns)
        has_weekday = any(c.get("name", "").lower() in ["weekday", "week_day"] for c in columns)
        
        if monotonic_seq_col and (has_season or has_weekday or hour_col):
            return True, f"有单调递增索引{monotonic_seq_col['name']}+{ 'season' if has_season else ''}{ 'weekday' if has_weekday else ''}{ 'hour' if hour_col else ''}"
        
        # ========== 5. 否定信号（仅降低置信度，不强制排除） ==========
        # 5a. Time 列非单调（随机交易时刻，如信用卡欺诈）
        for c in columns:
            name = c.get("name", "").lower()
            if name in ["time", "timestamp"]:
                is_mono = c.get("isMonotonic", False)
                if not is_mono:
                    # Time 列非单调 → 不是时序
                    return False, f"{c['name']}列非单调（随机交易时刻，非时间序列）"
        
        # 5b. month 列无效（全是0）
        for c in columns:
            if c.get("name", "").lower() in ["month", "mnth"]:
                most_common = c.get("mostCommon", "")
                try:
                    val = int(most_common) if most_common else 0
                    if val == 0:
                        return False, "month列全是0（无效时间列）"
                except:
                    pass
        
        # ========== 6. 默认信任 LLM ==========
        if llm_is_time_series:
            return True, "LLM判定为时序（数据画像无明确否定信号）"
        return False, "无时间列特征"

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
                    complexity_reason="规则fallback推断",
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

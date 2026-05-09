"""
知识库加载器
根据 IntentResult（task_type + complexity）动态加载对应类型的建模建议
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from app.agents.intent_recognition import IntentResult

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent / "task_kb.yaml"


class KnowledgeBaseLoader:
    """
    任务类型知识库加载器

    使用方式：
        kb = KnowledgeBaseLoader()
        refs = kb.load_references(intent_result)
        # refs 为字符串列表，可直接注入 PlanCoding Agent prompt
    """

    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or KB_PATH
        self._kb: Optional[Dict] = None
        self._load()

    def _load(self):
        """加载 YAML 知识库"""
        if not self.kb_path.exists():
            logger.warning(f"[KnowledgeBase] 知识库文件不存在: {self.kb_path}")
            self._kb = {"task_types": {}, "general": {"antipatterns": []}}
            return
        try:
            with open(self.kb_path, "r", encoding="utf-8") as f:
                self._kb = yaml.safe_load(f) or {}
            logger.info(f"[KnowledgeBase] 知识库加载完成: {len(self._kb.get('task_types', {}))} 个类型")
        except Exception as e:
            logger.error(f"[KnowledgeBase] 加载失败: {e}")
            self._kb = {"task_types": {}, "general": {"antipatterns": []}}

    def load_references(self, intent: IntentResult) -> List[str]:
        """
        根据意图识别结果加载知识库建议

        Args:
            intent: IntentResult（包含 task_type、complexity、is_time_series）

        Returns:
            字符串列表，每条为一个建议（可直接注入 prompt）
        """
        # simple 任务不加载知识库，避免噪声
        if intent.complexity == "simple":
            return []

        refs = []
        task_types = self._kb.get("task_types", {})

        # 根据 task_type + is_time_series 匹配知识库条目
        matched_key = self._match_task_type(intent, task_types)
        if matched_key and matched_key in task_types:
            entry = task_types[matched_key]
            common_failure = entry.get("common_failure", "")
            if common_failure:
                refs.append(f"[常见失败] {common_failure}")
            for item in entry.get("must_do", []):
                refs.append(f"[必须做] {item}")
            for item in entry.get("avoid", []):
                refs.append(f"[避免] {item}")
        else:
            logger.warning(f"[KnowledgeBase] 未找到匹配的知识库条目: task_type={intent.task_type.value}, is_time_series={intent.is_time_series}")

        # 通用反模式（所有 complex 任务都加载）
        general = self._kb.get("general", {})
        antipatterns = general.get("antipatterns", [])
        for ap in antipatterns:
            refs.append(f"[通用反模式] {ap}")

        return refs

    def _match_task_type(self, intent: IntentResult, task_types: Dict) -> Optional[str]:
        """
        根据 IntentResult 匹配知识库中的任务类型 key
        
        映射规则:
        - binary + 时序 → 不存在时序二分类，按标准二分类处理
        - binary + 不平衡 → binary_classification_imbalanced
        - binary + 标准 → binary_classification_standard
        - multiclass → multiclass_classification
        - regression + 时序 → time_series_regression
        - regression + 非时序 → standard_regression
        """
        from app.models.schemas import TaskType
        
        tt = intent.task_type
        is_ts = getattr(intent, 'is_time_series', False)
        
        if tt == TaskType.REGRESSION:
            return "time_series_regression" if is_ts else "standard_regression"
        
        if tt == TaskType.MULTICLASS_CLASSIFICATION:
            return "multiclass_classification"
        
        if tt == TaskType.BINARY_CLASSIFICATION:
            # 二分类：根据 data_profile_hint 判断是否不平衡
            # 当前简化：如果有 is_time_series 标记，但不是时序二分类知识库条目
            # 默认按标准/不平衡处理。不平衡的判断需要额外的数据画像信息
            # 这里先用一个启发式：如果存在 binary_classification_imbalanced 条目，尝试匹配
            # 但区分不平衡和标准需要更多信号（如最大类占比）
            # 简化：都加载 binary_classification_imbalanced（因为标准二分类的 must_do 更宽松）
            # 或者反过来：优先加载标准，如果不平衡再覆盖？
            # 
            # 实际策略：binary_classification_imbalanced 的 triggers 里有 "最大类占比>90%"
            # 但 loader 没有数据画像，只能通过 intent 的属性判断
            # 
            # 当前方案：若 data_profile 中最大类占比>90% 或正例<10%，加载不平衡条目
            # 但 IntentResult 中没有这个数据。暂时都返回 standard，后续可扩展
            # 
            # 更合理的方案：IntentResult 增加不平衡标记，或 loader 接收更丰富的信号
            # 当前先返回标准条目，因为避免过度约束
            if "binary_classification_imbalanced" in task_types:
                return "binary_classification_imbalanced"
            return "binary_classification_standard"
        
        return None

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
            intent: IntentResult（包含 task_type 和 complexity）

        Returns:
            字符串列表，每条为一个建议（可直接注入 prompt）
        """
        # simple 任务不加载知识库，避免噪声
        if intent.complexity == "simple":
            return []

        refs = []
        task_types = self._kb.get("task_types", {})

        # TODO: 根据 task_type 匹配对应的类型条目
        # 当前为框架，匹配逻辑待实现
        _ = task_types  # suppress unused warning

        # 通用反模式（所有 complex 任务都加载）
        general = self._kb.get("general", {})
        antipatterns = general.get("antipatterns", [])
        for ap in antipatterns:
            refs.append(f"[通用反模式] {ap}")

        return refs

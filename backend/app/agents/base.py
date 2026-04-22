"""
LLM 抽象层
统一封装 OpenAI / Ollama / 本地 OpenAI 兼容服务的调用
"""

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    统一 LLM 客户端
    
    三种 provider 均通过 OpenAI 兼容格式调用：
    - openai: 云端 API（如 OpenAI, Moonshot, DeepSeek 等）
    - ollama: 本地 Ollama（兼容 /v1/chat/completions）
    - local-openai: 本地 LM Studio / vLLM 等
    """
    
    def __init__(
        self,
        provider: str = None,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        base_url = base_url or settings.LLM_BASE_URL
        api_key = api_key or settings.LLM_API_KEY
        
        # Ollama 和本地服务不需要真实 api_key，但需要占位符
        if not api_key:
            api_key = "not-needed"
        
        # 统一使用 openai 库的兼容客户端
        try:
            from openai import OpenAI
            # client 级别设置 120 秒超时，防止网络挂起
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
            logger.info(f"[LLMClient] 初始化完成: provider={self.provider}, model={self.model}, base_url={base_url}")
        except ImportError:
            logger.error("[LLMClient] 未安装 openai 库，请执行: pip install openai")
            raise
    
    @classmethod
    def from_settings(cls) -> "LLMClient":
        """从全局配置创建客户端"""
        return cls()
    
    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用 LLM 进行对话补全
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            LLM 生成的文本内容
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=120  # 每次请求 120 秒超时
            )
            content = response.choices[0].message.content
            logger.debug(f"[LLMClient] 调用成功，输出长度: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"[LLMClient] 调用失败: {e}")
            raise


class BaseAgent:
    """
    Agent 基类
    
    所有 Agent 继承此类，通过 self.llm 调用大模型
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient.from_settings()
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM"""
        return self.llm.chat_completion(system_prompt, user_prompt)

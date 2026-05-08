"""
LLM 抽象层
统一封装 OpenAI / Ollama / 本地 OpenAI 兼容服务的调用
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

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
        max_tokens: int = None,
        extra_body: Optional[Dict] = None
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.extra_body = extra_body or {}
        
        base_url = base_url or settings.LLM_BASE_URL
        api_key = api_key or settings.LLM_API_KEY
        
        # Ollama 和本地服务不需要真实 api_key，但需要占位符
        if not api_key:
            api_key = "not-needed"
        
        # 统一使用 openai 库的兼容客户端
        try:
            from openai import OpenAI
            import httpx
            # 【修复】分别设置连接/读取/写入超时，增加连接超时防止 ConnectTimeout
            timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
            logger.info(f"[LLMClient] 初始化完成: provider={self.provider}, model={self.model}, base_url={base_url}, timeout={timeout}")
        except ImportError:
            logger.error("[LLMClient] 未安装 openai 库，请执行: pip install openai")
            raise
        except ImportError:
            logger.error("[LLMClient] 未安装 openai 库，请执行: pip install openai")
            raise
    
    @classmethod
    def from_settings(cls) -> "LLMClient":
        """从全局配置创建客户端"""
        return cls()
    
    def chat_completion(self, system_prompt: str, user_prompt: str, max_retries: int = 5) -> tuple:
        """
        调用 LLM 进行对话补全（带重试）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            max_retries: 最大重试次数（默认5次）
            
        Returns:
            (LLM 生成的文本内容, LLMUsageInfo)
        """
        import time as _time
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 对 deepseek 模型默认关闭 thinking（避免 JSON 被思考内容截断）
        extra_body = dict(self.extra_body) if self.extra_body else {}
        if "deepseek" in self.model.lower() and "enable_thinking" not in extra_body:
            extra_body["enable_thinking"] = False
        
        last_error = None
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # 使用 client 级别的 timeout，不再每次覆盖
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                
                # 提取 token 消耗
                usage = getattr(response, 'usage', None)
                from app.models.schemas import LLMUsageInfo
                usage_info = LLMUsageInfo(
                    prompt_tokens=getattr(usage, 'prompt_tokens', 0) if usage else 0,
                    completion_tokens=getattr(usage, 'completion_tokens', 0) if usage else 0,
                    total_tokens=getattr(usage, 'total_tokens', 0) if usage else 0,
                    provider=self.provider,
                    model=self.model
                )
                
                if attempt > 0:
                    logger.info(f"[LLMClient] 第 {attempt+1} 次重试成功")
                
                logger.debug(
                    f"[LLMClient] 调用成功，输出长度: {len(content)}, "
                    f"tokens: prompt={usage_info.prompt_tokens}, completion={usage_info.completion_tokens}, total={usage_info.total_tokens}"
                )
                return content, usage_info
            except Exception as e:
                last_error = e
                logger.warning(f"[LLMClient] 第 {attempt+1}/{max_retries} 次调用失败: {e}")
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt, 16)  # 指数退避: 1s, 2s, 4s, 8s, 16s，上限16秒
                    logger.info(f"[LLMClient] {wait}秒后重试...")
                    _time.sleep(wait)
        
        logger.error(f"[LLMClient] 调用失败，已重试 {max_retries} 次: {last_error}")
        raise last_error


class BaseAgent:
    """
    Agent 基类
    
    所有 Agent 继承此类，通过 self.llm 调用大模型
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient.from_settings()
        self._usage_records: List = []  # 记录每次 LLM 调用的 token 消耗
        self._llm_call_logs: List[Dict] = []  # 详细记录每次 LLM 调用的 prompt/response
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM，自动记录 token 消耗和完整对话日志，返回 content"""
        from datetime import datetime
        content, usage_info = self.llm.chat_completion(system_prompt, user_prompt)
        self._usage_records.append(usage_info)
        self._llm_call_logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": content,
            "usage": {
                "prompt_tokens": usage_info.prompt_tokens,
                "completion_tokens": usage_info.completion_tokens,
                "total_tokens": usage_info.total_tokens,
                "provider": usage_info.provider,
                "model": usage_info.model
            }
        })
        return content
    
    def get_usage_summary(self) -> Dict[str, int]:
        """获取累计 token 消耗汇总"""
        total_prompt = sum(u.prompt_tokens for u in self._usage_records)
        total_completion = sum(u.completion_tokens for u in self._usage_records)
        total = sum(u.total_tokens for u in self._usage_records)
        return {
            "call_count": len(self._usage_records),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total
        }
    
    def reset_usage(self):
        """重置 token 消耗记录"""
        self._usage_records = []
    
    def get_llm_call_logs(self) -> List[Dict]:
        """获取所有 LLM 调用日志"""
        return self._llm_call_logs
    
    def clear_llm_call_logs(self):
        """清空 LLM 调用日志"""
        self._llm_call_logs = []
    
    def save_llm_logs_to_dir(self, log_dir: Path, agent_name: str = "agent"):
        """将所有 LLM 调用日志保存到指定目录"""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for i, log in enumerate(self._llm_call_logs):
            prefix = f"{agent_name}_call_{i+1:02d}"
            # system prompt
            (log_dir / f"{prefix}_system_prompt.txt").write_text(
                log.get("system_prompt", ""), encoding='utf-8'
            )
            # user prompt
            (log_dir / f"{prefix}_user_prompt.txt").write_text(
                log.get("user_prompt", ""), encoding='utf-8'
            )
            # response
            (log_dir / f"{prefix}_response.txt").write_text(
                log.get("response", ""), encoding='utf-8'
            )
            # metadata
            meta = {
                "timestamp": log.get("timestamp"),
                "usage": log.get("usage")
            }
            (log_dir / f"{prefix}_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
            )

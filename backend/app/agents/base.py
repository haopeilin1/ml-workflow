"""
LLM 抽象层
统一封装 OpenAI / Ollama / 本地 OpenAI 兼容服务的调用
支持多层 Fallback 机制：主模型 → 本地 VLLM → 阿里云 → 循环回主模型
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class _FallbackClient:
    """Fallback 客户端封装"""
    def __init__(self, client, model: str, temperature: float, max_tokens: int,
                 extra_body: Optional[Dict], name: str, timeout: int):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.name = name
        self.timeout = timeout


class LLMClient:
    """
    统一 LLM 客户端（支持三层 Fallback + 快速连接重试）
    
    Fallback 层级（确保一定能跑通）：
    1. 主模型（外部 API）：connect=5s, read=180s
       - 连接失败 → 快速重试 1 次 → 仍失败 → 切 fallback1
       - 读取超时 → 切 fallback1
    2. Fallback1（本地 VLLM）
       - 超时 → 切 fallback2
    3. Fallback2（阿里云）
       - 超时 → 指数退避 → 下一次 attempt 循环回主模型
    """
    
    def __init__(
        self,
        provider: str = None,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        extra_body: Optional[Dict] = None,
        fallback_config: Optional[Dict] = None
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.extra_body = extra_body or {}
        
        base_url = base_url or settings.LLM_BASE_URL
        api_key = api_key or settings.LLM_API_KEY
        
        if not api_key:
            api_key = "not-needed"
        
        # 主客户端：connect=5s（快速连接），read=180s（正常读取）
        try:
            from openai import OpenAI
            import httpx
            primary_timeout = httpx.Timeout(
                connect=float(settings.EXTERNAL_API_CONNECT_TIMEOUT),
                read=float(settings.EXTERNAL_API_READ_TIMEOUT),
                write=30.0,
                pool=30.0
            )
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=primary_timeout)
            self.base_url = base_url
            logger.info(
                f"[LLMClient] 主模型初始化: model={self.model}, "
                f"base_url={base_url}, connect={settings.EXTERNAL_API_CONNECT_TIMEOUT}s, "
                f"read={settings.EXTERNAL_API_READ_TIMEOUT}s"
            )
        except ImportError:
            logger.error("[LLMClient] 未安装 openai 库，请执行: pip install openai")
            raise
        
        # 初始化 fallback 客户端
        self.fallback1: Optional[_FallbackClient] = None
        self.fallback2: Optional[_FallbackClient] = None
        self._init_fallbacks(fallback_config)
    
    def _build_fallback(self, level: int, fallback_config: Optional[Dict]) -> Optional[_FallbackClient]:
        """构建单个 fallback 客户端"""
        cfg = fallback_config or {}
        prefix = "FALLBACK_LLM_" if level == 1 else "FALLBACK_LLM2_"
        
        fb_provider = cfg.get("provider") or getattr(settings, f"{prefix}PROVIDER", "")
        fb_base_url = cfg.get("base_url") or getattr(settings, f"{prefix}BASE_URL", "")
        fb_api_key = cfg.get("api_key") or getattr(settings, f"{prefix}API_KEY", "")
        fb_model = cfg.get("model") or getattr(settings, f"{prefix}MODEL", "")
        fb_temperature = cfg.get("temperature")
        if fb_temperature is None:
            val = getattr(settings, f"{prefix}TEMPERATURE", -1.0)
            fb_temperature = val if val >= 0 else None
        fb_max_tokens = cfg.get("max_tokens") or getattr(settings, f"{prefix}MAX_TOKENS", 0)
        fb_max_tokens = fb_max_tokens if fb_max_tokens > 0 else None
        fb_extra_body = cfg.get("extra_body")
        if fb_extra_body is None:
            body_str = getattr(settings, f"{prefix}EXTRA_BODY", "")
            if body_str:
                try:
                    fb_extra_body = json.loads(body_str)
                except json.JSONDecodeError:
                    pass
        fb_timeout = getattr(settings, f"{prefix}TIMEOUT", 180)
        
        if not fb_base_url or not fb_model:
            return None
        
        if not fb_api_key:
            fb_api_key = "not-needed"
        
        try:
            from openai import OpenAI
            import httpx
            timeout = httpx.Timeout(connect=5.0, read=fb_timeout, write=30.0, pool=30.0)
            client = OpenAI(base_url=fb_base_url, api_key=fb_api_key, timeout=timeout)
            name = "fallback1-local" if level == 1 else "fallback2-aliyun"
            logger.info(
                f"[LLMClient] {name} 初始化: model={fb_model}, "
                f"base_url={fb_base_url}, timeout={timeout}"
            )
            return _FallbackClient(
                client=client,
                model=fb_model,
                temperature=fb_temperature if fb_temperature is not None else self.temperature,
                max_tokens=fb_max_tokens or self.max_tokens,
                extra_body=fb_extra_body or {},
                name=name,
                timeout=fb_timeout
            )
        except Exception as e:
            logger.warning(f"[LLMClient] Fallback{level} 初始化失败: {e}")
            return None
    
    def _init_fallbacks(self, fallback_config: Optional[Dict] = None):
        """初始化所有 fallback 客户端"""
        self.fallback1 = self._build_fallback(1, fallback_config)
        self.fallback2 = self._build_fallback(2, fallback_config)
    
    @classmethod
    def from_settings(cls) -> "LLMClient":
        """从全局配置创建客户端"""
        return cls()
    
    def _do_chat_completion(
        self,
        client,
        model: str,
        temperature: float,
        max_tokens: int,
        extra_body: Optional[Dict],
        messages: List[Dict],
        is_fallback: bool = False,
        fallback_name: str = ""
    ) -> tuple:
        """执行单次 LLM 调用，返回 (content, usage_info)"""
        extra = dict(extra_body) if extra_body else {}
        model_lower = model.lower()
        is_reasoning = any(k in model_lower for k in ("deepseek", "qwq", "r1", "reasoner", "qwen"))
        if is_reasoning:
            base_url_str = str(client.base_url) if hasattr(client, 'base_url') else ""
            if "deepseek.com" in base_url_str:
                if "thinking" not in extra:
                    extra["thinking"] = {"type": "disabled"}
            elif "enable_thinking" not in extra:
                extra["enable_thinking"] = False
        
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if extra:
            kwargs["extra_body"] = extra
        
        import time as _time
        start = _time.time()
        response = client.chat.completions.create(**kwargs)
        elapsed = _time.time() - start
        
        content = response.choices[0].message.content
        
        if content and "<think>" in content:
            import re as _re
            original_len = len(content)
            content = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()
            if len(content) < original_len:
                logger.info(f"[LLMClient] 已过滤 think 标签: {original_len - len(content)} 字符")
        
        usage = getattr(response, 'usage', None)
        from app.models.schemas import LLMUsageInfo
        usage_info = LLMUsageInfo(
            prompt_tokens=getattr(usage, 'prompt_tokens', 0) if usage else 0,
            completion_tokens=getattr(usage, 'completion_tokens', 0) if usage else 0,
            total_tokens=getattr(usage, 'total_tokens', 0) if usage else 0,
            provider=self.provider if not is_fallback else fallback_name,
            model=model,
            latency_seconds=round(elapsed, 2)
        )
        
        prefix = f"[{fallback_name}] " if fallback_name else ""
        logger.debug(
            f"[LLMClient] {prefix}调用成功，耗时={elapsed:.1f}s，"
            f"tokens: prompt={usage_info.prompt_tokens}, completion={usage_info.completion_tokens}"
        )
        return content, usage_info
    
    def _try_single_call(
        self,
        client,
        model: str,
        temperature: float,
        max_tokens: int,
        extra_body: Optional[Dict],
        messages: List[Dict],
        fallback_name: str = ""
    ) -> Tuple[bool, Optional[str], Optional[object], Optional[Exception]]:
        """
        尝试单次 LLM 调用，返回 (是否成功, content, usage_info, 错误)
        """
        try:
            content, usage = self._do_chat_completion(
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                messages=messages,
                is_fallback=bool(fallback_name),
                fallback_name=fallback_name
            )
            return True, content, usage, None
        except Exception as e:
            return False, None, None, e
    
    def _classify_error(self, error: Exception) -> Tuple[bool, bool, bool]:
        """
        分类错误类型
        返回: (是否连接超时, 是否读取超时, 是否其他错误)
        """
        error_str = str(error).lower()
        is_connect = "connecttimeout" in error_str or ("connect" in error_str and "timeout" in error_str)
        is_read = "readtimeout" in error_str or ("read" in error_str and "timeout" in error_str)
        is_other = not is_connect and not is_read
        return is_connect, is_read, is_other
    
    def chat_completion(self, system_prompt: str, user_prompt: str, max_retries: int = 5) -> tuple:
        """
        调用 LLM 进行对话补全（三层 Fallback + 快速连接重试 + 循环兜底）
        
        调用链（每个 attempt）：
        1. 主模型 → 连接失败则快速重试 → 读取超时则切 fallback1
        2. Fallback1（本地） → 超时则切 fallback2
        3. Fallback2（阿里云） → 超时则指数退避，下一次 attempt 循环回主模型
        
        返回: (content, LLMUsageInfo)
        """
        import time as _time
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        last_error = None
        
        for attempt in range(max_retries):
            # ========== 第 1 步：主模型（带快速连接重试）==========
            logger.debug(f"[LLMClient] Attempt {attempt+1}/{max_retries}: 尝试主模型 {self.model}")
            
            success, content, usage, error = self._try_single_call(
                client=self.client,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=self.extra_body,
                messages=messages,
                fallback_name=""
            )
            if success:
                if attempt > 0:
                    logger.info(f"[LLMClient] 第 {attempt+1} 次重试成功（主模型）")
                return content, usage
            
            last_error = error
            is_connect, is_read, is_other = self._classify_error(error)
            
            # 连接超时：快速重试
            if is_connect:
                for fast_retry in range(settings.EXTERNAL_API_FAST_RETRIES):
                    logger.warning(
                        f"[LLMClient] 主模型连接超时（{settings.EXTERNAL_API_CONNECT_TIMEOUT}s），"
                        f"快速重试 {fast_retry+1}/{settings.EXTERNAL_API_FAST_RETRIES}..."
                    )
                    success, content, usage, error = self._try_single_call(
                        client=self.client, model=self.model,
                        temperature=self.temperature, max_tokens=self.max_tokens,
                        extra_body=self.extra_body, messages=messages
                    )
                    if success:
                        logger.info(f"[LLMClient] 主模型快速重试成功")
                        return content, usage
                logger.warning("[LLMClient] 主模型快速重试均失败，转入 fallback1")
            elif is_read:
                logger.warning(
                    f"[LLMClient] 主模型读取超时（{settings.EXTERNAL_API_READ_TIMEOUT}s），"
                    f"转入 fallback1"
                )
            
            # ========== 第 2 步：Fallback1（本地 VLLM）==========
            if self.fallback1:
                logger.info(f"[LLMClient] 尝试 fallback1: {self.fallback1.model}")
                success, content, usage, error = self._try_single_call(
                    client=self.fallback1.client,
                    model=self.fallback1.model,
                    temperature=self.fallback1.temperature,
                    max_tokens=self.fallback1.max_tokens,
                    extra_body=self.fallback1.extra_body,
                    messages=messages,
                    fallback_name=self.fallback1.name
                )
                if success:
                    logger.info(f"[LLMClient] fallback1 调用成功")
                    return content, usage
                
                last_error = error
                is_connect, is_read, _ = self._classify_error(error)
                if is_connect:
                    logger.warning(f"[LLMClient] fallback1 连接超时")
                elif is_read:
                    logger.warning(f"[LLMClient] fallback1 读取超时（{self.fallback1.timeout}s）")
                else:
                    logger.warning(f"[LLMClient] fallback1 调用失败: {error}")
            
            # ========== 第 3 步：Fallback2（阿里云）==========
            if self.fallback2:
                logger.info(f"[LLMClient] 尝试 fallback2: {self.fallback2.model}")
                success, content, usage, error = self._try_single_call(
                    client=self.fallback2.client,
                    model=self.fallback2.model,
                    temperature=self.fallback2.temperature,
                    max_tokens=self.fallback2.max_tokens,
                    extra_body=self.fallback2.extra_body,
                    messages=messages,
                    fallback_name=self.fallback2.name
                )
                if success:
                    logger.info(f"[LLMClient] fallback2 调用成功")
                    return content, usage
                
                last_error = error
                is_connect, is_read, _ = self._classify_error(error)
                if is_connect:
                    logger.warning(f"[LLMClient] fallback2 连接超时")
                elif is_read:
                    logger.warning(f"[LLMClient] fallback2 读取超时（{self.fallback2.timeout}s）")
                else:
                    logger.warning(f"[LLMClient] fallback2 调用失败: {error}")
            
            # ========== 第 4 步：指数退避，准备下一次 attempt ==========
            if attempt < max_retries - 1:
                if settings.FALLBACK_CYCLE_ENABLED and (self.fallback1 or self.fallback2):
                    logger.info(
                        f"[LLMClient] 所有模型均失败，{min(2**attempt, 16)}s 后"
                        f"循环回主模型重试（attempt {attempt+2}/{max_retries}）..."
                    )
                else:
                    logger.info(
                        f"[LLMClient] 所有模型均失败，{min(2**attempt, 16)}s 后重试..."
                    )
                _time.sleep(min(2 ** attempt, 16))
        
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
        """调用 LLM，自动记录 token 消耗、延迟和完整对话日志，返回 content"""
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
                "model": usage_info.model,
                "latency_seconds": usage_info.latency_seconds
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
        """将所有 LLM 调用日志保存到指定目录（含完整元信息）"""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for i, log in enumerate(self._llm_call_logs):
            prefix = f"{agent_name}_call_{i+1:02d}"
            (log_dir / f"{prefix}_system_prompt.txt").write_text(
                log.get("system_prompt", ""), encoding='utf-8'
            )
            (log_dir / f"{prefix}_user_prompt.txt").write_text(
                log.get("user_prompt", ""), encoding='utf-8'
            )
            (log_dir / f"{prefix}_response.txt").write_text(
                log.get("response", ""), encoding='utf-8'
            )
            usage = log.get("usage", {})
            meta = {
                "timestamp": log.get("timestamp"),
                "model": usage.get("model", "unknown"),
                "provider": usage.get("provider", "unknown"),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "latency_seconds": usage.get("latency_seconds", 0.0)
            }
            (log_dir / f"{prefix}_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
            )

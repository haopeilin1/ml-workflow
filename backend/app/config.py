"""
应用配置
支持环境变量覆盖
"""

import sys
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # 应用基础配置
    APP_NAME: str = "ML Agent Backend"
    DEBUG: bool = False
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    
    # LLM 配置（可被环境变量覆盖）
    LLM_PROVIDER: str = "openai"  # openai | ollama | local-openai
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: int = 300  # httpx read timeout (秒)，硅基流动长prompt响应慢
    
    # 评测系统 PlanCoding Agent LLM 配置（空字符串则回退到全局 LLM 配置）
    # 这是复杂任务和简单任务的共同父级回退配置
    EVAL_PLAN_CODING_PROVIDER: str = ""
    EVAL_PLAN_CODING_BASE_URL: str = ""
    EVAL_PLAN_CODING_API_KEY: str = ""
    EVAL_PLAN_CODING_MODEL: str = ""
    EVAL_PLAN_CODING_TEMPERATURE: float = -1.0  # <0 表示使用全局
    EVAL_PLAN_CODING_MAX_TOKENS: int = 0       # 0 表示使用全局
    EVAL_PLAN_CODING_EXTRA_BODY: str = ""      # JSON 格式额外参数，如 {"enable_thinking": false}
    
    # 【新增】Plan Agent 专用 LLM 配置（复杂任务第一步：生成结构化计划）
    # 回退链：EVAL_PLAN_* → EVAL_PLAN_CODING_* → LLM_*
    EVAL_PLAN_PROVIDER: str = ""
    EVAL_PLAN_BASE_URL: str = ""
    EVAL_PLAN_API_KEY: str = ""
    EVAL_PLAN_MODEL: str = ""
    EVAL_PLAN_TEMPERATURE: float = -1.0
    EVAL_PLAN_MAX_TOKENS: int = 0
    EVAL_PLAN_EXTRA_BODY: str = ""
    
    # 【新增】Coding Agent 专用 LLM 配置（复杂任务第二步：写代码）
    # 回退链：EVAL_CODING_* → EVAL_PLAN_CODING_* → LLM_*
    EVAL_CODING_PROVIDER: str = ""
    EVAL_CODING_BASE_URL: str = ""
    EVAL_CODING_API_KEY: str = ""
    EVAL_CODING_MODEL: str = ""
    EVAL_CODING_TEMPERATURE: float = -1.0
    EVAL_CODING_MAX_TOKENS: int = 0
    EVAL_CODING_EXTRA_BODY: str = ""
    
    # 【新增】Unified Agent 专用 LLM 配置（简单任务：单步 PlanCoding）
    # 回退链：EVAL_UNIFIED_* → EVAL_PLAN_CODING_* → LLM_*
    EVAL_UNIFIED_PROVIDER: str = ""
    EVAL_UNIFIED_BASE_URL: str = ""
    EVAL_UNIFIED_API_KEY: str = ""
    EVAL_UNIFIED_MODEL: str = ""
    EVAL_UNIFIED_TEMPERATURE: float = -1.0
    EVAL_UNIFIED_MAX_TOKENS: int = 0
    EVAL_UNIFIED_EXTRA_BODY: str = ""
    
    # 评测系统 Judge Agent LLM 配置（空字符串则回退到全局 LLM 配置）
    EVAL_JUDGE_PROVIDER: str = ""
    EVAL_JUDGE_BASE_URL: str = ""
    EVAL_JUDGE_API_KEY: str = ""
    EVAL_JUDGE_MODEL: str = ""
    EVAL_JUDGE_TEMPERATURE: float = -1.0
    EVAL_JUDGE_MAX_TOKENS: int = 0
    EVAL_JUDGE_EXTRA_BODY: str = ""            # JSON 格式额外参数
    
    # 评测系统 Intent Recognition Agent LLM 配置（空字符串则回退到全局 LLM 配置）
    EVAL_INTENT_PROVIDER: str = ""
    EVAL_INTENT_BASE_URL: str = ""
    EVAL_INTENT_API_KEY: str = ""
    EVAL_INTENT_MODEL: str = ""
    EVAL_INTENT_TEMPERATURE: float = -1.0
    EVAL_INTENT_MAX_TOKENS: int = 0
    EVAL_INTENT_EXTRA_BODY: str = ""           # JSON 格式额外参数
    
    # 【新增】Evaluation Agent 专用 LLM 配置（FastEngine 内部代码评审/优化决策）
    # 回退链：EVAL_EVALUATION_* → EVAL_PLAN_CODING_* → LLM_*
    EVAL_EVALUATION_PROVIDER: str = ""
    EVAL_EVALUATION_BASE_URL: str = ""
    EVAL_EVALUATION_API_KEY: str = ""
    EVAL_EVALUATION_MODEL: str = ""
    EVAL_EVALUATION_TEMPERATURE: float = -1.0
    EVAL_EVALUATION_MAX_TOKENS: int = 0
    EVAL_EVALUATION_EXTRA_BODY: str = ""       # JSON 格式额外参数
    
    # ========== Fallback LLM 配置（多层兜底，确保一定能跑通）==========
    # 主模型连接策略
    EXTERNAL_API_CONNECT_TIMEOUT: int = 5    # 快速连接超时（秒），连不上立即重试
    EXTERNAL_API_READ_TIMEOUT: int = 180     # 读取超时（秒），超过则转入 fallback
    EXTERNAL_API_FAST_RETRIES: int = 1       # 连接失败后的快速重试次数
    
    # Fallback 1：本地 VLLM（第一层兜底）
    FALLBACK_LLM_PROVIDER: str = ""
    FALLBACK_LLM_BASE_URL: str = ""
    FALLBACK_LLM_API_KEY: str = ""
    FALLBACK_LLM_MODEL: str = ""
    FALLBACK_LLM_TEMPERATURE: float = -1.0
    FALLBACK_LLM_MAX_TOKENS: int = 0
    FALLBACK_LLM_EXTRA_BODY: str = ""
    FALLBACK_LLM_TIMEOUT: int = 180          # fallback1 读取超时（秒）
    
    # Fallback 2：阿里云（第二层兜底）
    FALLBACK_LLM2_PROVIDER: str = ""
    FALLBACK_LLM2_BASE_URL: str = ""
    FALLBACK_LLM2_API_KEY: str = ""
    FALLBACK_LLM2_MODEL: str = ""
    FALLBACK_LLM2_TEMPERATURE: float = -1.0
    FALLBACK_LLM2_MAX_TOKENS: int = 0
    FALLBACK_LLM2_EXTRA_BODY: str = ""
    FALLBACK_LLM2_TIMEOUT: int = 180         # fallback2 读取超时（秒）
    
    # 循环策略：fallback2 也超时后是否循环回主模型继续尝试
    FALLBACK_CYCLE_ENABLED: bool = True
    
    # 多模态模型配置（阿里云，用于生成可视化产物）
    MULTIMODAL_LLM_PROVIDER: str = ""
    MULTIMODAL_LLM_BASE_URL: str = ""
    MULTIMODAL_LLM_API_KEY: str = ""
    MULTIMODAL_LLM_MODEL: str = ""
    MULTIMODAL_LLM_TEMPERATURE: float = -1.0
    MULTIMODAL_LLM_MAX_TOKENS: int = 0
    MULTIMODAL_LLM_EXTRA_BODY: str = ""
    
    # 沙箱配置
    SANDBOX_TIMEOUT: int = 300  # 秒
    SANDBOX_MEMORY_LIMIT: str = "2g"
    SANDBOX_CPU_LIMIT: float = 2.0
    # 优先使用项目 venv 中的 Python（如果存在），否则使用当前解释器
    PYTHON_EXECUTABLE: str = str(
        Path(__file__).parent.parent / "venv" / "bin" / "python"
    ) if (Path(__file__).parent.parent / "venv" / "bin" / "python").exists() else sys.executable
    
    # 快速模式限制
    FAST_MAX_OPTIMIZE_ROUNDS: int = 3
    FAST_MAX_DEBUG_ROUNDS: int = 5
    FAST_MAX_USER_FEEDBACK_ROUNDS: int = 3
    
    # 数据切分配置
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

def _get_or_fallback(value, fallback):
    """如果值为空/0/-1，则回退到 fallback"""
    if value in (None, "", 0) or (isinstance(value, float) and value < 0):
        return fallback
    return value


def build_eval_llm_config(which: str, fallback_chain: list = None):
    """
    从 settings 构建评测用 LLMConfig 字典。
    
    Args:
        which: 'plan_coding' | 'plan' | 'coding' | 'simple' | 'judge' | 'intent'
        fallback_chain: 自定义回退链，如 ['plan_coding'] 表示 which 没配置时回退到 plan_coding
    
    默认回退链：
        plan → plan_coding → global
        coding → plan_coding → global
        simple → plan_coding → global
        plan_coding/judge/intent → global
    """
    import json
    
    # 确定回退链
    if fallback_chain is None:
        if which in ('plan', 'coding', 'unified'):
            fallback_chain = ['plan_coding']
        else:
            fallback_chain = []
    
    # 收集所有候选配置（按优先级）
    candidates = [which] + fallback_chain
    
    # 按优先级尝试每个候选
    for candidate in candidates:
        prefix = f"EVAL_{candidate.upper()}_"
        provider = getattr(settings, f"{prefix}PROVIDER", "")
        base_url = getattr(settings, f"{prefix}BASE_URL", "")
        api_key = getattr(settings, f"{prefix}API_KEY", "")
        model = getattr(settings, f"{prefix}MODEL", "")
        temperature = getattr(settings, f"{prefix}TEMPERATURE", -1.0)
        max_tokens = getattr(settings, f"{prefix}MAX_TOKENS", 0)
        extra_body_str = getattr(settings, f"{prefix}EXTRA_BODY", "")
        
        # 检查是否有有效配置
        has_config = any([
            provider, base_url, api_key, model,
            (isinstance(temperature, float) and temperature >= 0),
            (isinstance(max_tokens, int) and max_tokens > 0),
            extra_body_str
        ])
        
        if has_config:
            config = {
                "provider": _get_or_fallback(provider, settings.LLM_PROVIDER),
                "base_url": _get_or_fallback(base_url, settings.LLM_BASE_URL),
                "api_key": _get_or_fallback(api_key, settings.LLM_API_KEY),
                "model": _get_or_fallback(model, settings.LLM_MODEL),
                "temperature": _get_or_fallback(temperature, settings.LLM_TEMPERATURE),
                "max_tokens": _get_or_fallback(max_tokens, settings.LLM_MAX_TOKENS),
            }
            if extra_body_str:
                try:
                    extra_body = json.loads(extra_body_str)
                    if isinstance(extra_body, dict):
                        config["extra_body"] = extra_body
                except json.JSONDecodeError:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"[{prefix}EXTRA_BODY] JSON 解析失败，已忽略: {extra_body_str[:100]}"
                    )
            return config
    
    # 所有候选都没配置，回退到全局
    config = {
        "provider": settings.LLM_PROVIDER,
        "base_url": settings.LLM_BASE_URL,
        "api_key": settings.LLM_API_KEY,
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }
    return config


# 确保目录存在
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

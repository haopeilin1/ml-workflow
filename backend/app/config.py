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
    
    # 评测系统 PlanCoding Agent LLM 配置（空字符串则回退到全局 LLM 配置）
    EVAL_PLAN_CODING_PROVIDER: str = ""
    EVAL_PLAN_CODING_BASE_URL: str = ""
    EVAL_PLAN_CODING_API_KEY: str = ""
    EVAL_PLAN_CODING_MODEL: str = ""
    EVAL_PLAN_CODING_TEMPERATURE: float = -1.0  # <0 表示使用全局
    EVAL_PLAN_CODING_MAX_TOKENS: int = 0       # 0 表示使用全局
    EVAL_PLAN_CODING_EXTRA_BODY: str = ""      # JSON 格式额外参数，如 {"enable_thinking": false}
    
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
    
    # 沙箱配置
    SANDBOX_TIMEOUT: int = 300  # 秒
    SANDBOX_MEMORY_LIMIT: str = "2g"
    SANDBOX_CPU_LIMIT: float = 2.0
    PYTHON_EXECUTABLE: str = sys.executable
    
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


def build_eval_llm_config(which: str):
    """
    从 settings 构建评测用 LLMConfig 字典。
    which: 'plan_coding' | 'judge' | 'intent'
    空值自动回退到全局 LLM 配置。
    """
    import json
    prefix = f"EVAL_{which.upper()}_"
    config = {
        "provider": _get_or_fallback(getattr(settings, f"{prefix}PROVIDER"), settings.LLM_PROVIDER),
        "base_url": _get_or_fallback(getattr(settings, f"{prefix}BASE_URL"), settings.LLM_BASE_URL),
        "api_key": _get_or_fallback(getattr(settings, f"{prefix}API_KEY"), settings.LLM_API_KEY),
        "model": _get_or_fallback(getattr(settings, f"{prefix}MODEL"), settings.LLM_MODEL),
        "temperature": _get_or_fallback(getattr(settings, f"{prefix}TEMPERATURE"), settings.LLM_TEMPERATURE),
        "max_tokens": _get_or_fallback(getattr(settings, f"{prefix}MAX_TOKENS"), settings.LLM_MAX_TOKENS),
    }
    # 解析 extra_body JSON
    extra_body_str = getattr(settings, f"{prefix}EXTRA_BODY", "")
    if extra_body_str:
        try:
            extra_body = json.loads(extra_body_str)
            if isinstance(extra_body, dict):
                config["extra_body"] = extra_body
        except json.JSONDecodeError:
            import logging
            logging.getLogger(__name__).warning(f"[{prefix}EXTRA_BODY] JSON 解析失败，已忽略: {extra_body_str[:100]}")
    return config


# 确保目录存在
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# 确保目录存在
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

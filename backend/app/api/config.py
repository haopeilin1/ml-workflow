"""
系统配置 API
用于前端动态获取后端配置（如 LLM 配置）
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter(prefix="/api/config", tags=["系统配置"])


class LLMConfigResponse(BaseModel):
    enabled: bool = True
    provider: str
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    message: str = ""


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_config():
    """
    获取后端 LLM 配置，供前端自动填充使用。
    
    安全说明：
    - 仅当 ALLOW_FRONTEND_LLM_CONFIG=true 时才会返回 api_key
    - 生产环境应关闭此开关，强制用户在前端手动配置
    """
    allowed = settings.ALLOW_FRONTEND_LLM_CONFIG
    
    return LLMConfigResponse(
        enabled=True,
        provider=settings.LLM_PROVIDER,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY if allowed else "",
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        message="配置已同步" if allowed else "后端配置同步已关闭，请手动配置 LLM"
    )

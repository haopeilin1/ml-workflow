"""
LLM 代理 API
前端不直接调用外部 LLM，而是通过后端代理，解决 CORS 和 API Key 暴露问题
"""

import json
import logging
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.agents.base import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM 代理"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    stream: bool = False


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str = "llm-proxy-0"
    object: str = "chat.completion"
    choices: List[ChatChoice]


@router.post("/chat", response_model=ChatResponse)
async def chat_proxy(request: ChatRequest):
    """
    LLM 聊天代理接口
    
    前端将请求发给后端，后端使用配置好的 LLM（含 fallback）代为调用，
    避免前端直接暴露 API Key 和遇到 CORS 问题。
    
    支持所有配置在 .env 中的 LLM  provider（openai / ollama / local-openai）。
    """
    try:
        # 使用后端配置的 LLM 客户端
        llm = LLMClient(
            model=request.model or settings.LLM_MODEL,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # 提取 system prompt 和 user prompt
        system_prompt = ""
        user_messages = []
        
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_messages.append(msg.content)
            elif msg.role == "assistant":
                user_messages.append(f"[AI]: {msg.content}")
        
        # 合并 user prompt（多轮对话拼接）
        user_prompt = "\n".join(user_messages) if user_messages else "Hello"
        
        # 调用 LLM（返回 (content, usage) tuple）
        result = llm.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        content = result[0] if isinstance(result, tuple) else result
        
        if not content:
            raise HTTPException(status_code=503, detail="LLM 服务暂时不可用，请稍后重试")
        
        return ChatResponse(
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=content)
                )
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LLM Proxy] 调用失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")

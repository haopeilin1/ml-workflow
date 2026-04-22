"""
FastAPI 应用入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api import files, tasks


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="ML Agent 后端 API - 支持快速模式与深度模式的自动化机器学习建模",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS 配置（允许前端跨域访问）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制为前端域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(files.router)
    app.include_router(tasks.router)
    
    # 挂载产物文件目录，供前端下载
    app.mount("/artifacts", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="artifacts")
    
    @app.get("/")
    async def root():
        return {
            "app": settings.APP_NAME,
            "status": "running",
            "version": "0.1.0"
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

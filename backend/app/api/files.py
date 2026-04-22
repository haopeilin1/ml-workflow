"""
文件服务 API
- 上传数据文件
- 下载产物文件
- 查询已上传文件列表
"""

import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import FileUploadResponse, FileRole, UploadedFile

router = APIRouter(prefix="/api/files", tags=["文件服务"])


def detect_file_role(filename: str) -> FileRole:
    """根据文件名关键词自动识别文件角色"""
    name_lower = filename.lower()
    
    train_keywords = ["train", "training", "训练", "学习", "learn"]
    test_keywords = ["test", "testing", "预测", "predict", "submission"]
    val_keywords = ["val", "validation", "valid", "验证", "dev"]
    
    for kw in train_keywords:
        if kw in name_lower:
            return FileRole.TRAIN
    for kw in test_keywords:
        if kw in name_lower:
            return FileRole.TEST
    for kw in val_keywords:
        if kw in name_lower:
            return FileRole.VALIDATION
    
    return FileRole.UNKNOWN


@router.post("/upload", response_model=List[FileUploadResponse])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    上传数据文件（CSV / XLSX / XLS）
    自动根据文件名识别文件角色
    """
    responses = []
    
    for file in files:
        # 生成唯一文件名避免冲突
        file_id = uuid.uuid4().hex[:16]
        original_name = file.filename or "unknown"
        suffix = Path(original_name).suffix.lower()
        safe_name = f"{file_id}{suffix}"
        save_path = settings.UPLOAD_DIR / safe_name
        
        # 保存文件
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 自动识别角色
        role = detect_file_role(original_name)
        
        responses.append(FileUploadResponse(
            file_id=file_id,
            name=original_name,
            role=role,
            size=save_path.stat().st_size,
            path=str(save_path),
            message=f"上传成功，自动识别为: {role.value}"
        ))
    
    return responses


@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """下载文件"""
    # 在 uploads 和 outputs 目录中查找
    for search_dir in [settings.UPLOAD_DIR, settings.OUTPUT_DIR]:
        for file_path in search_dir.rglob(f"*{file_id}*"):
            if file_path.is_file():
                return FileResponse(
                    path=file_path,
                    filename=file_path.name,
                    media_type="application/octet-stream"
                )
    
    raise HTTPException(status_code=404, detail="文件未找到")


@router.get("/list")
async def list_uploaded_files():
    """列出已上传的文件"""
    files = []
    for file_path in settings.UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append(UploadedFile(
                name=file_path.name,
                path=str(file_path),
                role=detect_file_role(file_path.name)
            ))
    return files

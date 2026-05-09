"""
快速模式任务 API
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    StartFastTaskRequest, StartFastTaskResponse,
    TaskStatusResponse, UserFeedbackRequest,
    FastTaskPhase
)
from app.core.state import task_manager
from app.core.data_splitter import DataSplitter
from app.core.fast_engine import get_or_create_engine, remove_engine
from app.config import settings

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])

data_splitter = DataSplitter(settings.UPLOAD_DIR, settings.OUTPUT_DIR)


@router.post("/fast/start", response_model=StartFastTaskResponse)
async def start_fast_task(request: StartFastTaskRequest):
    """
    启动快速模式任务
    
    流程：
    1. 创建任务会话
    2. 数据切分与验证集准备
    3. 启动 FastEngine 后台线程
    """
    # 创建任务
    state = task_manager.create_task(request.task_config)
    task_id = state.task_id
    
    # 数据切分准备
    try:
        tc = request.task_config
        datasets = data_splitter.prepare_datasets(
            files=[f.model_dump() for f in tc.uploaded_files],
            target_column=tc.extracted_slots.target_column or "target",
            task_type=tc.extracted_slots.task_type or "binary_classification",
            task_id=task_id,
            is_time_series=tc.extracted_slots.is_time_series or False
        )
        
        # 记录数据集路径到任务状态
        task_manager.update_task(
            task_id,
            plan=f"数据集准备完成: train={datasets['train'].name}, validation={datasets['validation'].name}"
        )
        
    except Exception as e:
        task_manager.update_phase(task_id, FastTaskPhase.FAILED)
        raise HTTPException(status_code=400, detail=f"数据准备失败: {str(e)}")
    
    # 启动 FastEngine（后台线程）
    try:
        engine = get_or_create_engine(task_id)
        engine.start()
    except Exception as e:
        task_manager.update_phase(task_id, FastTaskPhase.FAILED)
        raise HTTPException(status_code=500, detail=f"引擎启动失败: {str(e)}")
    
    return StartFastTaskResponse(
        task_id=task_id,
        phase=FastTaskPhase.PLANNING,
        message="快速模式任务已启动，Plan & Coding Agent 正在生成基线代码"
    )


@router.get("/fast/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询快速模式任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    # 优先展示历史最佳评估结果，确保用户看到的是得分最高的版本
    display_metrics = task.best_metrics or task.metrics
    display_evaluation = task.best_evaluation or task.evaluation
    
    return TaskStatusResponse(
        task_id=task.task_id,
        phase=task.phase,
        metrics=display_metrics,
        evaluation=display_evaluation,
        optimize_round=task.optimize_round,
        debug_round=task.debug_round,
        user_feedback_round=task.user_feedback_round,
        execution_error=task.execution_error,
        code=task.code,
        plan=task.plan,
        logs=task.logs,
        best_code=task.best_code,
        artifacts=task.artifacts,
        has_test_set=task.has_test_set
    )


@router.post("/fast/{task_id}/feedback")
async def submit_user_feedback(task_id: str, request: UserFeedbackRequest):
    """
    提交用户反馈（满意 / 不满意）
    
    当任务处于 PRESENTING 阶段时，用户提交反馈后引擎继续执行。
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    if task.phase != FastTaskPhase.PRESENTING:
        raise HTTPException(
            status_code=400,
            detail=f"当前任务阶段为 {task.phase.value}，不在等待反馈状态"
        )
    
    try:
        engine = get_or_create_engine(task_id)
        engine.continue_with_feedback(
            satisfied=request.satisfied,
            suggestion=request.suggestion or ""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反馈处理失败: {str(e)}")
    
    if request.satisfied:
        return {
            "task_id": task_id,
            "status": "completed",
            "message": "用户已确认满意，正在生成最终产物"
        }
    else:
        return {
            "task_id": task_id,
            "status": "optimizing",
            "message": "收到用户反馈，进入优化阶段",
            "suggestion": request.suggestion
        }


@router.post("/fast/{task_id}/stop")
async def stop_task(task_id: str):
    """停止任务"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    try:
        engine = get_or_create_engine(task_id)
        engine.stop()
        task_manager.update_phase(task_id, FastTaskPhase.FAILED)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")
    
    return {"task_id": task_id, "status": "stopped", "message": "任务已停止"}

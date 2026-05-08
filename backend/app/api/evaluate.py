"""
评测系统 API
提供评测启动、状态查询、报告获取接口
"""

import logging
import threading
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from app.core.evaluator import BenchmarkEvaluator
from app.models.evaluate_schemas import (
    StartBenchmarkRequest, StartBenchmarkResponse,
    BenchmarkStatusResponse, BenchmarkReport
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluate", tags=["评测系统"])

# 全局评测实例存储
_evaluators: Dict[str, BenchmarkEvaluator] = {}
_evaluator_lock = threading.Lock()


@router.post("/run", response_model=StartBenchmarkResponse)
async def start_benchmark(request: StartBenchmarkRequest):
    """
    启动自动化评测

    Args:
        request: 评测配置（benchmark_dir, num_runs, judge_llm_config）

    Returns:
        eval_id 和初始状态
    """
    import os
    benchmark_dir = request.benchmark_dir
    if not os.path.exists(benchmark_dir):
        raise HTTPException(status_code=400, detail=f"评测目录不存在: {benchmark_dir}")

    try:
        evaluator = BenchmarkEvaluator(
            benchmark_dir=benchmark_dir,
            num_runs=request.num_runs,
            plan_coding_llm_config=request.plan_coding_llm_config,
            judge_llm_config=request.judge_llm_config,
            max_wait_seconds=request.max_wait_seconds
        )

        with _evaluator_lock:
            _evaluators[evaluator.eval_id] = evaluator

        # 在后台线程中执行评测
        def _run():
            try:
                report = evaluator.run_benchmark()
                logger.info(f"[EvaluateAPI] 评测 {evaluator.eval_id} 完成")
            except Exception as e:
                logger.exception(f"[EvaluateAPI] 评测 {evaluator.eval_id} 异常: {e}")
            finally:
                # 评测结束后从全局字典中移除，防止内存泄漏
                with _evaluator_lock:
                    _evaluators.pop(evaluator.eval_id, None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # 扫描任务数量
        tasks = evaluator._discover_tasks()

        return StartBenchmarkResponse(
            eval_id=evaluator.eval_id,
            status="running",
            message="评测已启动",
            task_count=len(tasks)
        )

    except Exception as e:
        logger.exception("[EvaluateAPI] 启动评测失败")
        raise HTTPException(status_code=500, detail=f"启动评测失败: {str(e)}")


@router.get("/{eval_id}/status", response_model=BenchmarkStatusResponse)
async def get_benchmark_status(eval_id: str):
    """
    查询评测状态
    """
    with _evaluator_lock:
        evaluator = _evaluators.get(eval_id)

    if not evaluator:
        raise HTTPException(status_code=404, detail="评测未找到")

    status = evaluator.get_status()
    return BenchmarkStatusResponse(
        eval_id=eval_id,
        status=status["status"],
        current_task=status["current_task"],
        current_run=status["current_run"],
        total_tasks=status["total_tasks"],
        total_runs=status["total_runs"],
        completed_runs=status["completed_runs"],
        progress_percent=status["progress_percent"]
    )


@router.get("/{eval_id}/report")
async def get_benchmark_report(eval_id: str):
    """
    获取评测报告

    如果评测已完成，返回完整报告；
    如果评测进行中，返回当前进度和已完成的任务结果。
    """
    with _evaluator_lock:
        evaluator = _evaluators.get(eval_id)

    if not evaluator:
        raise HTTPException(status_code=404, detail="评测未找到")

    if evaluator._report:
        return evaluator._report.model_dump()

    # 评测进行中，返回当前状态
    return {
        "eval_id": eval_id,
        "status": "running",
        "progress": evaluator.get_status()
    }


@router.post("/{eval_id}/stop")
async def stop_benchmark(eval_id: str):
    """
    停止评测
    """
    with _evaluator_lock:
        evaluator = _evaluators.get(eval_id)

    if not evaluator:
        raise HTTPException(status_code=404, detail="评测未找到")

    evaluator.stop()
    return {"eval_id": eval_id, "status": "stopping", "message": "已发送停止信号"}

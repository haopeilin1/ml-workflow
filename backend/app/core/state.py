"""
任务状态管理（内存存储，后续可替换为 Redis）
"""

import uuid
from typing import Dict, Optional
from threading import Lock

from app.models.schemas import FastTaskState, FastTaskPhase, TaskConfig


class TaskManager:
    """
    任务管理器
    
    目前使用内存字典存储任务状态，生产环境建议替换为 Redis + 持久化数据库
    """
    
    def __init__(self):
        self._tasks: Dict[str, FastTaskState] = {}
        self._lock = Lock()
    
    def create_task(self, task_config: TaskConfig) -> FastTaskState:
        """创建新任务"""
        task_id = f"fast_{uuid.uuid4().hex[:12]}"
        state = FastTaskState(
            task_id=task_id,
            task_config=task_config,
            phase=FastTaskPhase.IDLE
        )
        with self._lock:
            self._tasks[task_id] = state
        return state
    
    def get_task(self, task_id: str) -> Optional[FastTaskState]:
        """获取任务状态"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def update_phase(self, task_id: str, phase: FastTaskPhase) -> bool:
        """更新任务阶段"""
        with self._lock:
            if task_id not in self._tasks:
                return False
            self._tasks[task_id].phase = phase
            return True
    
    def update_task(self, task_id: str, **kwargs) -> bool:
        """更新任务字段"""
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            return True
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False
    
    def list_tasks(self) -> Dict[str, FastTaskState]:
        """列出所有任务"""
        with self._lock:
            return dict(self._tasks)


# 全局任务管理器实例
task_manager = TaskManager()

"""
数据切分与验证集管理
- 有 validation 文件时直接使用
- 无 validation 时从 train 自动 8:2 切分
- test 文件始终隔离，仅用于最终预测
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import shutil
import logging

from app.models.schemas import FileRole, TaskType
from app.config import settings

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    数据切分器
    
    职责：
    1. 根据文件角色识别 train / validation / test
    2. 若无 validation，从 train 自动 8:2 切分（分类任务 stratify）
    3. test 文件始终隔离，不进入训练流程
    4. 切分结果持久化到输出目录，供沙箱中的代码读取
    """
    
    def __init__(self, upload_dir: Path, output_dir: Path):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_datasets(
        self,
        files: List[Dict],
        target_column: str,
        task_type: TaskType,
        task_id: str,
        is_time_series: bool = False
    ) -> Dict[str, Optional[Path]]:
        """
        准备数据集，返回处理后的文件路径映射
        
        Args:
            files: [{name, path, role, size}, ...]
            target_column: 目标列名
            task_type: 任务类型
            task_id: 任务ID，用于构建输出子目录
            is_time_series: 是否为时序任务，时序任务按时间顺序切分（前train后val）
            
        Returns:
            {
                'train': Path,           # 训练集（始终存在）
                'validation': Path,      # 验证集（可能由train切分产生）
                'test': Optional[Path]   # 测试集（可能为None）
            }
        """
        task_output_dir = self.output_dir / task_id / "data"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 按角色分类文件
        train_files = [f for f in files if f.get("role") == FileRole.TRAIN]
        val_files = [f for f in files if f.get("role") == FileRole.VALIDATION]
        test_files = [f for f in files if f.get("role") == FileRole.TEST]
        
        result = {}
        
        # --- 处理训练集 ---
        if not train_files:
            raise ValueError("未找到训练集（role=train），请至少上传一个训练数据文件")
        
        # 目前只支持单文件训练集（后续可扩展多文件合并）
        train_path = self._resolve_path(train_files[0]["path"])
        train_df = self._read_file(train_path)
        logger.info(f"[DataSplitter] 加载训练集: {train_path}, shape={train_df.shape}")
        
        # --- 处理验证集 ---
        if val_files:
            # 用户上传了验证集，直接使用
            val_path = self._resolve_path(val_files[0]["path"])
            val_df = self._read_file(val_path)
            logger.info(f"[DataSplitter] 使用用户上传的验证集: {val_path}, shape={val_df.shape}")
            
            # 保存到任务目录（统一路径供沙箱使用）
            result["train"] = self._save_df(train_df, task_output_dir / "train.csv")
            result["validation"] = self._save_df(val_df, task_output_dir / "validation.csv")
        else:
            # 无验证集，从训练集自动 8:2 切分
            if is_time_series:
                logger.info("[DataSplitter] 时序任务，按时间顺序前80%训练、后20%验证切分")
            else:
                logger.info("[DataSplitter] 未找到验证集，从训练集自动 8:2 切分")
            train_split, val_split = self._split_train_validation(
                train_df, target_column, task_type, is_time_series
            )
            result["train"] = self._save_df(train_split, task_output_dir / "train.csv")
            result["validation"] = self._save_df(val_split, task_output_dir / "validation.csv")
        
        # --- 处理测试集（始终隔离）---
        if test_files:
            test_path = self._resolve_path(test_files[0]["path"])
            if test_path.exists():
                test_df = self._read_file(test_path)
                logger.info(f"[DataSplitter] 加载测试集（仅用于最终预测）: {test_path}, shape={test_df.shape}")
                result["test"] = self._save_df(test_df, task_output_dir / "test.csv")
            else:
                logger.info(f"[DataSplitter] 测试集文件不存在（可能已被隔离）: {test_path}")
                result["test"] = None
        else:
            result["test"] = None
            logger.info("[DataSplitter] 未找到测试集")
        
        return result
    
    def _split_train_validation(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: TaskType,
        is_time_series: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        从训练集切分出验证集（8:2）
        
        - 时序任务：按数据原有顺序切分（前80%训练，后20%验证），禁止打乱
        - 非时序分类任务：使用 stratify 保证类别比例一致
        - 非时序回归任务：随机切分
        """
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在数据集中")
        
        if is_time_series:
            # 时序任务：按原有顺序切分，前 80% 为训练，后 20% 为验证
            n_total = len(df)
            n_train = int(n_total * (1 - settings.DEFAULT_TEST_SIZE))
            train_df = df.iloc[:n_train].copy()
            val_df = df.iloc[n_train:].copy()
            logger.info(
                f"[DataSplitter] 时序顺序切分: train={train_df.shape} (前{n_train}行), "
                f"val={val_df.shape} (后{n_total - n_train}行)，未打乱"
            )
            return train_df, val_df
        
        y = df[target_column]
        
        stratify = None
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            # 检查是否满足 stratify 条件：每个类别至少2个样本
            value_counts = y.value_counts()
            if (value_counts >= 2).all():
                stratify = y
                logger.info(f"[DataSplitter] 分类任务，启用 stratify 切分")
            else:
                logger.warning(
                    f"[DataSplitter] 某些类别样本数不足（最小={value_counts.min()}），跳过 stratify"
                )
        
        train_df, val_df = train_test_split(
            df,
            test_size=settings.DEFAULT_TEST_SIZE,
            random_state=settings.DEFAULT_RANDOM_STATE,
            stratify=stratify
        )
        
        logger.info(
            f"[DataSplitter] 切分完成: train={train_df.shape}, val={val_df.shape}, "
            f"stratify={'Yes' if stratify is not None else 'No'}"
        )
        return train_df, val_df
    
    def _read_file(self, path: Path) -> pd.DataFrame:
        """读取 CSV 或 Excel 文件"""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _save_df(self, df: pd.DataFrame, path: Path) -> Path:
        """保存 DataFrame 到 CSV"""
        df.to_csv(path, index=False)
        return path
    
    def _resolve_path(self, path_str: str) -> Path:
        """解析文件路径（支持相对路径和绝对路径）"""
        path = Path(path_str)
        # 如果路径已存在（相对或绝对），直接使用
        if path.exists():
            return path
        # 否则回退到 uploads 目录（兼容前端上传场景）
        if not path.is_absolute():
            fallback = self.upload_dir / path.name
            if fallback.exists():
                return fallback
        return path
    
    def get_sandbox_paths(self, task_id: str) -> Dict[str, str]:
        """
        获取沙箱中使用的统一路径（Docker 容器内路径）
        
        Returns:
            {
                'train': '/data/train.csv',
                'validation': '/data/validation.csv',
                'test': '/data/test.csv'  # 可能为 None
            }
        """
        base = "/data"
        return {
            "train": f"{base}/train.csv",
            "validation": f"{base}/validation.csv",
            "test": f"{base}/test.csv"
        }

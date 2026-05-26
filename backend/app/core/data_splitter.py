"""
数据切分与验证集管理
- 有 validation 文件时直接使用
- 无 validation 时从 train 自动 8:2 切分
- test 文件同时进入训练流程，用于与训练/验证集一致的预处理与特征工程
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
    3. test 文件同时进入训练流程，用于一致的预处理与特征工程（fit 只在训练集上执行，transform 应用到验证集和测试集）
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
        is_time_series: bool = False,
        data_profile: Optional[Dict] = None
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
                train_df, target_column, task_type, is_time_series, data_profile
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
        is_time_series: bool = False,
        data_profile: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        智能切分训练集/验证集（SmartSplitStrategy）
        
        - 动态验证集比例：基于样本量和类别分布自动计算
        - 时序任务：检测时间列并排序后切分（前train后val）
        - 分类任务：stratify + 验证集少数类样本数校验
        - 回归任务：随机切分
        """
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在数据集中")
        
        n_samples = len(df)
        
        # ========== 1. 计算动态验证集比例 ==========
        minority_ratio = None
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            y = df[target_column]
            value_counts = y.value_counts()
            minority_ratio = value_counts.min() / n_samples
        
        val_ratio = self._compute_validation_ratio(n_samples, minority_ratio, task_type)
        ratio_str = f"{minority_ratio:.4f}" if minority_ratio is not None else "N/A"
        logger.info(
            f"[SmartSplit] 样本量={n_samples}, minority_ratio={ratio_str}, "
            f"验证集比例={val_ratio:.2f}"
        )
        
        # ========== 2. 时序任务：检测时间列并排序 ==========
        if is_time_series:
            time_col = self._detect_time_column(df, data_profile)
            if time_col:
                df = df.sort_values(by=time_col).reset_index(drop=True)
                logger.info(f"[SmartSplit] 时序任务：按时间列 '{time_col}' 排序后切分")
            else:
                logger.warning("[SmartSplit] 时序任务但未检测到时间列，按原始行顺序切分（可能不准确）")
            
            n_train = int(n_samples * (1 - val_ratio))
            train_df = df.iloc[:n_train].copy()
            val_df = df.iloc[n_train:].copy()
            logger.info(
                f"[SmartSplit] 时序顺序切分: train={train_df.shape} (前{n_train}行), "
                f"val={val_df.shape} (后{n_samples - n_train}行)"
            )
            return train_df, val_df
        
        # ========== 3. 非时序任务切分 ==========
        y = df[target_column]
        
        stratify = None
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            value_counts = y.value_counts()
            if (value_counts >= 2).all():
                stratify = y
                logger.info(f"[SmartSplit] 分类任务，启用 stratify 切分")
            else:
                logger.warning(
                    f"[SmartSplit] 某些类别样本数不足（最小={value_counts.min()}），跳过 stratify"
                )
        
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=settings.DEFAULT_RANDOM_STATE,
            stratify=stratify
        )
        
        # ========== 4. 不平衡校验：验证集少数类样本数是否足够 ==========
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            val_y = val_df[target_column]
            min_class_count = val_y.value_counts().min()
            if min_class_count < 50:
                logger.warning(
                    f"[SmartSplit] 验证集少数类仅 {min_class_count} 个样本，"
                    f"AUC/F1 评估可能不稳定（建议≥50）"
                )
            else:
                logger.info(f"[SmartSplit] 验证集少数类样本数={min_class_count}，评估稳定性良好")
        
        logger.info(
            f"[SmartSplit] 切分完成: train={train_df.shape}, val={val_df.shape}, "
            f"stratify={'Yes' if stratify is not None else 'No'}"
        )
        return train_df, val_df
    
    def _compute_validation_ratio(
        self,
        n_samples: int,
        minority_ratio: Optional[float],
        task_type: TaskType
    ) -> float:
        """
        基于样本量和类别分布计算最优验证集比例。
        原则：验证集中少数类至少保证 50 个样本用于稳定评估。
        """
        # 基础比例由样本量决定
        if n_samples < 1000:
            base_ratio = 0.30
        elif n_samples < 50000:
            base_ratio = 0.20
        else:
            base_ratio = 0.10
        
        # 极度不平衡时增加比例，确保验证集少数类 ≥50 个
        if minority_ratio is not None and minority_ratio < 0.05:
            needed = 50.0 / (n_samples * minority_ratio)
            base_ratio = max(base_ratio, min(needed, 0.40))
            logger.info(
                f"[SmartSplit] 极度不平衡（minority={minority_ratio:.4f}），"
                f"验证集比例上调至 {base_ratio:.2f}"
            )
        
        return base_ratio
    
    def _detect_time_column(
        self,
        df: pd.DataFrame,
        data_profile: Optional[Dict] = None
    ) -> Optional[str]:
        """
        检测时间列，用于时序任务切分前排序。
        优先级：IntentAgent 判定 > isMonotonic 索引列 > isDateParseable 日期列
        """
        if data_profile:
            columns_info = data_profile.get("columns", [])
            # 优先级1：单调递增的数值索引列（instant, No, id）
            for col_info in columns_info:
                name = col_info.get("name", "")
                n_samples = data_profile.get("rowCount", 0)
                if col_info.get("isMonotonic") and n_samples > 10:
                    if col_info.get("uniqueCount") == n_samples:
                        # 列名含时间相关关键词，或就是简单的序号列
                        if any(kw in name.lower() for kw in ["instant", "no", "id", "index", "seq"]):
                            if name in df.columns:
                                return name
            
            # 优先级2：可解析为日期的字符串列
            for col_info in columns_info:
                name = col_info.get("name", "")
                if col_info.get("isDateParseable") and name in df.columns:
                    return name
        
        # 回退：按列名启发式匹配
        time_keywords = ["dteday", "date", "datetime", "timestamp", "time", "year", "month", "day"]
        for col in df.columns:
            if any(kw in col.lower() for kw in time_keywords):
                # 确认该列确实能解析为日期或单调递增
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return col
                try:
                    pd.to_datetime(df[col], errors='raise')
                    return col
                except:
                    pass
        
        return None
    
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

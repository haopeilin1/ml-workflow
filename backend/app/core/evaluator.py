"""
自动化评测引擎
批量运行建模任务，自动评估模型质量
"""

import csv
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
# sklearn >= 1.6 使用 root_mean_squared_error 替代 mean_squared_error(squared=False)
try:
    from sklearn.metrics import root_mean_squared_error as _rmse_func
except ImportError:
    def _rmse_func(y_true, y_pred):
        import math
        return math.sqrt(mean_squared_error(y_true, y_pred))

from app.agents.evaluate_judge import EvaluateJudgeAgent
from app.agents.intent_recognition import IntentRecognitionAgent, IntentResult
from app.config import settings
from app.core.data_splitter import DataSplitter
from app.core.fast_engine import get_or_create_engine, remove_engine
from app.core.state import task_manager
from app.models.evaluate_schemas import (
    BenchmarkTaskConfig, BenchmarkTaskResult, BenchmarkRoundResult,
    BenchmarkReport, JudgeResult, TestSetMetrics, TaskType,
    TimingBreakdown, TokenUsageSummary
)
from app.models.schemas import (
    TaskConfig, ExtractedSlots, UploadedFile, FileRole,
    FastTaskPhase, LLMConfig
)
from app.sandbox.executor import sandbox_executor

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    自动化评测引擎

    使用方式：
        evaluator = BenchmarkEvaluator(
            benchmark_dir="/path/to/benchmark",
            num_runs=3,
            judge_llm_config=LLMConfig(...)
        )
        report = evaluator.run_benchmark()
    """

    def __init__(
        self,
        benchmark_dir: str,
        num_runs: int = 3,
        judge_llm_config: Optional[LLMConfig] = None,
        plan_coding_llm_config: Optional[LLMConfig] = None,
        max_wait_seconds: int = 1200,  # 从 600 提升到 1200，给完整流程（3轮优化+产物生成）留出足够时间
        eval_id: Optional[str] = None
    ):
        self.plan_coding_llm_config = plan_coding_llm_config
        self.benchmark_dir = Path(benchmark_dir)
        self.num_runs = num_runs
        self.max_wait_seconds = max_wait_seconds
        self.eval_id = eval_id or f"eval_{uuid.uuid4().hex[:12]}"
        self.judge_agent = EvaluateJudgeAgent(llm_config=judge_llm_config)
        self.intent_agent = IntentRecognitionAgent()
        self._intent_cache: Dict[str, IntentResult] = {}
        self.data_splitter = DataSplitter(settings.UPLOAD_DIR, settings.OUTPUT_DIR)
        self.result_base_dir = settings.OUTPUT_DIR / self.eval_id
        self.result_base_dir.mkdir(parents=True, exist_ok=True)

        # 运行状态跟踪
        self._current_task: Optional[str] = None
        self._current_run: int = 0
        self._completed_runs: int = 0
        self._total_runs: int = 0
        self._running: bool = False
        self._report: Optional[BenchmarkReport] = None

    def run_benchmark(self) -> BenchmarkReport:
        """
        执行完整评测流程

        Returns:
            BenchmarkReport: 评测报告
        """
        start_time = time.time()
        self._running = True

        # 1. 发现所有任务
        tasks = self._discover_tasks()
        if not tasks:
            logger.error(f"[BenchmarkEvaluator] 在 {self.benchmark_dir} 下未找到任何任务")
            return self._build_empty_report("未找到任务")

        self._total_runs = len(tasks) * self.num_runs
        logger.info(f"[BenchmarkEvaluator] 发现 {len(tasks)} 个任务，每个运行 {self.num_runs} 次，共 {self._total_runs} 次运行")

        report = BenchmarkReport(
            eval_id=self.eval_id,
            benchmark_dir=str(self.benchmark_dir),
            num_runs=self.num_runs,
            task_names=[t.task_name for t in tasks],
            total_tasks=len(tasks),
            total_runs=self._total_runs,
            status="running"
        )
        self._report = report

        # 2. 对每个任务运行 num_runs 次
        round_results: List[BenchmarkRoundResult] = []
        for task_cfg in tasks:
            task_results: List[BenchmarkTaskResult] = []
            for run_idx in range(1, self.num_runs + 1):
                if not self._running:
                    logger.info("[BenchmarkEvaluator] 收到停止信号，中断评测")
                    break

                self._current_task = task_cfg.task_name
                self._current_run = run_idx

                logger.info(f"[BenchmarkEvaluator] 开始任务: {task_cfg.task_name}, 第 {run_idx}/{self.num_runs} 次运行")
                result = self._run_single_task(task_cfg, run_idx)
                task_results.append(result)
                self._completed_runs += 1

                logger.info(
                    f"[BenchmarkEvaluator] 任务 {task_cfg.task_name} 第 {run_idx} 次运行完成: "
                    f"success={result.success}, judge_accepted={result.judge_accepted}, "
                    f"duration={result.duration_seconds:.1f}s"
                )

            # 计算本轮聚合指标
            accepted_count = sum(1 for r in task_results if r.judge_accepted)
            success_rate = accepted_count / len(task_results) if task_results else 0.0
            
            scores = [r.best_score for r in task_results if r.best_score is not None]
            avg_score = sum(scores) / len(scores) if scores else None
            score_std = self._calc_std(scores) if scores else 0.0
            score_cv = score_std / avg_score if avg_score and avg_score != 0 else 0.0
            
            durations = [r.duration_seconds for r in task_results]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            min_duration = min(durations) if durations else 0.0
            max_duration = max(durations) if durations else 0.0
            duration_std = self._calc_std(durations) if durations else 0.0
            
            tokens = [r.token_usage for r in task_results if r.token_usage]
            avg_total_tokens = int(sum(t.total_tokens for t in tokens) / len(tokens)) if tokens else 0
            avg_plan_tokens = int(sum(t.plan_coding_total_tokens for t in tokens) / len(tokens)) if tokens else 0
            avg_eval_tokens = int(sum(t.evaluation_total_tokens for t in tokens) / len(tokens)) if tokens else 0

            round_result = BenchmarkRoundResult(
                round_index=len(round_results) + 1,
                task_results=task_results,
                success_rate=success_rate,
                avg_best_score=avg_score,
                success_count=accepted_count,
                fail_count=len(task_results) - accepted_count,
                avg_duration_seconds=avg_duration,
                min_duration_seconds=min_duration,
                max_duration_seconds=max_duration,
                duration_std=duration_std,
                avg_total_tokens=avg_total_tokens,
                avg_plan_coding_tokens=avg_plan_tokens,
                avg_evaluation_tokens=avg_eval_tokens,
                score_std=score_std,
                score_cv=score_cv
            )
            round_results.append(round_result)
            logger.info(
                f"[BenchmarkEvaluator] 任务 {task_cfg.task_name} 完成: "
                f"成功率={success_rate:.1%}, 平均耗时={avg_duration:.1f}s, "
                f"平均Token={avg_total_tokens}, score_std={score_std:.4f}"
            )

            if not self._running:
                break

        # 3. 生成最终报告
        total_accepted = sum(r.success_count for r in round_results)
        overall_rate = total_accepted / self._total_runs if self._total_runs > 0 else 0.0

        # 全局聚合：所有任务的平均耗时、Token、稳定性
        all_durations = [r.duration_seconds for rr in round_results for r in rr.task_results]
        all_tokens = [r.token_usage.total_tokens for rr in round_results for r in rr.task_results if r.token_usage]
        all_scores = [r.best_score for rr in round_results for r in rr.task_results if r.best_score is not None]
        
        overall_avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0.0
        overall_avg_tokens = int(sum(all_tokens) / len(all_tokens)) if all_tokens else 0
        overall_score_std = self._calc_std(all_scores) if all_scores else 0.0
        overall_duration_std = self._calc_std(all_durations) if all_durations else 0.0

        report.round_results = round_results
        report.overall_success_rate = overall_rate
        report.total_accepted = total_accepted
        report.overall_avg_duration_seconds = overall_avg_duration
        report.overall_avg_total_tokens = overall_avg_tokens
        report.overall_score_std = overall_score_std
        report.overall_duration_std = overall_duration_std
        report.status = "completed" if self._running else "stopped"
        report.completed_at = datetime.utcnow()
        self._running = False

        # 保存报告到文件
        report_path = self.result_base_dir / "benchmark_report.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding='utf-8')

        # 生成 CSV 表格
        csv_path = self._generate_csv_table(round_results)
        summary_csv_path = self._generate_summary_csv(round_results)
        logger.info(f"[BenchmarkEvaluator] 评测完成: 总成功率={overall_rate:.1%}")
        logger.info(f"[BenchmarkEvaluator] 全局平均耗时={overall_avg_duration:.1f}s, 全局平均Token={overall_avg_tokens}")
        logger.info(f"[BenchmarkEvaluator] 全局score_std={overall_score_std:.4f}, duration_std={overall_duration_std:.4f}")
        logger.info(f"[BenchmarkEvaluator] 报告保存至 {report_path}")
        logger.info(f"[BenchmarkEvaluator] 明细 CSV 保存至 {csv_path}")
        logger.info(f"[BenchmarkEvaluator] 汇总 CSV 保存至 {summary_csv_path}")

        return report

    @staticmethod
    def _calc_std(values: list) -> float:
        """计算标准差（样本标准差，分母 n-1）"""
        if len(values) < 2:
            return 0.0
        import math
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def stop(self):
        """停止评测"""
        self._running = False
        logger.info("[BenchmarkEvaluator] 收到停止信号")

    def get_status(self) -> Dict[str, Any]:
        """获取当前评测状态"""
        progress = (self._completed_runs / self._total_runs * 100) if self._total_runs > 0 else 0.0
        return {
            "eval_id": self.eval_id,
            "status": "running" if self._running else ("completed" if self._report else "idle"),
            "current_task": self._current_task,
            "current_run": self._current_run,
            "total_tasks": len(self._report.task_names) if self._report else 0,
            "total_runs": self._total_runs,
            "completed_runs": self._completed_runs,
            "progress_percent": round(progress, 1)
        }

    def _discover_tasks(self) -> List[BenchmarkTaskConfig]:
        """扫描 benchmark 目录，发现所有任务
        
        支持两种结构：
        1. benchmark_root/task_name/建模/ + benchmark_root/task_name/评估/
        2. benchmark_root/建模/ + benchmark_root/评估/（单任务直接评测）
        """
        tasks = []
        
        # 启动前清理：恢复上次被中断评测遗留的 hidden 测试集
        self._restore_all_hidden_test_csvs()
        
        # 检查是否是单任务直接传入（根目录下直接有 建模/ 和 评估/）
        direct_modeling = self.benchmark_dir / "建模"
        direct_eval = self.benchmark_dir / "评估"
        if direct_modeling.exists() and direct_eval.exists():
            task_cfg = self._parse_task_dir(self.benchmark_dir, self.benchmark_dir.name)
            if task_cfg:
                tasks.append(task_cfg)
                logger.info(f"[BenchmarkEvaluator] 单任务模式: {self.benchmark_dir.name}")
            return tasks
        
        # 多任务目录扫描
        for task_dir in sorted(self.benchmark_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_cfg = self._parse_task_dir(task_dir, task_dir.name)
            if task_cfg:
                tasks.append(task_cfg)
        
        return tasks
    
    def _restore_all_hidden_test_csvs(self):
        """恢复 benchmark 目录下所有遗留的 test.csv.hidden 文件
        
        防止上次评测被强制 kill 后，source 中的 test.csv 永远处于 hidden 状态。
        支持两种结构：
        1. 单任务: benchmark_dir/建模/test.csv.hidden
        2. 多任务: benchmark_dir/task_name/建模/test.csv.hidden
        """
        restored_count = 0
        modeling_dirs = set()
        
        # 直接子目录（单任务模式）
        for child in self.benchmark_dir.iterdir():
            if child.is_dir() and '建模' in child.name:
                modeling_dirs.add(child)
        
        # 孙目录（多任务模式）
        for task_dir in self.benchmark_dir.iterdir():
            if task_dir.is_dir():
                for grandchild in task_dir.iterdir():
                    if grandchild.is_dir() and '建模' in grandchild.name:
                        modeling_dirs.add(grandchild)
        
        for modeling_dir in modeling_dirs:
            hidden = modeling_dir / "test.csv.hidden"
            original = modeling_dir / "test.csv"
            if hidden.exists():
                if original.exists():
                    original.unlink()
                hidden.rename(original)
                restored_count += 1
                logger.info(f"[BenchmarkEvaluator] 恢复遗留的 hidden 测试集: {original}")
        
        if restored_count > 0:
            logger.info(f"[BenchmarkEvaluator] 共恢复 {restored_count} 个遗留的 hidden 测试集")
    
    def _parse_task_dir(self, task_dir: Path, task_name: str) -> Optional[BenchmarkTaskConfig]:
        """解析单个任务目录（兼容 建模/评估 和 数据建模/评估结果 两种命名）"""
        # 动态发现子目录（避免 Windows 路径编码问题）
        modeling_dirs = []
        eval_dirs = []
        try:
            for name in os.listdir(task_dir):
                child = task_dir / name
                if not child.is_dir():
                    continue
                # 匹配建模目录（含"建模"关键字）
                if '建模' in name:
                    modeling_dirs.append(child)
                # 匹配评估目录（含"评估"关键字）
                elif '评估' in name:
                    eval_dirs.append(child)
        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator] 读取目录失败 {task_name}: {e}")
            return None
        
        if not modeling_dirs or not eval_dirs:
            logger.warning(f"[BenchmarkEvaluator] 跳过 {task_name}: 缺少含'建模'或'评估'的子目录")
            return None
        
        # 【修复】优先使用含"数据"的建模目录，否则使用第一个
        modeling_dir = None
        for md in modeling_dirs:
            if '数据' in md.name:
                modeling_dir = md
                break
        if not modeling_dir:
            modeling_dir = modeling_dirs[0]
        
        eval_dir = eval_dirs[0]

        # 查找文件（灵活匹配，支持多种命名习惯）
        csv_files = list(modeling_dir.glob("*.csv"))
        train_files = [f for f in csv_files if "train" in f.name.lower()]
        test_files = [f for f in csv_files if "test" in f.name.lower() and "train" not in f.name.lower()]
        
        # 任务描述：优先匹配以"任务描述"开头的 txt，否则取第一个 txt
        all_txt = list(modeling_dir.glob("*.txt"))
        desc_files = [f for f in all_txt if f.name.startswith("任务描述")] or all_txt
        
        # ground_truth：评估目录下所有 csv（通常只有一个）
        gt_files = list(eval_dir.glob("*.csv"))

        if not train_files:
            logger.warning(f"[BenchmarkEvaluator] 跳过 {task_name}: 未找到训练集")
            return None
        if not gt_files:
            logger.warning(f"[BenchmarkEvaluator] 跳过 {task_name}: 未找到 ground_truth")
            return None

        train_path = train_files[0]
        test_path = test_files[0] if test_files else None
        desc_path = desc_files[0] if desc_files else None
        gt_path = gt_files[0]

        # 优先使用 LLM IntentAgent 识别任务信息，失败则回退到规则推断
        task_type, target_column, eval_metric, id_column, complexity = self._recognize_task_info_llm(
            train_path, gt_path, desc_path, task_name
        )
        if not target_column:
            task_type, target_column, id_column = self._infer_task_info(train_path, gt_path)
            eval_metric = None
            complexity = "simple"
            logger.info(f"[BenchmarkEvaluator] {task_name}: 回退到规则推断")

        return BenchmarkTaskConfig(
            task_name=task_name,
            task_dir=str(task_dir),
            train_path=str(train_path),
            test_path=str(test_path) if test_path else "",
            desc_path=str(desc_path) if desc_path else "",
            ground_truth_path=str(gt_path),
            target_column=target_column,
            task_type=task_type,
            eval_metric=eval_metric,
            id_column=id_column
        )

    def _build_data_profile(self, train_path: Path, gt_path: Path) -> Dict[str, Any]:
        """构建数据画像（供 IntentAgent 使用），包含丰富的列统计信息"""
        train_df = pd.read_csv(train_path)
        gt_df = pd.read_csv(gt_path)
        gt_cols = list(gt_df.columns)
        n_rows = int(len(train_df))

        profile = {
            "fileName": train_path.name,
            "rowCount": n_rows,
            "columnCount": int(len(train_df.columns)),
            "columns": []
        }

        for col in train_df.columns:
            series = train_df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            unique_count = int(series.nunique())
            missing_count = int(series.isna().sum())

            col_info = {
                "name": col,
                "type": "numeric" if is_numeric else "categorical",
                "uniqueCount": unique_count,
                "missingCount": missing_count,
            }

            # 标记可能的 id 列（唯一值 ≈ 行数）
            if n_rows > 0 and unique_count > n_rows * 0.9:
                col_info["isLikelyId"] = True

            # 数值列：增加统计特征
            if is_numeric:
                non_null = series.dropna()
                if len(non_null) > 0:
                    col_info["sampleValues"] = [float(v) for v in non_null.head(5).tolist()]
                    col_info["min"] = float(non_null.min())
                    col_info["max"] = float(non_null.max())
                    col_info["mean"] = float(non_null.mean())
            else:
                # 类别列：增加采样值和最常见值
                non_null = series.dropna()
                if len(non_null) > 0:
                    col_info["sampleValues"] = non_null.head(5).astype(str).tolist()
                    vc = non_null.value_counts()
                    if len(vc) > 0:
                        col_info["mostCommon"] = str(vc.index[0])
                        col_info["mostCommonFreq"] = int(vc.iloc[0])

            profile["columns"].append(col_info)

        return profile

    def _recognize_task_info_llm(
        self,
        train_path: Path,
        gt_path: Path,
        desc_path: Optional[Path],
        task_name: str,
    ) -> tuple:
        """
        使用 LLM IntentAgent 识别任务信息。
        返回 (task_type, target_column, eval_metric, id_column, complexity)。
        任一环节失败则返回全 None，由调用方回退到规则推断。
        """
        cache_key = task_name
        if cache_key in self._intent_cache:
            cached = self._intent_cache[cache_key]
            task_type_map = {
                "binary_classification": TaskType.BINARY_CLASSIFICATION,
                "multiclass_classification": TaskType.MULTICLASS_CLASSIFICATION,
                "regression": TaskType.REGRESSION,
            }
            task_type = task_type_map.get(cached.task_type, TaskType.BINARY_CLASSIFICATION)
            gt_df = pd.read_csv(gt_path)
            gt_cols = list(gt_df.columns)
            id_column = gt_cols[0] if len(gt_cols) >= 1 else None
            logger.info(f"[BenchmarkEvaluator] {task_name}: 命中 IntentAgent 缓存")
            return task_type, cached.target_column, cached.eval_metric, id_column, cached.complexity

        # 读取任务描述
        user_description = ""
        if desc_path and desc_path.exists():
            try:
                user_description = desc_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        # 构建数据画像
        try:
            profile = self._build_data_profile(train_path, gt_path)
        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator] 构建数据画像失败 {task_name}: {e}")
            return None, None, None, None

        # 调用 IntentAgent
        try:
            result = self.intent_agent.recognize(
                columns=profile.get("columns", []),
                task_description=user_description,
                row_count=profile.get("row_count", 0),
                col_count=profile.get("column_count", 0),
            )
        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator] IntentAgent 调用失败 {task_name}: {e}")
            return None, None, None, None

        if not result or not result.target_column:
            logger.info(f"[BenchmarkEvaluator] IntentAgent 未返回 target_column {task_name}")
            return None, None, None, None

        # 校验 target_column 是否在训练集中
        train_df = pd.read_csv(train_path)
        if result.target_column not in train_df.columns:
            logger.warning(
                f"[BenchmarkEvaluator] LLM 返回的 target_column '{result.target_column}' "
                f"不在训练集中，回退到规则推断"
            )
            return None, None, None, None

        # 映射 task_type
        task_type_map = {
            "binary_classification": TaskType.BINARY_CLASSIFICATION,
            "multiclass_classification": TaskType.MULTICLASS_CLASSIFICATION,
            "regression": TaskType.REGRESSION,
        }
        task_type = task_type_map.get(result.task_type, TaskType.BINARY_CLASSIFICATION)

        # id_column 从 ground_truth 推断
        gt_df = pd.read_csv(gt_path)
        gt_cols = list(gt_df.columns)
        id_column = gt_cols[0] if len(gt_cols) >= 1 else None

        self._intent_cache[cache_key] = result
        logger.info(
            f"[BenchmarkEvaluator] LLM 识别任务信息 {task_name}: "
            f"type={task_type.value}, target={result.target_column}, "
            f"metric={result.eval_metric}, complexity={result.complexity}"
        )
        return task_type, result.target_column, result.eval_metric, id_column, result.complexity

    def _infer_task_info(self, train_path: Path, gt_path: Path) -> tuple:
        """
        自动推断任务类型、目标列名和 id 列名
        """
        train_df = pd.read_csv(train_path)
        gt_df = pd.read_csv(gt_path)

        # id 列：ground_truth 中除目标列外的列（通常第一列）
        # 但我们需要先知道目标列才能确定 id 列...
        # 策略：假设 ground_truth 有 2 列，第一列是 id，第二列是 target
        gt_cols = list(gt_df.columns)
        id_column = gt_cols[0] if len(gt_cols) >= 1 else None
        target_column = gt_cols[-1] if len(gt_cols) >= 2 else gt_cols[0]

        # 如果训练集中没有 target_column，尝试找训练集中最可能是目标列的列
        if target_column not in train_df.columns:
            # 找训练集中不在 ground_truth 中的列
            possible_targets = [c for c in train_df.columns if c not in gt_cols]
            if possible_targets:
                target_column = possible_targets[0]
            else:
                #  fallback：用训练集的最后一列
                target_column = train_df.columns[-1]

        # 推断任务类型
        target_series = train_df[target_column]
        unique_vals = target_series.nunique()

        if pd.api.types.is_numeric_dtype(target_series):
            if unique_vals <= 10:
                # 可能是分类（整数编码）
                task_type = TaskType.BINARY_CLASSIFICATION if unique_vals == 2 else TaskType.MULTICLASS_CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION
        else:
            # 非数值型 → 分类
            task_type = TaskType.BINARY_CLASSIFICATION if unique_vals == 2 else TaskType.MULTICLASS_CLASSIFICATION

        logger.info(
            f"[BenchmarkEvaluator] 推断任务信息: type={task_type.value}, "
            f"target={target_column}, id={id_column}"
        )
        return task_type, target_column, id_column



    def _run_single_task(self, task_cfg: BenchmarkTaskConfig, run_index: int) -> BenchmarkTaskResult:
        """执行单个任务单次运行"""
        task_start = time.time()
        result = BenchmarkTaskResult(
            task_name=task_cfg.task_name,
            run_index=run_index,
            success=False,
            judge_accepted=False
        )

        try:
            # 1. 读取任务描述
            user_description = ""
            if task_cfg.desc_path and Path(task_cfg.desc_path).exists():
                user_description = Path(task_cfg.desc_path).read_text(encoding='utf-8').strip()

            # 2. 创建 TaskConfig（冷启动，不传入建模建议）
            agent_configs = {}
            if self.plan_coding_llm_config:
                agent_configs["plan_coding"] = self.plan_coding_llm_config
                # 评测系统内 EvaluationAgent 复用 PlanCoding 配置（同属代码理解类任务）
                agent_configs["evaluation"] = self.plan_coding_llm_config
            
            uploaded_files = [
                UploadedFile(name="train.csv", path=task_cfg.train_path, role=FileRole.TRAIN),
            ]
            if task_cfg.test_path:
                uploaded_files.append(UploadedFile(name="test.csv", path=task_cfg.test_path, role=FileRole.TEST))
            
            # 从 IntentAgent 缓存获取 complexity（核心层数据，不经过评测层 BenchmarkTaskConfig）
            cached_intent = self._intent_cache.get(task_cfg.task_name)
            complexity = cached_intent.complexity if cached_intent else "simple"
            
            tc = TaskConfig(
                extracted_slots=ExtractedSlots(
                    target_column=task_cfg.target_column,
                    task_type=task_cfg.task_type,
                    eval_metric=task_cfg.eval_metric,
                    complexity=complexity,
                    feature_constraints=[],
                    user_modeling_suggestions=None  # 冷启动：不传入建模建议
                ),
                uploaded_files=uploaded_files,
                user_description=user_description,
                agent_llm_configs=agent_configs if agent_configs else None
            )

            # 3. 创建任务并启动 FastEngine
            state = task_manager.create_task(tc)
            task_id = state.task_id
            result.task_id = task_id

            # 【关键】物理隔离测试集（source 目录）：
            # 在 prepare_datasets 之前隔离 source/test.csv，确保 DataSplitter 不会把 test.csv
            # 复制到 outputs/data/，从而保证 FastEngine 优化阶段 test.csv 完全不可见
            source_test_path = Path(task_cfg.test_path) if task_cfg.test_path else None
            source_test_hidden = None
            source_test_was_hidden = False
            if source_test_path and source_test_path.exists():
                source_test_hidden = source_test_path.parent / "test.csv.hidden"
                source_test_path.rename(source_test_hidden)
                source_test_was_hidden = True
                logger.info(f"[BenchmarkEvaluator] 已隔离 source 测试集: {source_test_path} -> {source_test_hidden}")

            # 数据切分准备（source 中无 test.csv，outputs/data/ 中也不会创建）
            datasets = self.data_splitter.prepare_datasets(
                files=[f.model_dump() for f in tc.uploaded_files],
                target_column=tc.extracted_slots.target_column or "target",
                task_type=tc.extracted_slots.task_type,
                task_id=task_id
            )
            task_manager.update_task(
                task_id,
                plan=f"评测数据集准备完成: train={datasets['train'].name}"
            )
            data_dir = datasets["train"].parent

            # 启动 FastEngine
            engine = get_or_create_engine(task_id)
            engine.start()

            # 4. 轮询等待 PRESENTING / FAILED / 超时
            presenting = self._wait_for_phase(task_id, [FastTaskPhase.PRESENTING, FastTaskPhase.FAILED], timeout=self.max_wait_seconds)
            if not presenting:
                result.error_message = "等待 PRESENTING 阶段超时或任务失败"
                result.phase = task_manager.get_task(task_id).phase.value if task_manager.get_task(task_id) else "unknown"
                if source_test_was_hidden:
                    self._restore_test_csv(source_test_hidden, source_test_path)
                self._cleanup_task(task_id)
                result.duration_seconds = time.time() - task_start
                return result

            task_state = task_manager.get_task(task_id)
            if task_state.phase == FastTaskPhase.FAILED:
                result.error_message = task_state.execution_error or "FastEngine 进入 FAILED 阶段"
                result.phase = "failed"
                result.logs = task_state.logs or []
                if source_test_was_hidden:
                    self._restore_test_csv(source_test_hidden, source_test_path)
                self._cleanup_task(task_id)
                result.duration_seconds = time.time() - task_start
                return result

            # 5. 到达 PRESENTING → 先恢复测试集到 source 和 data_dir，再提交满意反馈
            # （产物代码执行需要 test.csv，必须在提交反馈前准备好）
            if source_test_was_hidden:
                self._restore_test_csv(source_test_hidden, source_test_path)
                shutil.copy2(source_test_path, data_dir / "test.csv")
                logger.info(f"[BenchmarkEvaluator] 已恢复测试集到 source 并复制到 data_dir，供产物生成使用")
                print(f"[DEBUG] Copied test.csv to {data_dir / 'test.csv'}, exists={(data_dir / 'test.csv').exists()}")

            logger.info(f"[BenchmarkEvaluator] 任务 {task_id} 到达 PRESENTING，自动提交满意反馈")
            engine.continue_with_feedback(satisfied=True, suggestion="")

            # 6. 轮询等待 COMPLETED / FAILED
            completed = self._wait_for_phase(task_id, [FastTaskPhase.COMPLETED, FastTaskPhase.FAILED], timeout=self.max_wait_seconds)
            task_state = task_manager.get_task(task_id)
            result.phase = task_state.phase.value if task_state else "unknown"
            result.logs = task_state.logs or [] if task_state else []
            result.best_score = task_state.best_score if task_state else None
            result.val_metrics = task_state.best_metrics if task_state else None

            if not completed or task_state.phase == FastTaskPhase.FAILED:
                result.error_message = task_state.execution_error or "产物生成阶段失败或超时"
                self._cleanup_task(task_id)
                result.duration_seconds = time.time() - task_start
                return result

            result.success = True

            # 收集维度评分
            if task_state.best_evaluation and task_state.best_evaluation.dimension_scores:
                result.dimension_scores = [
                    ds.model_dump() for ds in task_state.best_evaluation.dimension_scores
                ]

            # 收集各阶段耗时（先初始化，test_prediction_seconds 在预测后填充）
            result.timing = TimingBreakdown(
                code_generation_seconds=engine.timings.get("code_generation_seconds", 0.0),
                sandbox_execution_seconds=engine.timings.get("sandbox_execution_seconds", 0.0),
                evaluation_seconds=engine.timings.get("evaluation_seconds", 0.0),
                artifact_generation_seconds=engine.timings.get("artifact_generation_seconds", 0.0),
                total_seconds=time.time() - task_start
            )

            # 收集 Token 消耗（除 Judge 外）
            plan_usage = engine.plan_coding_agent.get_usage_summary()
            eval_usage = engine.evaluation_agent.get_usage_summary()
            result.token_usage = TokenUsageSummary(
                plan_coding_calls=plan_usage["call_count"],
                plan_coding_prompt_tokens=plan_usage["prompt_tokens"],
                plan_coding_completion_tokens=plan_usage["completion_tokens"],
                plan_coding_total_tokens=plan_usage["total_tokens"],
                evaluation_calls=eval_usage["call_count"],
                evaluation_prompt_tokens=eval_usage["prompt_tokens"],
                evaluation_completion_tokens=eval_usage["completion_tokens"],
                evaluation_total_tokens=eval_usage["total_tokens"],
                total_calls=plan_usage["call_count"] + eval_usage["call_count"],
                total_prompt_tokens=plan_usage["prompt_tokens"] + eval_usage["prompt_tokens"],
                total_completion_tokens=plan_usage["completion_tokens"] + eval_usage["completion_tokens"],
                total_tokens=plan_usage["total_tokens"] + eval_usage["total_tokens"]
            )

            # 7. 使用 best_model.pkl 对测试集预测（不重新训练）
            # 【关键修复】提前保存训练代码到 data_dir.parent，供注入式脚本查找
            task_state = task_manager.get_task(task_id)
            if task_state and task_state.best_code:
                try:
                    (data_dir.parent / "code_best.py").write_text(task_state.best_code, encoding='utf-8')
                    print(f"[DEBUG] Saved code_best.py to {data_dir.parent / 'code_best.py'}")
                except Exception as e:
                    print(f"[DEBUG] Failed to save code_best.py: {e}")
            # 调试：打印 data_dir 内容
            try:
                files_in_data = list(data_dir.iterdir())
                print(f"[DEBUG] data_dir={data_dir} files: {[f.name for f in files_in_data]}")
            except Exception as e:
                print(f"[DEBUG] Failed to list data_dir: {e}")
            pred_start = time.time()
            pred_path = self._run_test_prediction(data_dir, task_cfg)
            pred_seconds = time.time() - pred_start
            if result.timing:
                result.timing.test_prediction_seconds = pred_seconds

            # 8. 计算测试集指标（与验证集指标保持一致）
            if pred_path and Path(pred_path).exists():
                test_metrics = self._compute_test_metrics(
                    pred_path, task_cfg.ground_truth_path, task_cfg.task_type, result.val_metrics, task_cfg.eval_metric
                )
                result.test_metrics = test_metrics
            else:
                logger.warning(f"[BenchmarkEvaluator] 任务 {task_id} 未生成测试集预测结果")

            # 9. LLM Judge 评估
            if result.success:
                judge_result = self.judge_agent.judge(
                    task_type=task_cfg.task_type,
                    target_column=task_cfg.target_column,
                    eval_metric=task_cfg.eval_metric,
                    val_metrics=result.val_metrics,
                    test_metrics=result.test_metrics
                )
                result.judge_accepted = judge_result.accepted
                result.judge_analysis = judge_result.analysis
                result.judge_reason = judge_result.reason

            # 10. 保存中间结果（含详细日志）
            result_dir = self._save_intermediate_results(
                result, task_cfg, run_index, task_state, data_dir, engine=engine
            )
            result.result_dir = str(result_dir)

            # 清理
            self._cleanup_task(task_id)

        except Exception as e:
            logger.exception(f"[BenchmarkEvaluator] 任务 {task_cfg.task_name} 第 {run_index} 次运行异常")
            result.error_message = f"运行异常: {str(e)}"
            # 异常时也要恢复 source 中的测试集
            try:
                if source_test_was_hidden and source_test_hidden and source_test_hidden.exists():
                    self._restore_test_csv(source_test_hidden, source_test_path)
            except:
                pass
            if result.task_id:
                try:
                    self._cleanup_task(result.task_id)
                except:
                    pass

        result.duration_seconds = time.time() - task_start
        return result

    @staticmethod
    def _restore_test_csv(test_csv_hidden: Path, test_csv_path: Path) -> bool:
        """安全恢复测试集：如果目标文件已存在则先删除"""
        if test_csv_hidden.exists():
            if test_csv_path.exists():
                test_csv_path.unlink()
            test_csv_hidden.rename(test_csv_path)
            return True
        return False

    def _wait_for_phase(self, task_id: str, target_phases: List[FastTaskPhase], timeout: int = 1200, interval: int = 2) -> bool:
        """轮询等待任务到达目标阶段之一"""
        start = time.time()
        while time.time() - start < timeout:
            task = task_manager.get_task(task_id)
            if not task:
                return False
            if task.phase in target_phases:
                return True
            time.sleep(interval)
        return False

    @staticmethod
    def _extract_definitions_from_code(code: str) -> str:
        """
        从训练代码中提取所有顶层 class/function 定义 + "安全"赋值语句 + import 语句，用于注入预测脚本。
        【关键过滤】
        1. 只提取目标为简单变量名的赋值（排除 pipeline.named_steps['model'] = x 这种）
        2. 赋值语句右侧不能包含函数调用（如 pipeline.predict()），否则会在预测阶段执行训练代码
        3. 排除可能覆盖预测脚本已有变量的赋值（pipeline/model/preprocessor 等）
        4. 排除危险 import（os/subprocess/socket 等）
        这样即使模型 pickle 中引用了自定义类/函数，预测脚本中也会有这些定义。
        """
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ""
        
        # 预测脚本中已有的变量名，避免被训练代码赋值覆盖
        _FORBIDDEN_ASSIGN_NAMES = {
            'pipeline', 'model', 'preprocessor', 'estimator', 'clf', 'regressor',
            'classifier', 'df', 'data', 'train', 'test', 'valid', 'X', 'y',
            'X_train', 'X_test', 'y_train', 'y_test', 'train_df', 'test_df',
            'train_pred', 'valid_pred', 'preds', 'predictions', 'probas',
            'train_auc', 'valid_auc', 'result', 'submission', 'output',
            'features', 'target', 'label', 'id_col', 'id_column',
            'model_obj', 'load_error', 'last_error', 'strategies', 'probs',
            'proba_matrix', 'X_pred', 'X_pipe',
        }
        
        # 危险模块，不允许在注入脚本中 import
        _DANGEROUS_MODULES = {'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 'http'}
        
        def _has_call(node) -> bool:
            """递归检查 AST 节点中是否包含函数调用"""
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    return True
            return False
        
        def _is_simple_name_target(node) -> bool:
            """检查赋值目标是否都是简单变量名"""
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    return False
            return True
        
        def _is_safe_import(node) -> bool:
            """检查 import 语句是否安全（不涉及危险模块）"""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split('.')[0]
                    if top_module in _DANGEROUS_MODULES:
                        return False
                return True
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_module = node.module.split('.')[0]
                    if top_module in _DANGEROUS_MODULES:
                        return False
                return True
            return False
        
        lines = code.split('\n')
        segments = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                # 提取 class/function 定义
                start_line = node.lineno - 1  # 0-based
                end_line = getattr(node, 'end_lineno', node.lineno)
                if end_line and end_line <= len(lines):
                    def_text = '\n'.join(lines[start_line:end_line])
                    segments.append(def_text)
            elif isinstance(node, ast.Assign):
                # 【关键过滤】只提取"安全"赋值：
                # 1. 目标必须是简单变量名（排除 Subscript/Attribute 目标）
                # 2. 右侧不能包含函数调用
                # 3. 变量名不在禁止列表中
                if not _is_simple_name_target(node):
                    continue
                if _has_call(node):
                    continue
                # 检查所有目标变量名
                target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
                if target_names & _FORBIDDEN_ASSIGN_NAMES:
                    continue
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', node.lineno)
                if end_line and end_line <= len(lines):
                    assign_text = '\n'.join(lines[start_line:end_line])
                    segments.append(assign_text)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # 提取安全的 import 语句，确保注入的代码有必要的导入
                if not _is_safe_import(node):
                    continue
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', node.lineno)
                if end_line and end_line <= len(lines):
                    import_text = '\n'.join(lines[start_line:end_line])
                    segments.append(import_text)
        
        return '\n\n'.join(segments) if segments else ""

    def _run_test_prediction(self, data_dir: Path, task_cfg: BenchmarkTaskConfig) -> Optional[Path]:
        """
        使用 best_model.pkl 对测试集进行预测（不重新训练）

        策略优先级：
        1. 执行 LLM 生成的 predict.py
        2. 【新增】从训练代码提取自定义定义，构造注入式预测脚本
        3. 回退到内置通用预测模板
        """
        model_path = data_dir / "best_model.pkl"
        # 如果 data_dir 中没有 best_model.pkl，尝试从 artifacts 复制
        if not model_path.exists():
            artifacts_model = data_dir.parent / "artifacts" / "model.pkl"
            if artifacts_model.exists():
                shutil.copy2(artifacts_model, model_path)
                logger.info(f"[BenchmarkEvaluator] 从 artifacts 复制模型到 {model_path}")
        
        if not model_path.exists():
            logger.warning(f"[BenchmarkEvaluator] 未找到 best_model.pkl，跳过测试集预测")
            return None
        
        # 【关键修复】确保 test.csv 在 data_dir 中
        test_csv_path = data_dir / "test.csv"
        if not test_csv_path.exists() and task_cfg.test_path:
            source_test = Path(task_cfg.test_path)
            if source_test.exists():
                shutil.copy2(source_test, test_csv_path)
                logger.info(f"[BenchmarkEvaluator] 从 source 复制测试集到 {test_csv_path}")

        # ========== 策略1: 执行 LLM 生成的 predict.py ==========
        predict_py_path = data_dir / "predict.py"
        if not predict_py_path.exists():
            predict_py_path = data_dir.parent / "artifacts" / "predict.py"
        if predict_py_path.exists():
            logger.info(f"[BenchmarkEvaluator] 发现 LLM 生成的 predict.py，优先执行")
            # 【修复】兼容 predict.py 中可能硬编码的 output/model.pkl
            output_dir = data_dir / "output"
            output_dir.mkdir(exist_ok=True)
            best_model = data_dir / "best_model.pkl"
            if best_model.exists() and not (output_dir / "model.pkl").exists():
                try:
                    shutil.copy2(best_model, output_dir / "model.pkl")
                    logger.info(f"[BenchmarkEvaluator] 已复制 best_model.pkl 到 output/model.pkl 以兼容 predict.py")
                except Exception as e:
                    logger.warning(f"[BenchmarkEvaluator] 复制模型到 output 失败: {e}")
            try:
                with open(predict_py_path, 'r', encoding='utf-8') as f:
                    llm_predict_code = f.read()
                
                wrapped_code = f"""
import pandas as pd
import sys
import os
try:
    import dill
except ImportError:
    pass

{llm_predict_code}
"""
                result = sandbox_executor.execute(
                    code=wrapped_code,
                    data_dir=data_dir,
                    task_type=task_cfg.task_type.value,
                    artifact_mode=True,
                    artifact_output_dir=data_dir
                )
                
                if result.success:
                    pred_path = data_dir / "eval_predictions.csv"
                    if pred_path.exists():
                        logger.info(f"[BenchmarkEvaluator] predict.py 执行成功，测试集预测完成: {pred_path}")
                        return pred_path
                    else:
                        logger.warning(f"[BenchmarkEvaluator] predict.py 执行成功但未生成 eval_predictions.csv")
                else:
                    logger.warning(f"[BenchmarkEvaluator] predict.py 执行失败: {result.error_message}")
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] predict.py 执行异常: {e}")

        # ========== 策略2: 【核心新增】从训练代码提取自定义定义，构造注入式预测脚本 ==========
        print(f"[DEBUG] Strategy2: data_dir={data_dir}, parent files={list(data_dir.parent.iterdir()) if data_dir.parent.exists() else 'N/A'}")
        code_best_path = data_dir.parent / "code_best.py"
        print(f"[DEBUG] code_best_path={code_best_path}, exists={code_best_path.exists()}")
        if not code_best_path.exists():
            # 尝试从 result_dir 查找
            result_dirs = list(data_dir.parent.glob("run_*"))
            if result_dirs:
                code_best_path = result_dirs[0] / "code_best.py"
                print(f"[DEBUG] Fallback code_best_path={code_best_path}")
        
        injected_defs = ""
        train_code = ""
        if code_best_path.exists():
            try:
                with open(code_best_path, 'r', encoding='utf-8') as f:
                    train_code = f.read()
                print(f"[DEBUG] Loaded code_best.py ({len(train_code)} chars)")
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] 读取训练代码失败: {e}")
        
        # 如果文件找不到，尝试从 task_manager 获取
        if not train_code:
            task_id_from_dir = data_dir.parent.name
            try:
                from app.core.state import task_manager
                tm_state = task_manager.get_task(task_id_from_dir)
                if tm_state and tm_state.best_code:
                    train_code = tm_state.best_code
                    print(f"[DEBUG] Got code from task_manager ({len(train_code)} chars)")
                else:
                    print(f"[DEBUG] task_manager has no best_code for {task_id_from_dir}")
            except Exception as e:
                print(f"[DEBUG] Failed to get code from task_manager: {e}")
        
        if train_code:
            try:
                injected_defs = self._extract_definitions_from_code(train_code)
                if injected_defs:
                    print(f"[DEBUG] Extracted {len(injected_defs)} chars of definitions")
                else:
                    print(f"[DEBUG] No definitions extracted")
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] 提取训练代码定义失败: {e}")
        else:
            print(f"[DEBUG] No train_code available for injection")
        
        if injected_defs:
            logger.info(f"[BenchmarkEvaluator] 尝试使用注入式预测脚本（含训练代码自定义定义）")
            id_col = task_cfg.id_column or "id"
            
            # 构造注入代码块：非空时才包 try-except，防止空 try 语法错误
            injected_defs_clean = injected_defs.strip()
            if injected_defs_clean:
                # 【关键修复】将每个顶层定义放入独立的 try 块，避免一个赋值失败导致所有后续定义被跳过
                injected_blocks = []
                for segment in injected_defs.split('\n\n'):
                    segment = segment.strip()
                    if not segment:
                        continue
                    indented = '\n'.join('    ' + line for line in segment.split('\n'))
                    injected_blocks.append(f"""try:
{indented}
except Exception as _e:
    print('INJECTED_DEFS_SKIPPED: ' + str(_e))""")
                injected_defs_try_block = '\n\n'.join(injected_blocks)
            else:
                injected_defs_try_block = "pass  # no definitions to inject"
            
            injected_predict_code = f"""
import pandas as pd
import dill
import numpy as np
import sys
import types

# ========== 【关键】注入训练代码中的自定义类/函数定义 ==========
# 在 try 块中执行注入的赋值，防止缺失变量导致脚本崩溃
{injected_defs_try_block}

# ========== 加载模型（优先 dill，支持自定义函数序列化）==========
model_obj = None
load_error = None
try:
    with open('data/best_model.pkl', 'rb') as f:
        model_obj = dill.load(f)
except Exception as e:
    load_error = e
    try:
        import pickle
        with open('data/best_model.pkl', 'rb') as f:
            model_obj = pickle.load(f)
    except Exception as e2:
        load_error = f"dill: {{e}}, pickle: {{e2}}"

if model_obj is None:
    raise RuntimeError(f"模型加载失败: {{load_error}}")

preprocessor = None
model = model_obj
if isinstance(model_obj, dict):
    preprocessor = model_obj.get('preprocessor')
    model = model_obj.get('model') or model_obj
    print(f'MODEL_DICT keys={{list(model_obj.keys())}}')

# ========== 加载测试集 ==========
test = pd.read_csv('data/test.csv')

# ========== 【修复】如果注入了 extract_time_features，在测试集上显式调用 ==========
if 'extract_time_features' in dir():
    try:
        _time_col = 'dteday' if 'dteday' in test.columns else ('datetime' if 'datetime' in test.columns else None)
        if _time_col:
            test = extract_time_features(test, _time_col)
            print('EXTRACT_TIME_FEATURES_APPLIED')
    except Exception as _e:
        print('EXTRACT_TIME_FEATURES_FAILED: ' + str(_e))

# ========== 【关键】如果训练代码定义了 prepare_for_prediction，先调用它 ==========
if 'prepare_for_prediction' in dir():
    try:
        test = prepare_for_prediction(test)
        print('PREPARE_FOR_PREDICTION_APPLIED')
    except Exception as e:
        print('PREPARE_FOR_PREDICTION_FAILED: ' + str(e))

# ========== 时间特征自动提取（应对训练时手动提取但未 Pipeline 化）==========
for col in list(test.columns):
    if test[col].dtype == 'object':
        try:
            dt = pd.to_datetime(test[col], errors='coerce')
            if dt.notna().sum() > len(test) * 0.3:
                test[f"{{col}}_year"] = dt.dt.year
                test[f"{{col}}_month"] = dt.dt.month
                test[f"{{col}}_day"] = dt.dt.day
                test[f"{{col}}_hour"] = dt.dt.hour
                test[f"{{col}}_dayofweek"] = dt.dt.dayofweek
                # 【修复】同时生成不带前缀的版本，兼容训练代码手动提取的命名
                if 'year' not in test.columns:
                    test['year'] = dt.dt.year
                if 'month' not in test.columns:
                    test['month'] = dt.dt.month
                if 'day' not in test.columns:
                    test['day'] = dt.dt.day
                if 'hour' not in test.columns:
                    test['hour'] = dt.dt.hour
                if 'weekday' not in test.columns:
                    test['weekday'] = dt.dt.dayofweek
                print(f'TIME_FEATURE_EXTRACTED from {{col}}')
        except Exception:
            pass

# ========== 数值型时间特征自动推断（应对训练代码生成 month_sin/hour_cos 等但未 Pipeline 化）==========
import math
if 'month' in test.columns and 'month_sin' not in test.columns:
    test['month_sin'] = np.sin(2 * math.pi * test['month'] / 12)
    test['month_cos'] = np.cos(2 * math.pi * test['month'] / 12)
    print('AUTO_FEATURE: month_sin, month_cos')
if 'hour' in test.columns and 'hour_sin' not in test.columns:
    test['hour_sin'] = np.sin(2 * math.pi * test['hour'] / 24)
    test['hour_cos'] = np.cos(2 * math.pi * test['hour'] / 24)
    print('AUTO_FEATURE: hour_sin, hour_cos')
if 'day' in test.columns and 'day_sin' not in test.columns:
    test['day_sin'] = np.sin(2 * math.pi * test['day'] / 31)
    test['day_cos'] = np.cos(2 * math.pi * test['day'] / 31)
    print('AUTO_FEATURE: day_sin, day_cos')
if 'year' in test.columns and 'year_norm' not in test.columns:
    y_min = test['year'].min()
    y_max = test['year'].max()
    test['year_norm'] = (test['year'] - y_min) / (y_max - y_min + 1e-8)
    print('AUTO_FEATURE: year_norm')

# ========== 预测 ==========
X_test = None
preds = None
strategies = []

if hasattr(model, 'feature_names_in_'):
    strategies.append(('feature_names_in_', lambda: test[[c for c in model.feature_names_in_ if c in test.columns]]))
if hasattr(model, 'feature_name_'):
    strategies.append(('feature_name_', lambda: test[[c for c in model.feature_name_ if c in test.columns]]))
if hasattr(model, 'booster_'):
    try:
        fn = model.booster_.feature_name()
        strategies.append(('booster_feature_name', lambda: test[[c for c in fn if c in test.columns]]))
    except Exception:
        pass
strategies.append(('all_columns', lambda: test))
if 'id' in test.columns:
    strategies.append(('drop_id', lambda: test.drop(columns=['id'])))

def _encode_objects(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
    return df
strategies.append(('encode_objects', lambda: _encode_objects(test)))

last_error = None
for name, strategy in strategies:
    try:
        X_test = strategy()
        if X_test is None:
            continue
        X_pred = preprocessor.transform(X_test) if preprocessor else X_test
        try:
            preds = model.predict(X_pred)
        except TypeError as te:
            if "is not an estimator instance" in str(te) and hasattr(model, 'steps'):
                X_pipe = X_pred
                for step_name, step in model.steps[:-1]:
                    if hasattr(step, 'transform'):
                        X_pipe = step.transform(X_pipe)
                last_step = model.steps[-1][1]
                preds = last_step.predict(X_pipe)
                print(f'PREDICT_OK strategy={{name}}_pipeline_fallback shape={{X_test.shape}}')
            elif "'<' not supported between instances of" in str(te):
                X_test_str = X_test.copy()
                for col in X_test_str.columns:
                    X_test_str[col] = X_test_str[col].astype(str)
                X_pred_str = preprocessor.transform(X_test_str) if preprocessor else X_test_str
                preds = model.predict(X_pred_str)
                print(f'PREDICT_OK strategy={{name}}_typefix shape={{X_test_str.shape}}')
            else:
                raise
        print(f'PREDICT_OK strategy={{name}} shape={{X_test.shape}}')
        break
    except Exception as e:
        last_error = e
        print(f'PREDICT_FAIL strategy={{name}}: {{e}}')
        continue

if preds is None:
    raise last_error or RuntimeError('所有预测策略均失败')

# 概率预测
probs = None
proba_matrix = None
try:
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
        if probas.ndim > 1 and probas.shape[1] >= 2:
            probs = probas[:, -1]
            if probas.shape[1] > 2:
                proba_matrix = probas
        else:
            probs = probas.flatten()
except Exception:
    pass

# 获取 id 列并保存结果
id_col = '{id_col}'
if id_col not in test.columns:
    id_col = test.columns[0]

result = pd.DataFrame({{id_col: test[id_col], 'prediction': preds}})
if probs is not None:
    result['probability'] = probs
if proba_matrix is not None:
    for i in range(proba_matrix.shape[1]):
        result[f'proba_{{i}}'] = proba_matrix[:, i]
result.to_csv('output/eval_predictions.csv', index=False)
print('EVAL_PREDICTIONS_SAVED')
"""
            try:
                result = sandbox_executor.execute(
                    code=injected_predict_code,
                    data_dir=data_dir,
                    task_type=task_cfg.task_type.value,
                    artifact_mode=True,
                    artifact_output_dir=data_dir
                )
                if result.error_message and "安全检查未通过" in result.error_message:
                    # 语法错误时打印注入代码以便调试
                    logger.error(f"[BenchmarkEvaluator] 注入式脚本安全检查失败，完整代码:\n{injected_predict_code}")
                
                if result.success:
                    pred_path = data_dir / "eval_predictions.csv"
                    if pred_path.exists():
                        logger.info(f"[BenchmarkEvaluator] 注入式预测脚本执行成功，测试集预测完成: {pred_path}")
                        return pred_path
                    else:
                        logger.warning(f"[BenchmarkEvaluator] 注入式预测脚本执行成功但未生成 eval_predictions.csv")
                else:
                    logger.warning(f"[BenchmarkEvaluator] 注入式预测脚本执行失败: {result.error_message}")
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] 注入式预测脚本执行异常: {e}")

        # ========== 策略3: 回退到内置通用预测模板 ==========
        id_col = task_cfg.id_column or "id"

        predict_code = f"""
import pandas as pd
import pickle
import numpy as np
import sys
import types

# 加载模型（可能为 dict 格式 {{'preprocessor': ..., 'model': ...}}）
# 【修复】优先使用 dill 加载（支持自定义函数序列化），失败再回退到 pickle
model_obj = None
load_error = None
try:
    import dill
    with open('data/best_model.pkl', 'rb') as f:
        model_obj = dill.load(f)
except Exception as e:
    load_error = e
    # 回退到 pickle（处理训练脚本里自定义函数引用）
    class _DummyMain(types.ModuleType):
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    _old_main = sys.modules.get('__main__')
    try:
        sys.modules['__main__'] = _DummyMain('__main__')
        with open('data/best_model.pkl', 'rb') as f:
            model_obj = pickle.load(f)
    except AttributeError as _ae:
        # 如果仍失败，尝试用 joblib 加载
        try:
            import joblib
            model_obj = joblib.load('data/best_model.pkl')
        except Exception:
            raise _ae
    finally:
        if _old_main is not None:
            sys.modules['__main__'] = _old_main

if model_obj is None:
    raise RuntimeError("模型加载失败: dill=" + str(load_error))

preprocessor = None
model = model_obj
if isinstance(model_obj, dict):
    preprocessor = model_obj.get('preprocessor')
    model = model_obj.get('model') or model_obj
    print(f'MODEL_DICT keys={{list(model_obj.keys())}}')

# 加载测试集
test = pd.read_csv('data/test.csv')

# 【修复1】时间特征工程对齐：从 datetime 列自动提取 year/month/day/hour
# （应对训练代码手动做了时间拆分但未放进 Pipeline 的情况）
for col in list(test.columns):
    if test[col].dtype == 'object':
        try:
            dt = pd.to_datetime(test[col], errors='coerce')
            if dt.notna().sum() > len(test) * 0.3:  # 超过30%能解析为日期
                test[f"{{col}}_year"] = dt.dt.year
                test[f"{{col}}_month"] = dt.dt.month
                test[f"{{col}}_day"] = dt.dt.day
                test[f"{{col}}_hour"] = dt.dt.hour
                test[f"{{col}}_dayofweek"] = dt.dt.dayofweek
                # 【修复】同时生成不带前缀的版本，兼容训练代码手动提取的命名
                if 'year' not in test.columns:
                    test['year'] = dt.dt.year
                if 'month' not in test.columns:
                    test['month'] = dt.dt.month
                if 'day' not in test.columns:
                    test['day'] = dt.dt.day
                if 'hour' not in test.columns:
                    test['hour'] = dt.dt.hour
                if 'weekday' not in test.columns:
                    test['weekday'] = dt.dt.dayofweek
                print(f'TIME_FEATURE_EXTRACTED from {{col}}')
        except Exception:
            pass

# 【修复2】数值型时间特征自动推断（应对训练代码生成 month_sin/hour_cos 等但未 Pipeline 化）
import math
if 'month' in test.columns and 'month_sin' not in test.columns:
    test['month_sin'] = np.sin(2 * math.pi * test['month'] / 12)
    test['month_cos'] = np.cos(2 * math.pi * test['month'] / 12)
    print('AUTO_FEATURE: month_sin, month_cos')
if 'hour' in test.columns and 'hour_sin' not in test.columns:
    test['hour_sin'] = np.sin(2 * math.pi * test['hour'] / 24)
    test['hour_cos'] = np.cos(2 * math.pi * test['hour'] / 24)
    print('AUTO_FEATURE: hour_sin, hour_cos')
if 'day' in test.columns and 'day_sin' not in test.columns:
    test['day_sin'] = np.sin(2 * math.pi * test['day'] / 31)
    test['day_cos'] = np.cos(2 * math.pi * test['day'] / 31)
    print('AUTO_FEATURE: day_sin, day_cos')
if 'year' in test.columns and 'year_norm' not in test.columns:
    y_min = test['year'].min()
    y_max = test['year'].max()
    test['year_norm'] = (test['year'] - y_min) / (y_max - y_min + 1e-8)
    print('AUTO_FEATURE: year_norm')

# 推断特征列：尝试多种策略，自动回退
X_test = None
preds = None
strategies = []

# 策略1: sklearn Pipeline / 部分模型保存的 feature_names_in_
if hasattr(model, 'feature_names_in_'):
    strategies.append(('feature_names_in_', lambda: test[[c for c in model.feature_names_in_ if c in test.columns]]))

# 策略2: LightGBM 的 feature_name_
if hasattr(model, 'feature_name_'):
    strategies.append(('feature_name_', lambda: test[[c for c in model.feature_name_ if c in test.columns]]))

# 策略3: XGBoost / LightGBM 底层 booster
if hasattr(model, 'booster_'):
    try:
        fn = model.booster_.feature_name()
        strategies.append(('booster_feature_name', lambda: test[[c for c in fn if c in test.columns]]))
    except Exception:
        pass

# 策略4: 使用测试集全部列（训练时可能把 id 也当作特征）
strategies.append(('all_columns', lambda: test))

# 策略5: 排除 id 列
if 'id' in test.columns:
    strategies.append(('drop_id', lambda: test.drop(columns=['id'])))

# 策略6: 对 object 列进行 factorize 编码（训练时可能已编码）
def _encode_objects(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
    return df
strategies.append(('encode_objects', lambda: _encode_objects(test)))

# 依次尝试，直到预测成功
last_error = None
for name, strategy in strategies:
    try:
        X_test = strategy()
        if X_test is None:
            continue
        # 如果有预处理器，先转换特征
        X_pred = preprocessor.transform(X_test) if preprocessor else X_test
        # 尝试预测
        try:
            preds = model.predict(X_pred)
        except TypeError as te:
            # 处理 Pipeline 的 last_step 不是 sklearn estimator 的情况
            if "is not an estimator instance" in str(te) and hasattr(model, 'steps'):
                # 手动执行 Pipeline 的前 n-1 步 transform，然后对 last_step 调用 predict
                X_pipe = X_pred
                for step_name, step in model.steps[:-1]:
                    if hasattr(step, 'transform'):
                        X_pipe = step.transform(X_pipe)
                last_step = model.steps[-1][1]
                preds = last_step.predict(X_pipe)
                print(f'PREDICT_OK strategy={{name}}_pipeline_fallback shape={{X_test.shape}}')
            # 【修复2】处理 int/str 混合类型导致 OneHotEncoder 报错
            elif "'<' not supported between instances of" in str(te):
                X_test_str = X_test.copy()
                for col in X_test_str.columns:
                    X_test_str[col] = X_test_str[col].astype(str)
                X_pred_str = preprocessor.transform(X_test_str) if preprocessor else X_test_str
                preds = model.predict(X_pred_str)
                print(f'PREDICT_OK strategy={{name}}_typefix shape={{X_test_str.shape}}')
            else:
                raise
        print(f'PREDICT_OK strategy={{name}} shape={{X_test.shape}}')
        break
    except Exception as e:
        last_error = e
        print(f'PREDICT_FAIL strategy={{name}}: {{e}}')
        continue

if preds is None:
    raise last_error or RuntimeError('所有预测策略均失败')

# 概率预测（如果模型支持，用于计算 AUC / Log Loss 等需要概率的指标）
probs = None
proba_matrix = None
try:
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
        if probas.ndim > 1 and probas.shape[1] >= 2:
            # 二分类：取正类概率（最后一列）
            probs = probas[:, -1]
            # 多分类：保存完整概率矩阵（用于 Log Loss 计算）
            if probas.shape[1] > 2:
                proba_matrix = probas
        else:
            probs = probas.flatten()
except Exception:
    pass

# 尝试获取 id 列
id_col = '{id_col}'
if id_col not in test.columns:
    id_col = test.columns[0]

# 保存预测结果到 output/ 目录（沙箱会自动收集）
result = pd.DataFrame({{id_col: test[id_col], 'prediction': preds}})
if probs is not None:
    result['probability'] = probs
# 多分类概率矩阵：保存为 proba_0, proba_1, ... 列
if proba_matrix is not None:
    for i in range(proba_matrix.shape[1]):
        result[f'proba_{{i}}'] = proba_matrix[:, i]
result.to_csv('output/eval_predictions.csv', index=False)
print('EVAL_PREDICTIONS_SAVED')
"""

        try:
            result = sandbox_executor.execute(
                code=predict_code,
                data_dir=data_dir,
                task_type=task_cfg.task_type.value,
                artifact_mode=True,  # 允许写入文件
                artifact_output_dir=data_dir
            )

            if not result.success:
                logger.error(f"[BenchmarkEvaluator] 测试集预测脚本执行失败: {result.error_message}")
                return None

            pred_path = data_dir / "eval_predictions.csv"
            if pred_path.exists():
                logger.info(f"[BenchmarkEvaluator] 测试集预测完成: {pred_path}")
                return pred_path
            else:
                logger.warning(f"[BenchmarkEvaluator] 预测脚本执行成功但未生成 eval_predictions.csv")
                return None

        except Exception as e:
            logger.exception(f"[BenchmarkEvaluator] 测试集预测异常: {e}")
            return None

    def _compute_test_metrics(self, pred_path: str, gt_path: str, task_type: TaskType, val_metrics, eval_metric: Optional[str] = None) -> TestSetMetrics:
        """
        计算测试集指标，确保与验证集使用的指标一致。

        策略：
        1. 如果指定了 eval_metric，优先计算该指标
        2. 否则根据 val_metrics 中非空字段推断验证时计算了哪些指标
        3. 二分类 AUC 优先使用 probability 列（需要概率值），回退到 prediction 列
        """
        pred_path_obj = Path(pred_path)
        gt_path_obj = Path(gt_path)
        if not pred_path_obj.exists():
            logger.warning(f"[BenchmarkEvaluator] 预测文件不存在: {pred_path}")
            return TestSetMetrics()
        if not gt_path_obj.exists():
            logger.warning(f"[BenchmarkEvaluator] 真实标签文件不存在: {gt_path}")
            return TestSetMetrics()
        
        pred_df = pd.read_csv(pred_path)
        gt_df = pd.read_csv(gt_path)

        # 对齐：根据 id 列合并
        # 【修复】智能推断 id 列：优先使用 pred_df 第一列，如果 gt_df 中没有该列，则尝试 gt_df 的第一列
        pred_id_col = pred_df.columns[0]
        gt_id_col = gt_df.columns[0]
        if pred_id_col in gt_df.columns:
            id_col = pred_id_col
        elif gt_id_col in pred_df.columns:
            id_col = gt_id_col
        else:
            # 查找共同列（排除 prediction/probability 等预测列）
            common_cols = [c for c in pred_df.columns if c in gt_df.columns and c not in ('prediction', 'probability')]
            if common_cols:
                id_col = common_cols[0]
            else:
                logger.warning(f"[BenchmarkEvaluator] 预测结果与 ground_truth 无共同列，无法计算指标")
                return TestSetMetrics()
        
        # 【关键修复】统一 id 列类型为字符串，防止 int64 vs object merge 失败
        pred_df[id_col] = pred_df[id_col].astype(str)
        gt_df[id_col] = gt_df[id_col].astype(str)
        
        merged = pd.merge(pred_df, gt_df, on=id_col, how="inner")

        if len(merged) == 0:
            logger.warning(f"[BenchmarkEvaluator] 预测结果与 ground_truth 无交集，无法计算指标")
            return TestSetMetrics()

        y_true = merged.iloc[:, -1]  # ground_truth 的 target 列
        y_pred = merged["prediction"]

        # 统一标签类型：处理 ground_truth 为字符串但预测为数值的情况
        #（如 ground_truth='No'/'Yes'，prediction=0/1）
        if y_true.dtype == 'object' and str(y_pred.dtype) in ['int64', 'int32', 'float64', 'bool', 'int']:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_true = le.fit_transform(y_true)
        elif str(y_pred.dtype) == 'object' and str(y_true.dtype) in ['int64', 'int32', 'float64', 'bool', 'int']:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_pred = le.fit_transform(y_pred)

        # 分类任务：确保预测值为整数标签（处理回归器误用于分类或浮点预测的情况）
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            try:
                y_pred = y_pred.round().astype(int)
                y_true = y_true.round().astype(int)
                logger.info(f'[BenchmarkEvaluator] 分类标签已转换为整数: y_true unique={sorted(y_true.unique())}, y_pred unique={sorted(y_pred.unique())}')
            except Exception as e:
                logger.warning(f'[BenchmarkEvaluator] 分类标签整数转换失败: {e}')

        # AUC 需要概率值：优先使用 probability 列
        y_proba = merged["probability"] if "probability" in merged.columns else None
        
        # 处理 NaN
        if y_true.isna().any():
            logger.warning(f"[BenchmarkEvaluator] y_true 包含 {y_true.isna().sum()} 个 NaN，已丢弃")
            valid_mask = y_true.notna()
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if y_proba is not None:
                y_proba = y_proba[valid_mask]
        if y_pred.isna().any():
            logger.warning(f"[BenchmarkEvaluator] y_pred 包含 {y_pred.isna().sum()} 个 NaN，已丢弃")
            valid_mask = y_pred.notna()
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if y_proba is not None:
                y_proba = y_proba[valid_mask]

        metrics = TestSetMetrics()

        # 推断验证集使用的主要指标：从 val_metrics 中非空字段判断
        has_val_auc = val_metrics is not None and val_metrics.val_auc is not None
        has_val_acc = val_metrics is not None and val_metrics.val_accuracy is not None
        has_val_rmse = val_metrics is not None and val_metrics.val_rmse is not None
        has_val_score = val_metrics is not None and val_metrics.val_score is not None

        # 如果指定了 eval_metric，记录日志
        if eval_metric:
            logger.info(f"[BenchmarkEvaluator] 使用任务指定评估指标: {eval_metric}")

        try:
            if task_type == TaskType.BINARY_CLASSIFICATION:
                # 二分类
                # AUC：需要概率值
                if has_val_auc or eval_metric == 'AUC':
                    try:
                        if y_proba is not None:
                            metrics.auc = float(roc_auc_score(y_true, y_proba))
                        else:
                            # 回退：使用标签（不推荐，但兼容不支持 predict_proba 的模型）
                            metrics.auc = float(roc_auc_score(y_true, y_pred))
                        logger.info(f"[BenchmarkEvaluator] 测试集 AUC={metrics.auc:.4f} (使用{'概率' if y_proba is not None else '标签'})")
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 AUC 计算失败: {e}")

                # Accuracy
                if has_val_acc or has_val_score or eval_metric == 'Accuracy':
                    try:
                        metrics.accuracy = float(accuracy_score(y_true, y_pred))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 Accuracy 计算失败: {e}")

                # F1
                if eval_metric in (None, 'F1', 'Accuracy', 'AUC'):
                    try:
                        metrics.f1 = float(f1_score(y_true, y_pred, average="binary"))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 F1 计算失败: {e}")

            elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                # 多分类
                if has_val_acc or has_val_score or eval_metric == 'Accuracy':
                    try:
                        metrics.accuracy = float(accuracy_score(y_true, y_pred))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 Accuracy 计算失败: {e}")

                if eval_metric in (None, 'F1-macro', 'F1', 'Accuracy'):
                    try:
                        metrics.f1_macro = float(f1_score(y_true, y_pred, average="macro"))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 F1-macro 计算失败: {e}")

                # Log Loss（多分类对数损失）：需要完整概率矩阵
                if eval_metric == 'Log Loss':
                    try:
                        # 从预测结果中读取 proba_0, proba_1, ... 列
                        proba_cols = [c for c in merged.columns if c.startswith('proba_')]
                        if proba_cols:
                            proba_cols = sorted(proba_cols, key=lambda x: int(x.split('_')[1]))
                            y_proba_matrix = merged[proba_cols].values
                            from sklearn.metrics import log_loss
                            metrics.log_loss = float(log_loss(y_true, y_proba_matrix))
                            logger.info(f"[BenchmarkEvaluator] 测试集 Log Loss={metrics.log_loss:.4f}")
                        else:
                            logger.warning("[BenchmarkEvaluator] 未找到概率矩阵列(proba_*)，无法计算 Log Loss")
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 Log Loss 计算失败: {e}")

            elif task_type == TaskType.REGRESSION:
                # 回归
                if has_val_rmse or has_val_score or eval_metric == 'RMSE':
                    try:
                        metrics.rmse = float(_rmse_func(y_true, y_pred))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 RMSE 计算失败: {e}")

                if eval_metric in (None, 'MAE', 'RMSE', 'R2'):
                    try:
                        metrics.mae = float(mean_absolute_error(y_true, y_pred))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 MAE 计算失败: {e}")

                if eval_metric in (None, 'R2', 'RMSE', 'MAE'):
                    try:
                        metrics.r2 = float(r2_score(y_true, y_pred))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 R² 计算失败: {e}")

            # 动态计算未知指标（用户指定的指标不在上述硬编码列表中时）
            if eval_metric and not self._is_metric_already_computed(metrics, eval_metric):
                self._compute_unknown_metric(eval_metric, y_true, y_pred, y_proba, merged, metrics, task_type)

            logger.info(f"[BenchmarkEvaluator] 测试集指标计算完成: {metrics.model_dump()}")

        except Exception as e:
            logger.exception(f"[BenchmarkEvaluator] 指标计算异常: {e}")

        return metrics

    def _is_metric_already_computed(self, metrics: TestSetMetrics, eval_metric: str) -> bool:
        """检查指定指标是否已经在 metrics 中计算完成"""
        if not eval_metric:
            return True
        normalized = eval_metric.strip().lower().replace('-', '_').replace(' ', '_')
        known_map = {
            'auc': metrics.auc is not None,
            'accuracy': metrics.accuracy is not None,
            'f1': metrics.f1 is not None,
            'f1_macro': metrics.f1_macro is not None,
            'log_loss': metrics.log_loss is not None,
            'rmse': metrics.rmse is not None,
            'mae': metrics.mae is not None,
            'r2': metrics.r2 is not None,
        }
        return known_map.get(normalized, False)

    def _compute_unknown_metric(
        self,
        eval_metric: str,
        y_true,
        y_pred,
        y_proba,
        merged,
        metrics: TestSetMetrics,
        task_type: TaskType
    ):
        """
        尝试动态计算未知指标。通过 sklearn.metrics 查找同名或近名函数。
        支持常见指标如 Cohen's Kappa、Matthews MCC 等。
        """
        import sklearn.metrics as skm

        # 标准化指标名：转小写、替换空格和连字符为下划线
        normalized = eval_metric.strip().lower().replace('-', '_').replace(' ', '_')

        # 别名映射：常见指标别名 → sklearn 标准名（去掉 _score 后缀）
        alias_map = {
            'roc_auc': 'roc_auc',
            'auc_roc': 'roc_auc',
            'auc': 'roc_auc',
            'accuracy': 'accuracy',
            'acc': 'accuracy',
            'f1': 'f1',
            'f1_macro': 'f1',
            'macro_f1': 'f1',
            'macro_f1_score': 'f1',
            'f1_micro': 'f1',
            'f1_weighted': 'f1',
            'log_loss': 'log_loss',
            'multi_class_log_loss': 'log_loss',
            'multiclass_logloss': 'log_loss',
            'rmse': 'mean_squared_error',  # 需要后续开方
            'mae': 'mean_absolute_error',
            'r2': 'r2',
            'r_squared': 'r2',
            'cohen_kappa': 'cohen_kappa',
            'kappa': 'cohen_kappa',
            'matthews': 'matthews_corrcoef',
            'mcc': 'matthews_corrcoef',
            'precision': 'precision',
            'recall': 'recall',
        }

        # 先尝试别名映射
        if normalized in alias_map:
            normalized = alias_map[normalized]
        # 去掉常见后缀后再试一次
        else:
            for suffix in ['_score', '_loss', '_error']:
                if normalized.endswith(suffix):
                    base = normalized[:-len(suffix)]
                    if base in alias_map:
                        normalized = alias_map[base]
                    break

        # 尝试查找 sklearn.metrics 中的函数
        candidate_names = [
            normalized,
            normalized + '_score',
            normalized + '_loss',
            normalized + '_error',
        ]

        metric_func = None
        for name in candidate_names:
            if hasattr(skm, name):
                metric_func = getattr(skm, name)
                break

        if metric_func is None:
            logger.warning(f"[BenchmarkEvaluator] 未知指标 '{eval_metric}'（标准化: {normalized}），无法自动计算。"
                          f"已计算通用指标作为参考。")
            return

        try:
            # 判断指标需要标签还是概率
            # 需要概率的指标（通过函数签名检测）
            import inspect
            sig = inspect.signature(metric_func)
            param_names = list(sig.parameters.keys())

            # Log Loss / Brier Score 等需要概率
            if 'y_prob' in param_names or 'y_proba' in param_names or normalized in ('log_loss', 'brier_score'):
                proba_cols = [c for c in merged.columns if c.startswith('proba_')]
                if proba_cols:
                    proba_cols = sorted(proba_cols, key=lambda x: int(x.split('_')[1]))
                    y_proba_matrix = merged[proba_cols].values
                    result = metric_func(y_true, y_proba_matrix)
                elif y_proba is not None:
                    # 二分类概率
                    result = metric_func(y_true, y_proba)
                else:
                    logger.warning(f"[BenchmarkEvaluator] 指标 '{eval_metric}' 需要概率值但未找到")
                    return
            else:
                # 大多数指标只需要标签
                kwargs = {}
                if 'average' in param_names and task_type == TaskType.MULTICLASS_CLASSIFICATION:
                    kwargs['average'] = 'macro'
                result = metric_func(y_true, y_pred, **kwargs)

            # 保存到动态字段（通过 setattr 绕过 Pydantic 的固定字段）
            # 优先尝试匹配标准字段名
            field_map = {
                'roc_auc': 'auc',
                'accuracy': 'accuracy',
                'f1': 'f1',
                'f1_macro': 'f1_macro',
                'log_loss': 'log_loss',
                'rmse': 'rmse',
                'mae': 'mae',
                'r2': 'r2',
            }
            field_name = field_map.get(normalized, normalized)
            if hasattr(metrics, field_name):
                setattr(metrics, field_name, float(result))
            else:
                # 保存到 metrics 的额外信息中（通过 model_extra 或自定义字段）
                if not hasattr(metrics, '_extra_metrics'):
                    metrics._extra_metrics = {}
                metrics._extra_metrics[eval_metric] = float(result)

            logger.info(f"[BenchmarkEvaluator] 动态计算指标 '{eval_metric}' = {float(result):.4f}")

        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator] 动态计算指标 '{eval_metric}' 失败: {e}")

    def _save_intermediate_results(
        self,
        result: BenchmarkTaskResult,
        task_cfg: BenchmarkTaskConfig,
        run_index: int,
        task_state,
        data_dir: Path,
        engine=None
    ) -> Path:
        """保存中间结果到独立目录（含详细日志：LLM prompt/response、沙箱输出、代码历史等）"""
        result_dir = self.result_base_dir / task_cfg.task_name / f"run_{run_index}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存代码（最佳代码 + 当前代码 + 代码历史）
        if task_state and task_state.best_code:
            (result_dir / "code_best.py").write_text(task_state.best_code, encoding='utf-8')
            # 【关键修复】同时保存到 data_dir.parent，供注入式预测脚本查找
            try:
                (data_dir.parent / "code_best.py").write_text(task_state.best_code, encoding='utf-8')
            except Exception:
                pass
        if task_state and task_state.code:
            (result_dir / "code_current.py").write_text(task_state.code, encoding='utf-8')
        if task_state and task_state.code_history:
            for i, hist in enumerate(task_state.code_history):
                hist_file = result_dir / "code_history" / f"round_{hist.get('round', i)}_{hist.get('type', 'unknown')}.py"
                hist_file.parent.mkdir(parents=True, exist_ok=True)
                hist_file.write_text(hist.get("code", ""), encoding='utf-8')

        # 2. 保存模型文件
        model_path = data_dir / "best_model.pkl"
        if model_path.exists():
            shutil.copy2(model_path, result_dir / "best_model.pkl")

        # 3. 保存预测结果
        pred_path = data_dir / "eval_predictions.csv"
        if pred_path.exists():
            shutil.copy2(pred_path, result_dir / "eval_predictions.csv")
        
        # 3.5 保存 LLM 生成的 predict.py（如有）
        predict_py_path = data_dir / "predict.py"
        if not predict_py_path.exists():
            predict_py_path = data_dir.parent / "artifacts" / "predict.py"
        if predict_py_path.exists():
            shutil.copy2(predict_py_path, result_dir / "predict.py")

        # 4. 保存 ground_truth
        if Path(task_cfg.ground_truth_path).exists():
            shutil.copy2(task_cfg.ground_truth_path, result_dir / "ground_truth.csv")

        # 5. 保存指标
        metrics_data = {
            "val_metrics": result.val_metrics.model_dump() if result.val_metrics else None,
            "test_metrics": result.test_metrics.model_dump() if result.test_metrics else None,
            "best_score": result.best_score,
            "task_type": task_cfg.task_type.value,
            "target_column": task_cfg.target_column,
            "eval_metric": task_cfg.eval_metric
        }
        (result_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2, ensure_ascii=False), encoding='utf-8')

        # 6. 保存 Judge 结果
        judge_data = {
            "accepted": result.judge_accepted,
            "analysis": result.judge_analysis,
            "reason": result.judge_reason
        }
        (result_dir / "judge_result.json").write_text(json.dumps(judge_data, indent=2, ensure_ascii=False), encoding='utf-8')

        # 7. 保存日志（FastEngine 运行日志 + 沙箱输出）
        if result.logs:
            (result_dir / "logs.txt").write_text("\n".join(result.logs), encoding='utf-8')
        # 保存沙箱 stdout（最后一次成功执行的输出）
        if task_state and task_state.execution_output:
            (result_dir / "sandbox_stdout.txt").write_text(task_state.execution_output, encoding='utf-8')
        # 保存错误信息
        if result.error_message:
            (result_dir / "error.txt").write_text(result.error_message, encoding='utf-8')
        if task_state and task_state.execution_error:
            (result_dir / "execution_error.txt").write_text(task_state.execution_error, encoding='utf-8')

        # 8. 保存完整结果
        (result_dir / "task_result.json").write_text(
            result.model_dump_json(indent=2), encoding='utf-8'
        )

        # 9. 保存各 Agent 的 LLM 调用日志
        llm_log_dir = result_dir / "llm_calls"
        llm_log_dir.mkdir(parents=True, exist_ok=True)
        
        # IntentAgent（可能因缓存未调用）
        if self.intent_agent.get_llm_call_logs():
            self.intent_agent.save_llm_logs_to_dir(llm_log_dir / "intent", "intent")
        
        # JudgeAgent
        if self.judge_agent.get_llm_call_logs():
            self.judge_agent.save_llm_logs_to_dir(llm_log_dir / "judge", "judge")
        
        # FastEngine 内的 Agent
        if engine:
            if hasattr(engine, 'plan_coding_agent') and engine.plan_coding_agent.get_llm_call_logs():
                engine.plan_coding_agent.save_llm_logs_to_dir(llm_log_dir / "plan_coding", "plan_coding")
            if hasattr(engine, 'evaluation_agent') and engine.evaluation_agent.get_llm_call_logs():
                engine.evaluation_agent.save_llm_logs_to_dir(llm_log_dir / "evaluation", "evaluation")

        logger.info(f"[BenchmarkEvaluator] 中间结果已保存至 {result_dir}")
        return result_dir

    def _cleanup_task(self, task_id: str):
        """清理任务资源：停止引擎、移除全局引用、删除中间产物目录"""
        try:
            engine = get_or_create_engine(task_id)
            engine.stop()
        except Exception:
            pass
        try:
            remove_engine(task_id)
        except Exception:
            pass
        try:
            task_manager.delete_task(task_id)
        except Exception:
            pass
        # 删除 FastEngine 产生的中间产物目录（保留 eval_id 报告目录）
        try:
            task_output_dir = settings.OUTPUT_DIR / task_id
            if task_output_dir.exists():
                import shutil
                shutil.rmtree(task_output_dir)
                logger.info(f"[BenchmarkEvaluator] 已清理中间产物目录: {task_output_dir}")
        except Exception:
            pass

    def _build_empty_report(self, reason: str) -> BenchmarkReport:
        """构建空报告"""
        return BenchmarkReport(
            eval_id=self.eval_id,
            benchmark_dir=str(self.benchmark_dir),
            num_runs=self.num_runs,
            status="failed",
            task_names=[],
            round_results=[]
        )




    def _generate_csv_table(self, round_results: List[BenchmarkRoundResult]) -> Path:
        """
        生成 CSV 结果表格，每行代表一个任务的一次运行
        支持后续追加评测案例后重新计算成功率
        """
        csv_path = self.result_base_dir / "benchmark_results.csv"

        # 定义 CSV 列
        fieldnames = [
            "eval_id", "task_name", "run_index", "timestamp",
            "success", "phase", "best_score",
            # 维度评分
            "dim_metric_performance", "dim_overfit_control", "dim_algorithm_choice",
            "dim_pipeline_completeness", "dim_task_alignment",
            # 验证集指标
            "val_auc", "val_accuracy", "val_rmse", "val_score",
            # 测试集指标
            "test_auc", "test_accuracy", "test_f1", "test_f1_macro",
            "test_rmse", "test_mae", "test_r2",
            # Judge
            "judge_accepted", "judge_analysis", "judge_reason",
            # 耗时
            "code_gen_seconds", "sandbox_seconds", "eval_seconds",
            "artifact_seconds", "test_pred_seconds", "total_seconds",
            # Token 消耗（除 Judge 外）
            "plan_coding_calls", "plan_coding_tokens",
            "evaluation_calls", "evaluation_tokens",
            "total_llm_calls", "total_tokens",
            # 该任务本轮聚合指标（每行重复，方便透视分析）
            "task_success_rate", "task_avg_best_score",
            "task_avg_duration", "task_min_duration", "task_max_duration", "task_duration_std",
            "task_avg_tokens", "task_avg_plan_tokens", "task_avg_eval_tokens",
            "task_score_std", "task_score_cv",
            # 错误信息
            "error_message"
        ]

        rows = []
        for round_result in round_results:
            for r in round_result.task_results:
                # 提取维度评分
                dim_scores = {d["name"]: d["score"] for d in r.dimension_scores}

                # 提取验证集指标
                val = r.val_metrics

                # 提取测试集指标
                test = r.test_metrics

                row = {
                    "eval_id": self.eval_id,
                    "task_name": r.task_name,
                    "run_index": r.run_index,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": "1" if r.success else "0",
                    "phase": r.phase or "",
                    "best_score": r.best_score if r.best_score is not None else "",
                    # 维度评分
                    "dim_metric_performance": dim_scores.get("metric_performance", ""),
                    "dim_overfit_control": dim_scores.get("overfit_control", ""),
                    "dim_algorithm_choice": dim_scores.get("algorithm_choice", ""),
                    "dim_pipeline_completeness": dim_scores.get("pipeline_completeness", ""),
                    "dim_task_alignment": dim_scores.get("task_alignment", ""),
                    # 验证集指标
                    "val_auc": val.val_auc if val and val.val_auc is not None else "",
                    "val_accuracy": val.val_accuracy if val and val.val_accuracy is not None else "",
                    "val_rmse": val.val_rmse if val and val.val_rmse is not None else "",
                    "val_score": val.val_score if val and val.val_score is not None else "",
                    # 测试集指标
                    "test_auc": test.auc if test and test.auc is not None else "",
                    "test_accuracy": test.accuracy if test and test.accuracy is not None else "",
                    "test_f1": test.f1 if test and test.f1 is not None else "",
                    "test_f1_macro": test.f1_macro if test and test.f1_macro is not None else "",
                    "test_rmse": test.rmse if test and test.rmse is not None else "",
                    "test_mae": test.mae if test and test.mae is not None else "",
                    "test_r2": test.r2 if test and test.r2 is not None else "",
                    # Judge
                    "judge_accepted": "1" if r.judge_accepted else "0",
                    "judge_analysis": (r.judge_analysis or "").replace("\n", " ")[:200],
                    "judge_reason": (r.judge_reason or "").replace("\n", " ")[:200],
                    # 耗时
                    "code_gen_seconds": round(r.timing.code_generation_seconds, 2) if r.timing else "",
                    "sandbox_seconds": round(r.timing.sandbox_execution_seconds, 2) if r.timing else "",
                    "eval_seconds": round(r.timing.evaluation_seconds, 2) if r.timing else "",
                    "artifact_seconds": round(r.timing.artifact_generation_seconds, 2) if r.timing else "",
                    "test_pred_seconds": round(r.timing.test_prediction_seconds, 2) if r.timing else "",
                    "total_seconds": round(r.duration_seconds, 2),
                    # Token
                    "plan_coding_calls": r.token_usage.plan_coding_calls if r.token_usage else "",
                    "plan_coding_tokens": r.token_usage.plan_coding_total_tokens if r.token_usage else "",
                    "evaluation_calls": r.token_usage.evaluation_calls if r.token_usage else "",
                    "evaluation_tokens": r.token_usage.evaluation_total_tokens if r.token_usage else "",
                    "total_llm_calls": r.token_usage.total_calls if r.token_usage else "",
                    "total_tokens": r.token_usage.total_tokens if r.token_usage else "",
                    # 该任务本轮聚合指标
                    "task_success_rate": round(round_result.success_rate, 4),
                    "task_avg_best_score": round(round_result.avg_best_score, 4) if round_result.avg_best_score is not None else "",
                    "task_avg_duration": round(round_result.avg_duration_seconds, 2),
                    "task_min_duration": round(round_result.min_duration_seconds, 2),
                    "task_max_duration": round(round_result.max_duration_seconds, 2),
                    "task_duration_std": round(round_result.duration_std, 4),
                    "task_avg_tokens": round_result.avg_total_tokens,
                    "task_avg_plan_tokens": round_result.avg_plan_coding_tokens,
                    "task_avg_eval_tokens": round_result.avg_evaluation_tokens,
                    "task_score_std": round(round_result.score_std, 4),
                    "task_score_cv": round(round_result.score_cv, 4),
                    # 错误
                    "error_message": (r.error_message or "").replace("\n", " ")[:200]
                }
                rows.append(row)

        # 写入 CSV
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return csv_path

    def _generate_summary_csv(self, round_results: List[BenchmarkRoundResult]) -> Path:
        """
        生成任务级汇总 CSV，每行代表一个任务的所有运行聚合
        """
        csv_path = self.result_base_dir / "benchmark_summary.csv"
        fieldnames = [
            "eval_id", "round_index", "task_name", "num_runs",
            "success_rate", "success_count", "fail_count",
            "avg_best_score", "score_std", "score_cv",
            "avg_duration_seconds", "min_duration_seconds", "max_duration_seconds", "duration_std",
            "avg_total_tokens", "avg_plan_coding_tokens", "avg_evaluation_tokens"
        ]

        rows = []
        for rr in round_results:
            rows.append({
                "eval_id": self.eval_id,
                "round_index": rr.round_index,
                "task_name": rr.task_results[0].task_name if rr.task_results else "",
                "num_runs": len(rr.task_results),
                "success_rate": round(rr.success_rate, 4),
                "success_count": rr.success_count,
                "fail_count": rr.fail_count,
                "avg_best_score": round(rr.avg_best_score, 4) if rr.avg_best_score is not None else "",
                "score_std": round(rr.score_std, 4),
                "score_cv": round(rr.score_cv, 4),
                "avg_duration_seconds": round(rr.avg_duration_seconds, 2),
                "min_duration_seconds": round(rr.min_duration_seconds, 2),
                "max_duration_seconds": round(rr.max_duration_seconds, 2),
                "duration_std": round(rr.duration_std, 4),
                "avg_total_tokens": rr.avg_total_tokens,
                "avg_plan_coding_tokens": rr.avg_plan_coding_tokens,
                "avg_evaluation_tokens": rr.avg_evaluation_tokens
            })

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return csv_path



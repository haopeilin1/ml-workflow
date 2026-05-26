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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

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
    TimingBreakdown, TokenUsageSummary, ArtifactInfo
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
        eval_id: Optional[str] = None,
        # 【新增】Plan / Coding / Unified / Evaluation Agent 独立 LLM 配置
        plan_llm_config: Optional[LLMConfig] = None,
        coding_llm_config: Optional[LLMConfig] = None,
        unified_llm_config: Optional[LLMConfig] = None,
        evaluation_llm_config: Optional[LLMConfig] = None,
    ):
        self.plan_coding_llm_config = plan_coding_llm_config
        self.plan_llm_config = plan_llm_config
        self.coding_llm_config = coding_llm_config
        self.unified_llm_config = unified_llm_config
        self.evaluation_llm_config = evaluation_llm_config
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
        self._lock = threading.Lock()  # 并发锁

    def _run_task_round(self, task_cfg: BenchmarkTaskConfig) -> tuple:
        """运行单个任务的所有轮次（同一任务内顺序执行，不同任务间可并行）"""
        task_results: List[BenchmarkTaskResult] = []
        for run_idx in range(1, self.num_runs + 1):
            if not self._running:
                logger.info(f"[BenchmarkEvaluator] 收到停止信号，任务 {task_cfg.task_name} 中断")
                break

            with self._lock:
                self._current_task = task_cfg.task_name
                self._current_run = run_idx

            logger.info(f"[BenchmarkEvaluator] 开始任务: {task_cfg.task_name}, 第 {run_idx}/{self.num_runs} 次运行")
            result = self._run_single_task(task_cfg, run_idx)
            task_results.append(result)

            with self._lock:
                self._completed_runs += 1

            logger.info(
                f"[BenchmarkEvaluator] 任务 {task_cfg.task_name} 第 {run_idx} 次运行完成: "
                f"success={result.success}, judge_accepted={result.judge_accepted}, "
                f"duration={result.duration_seconds:.1f}s, artifacts={result.artifacts.completeness}"
            )
        return task_cfg, task_results

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

        # 2. 对每个任务运行 num_runs 次（【修复】串行执行，确保每次运行彻底冷启动，避免并发污染）
        round_results_map: Dict[str, BenchmarkRoundResult] = {}
        for tc in tasks:
            task_cfg, task_results = self._run_task_round(tc)

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

            # 产物完整性聚合
            artifact_completenesses = [r.artifacts.completeness for r in task_results if r.artifacts]
            most_common_completeness = max(set(artifact_completenesses), key=artifact_completenesses.count) if artifact_completenesses else "none"

            round_result = BenchmarkRoundResult(
                round_index=len(round_results_map) + 1,
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
            round_results_map[task_cfg.task_name] = round_result
            logger.info(
                f"[BenchmarkEvaluator] 任务 {task_cfg.task_name} 完成: "
                f"成功率={success_rate:.1%}, 平均耗时={avg_duration:.1f}s, "
                f"平均Token={avg_total_tokens}, score_std={score_std:.4f}, "
                f"产物完整性={most_common_completeness}"
            )

        # 按原始任务顺序排序
        round_results = [round_results_map[t.task_name] for t in tasks if t.task_name in round_results_map]

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
        # 支持灵活命名：用于建模、数据建模、建模 等
        direct_modeling = None
        direct_eval = None
        for child in self.benchmark_dir.iterdir():
            if not child.is_dir():
                continue
            if '建模' in child.name:
                direct_modeling = child
            elif '评估' in child.name:
                direct_eval = child
        
        if direct_modeling and direct_eval:
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
            return None, None
        
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
        task_type, target_column, eval_metric, id_column, complexity, is_time_series, data_profile, complexity_reason = self._recognize_task_info_llm(
            train_path, gt_path, desc_path, task_name
        )
        if not target_column:
            task_type, target_column, id_column = self._infer_task_info(train_path, gt_path)
            eval_metric = None
            complexity = "simple"
            is_time_series = False
            complexity_reason = "规则推断fallback"
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
            id_column=id_column,
            data_profile=data_profile,
            complexity_reason=complexity_reason
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
            missing_rate = round(missing_count / n_rows * 100, 2) if n_rows > 0 else 0.0

            col_info = {
                "name": col,
                "type": "numeric" if is_numeric else "categorical",
                "originalDtype": str(series.dtype),
                "uniqueCount": unique_count,
                "missingCount": missing_count,
                "missingRate": missing_rate,
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
                    col_info["std"] = float(non_null.std())
                    col_info["median"] = float(non_null.median())
                    col_info["skewness"] = float(non_null.skew())
                    
                    # 【关键新增】检测单调性——时序数据的核心特征
                    # 对疑似时间/索引列检测是否单调递增
                    if unique_count == n_rows and n_rows > 10:
                        col_info["isMonotonic"] = bool(non_null.is_monotonic_increasing)
            else:
                # 类别列：增加采样值和最常见值
                non_null = series.dropna()
                if len(non_null) > 0:
                    col_info["sampleValues"] = non_null.head(5).astype(str).tolist()
                    vc = non_null.value_counts()
                    if len(vc) > 0:
                        col_info["mostCommon"] = str(vc.index[0])
                        col_info["mostCommonFreq"] = int(vc.iloc[0])
                    
                    # 基数分级：帮助 LLM 选择编码策略
                    if unique_count <= 10:
                        col_info["cardinality"] = "low"
                    elif unique_count <= 100:
                        col_info["cardinality"] = "medium"
                    else:
                        col_info["cardinality"] = "high"
                    
                    # 【关键新增】检测字符串列是否可解析为日期
                    try:
                        sample = str(non_null.iloc[0])
                        if len(sample) >= 6 and len(sample) <= 25:
                            pd.to_datetime(sample)
                            col_info["isDateParseable"] = True
                    except:
                        pass

            profile["columns"].append(col_info)
        
        # ========== 【新增】时间列连贯性检测 ==========
        time_series_signal = self._detect_time_series_signal(train_df, profile)
        if time_series_signal:
            profile["timeSeriesSignal"] = time_series_signal
        
        # ========== 【新增】类别不平衡量化 ==========
        target_col = None
        # 尝试从列名推断目标列（简单启发式：列名含 target/label/y/churn/fraud/class）
        for col in train_df.columns:
            if any(kw in col.lower() for kw in ["target", "label", "y", "churn", "fraud", "class", "bool"]):
                target_col = col
                break
        # 如果没有启发式匹配，取最后一列（常见做法）
        if target_col is None and len(train_df.columns) > 0:
            target_col = train_df.columns[-1]
        
        if target_col and target_col in train_df.columns:
            y = train_df[target_col]
            # 只针对类别型目标列计算不平衡度
            if y.dtype == 'object' or y.nunique() <= 20:
                vc = y.value_counts()
                if len(vc) >= 2:
                    max_ratio = vc.iloc[0] / len(y)
                    min_ratio = vc.iloc[-1] / len(y)
                    imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
                    profile["classBalance"] = {
                        "targetColumn": target_col,
                        "nClasses": int(len(vc)),
                        "maxClassRatio": float(round(max_ratio, 4)),
                        "minClassRatio": float(round(min_ratio, 4)),
                        "imbalanceRatio": float(round(imbalance_ratio, 2)),
                        "isSeverelyImbalanced": bool(imbalance_ratio > 10 or min_ratio < 0.05)
                    }
        
        return profile
    
    def _detect_time_series_signal(self, df: pd.DataFrame, profile: Dict) -> Optional[Dict]:
        """
        检测时间列连贯性信号。
        不仅检测单调递增，还检测相邻行间隔是否稳定（方差小）。
        返回：{"coherent": True/False, "column": "列名", "intervalStd": float, "reason": str}
        """
        columns_info = profile.get("columns", [])
        n_rows = profile.get("rowCount", 0)
        
        # 候选列：单调递增 + 唯一值=行数
        candidates = []
        for col_info in columns_info:
            name = col_info.get("name", "")
            if col_info.get("isMonotonic") and n_rows > 10:
                if col_info.get("uniqueCount") == n_rows:
                    if name in df.columns:
                        candidates.append(name)
        
        # 也检查可解析为日期的列
        for col_info in columns_info:
            name = col_info.get("name", "")
            if col_info.get("isDateParseable") and name in df.columns:
                if name not in candidates:
                    candidates.append(name)
        
        for col_name in candidates:
            series = df[col_name]
            try:
                # 数值列：检测差值稳定性
                if pd.api.types.is_numeric_dtype(series):
                    diffs = series.diff().dropna()
                    if len(diffs) > 1:
                        diff_std = float(diffs.std())
                        diff_mean = float(diffs.mean())
                        # 间隔稳定：标准差小（相对于均值），且无大跳跃
                        if diff_mean > 0 and diff_std / diff_mean < 0.1:
                            return {
                                "coherent": True,
                                "column": col_name,
                                "intervalStd": float(round(diff_std, 4)),
                                "intervalMean": float(round(diff_mean, 4)),
                                "reason": f"列 '{col_name}' 单调递增，相邻间隔稳定（mean={diff_mean:.2f}, std={diff_std:.2f}）"
                            }
                        else:
                            return {
                                "coherent": False,
                                "column": col_name,
                                "intervalStd": float(round(diff_std, 4)),
                                "intervalMean": float(round(diff_mean, 4)),
                                "reason": f"列 '{col_name}' 单调递增，但相邻间隔不稳定（std/mean={diff_std/diff_mean:.2f}）"
                            }
                
                # 日期列：检测时间差稳定性
                dt_series = pd.to_datetime(series, errors='coerce')
                if dt_series.notna().sum() / len(dt_series) > 0.9:
                    diffs = dt_series.diff().dropna()
                    if len(diffs) > 1:
                        # 转换为小时数
                        diff_hours = diffs.dt.total_seconds() / 3600
                        diff_std = float(diff_hours.std())
                        diff_mean = float(diff_hours.mean())
                        if diff_mean > 0 and diff_std / diff_mean < 0.5:
                            return {
                                "coherent": True,
                                "column": col_name,
                                "intervalStdHours": float(round(diff_std, 4)),
                                "intervalMeanHours": float(round(diff_mean, 4)),
                                "reason": f"列 '{col_name}' 为日期序列，时间间隔稳定（mean={diff_mean:.2f}h, std={diff_std:.2f}h）"
                            }
            except Exception:
                pass
        
        return None

    def _recognize_task_info_llm(
        self,
        train_path: Path,
        gt_path: Path,
        desc_path: Optional[Path],
        task_name: str,
    ) -> tuple:
        """
        使用 LLM IntentAgent 识别任务信息。
        返回 (task_type, target_column, eval_metric, id_column, complexity, is_time_series)。
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
            return task_type, cached.target_column, cached.eval_metric, id_column, cached.complexity, cached.is_time_series, None, cached.complexity_reason

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
            return None, None, None, None, None, None, None, None, None, None

        # 调用 IntentAgent
        try:
            result = self.intent_agent.recognize(
                columns=profile.get("columns", []),
                task_description=user_description,
                row_count=profile.get("rowCount", 0),
                col_count=profile.get("columnCount", 0),
                data_profile=profile,
            )
        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator] IntentAgent 调用失败 {task_name}: {e}")
            return None, None, None, None, None, None, None, None

        if not result or not result.target_column:
            logger.info(f"[BenchmarkEvaluator] IntentAgent 未返回 target_column {task_name}")
            return None, None, None, None, None, None, None, None

        # 校验 target_column 是否在训练集中
        train_df = pd.read_csv(train_path)
        if result.target_column not in train_df.columns:
            logger.warning(
                f"[BenchmarkEvaluator] LLM 返回的 target_column '{result.target_column}' "
                f"不在训练集中，回退到规则推断"
            )
            return None, None, None, None, None, None

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
            f"metric={result.eval_metric}, complexity={result.complexity}, ts={result.is_time_series}, "
            f"reason={result.complexity_reason}"
        )
        return task_type, result.target_column, result.eval_metric, id_column, result.complexity, result.is_time_series, profile, result.complexity_reason

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
        """执行单个任务单次运行（彻底冷启动模式）"""
        task_start = time.time()
        result = BenchmarkTaskResult(
            task_name=task_cfg.task_name,
            run_index=run_index,
            success=False,
            judge_accepted=False
        )

        try:
            # ========== 【冷启动】清空所有 Agent 状态和缓存 ==========
            logger.info(f"[BenchmarkEvaluator] 【冷启动】任务 {task_cfg.task_name} 第 {run_index} 次运行，清空所有缓存和 Agent 状态")
            # 1. 清空意图识别缓存（全部，不只是当前任务）
            self._intent_cache.clear()
            # 2. 清空 IntentAgent 的日志和用量
            if hasattr(self.intent_agent, 'clear_llm_call_logs'):
                self.intent_agent.clear_llm_call_logs()
            if hasattr(self.intent_agent, 'reset_usage'):
                self.intent_agent.reset_usage()
            # 3. 清空 JudgeAgent 的日志和用量
            if hasattr(self.judge_agent, 'clear_llm_call_logs'):
                self.judge_agent.clear_llm_call_logs()
            if hasattr(self.judge_agent, 'reset_usage'):
                self.judge_agent.reset_usage()
            logger.info(f"[BenchmarkEvaluator] 【冷启动】缓存和 Agent 状态已清空")
            
            # 1. 读取任务描述
            user_description = ""
            if task_cfg.desc_path and Path(task_cfg.desc_path).exists():
                user_description = Path(task_cfg.desc_path).read_text(encoding='utf-8').strip()

            # 2. 创建 TaskConfig（冷启动，不传入建模建议）
            agent_configs = {}
            # 【新增】Plan / Coding / Unified 独立 LLM 配置
            if self.plan_llm_config:
                agent_configs["plan"] = self.plan_llm_config
            if self.coding_llm_config:
                agent_configs["coding"] = self.coding_llm_config
            if self.unified_llm_config:
                agent_configs["unified"] = self.unified_llm_config
            if self.evaluation_llm_config:
                agent_configs["evaluation"] = self.evaluation_llm_config
            # 向后兼容：plan_coding_llm_config 作为共同回退
            if self.plan_coding_llm_config:
                agent_configs["plan_coding"] = self.plan_coding_llm_config
            
            uploaded_files = [
                UploadedFile(name="train.csv", path=task_cfg.train_path, role=FileRole.TRAIN),
            ]
            if task_cfg.test_path:
                uploaded_files.append(UploadedFile(name="test.csv", path=task_cfg.test_path, role=FileRole.TEST))
            
            # 每次运行都重新进行意图识别（缓存已在冷启动阶段全部清空）
            intent_start = time.time()
            recognized = self._recognize_task_info_llm(
                Path(task_cfg.train_path),
                Path(task_cfg.ground_truth_path),
                Path(task_cfg.desc_path) if task_cfg.desc_path else None,
                task_cfg.task_name,
            )
            intent_seconds = time.time() - intent_start
            # recognized 返回元组长度可能为 6/8/10，统一安全读取
            if recognized and recognized[0] is not None and len(recognized) >= 6:
                complexity = recognized[4] if recognized[4] else "simple"
                is_time_series = recognized[5] if recognized[5] is not None else False
                complexity_reason = recognized[7] if len(recognized) > 7 else None
            else:
                # 识别失败，回退到默认值
                complexity = "simple"
                is_time_series = False
                complexity_reason = None
            
            # 保存意图识别结果到 result
            result.complexity = complexity
            result.complexity_reason = complexity_reason
            result.is_time_series = is_time_series
            result.intent_recognized = recognized is not None and recognized[0] is not None
            
            tc = TaskConfig(
                extracted_slots=ExtractedSlots(
                    target_column=task_cfg.target_column,
                    task_type=task_cfg.task_type,
                    eval_metric=task_cfg.eval_metric,
                    id_column=task_cfg.id_column,  # 修复：补全 id_column 传递
                    complexity=complexity,
                    complexity_reason=complexity_reason,
                    is_time_series=is_time_series,
                    feature_constraints=[],
                    user_modeling_suggestions=None  # 冷启动：不传入建模建议
                ),
                uploaded_files=uploaded_files,
                user_description=user_description,
                data_profile=task_cfg.data_profile,
                agent_llm_configs=agent_configs if agent_configs else None
            )

            # 3. 创建任务并启动 FastEngine
            state = task_manager.create_task(tc)
            task_id = state.task_id
            result.task_id = task_id

            # 测试集在训练阶段即对代码可见，用于一致的预处理与特征工程
            source_test_path = Path(task_cfg.test_path) if task_cfg.test_path else None

            # 数据切分准备（test.csv 会正常复制到 outputs/data/）
            datasets = self.data_splitter.prepare_datasets(
                files=[f.model_dump() for f in tc.uploaded_files],
                target_column=tc.extracted_slots.target_column or "target",
                task_type=tc.extracted_slots.task_type,
                task_id=task_id,
                is_time_series=tc.extracted_slots.is_time_series or False,
                data_profile=task_cfg.data_profile
            )
            task_manager.update_task(
                task_id,
                plan=f"评测数据集准备完成: train={datasets['train'].name}"
            )
            data_dir = datasets["train"].parent

            # 启动 FastEngine（传入时间预算用于动态时间预算）
            # PRESENTING 等待 15 分钟，COMPLETED 等待 20 分钟
            presenting_timeout = min(self.max_wait_seconds, 1500) if self.max_wait_seconds else 1500
            completed_timeout = min(self.max_wait_seconds, 1200) if self.max_wait_seconds else 1200
            engine = get_or_create_engine(task_id, max_wait_seconds=presenting_timeout)
            engine.start()

            # 4. 轮询等待 PRESENTING / FAILED / 超时（15分钟）
            presenting = self._wait_for_phase(task_id, [FastTaskPhase.PRESENTING, FastTaskPhase.FAILED], timeout=presenting_timeout)
            if not presenting:
                # ========== 【超时降级提取】PRESENTING 等待超时，尝试提取最佳模型 ==========
                task_state = task_manager.get_task(task_id)
                current_phase = task_state.phase.value if task_state else "unknown"
                
                logger.warning(
                    f"[BenchmarkEvaluator] 任务 {task_id} PRESENTING 等待超时 "
                    f"(phase={current_phase}, elapsed={time.time()-task_start:.0f}s, timeout={presenting_timeout}s)，"
                    f"尝试提取最佳模型进行测试预测..."
                )
                
                # 尝试降级提取：只要有训练产物就尽力给用户结果
                extracted = self._try_extract_best_effort(
                    task_id, data_dir, task_cfg, result, task_state, intent_seconds, task_start,
                    timeout_reason="PRESENTING_TIMEOUT"
                )
                if extracted:
                    logger.info(f"[BenchmarkEvaluator] 任务 {task_id} 超时降级提取成功")
                    self._cleanup_task(task_id)
                    result.duration_seconds = time.time() - task_start
                    return result
                
                # 降级提取也失败，返回失败
                logger.error(f"[BenchmarkEvaluator] 任务 {task_id} 降级提取失败，无可用模型")
                result.error_message = f"等待 PRESENTING 阶段超时(phase={current_phase})，且无法提取最佳模型"
                result.phase = current_phase
                # 【修复】FAILED 任务也要保存中间结果
                try:
                    result_dir = self._save_intermediate_results(result, task_cfg, run_index, task_state, data_dir, engine=engine)
                    result.result_dir = str(result_dir)
                except Exception as save_e:
                    logger.warning(f"[BenchmarkEvaluator] FAILED 任务保存中间结果失败: {save_e}")
                self._cleanup_task(task_id)
                result.duration_seconds = time.time() - task_start
                return result

            task_state = task_manager.get_task(task_id)
            if task_state.phase == FastTaskPhase.FAILED:
                # ========== 【降级提取】FAILED 也尝试提取最佳模型 ==========
                logger.warning(
                    f"[BenchmarkEvaluator] 任务 {task_id} 进入 FAILED，"
                    f"尝试提取已训练的最佳模型..."
                )
                extracted = self._try_extract_best_effort(
                    task_id, data_dir, task_cfg, result, task_state, intent_seconds, task_start,
                    timeout_reason="FAILED"
                )
                if extracted:
                    logger.info(f"[BenchmarkEvaluator] 任务 {task_id} FAILED 降级提取成功")
                    self._cleanup_task(task_id)
                    result.duration_seconds = time.time() - task_start
                    return result
                
                result.error_message = task_state.execution_error or "FastEngine 进入 FAILED 阶段"
                result.phase = "failed"
                result.logs = task_state.logs or []
                # 【修复】FAILED 任务也要保存中间结果
                try:
                    result_dir = self._save_intermediate_results(result, task_cfg, run_index, task_state, data_dir, engine=engine)
                    result.result_dir = str(result_dir)
                except Exception as save_e:
                    logger.warning(f"[BenchmarkEvaluator] FAILED 任务保存中间结果失败: {save_e}")
                self._cleanup_task(task_id)
                result.duration_seconds = time.time() - task_start
                return result

            # 5. 到达 PRESENTING → 提交满意反馈（test.csv 已在训练阶段可用）

            logger.info(f"[BenchmarkEvaluator] 任务 {task_id} 到达 PRESENTING，自动提交满意反馈")
            engine.continue_with_feedback(satisfied=True, suggestion="")

            # 6. 轮询等待 COMPLETED / FAILED（20分钟）
            completed = self._wait_for_phase(task_id, [FastTaskPhase.COMPLETED, FastTaskPhase.FAILED], timeout=completed_timeout)
            task_state = task_manager.get_task(task_id)
            result.phase = task_state.phase.value if task_state else "unknown"
            result.logs = task_state.logs or [] if task_state else []
            result.best_score = task_state.best_score if task_state else None
            result.val_metrics = task_state.best_metrics if task_state else None

            if not completed or task_state.phase == FastTaskPhase.FAILED:
                # ========== 【超时降级提取】COMPLETED 等待超时或 FAILED ==========
                current_phase = task_state.phase.value if task_state else "unknown"
                logger.warning(
                    f"[BenchmarkEvaluator] 任务 {task_id} 产物阶段超时或 FAILED "
                    f"(phase={current_phase})，尝试提取最佳模型进行测试预测..."
                )
                
                extracted = self._try_extract_best_effort(
                    task_id, data_dir, task_cfg, result, task_state, intent_seconds, task_start,
                    timeout_reason="COMPLETED_TIMEOUT_OR_FAILED"
                )
                if extracted:
                    logger.info(f"[BenchmarkEvaluator] 任务 {task_id} 产物阶段降级提取成功")
                    self._cleanup_task(task_id)
                    result.duration_seconds = time.time() - task_start
                    return result
                
                result.error_message = task_state.execution_error or "产物生成阶段失败或超时"
                # 【修复】FAILED 任务也要保存中间结果
                try:
                    result_dir = self._save_intermediate_results(result, task_cfg, run_index, task_state, data_dir, engine=engine)
                    result.result_dir = str(result_dir)
                except Exception as save_e:
                    logger.warning(f"[BenchmarkEvaluator] FAILED 任务保存中间结果失败: {save_e}")
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
                intent_recognition_seconds=intent_seconds,
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
            pred_path, prediction_strategy = self._run_test_prediction(data_dir, task_cfg)
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
                    test_metrics=result.test_metrics,
                    prediction_strategy=prediction_strategy
                )
                result.judge_accepted = judge_result.accepted
                result.judge_analysis = judge_result.analysis
                result.judge_reason = judge_result.reason
                result.prediction_strategy = prediction_strategy

            # 10. 检测产物生成情况
            try:
                result.artifacts = self._detect_artifacts(data_dir)
                logger.info(
                    f"[BenchmarkEvaluator] 产物检测: {task_cfg.task_name} 第 {run_index} 次 "
                    f"completeness={result.artifacts.completeness}, "
                    f"pred={result.artifacts.prediction_file}, model={result.artifacts.model_file}, "
                    f"report={result.artifacts.report_html}, fig={result.artifacts.report_fig_png}"
                )
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] 产物检测失败: {e}")

            # 11. 【新增】提取各环节实际使用的 LLM 追踪（含 fallback 情况）
            try:
                result.llm_usage_trace = self._extract_llm_usage_trace(engine)
                logger.info(f"[BenchmarkEvaluator] LLM 使用追踪: {result.llm_usage_trace}")
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] LLM 使用追踪提取失败: {e}")
            
            # 12. 保存中间结果（含详细日志）
            result_dir = self._save_intermediate_results(
                result, task_cfg, run_index, task_state, data_dir, engine=engine
            )
            result.result_dir = str(result_dir)

            # 清理
            self._cleanup_task(task_id)

        except Exception as e:
            logger.exception(f"[BenchmarkEvaluator] 任务 {task_cfg.task_name} 第 {run_index} 次运行异常")
            result.error_message = f"运行异常: {str(e)}"
            # 异常处理
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

    def _detect_artifacts(self, data_dir: Path) -> ArtifactInfo:
        """检测产物生成情况（支持多种任务类型的差异化产物）
        
        【修复】同时检测 FastEngine 产物目录 artifacts/ 和传统 output/ 目录
        """
        # FastEngine 产物保存到 artifacts/ 目录
        artifact_dir = data_dir.parent / "artifacts"
        # 传统 output/ 目录（兼容旧逻辑）
        output_dir = data_dir.parent / "output"
        
        # 优先使用存在的目录，否则默认 artifacts/
        primary_dir = artifact_dir if artifact_dir.exists() else (output_dir if output_dir.exists() else artifact_dir)

        info = ArtifactInfo()
        
        def _check_file(filename: str) -> bool:
            """检查文件是否存在于 primary_dir 或 fallback 目录"""
            for check_dir in [primary_dir, artifact_dir, output_dir]:
                fpath = check_dir / filename
                if fpath.exists() and fpath.stat().st_size > 0:
                    return True
            return False
        
        files = {
            "prediction_file": "test_predictions.csv",
            "model_file": "model.pkl",
            "feature_importance_csv": "feature_importance.csv",
            "feature_importance_png": "feature_importance.png",
            "report_html": "report.html",
            "report_fig_png": "report_fig.png",
            "predict_script": "predict.py",
            "pipeline_py": "pipeline.py",
            "residual_png": "residual.png",
            "cluster_png": "cluster.png",
            "ts_forecast_png": "ts_forecast.png",
        }
        for attr, filename in files.items():
            setattr(info, attr, _check_file(filename))
        
        # 收集实际存在的产物文件列表（含完整路径信息）
        generated_files = []
        for attr, filename in files.items():
            if _check_file(filename):
                generated_files.append(filename)
        info.generated_files = generated_files

        # 判断完整性（根据存在的产物数量）
        has_model = info.model_file
        has_pred = info.prediction_file
        has_fi_csv = info.feature_importance_csv
        has_fi_png = info.feature_importance_png
        has_report = info.report_html
        has_fig = info.report_fig_png or info.residual_png or info.cluster_png or info.ts_forecast_png
        has_predict = info.predict_script or info.pipeline_py

        if has_model and has_pred and has_fi_csv and has_fi_png and has_report and has_fig and has_predict:
            info.completeness = "full"
        elif has_model and has_pred and has_fi_csv and has_report:
            info.completeness = "simplified"
        elif has_model and (has_pred or has_fi_csv or has_report):
            info.completeness = "partial"
        elif has_model:
            info.completeness = "minimal"
        else:
            info.completeness = "none"

        return info

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

    def _extract_llm_usage_trace(self, engine) -> Dict[str, Any]:
        """
        提取各环节实际使用的 LLM 追踪（含 fallback 情况）
        
        返回格式:
        {
            "intent": {"model": "qwen3.6-27b", "provider": "fallback1-local", "calls": 1, "tokens": 1000},
            "plan": {"model": "deepseek-v4-pro", "provider": "openai", "calls": 2, "tokens": 5000},
            "coding": {"model": "deepseek-v4-pro", "provider": "openai", "calls": 5, "tokens": 15000},
            "evaluation": {"model": "qwen3.6-27b", "provider": "fallback1-local", "calls": 4, "tokens": 8000},
            "judge": {"model": "qwen3.5-flash", "provider": "openai", "calls": 1, "tokens": 2000}
        }
        """
        trace = {}
        
        # 1. Intent Agent
        intent_logs = self.intent_agent.get_llm_call_logs()
        if intent_logs:
            trace["intent"] = self._summarize_logs(intent_logs)
        
        # 2. Judge Agent
        judge_logs = self.judge_agent.get_llm_call_logs()
        if judge_logs:
            trace["judge"] = self._summarize_logs(judge_logs)
        
        # 3. FastEngine 内的 Agent
        if engine:
            # PlanCoding Agent（包含 Plan/Coding/Unified/产物生成 的所有调用）
            if hasattr(engine, 'plan_coding_agent'):
                pc_logs = engine.plan_coding_agent.get_llm_call_logs()
                if pc_logs:
                    # 按 model 分组统计，区分不同阶段
                    trace["plan_coding"] = self._summarize_logs(pc_logs)
                    # 进一步按 model 分组，识别不同阶段使用的模型
                    model_groups = {}
                    for log in pc_logs:
                        usage = log.get("usage", {})
                        model = usage.get("model", "unknown")
                        provider = usage.get("provider", "unknown")
                        key = f"{provider}/{model}"
                        if key not in model_groups:
                            model_groups[key] = {"model": model, "provider": provider, "calls": 0, "tokens": 0}
                        model_groups[key]["calls"] += 1
                        model_groups[key]["tokens"] += usage.get("total_tokens", 0)
                    trace["plan_coding_breakdown"] = list(model_groups.values())
            
            # Evaluation Agent
            if hasattr(engine, 'evaluation_agent'):
                eval_logs = engine.evaluation_agent.get_llm_call_logs()
                if eval_logs:
                    trace["evaluation"] = self._summarize_logs(eval_logs)
        
        return trace
    
    def _summarize_logs(self, logs: List[Dict]) -> Dict[str, Any]:
        """汇总 LLM 调用日志（含延迟统计）"""
        if not logs:
            return {}
        
        # 统计每个 model/provider 组合的调用次数、token、延迟
        model_counts = {}
        total_tokens = 0
        total_latency = 0.0
        fallback_count = 0
        for log in logs:
            usage = log.get("usage", {})
            model = usage.get("model", "unknown")
            provider = usage.get("provider", "unknown")
            latency = usage.get("latency_seconds", 0.0)
            key = f"{provider}/{model}"
            if key not in model_counts:
                model_counts[key] = {"model": model, "provider": provider, "calls": 0, "tokens": 0, "latency": 0.0}
            model_counts[key]["calls"] += 1
            model_counts[key]["tokens"] += usage.get("total_tokens", 0)
            model_counts[key]["latency"] += latency
            total_tokens += usage.get("total_tokens", 0)
            total_latency += latency
            if provider not in ("openai", "") and "fallback" not in provider.lower():
                # 标记非主提供商的调用（fallback）
                fallback_count += 1
        
        # 找出使用最多的 model（主模型）
        primary = max(model_counts.values(), key=lambda x: x["calls"])
        
        return {
            "primary_model": primary["model"],
            "primary_provider": primary["provider"],
            "total_calls": len(logs),
            "total_tokens": total_tokens,
            "total_latency_seconds": round(total_latency, 2),
            "avg_latency_seconds": round(total_latency / len(logs), 2) if logs else 0,
            "fallback_triggers": fallback_count,
            "model_breakdown": list(model_counts.values())
        }

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

    def _run_test_prediction(self, data_dir: Path, task_cfg: BenchmarkTaskConfig) -> Tuple[Optional[Path], str]:
        """
        对测试集进行预测。

        策略优先级：
        0. 【核心新增】直接读取训练代码输出的 test_predictions.csv（避免序列化问题）
        1. 执行 LLM 生成的 predict.py
        2. 从训练代码提取自定义定义，构造注入式预测脚本
        3. 回退到内置通用预测模板
        """
        # ========== 策略0: 直接读取训练代码输出的测试预测（优先级最高）==========
        # FastEngine 在 best_score 更新时会快照保存 best_test_predictions.csv
        # 【修复】同时检查 data_dir 和 artifacts 目录（产物可能被收集到 artifacts/）
        artifact_dir = data_dir.parent / "artifacts"
        
        # 【调试日志】列出 data_dir 中所有文件
        try:
            data_files = {f.name: f.stat().st_size for f in data_dir.iterdir() if f.is_file()}
            logger.info(f"[BenchmarkEvaluator][DEBUG] _run_test_prediction data_dir 文件: {data_files}")
        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator][DEBUG] 无法列出 data_dir 文件: {e}")
        
        # 【关键修复】检查文件不仅存在，还必须包含 prediction 列
        def _check_pred_file(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                df = pd.read_csv(path)
                if 'prediction' not in df.columns:
                    logger.warning(f"[BenchmarkEvaluator] {path.name} 缺少 prediction 列，跳过")
                    return False
                return True
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] 检查 {path.name} 失败: {e}")
                return False
        
        best_pred_path = data_dir / "best_test_predictions.csv"
        if _check_pred_file(best_pred_path):
            logger.info(f"[BenchmarkEvaluator] 发现最佳测试预测快照，直接使用: {best_pred_path} (size={best_pred_path.stat().st_size})")
            return best_pred_path, "embedded_best"
        # 也检查 artifacts 目录
        best_pred_path_art = artifact_dir / "best_test_predictions.csv"
        if _check_pred_file(best_pred_path_art):
            logger.info(f"[BenchmarkEvaluator] 从 artifacts 发现最佳测试预测快照: {best_pred_path_art} (size={best_pred_path_art.stat().st_size})")
            return best_pred_path_art, "embedded_best"
        
        # 如果快照不存在，检查当前 test_predictions.csv（单轮执行或最终轮次）
        pred_path = data_dir / "test_predictions.csv"
        if _check_pred_file(pred_path):
            logger.info(f"[BenchmarkEvaluator] 发现训练代码直接输出的测试预测: {pred_path} (size={pred_path.stat().st_size})")
            return pred_path, "embedded"
        # 也检查 artifacts 目录
        pred_path_art = artifact_dir / "test_predictions.csv"
        if _check_pred_file(pred_path_art):
            logger.info(f"[BenchmarkEvaluator] 从 artifacts 发现测试预测: {pred_path_art} (size={pred_path_art.stat().st_size})")
            return pred_path_art, "embedded"
        
        logger.warning(f"[BenchmarkEvaluator][DEBUG] 策略0失败: 未找到 best_test_predictions.csv 或 test_predictions.csv")
        
        model_path = data_dir / "best_model.pkl"
        # 如果 data_dir 中没有 best_model.pkl，尝试从 artifacts 复制
        if not model_path.exists():
            artifacts_model = data_dir.parent / "artifacts" / "model.pkl"
            if artifacts_model.exists():
                shutil.copy2(artifacts_model, model_path)
                logger.info(f"[BenchmarkEvaluator] 从 artifacts 复制模型到 {model_path}")
        
        if not model_path.exists():
            logger.warning(f"[BenchmarkEvaluator] 未找到 best_model.pkl，跳过策略1-3")
            # 直接跳到策略4（LLM改写训练代码做预测）
            logger.info("[BenchmarkEvaluator] 尝试策略4: LLM改写训练代码生成预测")
            pred_path, strategy = self._strategy4_llm_predict(data_dir, task_cfg)
            if pred_path:
                return pred_path, strategy
            return None, None
        
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
                        return pred_path, "llm_predict"
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
                        return pred_path, "injected"
                    else:
                        logger.warning(f"[BenchmarkEvaluator] 注入式预测脚本执行成功但未生成 eval_predictions.csv")
                else:
                    logger.warning(f"[BenchmarkEvaluator] 注入式预测脚本执行失败: {result.error_message}")
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] 注入式预测脚本执行异常: {e}")

        # ========== 策略3已删除（通用预测模板不可靠）==========
        
        # ========== 策略4: LLM改写训练代码做预测（终极兜底）==========
        logger.info("[BenchmarkEvaluator] 策略0-3均失败，尝试策略4: LLM改写训练代码生成预测")
        pred_path, strategy = self._strategy4_llm_predict(data_dir, task_cfg)
        if pred_path:
            return pred_path, strategy
        
        return None, None

    def _strategy4_llm_predict(self, data_dir: Path, task_cfg: BenchmarkTaskConfig) -> Tuple[Optional[Path], str]:
        """
        策略4: 当所有策略都失败时，用LLM把训练代码改成只进行测试集预测的版本，然后执行。
        
        核心思想：训练代码已经包含了所有预处理逻辑，LLM只需要去掉训练部分，加上模型加载和预测。
        """
        # 1. 获取训练代码
        train_code = ""
        code_best_path = data_dir.parent / "code_best.py"
        if not code_best_path.exists():
            result_dirs = list(data_dir.parent.glob("run_*"))
            if result_dirs:
                code_best_path = result_dirs[0] / "code_best.py"
        
        if code_best_path.exists():
            try:
                train_code = code_best_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning(f"[Strategy4] 读取训练代码失败: {e}")
        
        if not train_code:
            # 尝试从 task_manager 获取
            task_id_from_dir = data_dir.parent.name
            try:
                tm_state = task_manager.get_task(task_id_from_dir)
                if tm_state and tm_state.best_code:
                    train_code = tm_state.best_code
            except Exception:
                pass
        
        if not train_code:
            logger.warning("[Strategy4] 无训练代码可用，跳过")
            return None, None
        
        # 2. 检查是否有 LLM 配置
        if not hasattr(self, 'coding_llm_config') or not self.coding_llm_config:
            logger.warning("[Strategy4] 无 coding_llm_config，无法调用LLM")
            return None, None
        
        # 3. 创建 LLM client 并调用
        try:
            from app.agents.base import LLMClient
            llm = LLMClient(
                provider=self.coding_llm_config.provider,
                base_url=self.coding_llm_config.base_url,
                api_key=self.coding_llm_config.api_key,
                model=self.coding_llm_config.model,
                temperature=0.1,
                max_tokens=8192,
            )
            
            system_prompt = """你是一名资深机器学习工程师。你的任务是将一段训练代码修改为只进行测试集预测的版本。

关键规则：
1. 保留所有数据预处理逻辑（列丢弃、缺失值处理、编码、缩放、特征工程）
2. 加载已保存的模型 data/best_model.pkl（优先用dill，失败用pickle）
3. 加载 data/test.csv，用同样的预处理+模型进行预测
4. 保存预测结果到 data/test_predictions.csv（必须包含id列和prediction列）
5. 代码必须自包含，不要依赖外部变量
6. 严禁重新训练模型
7. 只输出Python代码，不要解释"""

            # 截断训练代码到8000字符（避免超出LLM上下文）
            code_for_prompt = train_code[:8000]
            if len(train_code) > 8000:
                code_for_prompt += "\n\n# ... (代码截断，剩余部分请根据上下文推断)"
            
            user_prompt = f"""请将以下训练代码修改为只进行测试集预测的版本。

原始训练代码：
```python
{code_for_prompt}
```

目标列: {task_cfg.target_column}
任务类型: {task_cfg.task_type.value}

请输出完整的Python代码。"""

            logger.info(f"[Strategy4] 调用LLM生成预测代码，训练代码长度={len(train_code)}")
            content, usage = llm.chat_completion(system_prompt, user_prompt, max_retries=2)
            
            if not content:
                logger.warning("[Strategy4] LLM返回空内容")
                return None, None
            
            # 从LLM响应中提取代码
            import re
            code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
            if code_match:
                predict_code = code_match.group(1).strip()
            else:
                # 没有代码块标记，尝试取全部内容
                predict_code = content.strip()
            
            if not predict_code:
                logger.warning("[Strategy4] 无法从LLM响应中提取代码")
                return None, None
            
            logger.info(f"[Strategy4] LLM生成预测代码，长度={len(predict_code)}")
            
            # 4. 沙箱执行
            result = sandbox_executor.execute(
                code=predict_code,
                data_dir=data_dir,
                task_type=task_cfg.task_type.value,
                artifact_mode=True,
                artifact_output_dir=data_dir
            )
            
            if result.success:
                # 检查是否生成了 test_predictions.csv
                pred_path = data_dir / "test_predictions.csv"
                if pred_path.exists():
                    logger.info(f"[Strategy4] 成功生成测试预测: {pred_path}")
                    return pred_path, "llm_rewritten"
                else:
                    logger.warning("[Strategy4] 代码执行成功但未生成 test_predictions.csv")
            else:
                logger.warning(f"[Strategy4] 预测代码执行失败: {result.error_message}")
            
            return None, None
            
        except Exception as e:
            logger.exception(f"[Strategy4] 异常: {e}")
            return None, None

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
        
        # 【关键修复】检查预测文件是否包含必要的列
        if 'prediction' not in pred_df.columns:
            logger.error(f"[BenchmarkEvaluator] 预测文件缺少 prediction 列，无法计算指标。可用列: {list(pred_df.columns)}")
            return TestSetMetrics()

        # 对齐：根据 id 列合并
        # 【修复】智能推断 id 列：优先查找名为 'id' 的列，其次使用 pred_df 第一列
        # 避免产物代码生成不同列顺序的预测文件时导致对齐错误
        if 'id' in pred_df.columns and 'id' in gt_df.columns:
            id_col = 'id'
        elif 'ID' in pred_df.columns and 'ID' in gt_df.columns:
            id_col = 'ID'
        else:
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
                    # 【关键修复】无共同列时，退而使用位置匹配：将两文件第一列视为 id 列
                    logger.warning(f"[BenchmarkEvaluator] 预测结果与 ground_truth 无共同列，尝试用第一列对齐: pred={pred_df.columns[0]}, gt={gt_df.columns[0]}")
                    pred_id_col = pred_df.columns[0]
                    gt_id_col = gt_df.columns[0]
                    pred_df = pred_df.rename(columns={pred_id_col: "_auto_id"})
                    gt_df = gt_df.rename(columns={gt_id_col: "_auto_id"})
                    id_col = "_auto_id"
        
        # 【关键修复】统一 id 列类型为字符串，防止 int64 vs object merge 失败
        pred_df[id_col] = pred_df[id_col].astype(str)
        gt_df[id_col] = gt_df[id_col].astype(str)
        
        merged = pd.merge(pred_df, gt_df, on=id_col, how="inner")

        if len(merged) == 0:
            logger.warning(f"[BenchmarkEvaluator] 预测结果与 ground_truth 无交集，无法计算指标")
            return TestSetMetrics()

        y_true = merged.iloc[:, -1]  # ground_truth 的 target 列
        y_pred = merged["prediction"]

        # 【调试日志】merge 后的数据统计
        logger.info(f"[BenchmarkEvaluator][DEBUG] _compute_test_metrics: merged shape={merged.shape}, columns={list(merged.columns)}")
        logger.info(f"[BenchmarkEvaluator][DEBUG] y_true dtype={y_true.dtype}, unique={sorted(y_true.unique())[:10]}")
        logger.info(f"[BenchmarkEvaluator][DEBUG] y_pred dtype={y_pred.dtype}, unique={sorted(y_pred.unique())}")

        # 【关键调试】保存 merged DataFrame 到文件供事后分析
        try:
            debug_dir = Path("outputs/debug_test_metrics")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"merged_{pd.Timestamp.now().strftime('%H%M%S')}.csv"
            merged.to_csv(debug_file, index=False)
            logger.info(f"[BenchmarkEvaluator][DEBUG] 已保存 merged DataFrame 到 {debug_file}")
        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator][DEBUG] 保存 merged DataFrame 失败: {e}")

        # 统一标签类型：处理 ground_truth 为字符串但预测为数值的情况
        #（如 ground_truth='No'/'Yes'，prediction=0/1）
        _y_true_is_str = y_true.dtype == object or str(y_true.dtype).startswith('str') or str(y_true.dtype) == 'category'
        _y_pred_is_str = y_pred.dtype == object or str(y_pred.dtype).startswith('str') or str(y_pred.dtype) == 'category'
        _y_true_is_num = str(y_true.dtype) in ['int64', 'int32', 'float64', 'bool', 'int']
        _y_pred_is_num = str(y_pred.dtype) in ['int64', 'int32', 'float64', 'bool', 'int']
        if _y_true_is_str and _y_pred_is_num:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_true = pd.Series(le.fit_transform(y_true))
            logger.info(f'[BenchmarkEvaluator] ground_truth 为字符串，预测为数值，已用 LabelEncoder 对齐: {dict(zip(le.classes_, le.transform(le.classes_)))}')
        elif _y_pred_is_str and _y_true_is_num:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_pred = pd.Series(le.fit_transform(y_pred))

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
        
        # 【关键修复】如果当前预测文件没有 probability 列，尝试从备份恢复
        if y_proba is None and task_type == TaskType.BINARY_CLASSIFICATION:
            backup_sources = [
                (pred_path_obj.parent / "best_test_predictions_backup.csv", "best_test_predictions_backup"),
                (pred_path_obj.parent / "test_predictions.csv", "test_predictions"),
                (pred_path_obj.parent.parent / "artifacts" / "best_test_predictions.csv", "artifacts_best"),
            ]
            for backup_path, backup_name in backup_sources:
                if backup_path.exists():
                    try:
                        backup_df = pd.read_csv(backup_path)
                        if "probability" in backup_df.columns and "id" in backup_df.columns:
                            # 按 id merge 恢复 probability 列
                            backup_df["id"] = backup_df["id"].astype(str)
                            merged_backup = pd.merge(merged, backup_df[["id", "probability"]], on=id_col, how="left")
                            if merged_backup["probability"].notna().sum() > 0:
                                y_proba = merged_backup["probability"]
                                logger.warning(f"[BenchmarkEvaluator][关键修复] 从 {backup_name} 恢复 probability 列，覆盖率={y_proba.notna().sum()}/{len(y_proba)}")
                                break
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 从 {backup_name} 恢复 probability 失败: {e}")
        
        # 【调试日志】probability 列统计
        if y_proba is not None:
            logger.info(f"[BenchmarkEvaluator][DEBUG] y_proba dtype={y_proba.dtype}, count={len(y_proba)}, min={y_proba.min():.6f}, max={y_proba.max():.6f}, mean={y_proba.mean():.6f}, median={y_proba.median():.6f}")
            logger.info(f"[BenchmarkEvaluator][DEBUG] y_proba < 0.01: {(y_proba < 0.01).sum()}, y_proba > 0.5: {(y_proba > 0.5).sum()}")
        else:
            logger.warning(f"[BenchmarkEvaluator][DEBUG] 未找到 probability 列，将使用 prediction 标签计算 AUC")
        
        # 【关键调试】保存计算AUC前的实际数据
        try:
            debug_dir = Path("outputs/debug_test_metrics")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_pred = pd.DataFrame({'y_true': y_true.values, 'y_proba': y_proba.values if y_proba is not None else y_pred.values, 'y_pred': y_pred.values})
            debug_file = debug_dir / f"auc_input_{pd.Timestamp.now().strftime('%H%M%S')}.csv"
            debug_pred.to_csv(debug_file, index=False)
            print(f"[DEBUG] Saved AUC input to {debug_file}, auc_should_be={roc_auc_score(y_true, y_proba if y_proba is not None else y_pred):.4f}")
        except Exception as e:
            print(f"[DEBUG] Failed to save AUC input: {e}")
        
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

        # 【关键修复】将数据转换为 numpy 数组副本，防止后续意外修改
        y_true_arr = y_true.values.copy() if hasattr(y_true, 'values') else np.array(y_true)
        y_pred_arr = y_pred.values.copy() if hasattr(y_pred, 'values') else np.array(y_pred)
        y_proba_arr = y_proba.values.copy() if y_proba is not None and hasattr(y_proba, 'values') else None
        
        # 【关键修复】直接计算并保存 AUC，确保 auc_input 和 metrics.auc 使用相同数据
        auc_should_be = None
        try:
            if y_proba_arr is not None:
                auc_should_be = float(roc_auc_score(y_true_arr, y_proba_arr))
            else:
                auc_should_be = float(roc_auc_score(y_true_arr, y_pred_arr))
        except Exception:
            pass

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
                        if y_proba_arr is not None:
                            metrics.auc = auc_should_be if auc_should_be is not None else float(roc_auc_score(y_true_arr, y_proba_arr))
                        else:
                            metrics.auc = auc_should_be if auc_should_be is not None else float(roc_auc_score(y_true_arr, y_pred_arr))
                        logger.info(f"[BenchmarkEvaluator] 测试集 AUC={metrics.auc:.4f} (使用{'概率' if y_proba_arr is not None else '标签'})")
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 AUC 计算失败: {e}")

                # Accuracy
                if has_val_acc or has_val_score or eval_metric == 'Accuracy':
                    try:
                        metrics.accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 Accuracy 计算失败: {e}")

                # F1
                if eval_metric in (None, 'F1', 'Accuracy', 'AUC'):
                    try:
                        metrics.f1 = float(f1_score(y_true_arr, y_pred_arr, average="binary"))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 F1 计算失败: {e}")

            elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                # 多分类
                if has_val_acc or has_val_score or eval_metric == 'Accuracy':
                    try:
                        metrics.accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 Accuracy 计算失败: {e}")

                if eval_metric in (None, 'F1-macro', 'F1', 'Accuracy'):
                    try:
                        metrics.f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
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
                            metrics.log_loss = float(log_loss(y_true_arr, y_proba_matrix))
                            logger.info(f"[BenchmarkEvaluator] 测试集 Log Loss={metrics.log_loss:.4f}")
                        else:
                            logger.warning("[BenchmarkEvaluator] 未找到概率矩阵列(proba_*)，无法计算 Log Loss")
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 Log Loss 计算失败: {e}")

            elif task_type == TaskType.REGRESSION:
                # 回归
                if has_val_rmse or has_val_score or eval_metric == 'RMSE':
                    try:
                        metrics.rmse = float(_rmse_func(y_true_arr, y_pred_arr))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 RMSE 计算失败: {e}")

                if eval_metric in (None, 'MAE', 'RMSE', 'R2'):
                    try:
                        metrics.mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 MAE 计算失败: {e}")

                if eval_metric in (None, 'R2', 'RMSE', 'MAE'):
                    try:
                        metrics.r2 = float(r2_score(y_true_arr, y_pred_arr))
                    except Exception as e:
                        logger.warning(f"[BenchmarkEvaluator] 测试集 R² 计算失败: {e}")

            # 动态计算未知指标（用户指定的指标不在上述硬编码列表中时）
            if eval_metric and not self._is_metric_already_computed(metrics, eval_metric):
                self._compute_unknown_metric(eval_metric, y_true, y_pred, y_proba, merged, metrics, task_type)

            logger.info(f"[BenchmarkEvaluator] 测试集指标计算完成: {metrics.model_dump()}")

        except Exception as e:
            logger.exception(f"[BenchmarkEvaluator] 指标计算异常: {e}")

        # 【关键调试】返回前保存 metrics
        try:
            import json
            debug_dir = Path("outputs/debug_test_metrics")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"metrics_return_{pd.Timestamp.now().strftime('%H%M%S')}.json"
            debug_file.write_text(json.dumps({
                'auc': metrics.auc,
                'accuracy': metrics.accuracy,
                'f1': metrics.f1,
                'pred_path': str(pred_path),
                'gt_path': str(gt_path)
            }, indent=2), encoding='utf-8')
            print(f"[DEBUG] metrics about to return: auc={metrics.auc}, auc_should_be={auc_should_be}, saved to {debug_file}")
        except Exception as e2:
            print(f"[DEBUG] failed to save metrics: {e2}")

        # 【关键修复】强制恢复 AUC，防止中间某行代码意外修改
        if auc_should_be is not None:
            metrics.auc = auc_should_be
        
        print(f"[DEBUG] Final check before return: metrics.auc={metrics.auc}, auc_should_be={auc_should_be}")
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
        
        # 【调试保留】保存关键的预测文件供事后分析
        for pred_name in ["test_predictions.csv", "best_test_predictions.csv"]:
            src = data_dir / pred_name
            if src.exists():
                try:
                    shutil.copy2(src, result_dir / pred_name)
                    logger.info(f"[BenchmarkEvaluator][DEBUG] 保留预测文件到结果目录: {pred_name} ({src.stat().st_size} bytes)")
                except Exception as e:
                    logger.warning(f"[BenchmarkEvaluator][DEBUG] 保留预测文件失败: {pred_name} - {e}")
        
        # 3.5 保存 LLM 生成的 predict.py（如有）
        predict_py_path = data_dir / "predict.py"
        if not predict_py_path.exists():
            predict_py_path = data_dir.parent / "artifacts" / "predict.py"
        if predict_py_path.exists():
            shutil.copy2(predict_py_path, result_dir / "predict.py")

        # 4. 保存 ground_truth
        if Path(task_cfg.ground_truth_path).exists():
            shutil.copy2(task_cfg.ground_truth_path, result_dir / "ground_truth.csv")

        # 5. 保存指标 + 意图识别结果
        metrics_data = {
            "val_metrics": result.val_metrics.model_dump() if result.val_metrics else None,
            "test_metrics": result.test_metrics.model_dump() if result.test_metrics else None,
            "best_score": result.best_score,
            "task_type": task_cfg.task_type.value,
            "target_column": task_cfg.target_column,
            "eval_metric": task_cfg.eval_metric,
            "complexity": result.complexity,
            "complexity_reason": result.complexity_reason,
            "is_time_series": result.is_time_series,
            "intent_recognized": result.intent_recognized,
            "prediction_strategy": result.prediction_strategy,
        }
        (result_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2, ensure_ascii=False), encoding='utf-8')
        
        # 5.5 保存产物总结
        if result.artifacts:
            artifact_summary = {
                "completeness": result.artifacts.completeness,
                "generated_files": result.artifacts.generated_files,
                "prediction_file": result.artifacts.prediction_file,
                "model_file": result.artifacts.model_file,
                "feature_importance_csv": result.artifacts.feature_importance_csv,
                "feature_importance_png": result.artifacts.feature_importance_png,
                "report_html": result.artifacts.report_html,
                "report_fig_png": result.artifacts.report_fig_png,
                "predict_script": result.artifacts.predict_script,
                "pipeline_py": result.artifacts.pipeline_py,
                "residual_png": result.artifacts.residual_png,
                "cluster_png": result.artifacts.cluster_png,
                "ts_forecast_png": result.artifacts.ts_forecast_png,
            }
            (result_dir / "artifact_summary.json").write_text(
                json.dumps(artifact_summary, indent=2, ensure_ascii=False), encoding='utf-8'
            )

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

        # 10. 生成易读的运行摘要（方便测试人员快速查看）
        try:
            summary_lines = self._generate_run_summary(result, task_cfg, run_index)
            (result_dir / "summary.txt").write_text("\n".join(summary_lines), encoding='utf-8')
        except Exception as e:
            logger.warning(f"[BenchmarkEvaluator] 生成 summary.txt 失败: {e}")

        logger.info(f"[BenchmarkEvaluator] 中间结果已保存至 {result_dir}")
        return result_dir

    def _generate_run_summary(
        self,
        result: BenchmarkTaskResult,
        task_cfg: BenchmarkTaskConfig,
        run_index: int
    ) -> List[str]:
        """生成易读的运行摘要（供测试人员快速查看）"""
        lines = []
        sep = "=" * 60
        
        # 头部
        lines.append(sep)
        lines.append(f"  任务运行摘要: {task_cfg.task_name} (第 {run_index} 次运行)")
        lines.append(sep)
        lines.append("")
        
        # 基本信息
        lines.append("【基本信息】")
        lines.append(f"  任务类型: {task_cfg.task_type.value if task_cfg.task_type else 'unknown'}")
        lines.append(f"  目标列: {task_cfg.target_column}")
        lines.append(f"  评估指标: {task_cfg.eval_metric or 'auto'}")
        lines.append(f"  复杂度判定: {result.complexity or 'unknown'} ({result.complexity_reason or 'N/A'})")
        lines.append(f"  时序任务: {'是' if result.is_time_series else '否'}")
        lines.append(f"  意图识别: {'✅ 成功' if result.intent_recognized else '❌ 失败/回退'}")
        lines.append(f"  预测策略: {result.prediction_strategy or 'N/A'}")
        lines.append(f"  运行结果: {'✅ 成功' if result.success else '❌ 失败'}")
        lines.append(f"  评审通过: {'✅ 是' if result.judge_accepted else '❌ 否'}")
        lines.append("")
        
        # 核心指标
        lines.append("【核心指标】")
        lines.append(f"  Best Score: {result.best_score:.2f}" if result.best_score is not None else "  Best Score: N/A")
        lines.append(f"  总耗时: {result.duration_seconds:.1f}s")
        if result.val_metrics:
            vm = result.val_metrics
            if vm.val_score is not None:
                lines.append(f"  验证集 Score: {vm.val_score:.4f}")
            if vm.val_auc is not None:
                lines.append(f"  验证集 AUC: {vm.val_auc:.4f}")
            if vm.val_rmse is not None:
                lines.append(f"  验证集 RMSE: {vm.val_rmse:.4f}")
        if result.test_metrics:
            tm = result.test_metrics
            if tm.f1 is not None:
                lines.append(f"  测试集 F1: {tm.f1:.4f}")
            if tm.auc is not None:
                lines.append(f"  测试集 AUC: {tm.auc:.4f}")
            if tm.rmse is not None:
                lines.append(f"  测试集 RMSE: {tm.rmse:.4f}")
        lines.append("")
        
        # 各阶段耗时
        if result.timing:
            t = result.timing
            lines.append("【各阶段耗时】")
            lines.append(f"  意图识别: {t.intent_recognition_seconds:.1f}s")
            lines.append(f"  代码生成: {t.code_generation_seconds:.1f}s")
            lines.append(f"  沙箱执行: {t.sandbox_execution_seconds:.1f}s")
            lines.append(f"  评估优化: {t.evaluation_seconds:.1f}s")
            lines.append(f"  产物生成: {t.artifact_generation_seconds:.1f}s")
            lines.append(f"  测试预测: {t.test_prediction_seconds:.1f}s")
            lines.append(f"  总计: {t.total_seconds:.1f}s")
            lines.append("")
        
        # Token 消耗
        if result.token_usage:
            tu = result.token_usage
            lines.append("【Token 消耗】")
            lines.append(f"  Plan/Coding 调用: {tu.plan_coding_calls} 次, {tu.plan_coding_total_tokens} tokens")
            lines.append(f"  Evaluation 调用: {tu.evaluation_calls} 次, {tu.evaluation_total_tokens} tokens")
            lines.append(f"  总计: {tu.total_calls} 次, {tu.total_tokens} tokens")
            lines.append("")
        
        # LLM 使用追踪
        if result.llm_usage_trace:
            lines.append("【LLM 使用追踪】")
            trace = result.llm_usage_trace
            for stage in ["intent", "plan_coding", "evaluation", "judge"]:
                info = trace.get(stage)
                if info:
                    model = info.get("primary_model", "unknown")
                    provider = info.get("primary_provider", "unknown")
                    calls = info.get("total_calls", 0)
                    tokens = info.get("total_tokens", 0)
                    latency = info.get("total_latency_seconds", 0)
                    fallback = info.get("fallback_triggers", 0)
                    fb_info = f" (fallback={fallback})" if fallback else ""
                    lines.append(f"  {stage:12s}: {model}@{provider} | {calls}calls | {tokens}tokens | {latency:.1f}s{fb_info}")
                    # 模型细分
                    breakdown = info.get("model_breakdown", [])
                    for bd in breakdown:
                        lines.append(f"    - {bd.get('model')}@{bd.get('provider')}: {bd.get('calls')} calls, {bd.get('tokens')} tokens, {bd.get('latency', 0):.1f}s")
            lines.append("")
        
        # 产物完整性
        if result.artifacts:
            art = result.artifacts
            lines.append("【产物生成】")
            lines.append(f"  完整性: {art.completeness or 'unknown'}")
            lines.append(f"  生成文件: {', '.join(art.generated_files) if art.generated_files else '无'}")
            lines.append("")
        
        # 评审结果
        if result.judge_analysis:
            lines.append("【评审分析】")
            lines.append(f"  {result.judge_analysis[:300]}...")
            lines.append("")
        
        # 错误信息
        if result.error_message:
            lines.append("【错误信息】")
            lines.append(f"  {result.error_message[:500]}")
            lines.append("")
        
        # 维度评分
        if result.dimension_scores:
            lines.append("【维度评分】")
            for ds in result.dimension_scores:
                name = ds.get("dimension", "unknown")
                score = ds.get("score", "N/A")
                weight = ds.get("weight", "")
                w_str = f" (weight={weight})" if weight else ""
                lines.append(f"  {name}: {score}{w_str}")
            lines.append("")
        
        # 尾部
        lines.append(sep)
        lines.append(f"  结果目录: {result.result_dir or 'N/A'}")
        lines.append(f"  生成时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append(sep)
        
        return lines

    def _try_extract_best_effort(
        self,
        task_id: str,
        data_dir: Path,
        task_cfg: BenchmarkTaskConfig,
        result: BenchmarkTaskResult,
        task_state,
        intent_seconds: float,
        task_start: float,
        timeout_reason: str = "TIMEOUT"
    ) -> bool:
        """
        【超时/失败降级提取】尝试使用已训练的最佳模型做测试预测。
        
        只要有 best_model.pkl 或 test_predictions.csv 存在，就尽力给用户结果。
        返回 True 表示提取成功，False 表示无法提取。
        """
        try:
            # 1. 提取已有信息
            if task_state:
                result.best_score = task_state.best_score
                result.val_metrics = task_state.best_metrics
            
            # 2. 保存最佳代码（供策略2注入式预测使用）
            best_code = task_state.best_code if task_state else None
            if best_code:
                try:
                    (data_dir.parent / "code_best.py").write_text(best_code, encoding='utf-8')
                    logger.info(f"[BenchmarkEvaluator] 已保存 best_code 到 code_best.py")
                except Exception as e:
                    logger.warning(f"[BenchmarkEvaluator] 保存 best_code 失败: {e}")
            
            # 3. 调试：打印 data_dir 内容
            try:
                files_in_data = list(data_dir.iterdir())
                logger.info(f"[BenchmarkEvaluator] data_dir 文件: {[f.name for f in files_in_data]}")
            except Exception as e:
                logger.warning(f"[BenchmarkEvaluator] 列出 data_dir 失败: {e}")
            
            # 4. 尝试测试预测
            pred_start = time.time()
            pred_path, prediction_strategy = self._run_test_prediction(data_dir, task_cfg)
            pred_seconds = time.time() - pred_start
            
            if pred_path and Path(pred_path).exists():
                # ===== 提取成功 =====
                result.success = True
                result.phase = "partial_success"
                result.prediction_strategy = prediction_strategy
                result.error_message = (
                    f"{timeout_reason}: 原流程未正常完成，"
                    f"但已成功提取最佳模型进行测试预测(strategy={prediction_strategy})"
                )
                
                # 计算测试指标
                try:
                    test_metrics = self._compute_test_metrics(
                        pred_path,
                        task_cfg.ground_truth_path,
                        task_cfg.task_type,
                        result.val_metrics,
                        task_cfg.eval_metric
                    )
                    result.test_metrics = test_metrics
                    # 在 error_message 中追加关键指标，方便一眼看到
                    if test_metrics:
                        if getattr(test_metrics, 'test_auc', None) is not None:
                            result.error_message += f" | test_auc={test_metrics.test_auc:.4f}"
                        elif getattr(test_metrics, 'test_rmse', None) is not None:
                            result.error_message += f" | test_rmse={test_metrics.test_rmse:.4f}"
                        elif getattr(test_metrics, 'auc', None) is not None:
                            result.error_message += f" | test_auc={test_metrics.auc:.4f}"
                        elif getattr(test_metrics, 'rmse', None) is not None:
                            result.error_message += f" | test_rmse={test_metrics.rmse:.4f}"
                except Exception as e:
                    logger.warning(f"[BenchmarkEvaluator] 降级提取时计算测试指标失败: {e}")
                
                # 记录时间（尽量收集引擎计时，不可用则为0）
                try:
                    engine = get_or_create_engine(task_id)
                    result.timing = TimingBreakdown(
                        intent_recognition_seconds=intent_seconds,
                        code_generation_seconds=engine.timings.get("code_generation_seconds", 0.0),
                        sandbox_execution_seconds=engine.timings.get("sandbox_execution_seconds", 0.0),
                        evaluation_seconds=engine.timings.get("evaluation_seconds", 0.0),
                        artifact_generation_seconds=engine.timings.get("artifact_generation_seconds", 0.0),
                        test_prediction_seconds=pred_seconds,
                        total_seconds=time.time() - task_start
                    )
                except Exception:
                    result.timing = TimingBreakdown(
                        intent_recognition_seconds=intent_seconds,
                        test_prediction_seconds=pred_seconds,
                        total_seconds=time.time() - task_start
                    )
                
                # 收集维度评分（如果有）
                if task_state and task_state.best_evaluation and task_state.best_evaluation.dimension_scores:
                    result.dimension_scores = [
                        ds.model_dump() for ds in task_state.best_evaluation.dimension_scores
                    ]
                
                # 产物检测
                try:
                    result.artifacts = self._detect_artifacts(data_dir)
                except Exception:
                    pass
                
                # 保存中间结果
                try:
                    result_dir = self._save_intermediate_results(
                        result, task_cfg, result.run_index, task_state, data_dir, engine=get_or_create_engine(task_id)
                    )
                    result.result_dir = str(result_dir)
                except Exception as e:
                    logger.warning(f"[BenchmarkEvaluator] 降级提取时保存中间结果失败: {e}")
                
                logger.info(
                    f"[BenchmarkEvaluator] 降级提取成功: {result.error_message}"
                )
                return True
            else:
                logger.warning(
                    f"[BenchmarkEvaluator] 降级提取失败: 未找到可用的预测文件或模型 "
                    f"(data_dir={data_dir})"
                )
                return False
                
        except Exception as e:
            logger.exception(f"[BenchmarkEvaluator] 降级提取过程异常: {e}")
            return False

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
            "dim_code_completeness", "dim_task_alignment",
            # 验证集指标
            "val_auc", "val_accuracy", "val_rmse", "val_score",
            # 测试集指标
            "test_auc", "test_accuracy", "test_f1", "test_f1_macro",
            "test_rmse", "test_mae", "test_r2",
            # Judge
            "judge_accepted", "judge_analysis", "judge_reason",
            # 耗时
            "intent_seconds", "code_gen_seconds", "sandbox_seconds", "eval_seconds",
            "artifact_seconds", "test_pred_seconds", "total_seconds",
            # Token 消耗（除 Judge 外）
            "plan_coding_calls", "plan_coding_tokens",
            "evaluation_calls", "evaluation_tokens",
            "total_llm_calls", "total_tokens",
            # LLM 使用追踪
            "intent_model", "intent_latency",
            "plan_coding_model", "plan_coding_latency",
            "evaluation_model", "evaluation_latency",
            "judge_model", "fallback_triggers",
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
                    "dim_code_completeness": dim_scores.get("code_completeness", ""),
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
                    "intent_seconds": round(r.timing.intent_recognition_seconds, 2) if r.timing else "",
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
                    # LLM 使用追踪
                    "intent_model": r.llm_usage_trace.get("intent", {}).get("primary_model", "") if r.llm_usage_trace else "",
                    "intent_latency": r.llm_usage_trace.get("intent", {}).get("total_latency_seconds", "") if r.llm_usage_trace else "",
                    "plan_coding_model": r.llm_usage_trace.get("plan_coding", {}).get("primary_model", "") if r.llm_usage_trace else "",
                    "plan_coding_latency": r.llm_usage_trace.get("plan_coding", {}).get("total_latency_seconds", "") if r.llm_usage_trace else "",
                    "evaluation_model": r.llm_usage_trace.get("evaluation", {}).get("primary_model", "") if r.llm_usage_trace else "",
                    "evaluation_latency": r.llm_usage_trace.get("evaluation", {}).get("total_latency_seconds", "") if r.llm_usage_trace else "",
                    "judge_model": r.llm_usage_trace.get("judge", {}).get("primary_model", "") if r.llm_usage_trace else "",
                    "fallback_triggers": r.llm_usage_trace.get("plan_coding", {}).get("fallback_triggers", "") if r.llm_usage_trace else "",
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



#!/usr/bin/env python3
"""
ML-Workflow 全量测评脚本
=========================

功能：
1. 扫描 test_benchmark/ 和 test_data/ 下的所有评测任务（共 20 个）
2. 每个任务运行 N 次（默认 3 次），每次彻底冷启动
3. 支持分批测试：先跑第一批 9 个重点验证任务，再跑第二批 11 个补充任务
4. 详细保存中间信息（时间、token、日志、产物、LLM 调用追踪）
5. 生成用户友好的 Markdown 报告

用法：
    cd backend && source venv/Scripts/activate

    # 跑全部 20 个任务
    python scripts/run_full_benchmark.py

    # 分批测试：先跑第一批（9 个重点验证任务）
    python scripts/run_full_benchmark.py --batch 1

    # 分批测试：再跑第二批（11 个补充任务）
    python scripts/run_full_benchmark.py --batch 2

    # 只跑 1 次（快速验证）
    python scripts/run_full_benchmark.py --batch 1 --runs 1

    # 手动指定任务（模糊匹配）
    python scripts/run_full_benchmark.py --tasks 信用卡 单车 房价

    # 自定义最大等待时间
    python scripts/run_full_benchmark.py --batch 1 --max-wait 1800

输出目录：outputs/benchmark_{timestamp}/
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 将项目根目录加入路径
# scripts/run_full_benchmark.py → backend/ → 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Python 路径也需要指向 backend/
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.core.evaluator import BenchmarkEvaluator
from app.models.schemas import LLMConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("run_full_benchmark")

# ============================================================
# 配置区：扫描目录和 LLM 配置
# ============================================================

# 待扫描的 benchmark 目录（相对于项目根目录）
DEFAULT_BENCHMARK_DIRS = [
    "test_benchmark",
    "test_data",
]

# ============================================================
# 全量任务清单（共 20 个任务，分 2 批）
# ============================================================
# 格式: (匹配关键字, 所在目录)
FULL_TASK_LIST = [
    # ----- 第一批：9 个重点验证任务 -----
    ("2026年股票", "test_data"),
    ("信用卡欺诈", "test_data"),
    ("共享单车", "test_data"),
    ("加利福尼亚房价", "test_data"),
    ("北京PM2.5", "test_data"),
    ("电商顾客退货", "test_data"),
    ("睡眠障碍", "test_data"),
    ("红酒品质", "test_data"),
    ("银行账户欺诈", "test_data"),
    # ----- 第二批：11 个补充任务 -----
    ("医疗保险费用", "test_data"),
    ("吸烟状况", "test_data"),
    ("垃圾邮件判别", "test_data"),
    ("成人收入预测", "test_data"),
    ("支付欺诈", "test_data"),
    ("机翼噪声", "test_data"),
    ("电子商务客户流失", "test_data"),
    ("糖尿病预测", "test_data"),
    ("肝硬化患者状态", "test_data"),
    ("鸢尾花种类", "test_data"),
    ("黑色素瘤种类", "test_data"),
]

BATCH_1_KEYWORDS = [t[0] for t in FULL_TASK_LIST[:9]]
BATCH_2_KEYWORDS = [t[0] for t in FULL_TASK_LIST[9:]]

# LLM 配置（从环境变量或 settings 读取，可覆盖）
def _get_llm_config(env_prefix: str, default_model: str) -> Optional[LLMConfig]:
    """从环境变量或 settings 构建 LLMConfig。
    
    优先级：os.environ > settings（.env 文件）> 默认值
    pydantic-settings 读取 .env 后不会写入 os.environ，因此必须同时检查两者。
    """
    def _get(field: str, default: str) -> str:
        # 1. 真正的环境变量（最高优先级）
        val = os.environ.get(f"{env_prefix}_{field}")
        if val:
            return val
        # 2. settings 中读取的 .env 配置
        settings_val = getattr(settings, f"{env_prefix}_{field}", None)
        if settings_val:
            return settings_val
        # 3. 默认值
        return default
    
    model = _get("MODEL", default_model)
    provider = _get("PROVIDER", "openai")
    base_url = _get("BASE_URL", settings.LLM_BASE_URL)
    api_key = _get("API_KEY", settings.LLM_API_KEY)
    
    if not model:
        return None
    
    return LLMConfig(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        max_tokens=8192,
    )


def get_default_llm_configs() -> dict:
    """获取默认的 LLM 配置映射（读取 .env 中的 EVAL_* 配置）"""
    # Plan/Coding Agent：读取 EVAL_PLAN_*, EVAL_CODING_*, EVAL_UNIFIED_*
    plan = _get_llm_config("EVAL_PLAN", settings.EVAL_PLAN_MODEL or settings.LLM_MODEL)
    coding = _get_llm_config("EVAL_CODING", settings.EVAL_CODING_MODEL or settings.LLM_MODEL)
    unified = _get_llm_config("EVAL_UNIFIED", settings.EVAL_UNIFIED_MODEL or settings.LLM_MODEL)
    
    # plan_coding 优先使用 EVAL_PLAN_CODING_*，未设置则完整继承 coding 配置
    plan_coding = _get_llm_config("EVAL_PLAN_CODING", settings.EVAL_PLAN_CODING_MODEL or settings.EVAL_CODING_MODEL or settings.LLM_MODEL)
    if plan_coding and not settings.EVAL_PLAN_CODING_BASE_URL and coding:
        plan_coding.base_url = coding.base_url
        plan_coding.api_key = coding.api_key
        plan_coding.provider = coding.provider
    
    # Evaluation Agent
    evaluation = _get_llm_config("EVAL_EVALUATION", settings.EVAL_EVALUATION_MODEL or settings.LLM_MODEL)
    
    # Judge Agent
    judge = _get_llm_config("EVAL_JUDGE", settings.EVAL_JUDGE_MODEL or settings.LLM_MODEL)
    
    # Intent Agent
    intent = _get_llm_config("EVAL_INTENT", settings.EVAL_INTENT_MODEL or settings.LLM_MODEL)
    
    return {
        "plan_coding": plan_coding,
        "plan": plan,
        "coding": coding,
        "unified": unified,
        "evaluation": evaluation,
        "judge": judge,
        "intent": intent,
    }


# ============================================================
# 目录扫描
# ============================================================

def discover_all_tasks(project_root: Path, benchmark_dirs: List[str]) -> List[Path]:
    """
    扫描多个 benchmark 目录，返回所有任务目录路径。
    
    任务目录特征：包含 "建模" 和 "评估" 子目录。
    """
    all_tasks: List[Path] = []
    
    for bench_name in benchmark_dirs:
        bench_path = project_root / bench_name
        if not bench_path.exists():
            logger.warning(f"[Scanner] 目录不存在，跳过: {bench_path}")
            continue
        
        logger.info(f"[Scanner] 扫描目录: {bench_path}")
        
        # 遍历 bench_path 下的一级子目录
        for item in sorted(bench_path.iterdir()):
            if not item.is_dir():
                continue
            
            # 检查是否同时包含 "建模" 和 "评估" 子目录
            has_modeling = False
            has_eval = False
            try:
                for sub in item.iterdir():
                    if not sub.is_dir():
                        continue
                    if "建模" in sub.name:
                        has_modeling = True
                    elif "评估" in sub.name:
                        has_eval = True
            except Exception as e:
                logger.warning(f"[Scanner] 读取目录失败 {item.name}: {e}")
                continue
            
            if has_modeling and has_eval:
                all_tasks.append(item)
                logger.info(f"[Scanner] 发现任务: {item.name}")
            else:
                logger.debug(f"[Scanner] 跳过 {item.name}: modeling={has_modeling}, eval={has_eval}")
    
    logger.info(f"[Scanner] 共发现 {len(all_tasks)} 个评测任务")
    return all_tasks


# ============================================================
# 测评系统自检
# ============================================================

def self_check(project_root: Path, tasks: List[Path]) -> dict:
    """
    测评系统自检：在正式运行前检测常见配置/环境问题。
    
    返回: {"passed": bool, "issues": List[str], "warnings": List[str]}
    """
    issues = []
    warnings = []
    
    # 1. 检查 venv（在项目根目录或 backend/ 下）
    venv_paths = [
        project_root / "venv" / "Scripts" / "python.exe",
        project_root / "backend" / "venv" / "Scripts" / "python.exe",
    ]
    venv_found = any(p.exists() for p in venv_paths)
    if not venv_found:
        issues.append(f"虚拟环境不存在，已查找: {[str(p) for p in venv_paths]}")
    
    # 2. 检查 .env / API Key
    if not settings.LLM_API_KEY:
        issues.append("LLM_API_KEY 未配置，无法调用 LLM")
    
    # 3. 检查 uploads / outputs 目录可写
    for dirname in ["uploads", "outputs"]:
        d = project_root / dirname
        try:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"目录不可写 {d}: {e}")
    
    # 4. 检查每个任务的基本文件完整性
    for task_dir in tasks:
        try:
            modeling_dir = None
            eval_dir = None
            for sub in task_dir.iterdir():
                if not sub.is_dir():
                    continue
                if "建模" in sub.name:
                    modeling_dir = sub
                elif "评估" in sub.name:
                    eval_dir = sub
            
            if not modeling_dir:
                issues.append(f"[{task_dir.name}] 缺少建模目录")
                continue
            if not eval_dir:
                issues.append(f"[{task_dir.name}] 缺少评估目录")
                continue
            
            train_files = list(modeling_dir.glob("*train*.csv"))
            if not train_files:
                issues.append(f"[{task_dir.name}] 建模目录缺少 train 文件")
            
            gt_files = list(eval_dir.glob("*.csv"))
            if not gt_files:
                issues.append(f"[{task_dir.name}] 评估目录缺少 ground_truth 文件")
            
            desc_files = list(modeling_dir.glob("*.txt"))
            if not desc_files:
                warnings.append(f"[{task_dir.name}] 缺少任务描述文件(task_desc.txt)")
        
        except Exception as e:
            issues.append(f"[{task_dir.name}] 自检异常: {e}")
    
    passed = len(issues) == 0
    return {"passed": passed, "issues": issues, "warnings": warnings}


# ============================================================
# 用户报告生成
# ============================================================

def generate_user_report(result_base_dir: Path, report: dict) -> Path:
    """
    生成面向用户的 Markdown 报告，包含时间消耗、Token 消耗、成功率等
    从用户角度关心的问题。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    
    lines.append("# ML-Workflow 全量测评报告")
    lines.append(f"\n生成时间: {now}")
    lines.append(f"测评 ID: {report.get('eval_id', 'N/A')}")
    lines.append(f"总任务数: {report.get('total_tasks', 0)}")
    lines.append(f"每任务运行次数: {report.get('num_runs', 0)}")
    lines.append(f"总运行次数: {report.get('total_runs', 0)}")
    lines.append("")
    
    # 总体统计
    lines.append("## 📊 总体统计")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 整体成功率 (Judge Accepted) | {report.get('overall_success_rate', 0):.1%} |")
    lines.append(f"| 总通过次数 | {report.get('total_accepted', 0)} / {report.get('total_runs', 0)} |")
    lines.append(f"| 全局平均耗时 | {report.get('overall_avg_duration_seconds', 0):.1f}s |")
    lines.append(f"| 全局平均 Token | {report.get('overall_avg_total_tokens', 0):,} |")
    lines.append(f"| 全局 Score 标准差 | {report.get('overall_score_std', 0):.4f} |")
    lines.append("")
    
    # 各任务详情
    lines.append("## 📋 各任务测评详情")
    lines.append("")
    
    round_results = report.get("round_results", [])
    task_names = report.get("task_names", [])
    for idx, round_result in enumerate(round_results):
        task_name = task_names[idx] if idx < len(task_names) else f"task_{idx}"
        success_rate = round_result.get("success_rate", 0)
        avg_duration = round_result.get("avg_duration_seconds", 0)
        avg_tokens = round_result.get("avg_total_tokens", 0)
        score_std = round_result.get("score_std", 0)
        
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append(f"- **成功率**: {success_rate:.1%} ({round_result.get('success_count', 0)}/{round_result.get('fail_count', 0) + round_result.get('success_count', 0)})")
        lines.append(f"- **平均耗时**: {avg_duration:.1f}s")
        lines.append(f"- **平均 Token**: {avg_tokens:,}")
        lines.append(f"- **Score 稳定性 (std)**: {score_std:.4f}")
        lines.append("")
        
        # 每次运行的明细
        lines.append("| 运行 | 结果 | Judge | 耗时(s) | Token | 产物完整度 | 预测策略 |")
        lines.append("|------|------|-------|---------|-------|-----------|----------|")
        
        for tr in round_result.get("task_results", []):
            run_idx = tr.get("run_index", 0)
            success = "✅" if tr.get("success") else "❌"
            judge = "✅" if tr.get("judge_accepted") else "❌"
            duration = tr.get("duration_seconds", 0)
            tokens = tr.get("token_usage", {}).get("total_tokens", 0) if tr.get("token_usage") else 0
            completeness = tr.get("artifacts", {}).get("completeness", "N/A") if tr.get("artifacts") else "N/A"
            strategy = tr.get("prediction_strategy", "N/A") or "N/A"
            lines.append(f"| {run_idx} | {success} | {judge} | {duration:.1f} | {tokens:,} | {completeness} | {strategy} |")
        
        lines.append("")
    
    # 测评系统自检
    self_check_result = report.get("self_check", {})
    if self_check_result:
        lines.append("## 🔧 测评系统自检")
        lines.append("")
        if self_check_result.get("passed"):
            lines.append("✅ **自检通过**")
        else:
            lines.append("❌ **自检未通过**")
        
        if self_check_result.get("issues"):
            lines.append("\n### 问题 (Issues)")
            for issue in self_check_result["issues"]:
                lines.append(f"- ❌ {issue}")
        
        if self_check_result.get("warnings"):
            lines.append("\n### 警告 (Warnings)")
            for warning in self_check_result["warnings"]:
                lines.append(f"- ⚠️ {warning}")
        lines.append("")
    
    # 附录：文件位置
    lines.append("## 📁 输出文件")
    lines.append("")
    lines.append(f"- JSON 报告: `{result_base_dir / 'benchmark_report.json'}`")
    lines.append(f"- CSV 明细: `{result_base_dir / 'benchmark_results.csv'}`")
    lines.append(f"- CSV 汇总: `{result_base_dir / 'benchmark_summary.csv'}`")
    lines.append(f"- Markdown 报告: `{result_base_dir / 'user_report.md'}`")
    lines.append("")
    
    content = "\n".join(lines)
    report_path = result_base_dir / "user_report.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path


# ============================================================
# 增强版 BenchmarkEvaluator：支持多目录 + 更完善的保存
# ============================================================

class FullBenchmarkEvaluator(BenchmarkEvaluator):
    """
    增强版评测器，支持：
    1. 从多个目录收集任务
    2. 更完善的中间结果保存（意图识别、复杂度、时序判定等）
    3. 用户报告生成
    """
    
    def __init__(self, task_dirs: List[Path], num_runs: int = 3, max_wait_seconds: int = 1200, **kwargs):
        # 使用第一个任务目录的父目录作为 benchmark_dir（dummy）
        dummy_dir = str(task_dirs[0].parent) if task_dirs else "."
        super().__init__(
            benchmark_dir=dummy_dir,
            num_runs=num_runs,
            max_wait_seconds=max_wait_seconds,
            **kwargs
        )
        self._task_dirs = task_dirs
    
    def _discover_tasks(self):
        """覆盖父类方法：从多个预扫描的目录中解析任务"""
        from app.models.evaluate_schemas import BenchmarkTaskConfig
        
        tasks = []
        self._restore_all_hidden_test_csvs()
        
        for task_dir in self._task_dirs:
            task_cfg = self._parse_task_dir(task_dir, task_dir.name)
            if task_cfg:
                tasks.append(task_cfg)
                logger.info(f"[FullBenchmarkEvaluator] 加载任务: {task_dir.name}")
        
        return tasks
    
    def _restore_all_hidden_test_csvs(self):
        """恢复所有被扫描目录下的 hidden 测试集"""
        restored_count = 0
        for task_dir in self._task_dirs:
            for sub in task_dir.iterdir():
                if not sub.is_dir():
                    continue
                if "建模" in sub.name:
                    hidden = sub / "test.csv.hidden"
                    original = sub / "test.csv"
                    if hidden.exists():
                        if original.exists():
                            original.unlink()
                        hidden.rename(original)
                        restored_count += 1
                        logger.info(f"[FullBenchmarkEvaluator] 恢复 hidden: {original}")
        if restored_count > 0:
            logger.info(f"[FullBenchmarkEvaluator] 共恢复 {restored_count} 个 hidden 测试集")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ML-Workflow 全量测评脚本")
    parser.add_argument("--runs", type=int, default=3, help="每个任务的运行次数（默认3）")
    parser.add_argument("--max-wait", type=int, default=1200, help="最大等待秒数（默认1200）")
    parser.add_argument("--benchmark-dirs", nargs="+", default=None, 
                        help="手动指定 benchmark 目录（默认扫描 test_benchmark 和 test_data）")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="只运行指定任务名（模糊匹配），与 --batch 互斥")
    parser.add_argument("--batch", type=int, choices=[1, 2], default=None,
                        help="分批运行: 1=第一批(9个重点验证任务), 2=第二批(11个补充任务)")
    parser.add_argument("--skip-check", action="store_true", help="跳过自检")
    parser.add_argument("--eval-id", type=str, default=None, help="自定义评测 ID")
    args = parser.parse_args()
    
    project_root = PROJECT_ROOT
    benchmark_dirs = args.benchmark_dirs or DEFAULT_BENCHMARK_DIRS
    
    logger.info("=" * 60)
    logger.info("ML-Workflow 全量测评启动")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"扫描目录: {benchmark_dirs}")
    logger.info(f"每任务运行次数: {args.runs}")
    logger.info(f"最大等待时间: {args.max_wait}s")
    logger.info("=" * 60)
    
    # ---- Step 1: 扫描任务 ----
    all_task_dirs = discover_all_tasks(project_root, benchmark_dirs)
    
    # 分批处理：--batch 与 --tasks 互斥，batch 优先级高
    if args.batch is not None:
        if args.batch == 1:
            batch_keywords = BATCH_1_KEYWORDS
            logger.info(f"[Batch] 选择第一批: 9 个重点验证任务")
        else:
            batch_keywords = BATCH_2_KEYWORDS
            logger.info(f"[Batch] 选择第二批: 11 个补充任务")
        
        filtered = []
        for t in all_task_dirs:
            for kw in batch_keywords:
                if kw in t.name:
                    filtered.append(t)
                    break
        all_task_dirs = filtered
        logger.info(f"[Batch] 匹配到 {len(all_task_dirs)} 个任务")
    
    elif args.tasks:
        filtered = []
        for t in all_task_dirs:
            for pattern in args.tasks:
                if pattern in t.name:
                    filtered.append(t)
                    break
        all_task_dirs = filtered
        logger.info(f"[Filter] 过滤后任务数: {len(all_task_dirs)}")
    
    if not all_task_dirs:
        logger.error("未发现任何评测任务，退出")
        sys.exit(1)
    
    # ---- Step 2: 自检 ----
    self_check_result = {"passed": True, "issues": [], "warnings": []}
    if not args.skip_check:
        logger.info("[SelfCheck] 开始测评系统自检...")
        self_check_result = self_check(project_root, all_task_dirs)
        
        if self_check_result["warnings"]:
            for w in self_check_result["warnings"]:
                logger.warning(f"[SelfCheck] ⚠️ {w}")
        
        if self_check_result["issues"]:
            for issue in self_check_result["issues"]:
                logger.error(f"[SelfCheck] ❌ {issue}")
            logger.error("[SelfCheck] 自检未通过，请修复上述问题后重试")
            
            # 非交互模式下记录警告但继续运行（允许用户手动跳过自检）
            logger.warning("[SelfCheck] 自检未通过，但 --skip-check 未设置，尝试继续运行...")
            logger.warning("[SelfCheck] 如遇问题请手动检查上述 issue")
        else:
            logger.info("[SelfCheck] ✅ 自检通过")
    
    # ---- Step 3: 构建 LLM 配置 ----
    llm_configs = get_default_llm_configs()
    logger.info(f"[LLM] Plan/Coding: {llm_configs.get('plan_coding', {}).model if llm_configs.get('plan_coding') else '默认'}")
    logger.info(f"[LLM] Evaluation: {llm_configs.get('evaluation', {}).model if llm_configs.get('evaluation') else '默认'}")
    logger.info(f"[LLM] Judge: {llm_configs.get('judge', {}).model if llm_configs.get('judge') else '默认'}")
    
    # ---- Step 4: 运行评测 ----
    evaluator = FullBenchmarkEvaluator(
        task_dirs=all_task_dirs,
        num_runs=args.runs,
        max_wait_seconds=args.max_wait,
        eval_id=args.eval_id,
        plan_coding_llm_config=llm_configs.get("plan_coding"),
        plan_llm_config=llm_configs.get("plan"),
        coding_llm_config=llm_configs.get("coding"),
        unified_llm_config=llm_configs.get("unified"),
        evaluation_llm_config=llm_configs.get("evaluation"),
        judge_llm_config=llm_configs.get("judge"),
    )
    
    start_time = time.time()
    try:
        report = evaluator.run_benchmark()
    except KeyboardInterrupt:
        logger.warning("[Main] 收到中断信号，正在停止...")
        evaluator.stop()
        raise
    except Exception as e:
        logger.exception(f"[Main] 评测运行异常: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # ---- Step 5: 生成用户报告 ----
    report_dict = json.loads(report.model_dump_json())
    report_dict["self_check"] = self_check_result
    report_dict["total_wall_time_seconds"] = total_time
    
    result_base_dir = evaluator.result_base_dir
    user_report_path = generate_user_report(result_base_dir, report_dict)
    
    # ---- Step 6: 输出摘要 ----
    logger.info("=" * 60)
    logger.info("评测完成!")
    logger.info(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"整体成功率: {report.overall_success_rate:.1%}")
    logger.info(f"通过/总运行: {report.total_accepted}/{args.runs * len(all_task_dirs)}")
    logger.info(f"全局平均耗时: {report.overall_avg_duration_seconds:.1f}s")
    logger.info(f"全局平均 Token: {report.overall_avg_total_tokens:,}")
    logger.info(f"结果目录: {result_base_dir}")
    logger.info(f"用户报告: {user_report_path}")
    logger.info("=" * 60)
    
    # 失败的详细列表
    failed_tasks = []
    for rr in report.round_results:
        for tr in rr.task_results:
            if not tr.judge_accepted:
                failed_tasks.append({
                    "task": rr.task_name,
                    "run": tr.run_index,
                    "reason": tr.judge_reason or tr.error_message or "Unknown",
                })
    
    if failed_tasks:
        logger.info("\n未通过 Judge 的任务列表:")
        for ft in failed_tasks:
            logger.info(f"  - {ft['task']} (run {ft['run']}): {ft['reason'][:100]}...")


if __name__ == "__main__":
    main()

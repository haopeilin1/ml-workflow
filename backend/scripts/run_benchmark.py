"""
命令行评测脚本
直接调用 BenchmarkEvaluator 执行评测，不走 HTTP API

Usage:
    python -m scripts.run_benchmark --benchmark-dir ./benchmarks --num-runs 3
    python -m scripts.run_benchmark --benchmark-dir ./benchmarks --judge-model gpt-4o
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 将 backend 目录加入路径
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.evaluator import BenchmarkEvaluator
from app.models.schemas import LLMConfig
from app.config import settings, build_eval_llm_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ML Workflow 自动化评测")
    parser.add_argument("--benchmark-dir", required=True, help="评测数据集根目录")
    parser.add_argument("--num-runs", type=int, default=3, help="每个任务运行次数（默认3）")
    parser.add_argument("--max-wait", type=int, default=1200, help="每个任务最大等待时间（秒，默认1200）")
    parser.add_argument("--judge-model", default="", help="覆盖 Judge LLM model（默认从 .env 读取）")
    parser.add_argument("--plan-model", default="", help="覆盖 PlanCoding LLM model（默认从 .env 读取）")
    parser.add_argument("--output", default="", help="报告输出路径（默认保存到 outputs/eval_xxx/）")

    args = parser.parse_args()

    # 从 settings/.env 构建 LLM 配置，支持命令行覆盖 model
    judge_cfg = build_eval_llm_config("judge")
    plan_cfg = build_eval_llm_config("plan_coding")
    if args.judge_model:
        judge_cfg["model"] = args.judge_model
    if args.plan_model:
        plan_cfg["model"] = args.plan_model

    judge_llm_config = LLMConfig(**judge_cfg)
    plan_llm_config = LLMConfig(**plan_cfg)

    logger.info("=" * 60)
    logger.info("ML Workflow 自动化评测启动")
    logger.info(f"评测目录: {args.benchmark_dir}")
    logger.info(f"每个任务运行次数: {args.num_runs}")
    logger.info(f"Judge LLM: {judge_cfg['provider']}/{judge_cfg['model']}")
    logger.info(f"PlanCoding LLM: {plan_cfg['provider']}/{plan_cfg['model']}")
    logger.info("=" * 60)

    # 创建评测器并执行
    evaluator = BenchmarkEvaluator(
        benchmark_dir=args.benchmark_dir,
        num_runs=args.num_runs,
        judge_llm_config=judge_llm_config,
        plan_coding_llm_config=plan_llm_config,
        max_wait_seconds=args.max_wait
    )

    report = evaluator.run_benchmark()

    # 输出结果
    logger.info("=" * 60)
    logger.info("评测完成")
    logger.info(f"总任务数: {report.total_tasks}")
    logger.info(f"总运行次数: {report.total_runs}")
    logger.info(f"通过次数: {report.total_accepted}")
    logger.info(f"整体成功率: {report.overall_success_rate:.1%}")
    logger.info("=" * 60)

    # 每轮详细结果
    for round_result in report.round_results:
        logger.info(f"\n任务: {round_result.task_results[0].task_name if round_result.task_results else 'unknown'}")
        logger.info(f"  成功率: {round_result.success_rate:.1%} ({round_result.success_count}/{len(round_result.task_results)})")
        logger.info(f"  平均 best_score: {round_result.avg_best_score:.2f}" if round_result.avg_best_score else "  平均 best_score: N/A")
        for tr in round_result.task_results:
            status = "✅ 通过" if tr.judge_accepted else "❌ 失败"
            logger.info(f"    第{tr.run_index}次: {status} | score={tr.best_score or 'N/A'} | duration={tr.duration_seconds:.1f}s")
            if tr.error_message:
                logger.info(f"      错误: {tr.error_message[:100]}")

    # 保存报告
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(evaluator.result_base_dir) / "benchmark_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding='utf-8')
    logger.info(f"\n评测报告已保存: {output_path}")


if __name__ == "__main__":
    main()

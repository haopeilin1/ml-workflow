"""
运行剩余4个任务的评测脚本
使用正确的 LLM 配置，与之前5个任务保持一致
"""

import json
import logging
import sys
from pathlib import Path

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
    benchmark_dir = Path("../test_data_remaining").resolve()
    num_runs = 1
    max_wait = 1800  # 30分钟
    eval_id = "eval_675d50863666_remaining"

    # 构建正确的 LLM 配置
    plan_cfg = LLMConfig(**build_eval_llm_config('plan'))
    coding_cfg = LLMConfig(**build_eval_llm_config('coding'))
    eval_cfg = LLMConfig(**build_eval_llm_config('evaluation'))
    judge_cfg = LLMConfig(**build_eval_llm_config('judge'))

    logger.info("=" * 60)
    logger.info("剩余4任务评测启动")
    logger.info(f"评测目录: {benchmark_dir}")
    logger.info(f"每个任务运行次数: {num_runs}")
    logger.info(f"Plan LLM: {plan_cfg.provider}/{plan_cfg.model}")
    logger.info(f"Coding LLM: {coding_cfg.provider}/{coding_cfg.model}")
    logger.info(f"Evaluation LLM: {eval_cfg.provider}/{eval_cfg.model}")
    logger.info(f"Judge LLM: {judge_cfg.provider}/{judge_cfg.model}")
    logger.info("=" * 60)

    evaluator = BenchmarkEvaluator(
        benchmark_dir=str(benchmark_dir),
        num_runs=num_runs,
        judge_llm_config=judge_cfg,
        plan_llm_config=plan_cfg,
        coding_llm_config=coding_cfg,
        evaluation_llm_config=eval_cfg,
        max_wait_seconds=max_wait,
        eval_id=eval_id
    )

    report = evaluator.run_benchmark()

    logger.info("=" * 60)
    logger.info("评测完成")
    logger.info(f"总任务数: {report.total_tasks}")
    logger.info(f"总运行次数: {report.total_runs}")
    logger.info(f"通过次数: {report.total_accepted}")
    logger.info(f"整体成功率: {report.overall_success_rate:.1%}")
    logger.info("=" * 60)

    for round_result in report.round_results:
        logger.info(f"\n任务: {round_result.task_results[0].task_name if round_result.task_results else 'unknown'}")
        logger.info(f"  成功率: {round_result.success_rate:.1%} ({round_result.success_count}/{len(round_result.task_results)})")
        logger.info(f"  平均 best_score: {round_result.avg_best_score:.2f}" if round_result.avg_best_score else "  平均 best_score: N/A")
        for tr in round_result.task_results:
            status = "✅ 通过" if tr.judge_accepted else "❌ 失败"
            logger.info(f"    第{tr.run_index}次: {status} | score={tr.best_score or 'N/A'} | duration={tr.duration_seconds:.1f}s")
            if tr.error_message:
                logger.info(f"      错误: {tr.error_message[:100]}")

    output_path = Path(evaluator.result_base_dir) / "benchmark_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding='utf-8')
    logger.info(f"\n评测报告已保存: {output_path}")


if __name__ == "__main__":
    main()

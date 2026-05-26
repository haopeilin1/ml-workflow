"""
Batch 2: 11个剩余任务的评测脚本
使用正确的 LLM 配置，与之前任务保持一致
"""

import json
import logging
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.evaluator import BenchmarkEvaluator
from app.models.schemas import LLMConfig
from app.config import build_eval_llm_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    benchmark_dir = Path("../test_data_batch2").resolve()
    num_runs = 1
    max_wait = 1800  # 30分钟每个任务
    eval_id = "eval_batch2_11tasks"

    plan_cfg = LLMConfig(**build_eval_llm_config('plan'))
    coding_cfg = LLMConfig(**build_eval_llm_config('coding'))
    eval_cfg = LLMConfig(**build_eval_llm_config('evaluation'))
    judge_cfg = LLMConfig(**build_eval_llm_config('judge'))

    logger.info("=" * 60)
    logger.info("Batch 2: 11任务评测启动")
    logger.info(f"评测目录: {benchmark_dir}")
    logger.info(f"每个任务运行次数: {num_runs}")
    logger.info(f"Plan LLM: {plan_cfg.model}")
    logger.info(f"Coding LLM: {coding_cfg.model}")
    logger.info(f"Evaluation LLM: {eval_cfg.model}")
    logger.info(f"Judge LLM: {judge_cfg.model}")
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
        trs = round_result.task_results
        task_name = trs[0].task_name if trs else 'unknown'
        logger.info(f"\n任务: {task_name}")
        logger.info(f"  成功率: {round_result.success_rate:.1%}")
        if round_result.avg_best_score:
            logger.info(f"  平均 best_score: {round_result.avg_best_score:.2f}")
        for tr in trs:
            status = "PASS" if tr.judge_accepted else "FAIL"
            logger.info(f"    Run {tr.run_index}: {status} | score={tr.best_score or 'N/A'} | duration={tr.duration_seconds:.1f}s")
            if tr.error_message:
                logger.info(f"      Error: {tr.error_message[:120]}")

    output_path = Path(evaluator.result_base_dir) / "benchmark_report.json"
    output_path.write_text(report.model_dump_json(indent=2), encoding='utf-8')
    logger.info(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()

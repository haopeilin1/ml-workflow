"""
验证吸烟状况任务修复的评测脚本
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
    benchmark_dir = Path("../test_data_smoking").resolve()
    eval_id = "eval_smoking_fix_test"

    plan_cfg = LLMConfig(**build_eval_llm_config('plan'))
    coding_cfg = LLMConfig(**build_eval_llm_config('coding'))
    eval_cfg = LLMConfig(**build_eval_llm_config('evaluation'))
    judge_cfg = LLMConfig(**build_eval_llm_config('judge'))

    logger.info("=" * 60)
    logger.info("吸烟状况修复验证评测")
    logger.info("=" * 60)

    evaluator = BenchmarkEvaluator(
        benchmark_dir=str(benchmark_dir),
        num_runs=1,
        judge_llm_config=judge_cfg,
        plan_llm_config=plan_cfg,
        coding_llm_config=coding_cfg,
        evaluation_llm_config=eval_cfg,
        max_wait_seconds=1800,
        eval_id=eval_id
    )

    report = evaluator.run_benchmark()

    logger.info("=" * 60)
    logger.info("评测完成")
    logger.info(f"总任务数: {report.total_tasks}")
    logger.info(f"通过次数: {report.total_accepted}")
    logger.info(f"整体成功率: {report.overall_success_rate:.1%}")
    logger.info("=" * 60)

    for round_result in report.round_results:
        task_name = round_result['task_results'][0]['task_name'] if round_result['task_results'] else 'unknown'
        logger.info(f"\n任务: {task_name}")
        tr = round_result['task_results'][0]
        status = "PASS" if tr['judge_accepted'] else "FAIL"
        logger.info(f"  结果: {status} | score={tr.get('best_score', 'N/A')} | duration={tr['duration_seconds']:.1f}s")
        if tr.get('error_message'):
            logger.info(f"  错误: {tr['error_message'][:120]}")

    output_path = Path(evaluator.result_base_dir) / "benchmark_report.json"
    output_path.write_text(report.model_dump_json(indent=2), encoding='utf-8')
    logger.info(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()

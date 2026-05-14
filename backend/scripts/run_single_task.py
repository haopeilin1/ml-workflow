"""
单任务快速评测脚本，支持 Plan/Coding 分离配置
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import logging
from app.core.evaluator import BenchmarkEvaluator
from app.models.schemas import LLMConfig
from app.config import build_eval_llm_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def build_llm_config(which: str) -> LLMConfig:
    cfg = build_eval_llm_config(which)
    return LLMConfig(**cfg)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True, help="任务目录（单任务模式）")
    parser.add_argument("--num-runs", type=int, default=1, help="运行次数")
    parser.add_argument("--max-wait", type=int, default=1800, help="单任务最大等待秒数")
    args = parser.parse_args()

    # 独立 LLM 配置
    plan_cfg = build_llm_config("plan")
    coding_cfg = build_llm_config("coding")
    unified_cfg = build_llm_config("unified")
    evaluation_cfg = build_llm_config("evaluation")
    judge_cfg = build_llm_config("judge")

    logger.info("=" * 60)
    logger.info("单任务快速评测")
    logger.info(f"任务目录: {args.benchmark_dir}")
    logger.info(f"Plan LLM:    {plan_cfg.provider}/{plan_cfg.model}")
    logger.info(f"Coding LLM:  {coding_cfg.provider}/{coding_cfg.model}")
    logger.info(f"Unified LLM: {unified_cfg.provider}/{unified_cfg.model}")
    logger.info(f"Eval LLM:    {evaluation_cfg.provider}/{evaluation_cfg.model}")
    logger.info(f"Judge LLM:   {judge_cfg.provider}/{judge_cfg.model}")
    logger.info("=" * 60)

    evaluator = BenchmarkEvaluator(
        benchmark_dir=args.benchmark_dir,
        num_runs=args.num_runs,
        max_wait_seconds=args.max_wait,
        plan_llm_config=plan_cfg,
        coding_llm_config=coding_cfg,
        unified_llm_config=unified_cfg,
        evaluation_llm_config=evaluation_cfg,
        judge_llm_config=judge_cfg,
    )

    report = evaluator.run_benchmark()

    logger.info("=" * 60)
    logger.info("评测完成")
    for rr in report.round_results:
        for tr in rr.task_results:
            status = "✅ 通过" if tr.judge_accepted else "❌ 失败"
            logger.info(f"  {tr.task_name} run_{tr.run_index}: {status} | score={tr.best_score or 'N/A'} | {tr.duration_seconds:.1f}s")
            if tr.error_message:
                logger.info(f"    错误: {tr.error_message}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

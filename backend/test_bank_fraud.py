"""
银行欺诈单任务端到端测试 — 验证所有修复是否奏效
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path

from app.core.evaluator import BenchmarkEvaluator
from app.models.schemas import LLMConfig
from app.config import build_eval_llm_config


def _get_llm_config(agent_type: str):
    cfg = build_eval_llm_config(agent_type)
    return {
        "provider": cfg["provider"],
        "base_url": cfg["base_url"],
        "api_key": cfg["api_key"],
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }


plan_llm_config = _get_llm_config("plan")
coding_llm_config = _get_llm_config("coding")
simple_llm_config = _get_llm_config("unified")

TASK_DIR = "/home/hpl/ml-workflow/test_data/银行账户欺诈"


def test_end_to_end():
    print("\n" + "=" * 70)
    print("银行欺诈端到端 Benchmark 测试")
    print("=" * 70)

    plan_llm = LLMConfig(**plan_llm_config)
    coding_llm = LLMConfig(**coding_llm_config)
    unified_llm = LLMConfig(**simple_llm_config)

    evaluator = BenchmarkEvaluator(
        benchmark_dir=TASK_DIR,
        num_runs=1,
        plan_llm_config=plan_llm,
        coding_llm_config=coding_llm,
        unified_llm_config=unified_llm,
        judge_llm_config=plan_llm,
        max_wait_seconds=1200,
        eval_id="test_bank_fraud_single"
    )

    print(f"\n启动 Benchmark，任务目录: {TASK_DIR}")
    print("这将运行完整的 FastEngine 流程（包括产物生成），预计需要 5-15 分钟...")
    print("=" * 70)

    tasks = evaluator._discover_tasks()
    if not tasks:
        print("未找到任务!")
        return

    task = tasks[0]
    print(f"发现任务: {task.task_name}")
    print(f"  train: {task.train_path}")
    print(f"  test: {task.test_path}")
    print(f"  gt: {task.ground_truth_path}")

    result = evaluator._run_single_task(task, run_index=1)

    print(f"\n{'='*70}")
    print("任务结果:")
    print(f"  success: {result.success}")
    print(f"  judge_accepted: {result.judge_accepted}")
    print(f"  best_score: {result.best_score}")
    print(f"  phase: {result.phase}")
    print(f"  duration: {result.duration_seconds:.1f}s")
    print(f"  error: {result.error_message or '无'}")
    print(f"  complexity: {result.complexity if hasattr(result, 'complexity') else 'N/A'}")

    if result.val_metrics:
        print(f"\n验证集指标:")
        m = result.val_metrics
        print(f"  val_auc: {m.val_auc}")
        print(f"  val_accuracy: {m.val_accuracy}")
        print(f"  train_score: {m.train_score}")
        print(f"  overfit_ratio: {m.overfit_ratio}")

    if result.test_metrics:
        print(f"\n测试集指标:")
        tm = result.test_metrics
        print(f"  test_auc: {tm.auc}")
        print(f"  test_accuracy: {tm.accuracy}")
        print(f"  test_f1: {tm.f1}")
        print(f"  test_f1_macro: {tm.f1_macro}")

    if result.artifacts:
        print(f"\n产物检测:")
        a = result.artifacts
        print(f"  completeness: {a.completeness}")
        print(f"  model_file: {a.model_file}")
        print(f"  prediction_file: {a.prediction_file}")
        print(f"  feature_importance_csv: {a.feature_importance_csv}")
        print(f"  feature_importance_png: {a.feature_importance_png}")
        print(f"  report_html: {a.report_html}")
        print(f"  predict_script: {a.predict_script}")
        print(f"  generated_files: {a.generated_files}")

    if result.logs:
        print(f"\n日志 ({len(result.logs)} 条):")
        for log in result.logs[-30:]:
            print(f"  {log[:180]}")

    # 保存完整结果
    output_dir = Path("/tmp/test_bank_fraud_output")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "benchmark_result.json").write_text(
        result.model_dump_json(indent=2), encoding='utf-8'
    )
    print(f"\n完整结果已保存到: {output_dir}/benchmark_result.json")

    return result


if __name__ == "__main__":
    test_end_to_end()

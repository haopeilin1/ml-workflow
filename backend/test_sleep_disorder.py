"""睡眠障碍预测 - 复杂多分类任务端到端测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from app.config import build_eval_llm_config
from app.models.schemas import TaskConfig, ExtractedSlots, UploadedFile, FileRole, TaskType, LLMConfig
from app.core.evaluator import BenchmarkEvaluator

TASK_DIR = "/home/hpl/ml-workflow/test_data/睡眠障碍预测"

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

print("=" * 70)
print("睡眠障碍预测 - 复杂多分类任务端到端测试")
print("=" * 70)
print(f"Plan LLM:    {plan_llm_config['model']}")
print(f"Coding LLM:  {coding_llm_config['model']}")
print(f"Unified LLM: {simple_llm_config['model']}")
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
    max_wait_seconds=1800,
    eval_id="test_sleep_disorder"
)

tasks = evaluator._discover_tasks()
if not tasks:
    print("未找到任务!")
    sys.exit(1)

task = tasks[0]
print(f"\n发现任务: {task.task_name}")
print(f"  train: {task.train_path}")
print(f"  test: {task.test_path}")
print(f"  gt: {task.ground_truth_path}")
print(f"  target: {task.target_column}")
print(f"  task_type: {task.task_type}")
print(f"  complexity: {task.complexity_reason}")

# 预跑意图识别
desc_path = Path(task.desc_path) if task.desc_path else None
desc = desc_path.read_text(encoding='utf-8').strip() if desc_path and desc_path.exists() else ""
intent = evaluator.intent_agent.recognize(
    task_description=desc,
    columns=task.data_profile.get("columns", []) if task.data_profile else [],
    row_count=task.data_profile.get("rowCount", 0) if task.data_profile else 0
)
evaluator._intent_cache[task.task_name] = intent
print(f"\n意图识别:")
print(f"  complexity={intent.complexity}, is_time_series={intent.is_time_series}")
print(f"  target={intent.target_column}, eval_metric={intent.eval_metric}")

result = evaluator._run_single_task(task, run_index=1)

print(f"\n{'='*70}")
print("任务结果:")
print(f"  success: {result.success}")
print(f"  judge_accepted: {result.judge_accepted}")
print(f"  best_score: {result.best_score}")
print(f"  phase: {result.phase}")
print(f"  duration: {result.duration_seconds:.1f}s")
print(f"  error: {result.error_message or '无'}")

if result.val_metrics:
    print(f"\n验证集指标:")
    m = result.val_metrics
    print(f"  val_accuracy: {m.val_accuracy}")
    print(f"  train_score: {m.train_score}")
    print(f"  overfit_ratio: {m.overfit_ratio}")

if result.test_metrics:
    print(f"\n测试集指标:")
    tm = result.test_metrics
    print(f"  test_accuracy: {tm.test_accuracy}")
    print(f"  test_f1: {tm.test_f1}")

if result.logs:
    print(f"\n日志 ({len(result.logs)} 条):")
    for log in result.logs[-30:]:
        print(f"  {log[:150]}")

print(f"\n{'='*70}")

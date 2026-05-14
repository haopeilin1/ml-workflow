"""肝硬化患者状态预测 - 复杂多分类任务端到端测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from app.config import build_eval_llm_config
from app.models.schemas import TaskConfig, ExtractedSlots, UploadedFile, FileRole, TaskType, LLMConfig
from app.core.evaluator import BenchmarkEvaluator

TASK_DIR = "/home/hpl/ml-workflow/test_data/肝硬化患者状态预测"

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

plan_cfg = _get_llm_config("plan")
coding_cfg = _get_llm_config("coding")
unified_cfg = _get_llm_config("unified")
eval_cfg = _get_llm_config("evaluation")

print("=" * 70)
print("肝硬化患者状态预测 - 复杂多分类任务端到端测试")
print("=" * 70)
print(f"Plan LLM:        {plan_cfg['model']} @ {plan_cfg['base_url']}")
print(f"Coding LLM:      {coding_cfg['model']} @ {coding_cfg['base_url']}")
print(f"Unified LLM:     {unified_cfg['model']} @ {unified_cfg['base_url']}")
print(f"Evaluation LLM:  {eval_cfg['model']} @ {eval_cfg['base_url']}")
print("=" * 70)

plan_llm = LLMConfig(**plan_cfg)
coding_llm = LLMConfig(**coding_cfg)
unified_llm = LLMConfig(**unified_cfg)
eval_llm = LLMConfig(**eval_cfg)

evaluator = BenchmarkEvaluator(
    benchmark_dir=TASK_DIR,
    num_runs=1,
    plan_llm_config=plan_llm,
    coding_llm_config=coding_llm,
    unified_llm_config=unified_llm,
    evaluation_llm_config=eval_llm,
    judge_llm_config=plan_llm,
    max_wait_seconds=1800,
    eval_id="test_liver_cirrhosis"
)

tasks = evaluator._discover_tasks()
if not tasks:
    print("未找到任务!")
    sys.exit(1)

task = tasks[0]
print(f"\n发现任务: {task.task_name}")
print(f"  train: {task.train_path}")
print(f"  test:  {task.test_path}")
print(f"  target: {task.target_column}")
print(f"  task_type: {task.task_type}")

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

print("\n[MAIN] 开始 _run_single_task...")
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

if result.logs:
    print(f"\n日志 ({len(result.logs)} 条):")
    for log in result.logs[-20:]:
        print(f"  {log[:200]}")

print(f"\n{'='*70}")

"""
单任务快速测试 — 验证 Plan/Coding 分离路由
只跑信用卡欺诈一个任务，观察 PlanAgent 和 CodingAgent 的输出
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path

from app.agents.intent_recognition import IntentRecognitionAgent, IntentResult
from app.agents.plan_agent import PlanAgent
from app.agents.coding_agent import CodingAgent
from app.agents.base import LLMClient
from app.models.schemas import TaskConfig, ExtractedSlots, UploadedFile, FileRole, TaskType
from app.core.evaluator import BenchmarkEvaluator

# 从 .env 读取 LLM 配置（优先使用独立 Agent 配置）
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

# 各 Agent 使用各自配置
plan_llm_config = _get_llm_config("plan")
coding_llm_config = _get_llm_config("coding")
simple_llm_config = _get_llm_config("unified")

TASK_DIR = "/home/hpl/ml-workflow/test_data/信用卡欺诈-二分类-类别极度不平衡"

def test_plan_agent_only():
    """只测试 PlanAgent 输出"""
    print("=" * 70)
    print("测试 1: PlanAgent 结构化计划生成")
    print("=" * 70)
    
    # 1. 构建数据画像
    from app.core.evaluator import BenchmarkEvaluator
    evaluator = BenchmarkEvaluator(
        benchmark_dir=TASK_DIR,
        num_runs=1,
        plan_coding_llm_config=None,
    )
    
    # 手动构建 data profile
    train_path = Path(TASK_DIR) / "用于建模" / "train.csv"
    gt_path = Path(TASK_DIR) / "用于评估" / "test_with_target.csv"
    profile = evaluator._build_data_profile(train_path, gt_path)
    
    # 2. 意图识别
    intent_agent = IntentRecognitionAgent()
    desc_path = Path(TASK_DIR) / "用于建模" / "任务描述-信用卡欺诈.txt"
    desc = desc_path.read_text(encoding='utf-8').strip() if desc_path.exists() else ""
    
    intent = intent_agent.recognize(
        task_description=desc,
        columns=profile.get("columns", []),
        row_count=profile.get("rowCount", 0)
    )
    
    print(f"\n意图识别结果:")
    print(f"  task_type: {intent.task_type}")
    print(f"  target_column: {intent.target_column}")
    print(f"  complexity: {intent.complexity}")
    print(f"  complexity_reason: {intent.complexity_reason}")
    print(f"  is_time_series: {intent.is_time_series}")
    print(f"  eval_metric: {intent.eval_metric}")
    
    # 3. 构建 TaskConfig
    tc = TaskConfig(
        extracted_slots=ExtractedSlots(
            target_column=intent.target_column or "IsFraud",
            task_type=intent.task_type or TaskType.BINARY_CLASSIFICATION,
            eval_metric=intent.eval_metric,
            complexity=intent.complexity,
            complexity_reason=intent.complexity_reason,
            is_time_series=intent.is_time_series,
            feature_constraints=[],
            user_modeling_suggestions=None,
        ),
        uploaded_files=[
            UploadedFile(name="train.csv", path=str(train_path), role=FileRole.TRAIN),
        ],
        user_description=desc,
        data_profile=profile,
    )
    
    # 4. 调用 PlanAgent
    llm = LLMClient(**plan_llm_config)
    plan_agent = PlanAgent(llm_client=llm)
    
    print(f"\n调用 PlanAgent 生成结构化计划... (model={plan_llm_config['model']})")
    plan_result = plan_agent.generate(tc)
    
    print(f"\n--- 结构化计划结果 ---")
    print(f"核心挑战 ({len(plan_result.core_challenges)}):")
    for c in plan_result.core_challenges:
        print(f"  - {c}")
    
    print(f"\nMUST DO ({len(plan_result.must_do)}):")
    for m in plan_result.must_do:
        marker = "[关键]" if m.critical else ""
        print(f"  - {marker} {m.item}")
        print(f"    原因: {m.reason}")
    
    print(f"\nAVOID ({len(plan_result.avoid)}):")
    for a in plan_result.avoid:
        print(f"  - {a.item}")
        print(f"    原因: {a.reason}")
    
    print(f"\nPipeline 计划 ({len(plan_result.pipeline_plan)}):")
    for s in plan_result.pipeline_plan:
        print(f"  ▶ {s.step}")
        for act in s.actions:
            print(f"    - {act}")
    
    print(f"\n模型选择: {plan_result.model_choice}")
    print(f"预期表现: {plan_result.expected_performance}")
    
    print(f"\n风险 ({len(plan_result.risks)}):")
    for r in plan_result.risks:
        print(f"  ⚠ {r}")
    
    # 保存完整 plan 文本
    output_dir = Path("/tmp/test_plan_output")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "plan_raw.txt").write_text(plan_result.raw_plan_text, encoding='utf-8')
    formatted = plan_agent.format_plan_for_coding(plan_result)
    (output_dir / "plan_formatted.txt").write_text(formatted, encoding='utf-8')
    print(f"\n计划已保存到: {output_dir}")
    
    return tc, plan_result, formatted


def test_coding_agent(tc, formatted_plan):
    """测试 CodingAgent 基于 plan 生成代码"""
    print("\n" + "=" * 70)
    print("测试 2: CodingAgent 代码生成")
    print("=" * 70)
    
    llm = LLMClient(**coding_llm_config)
    coding_agent = CodingAgent(llm_client=llm)
    
    print(f"\n调用 CodingAgent 生成代码... (model={coding_llm_config['model']})")
    code_output = coding_agent.generate(
        task_config=tc,
        structured_plan=formatted_plan,
        run_state="INIT",
        context_payload="",
        previous_code=""
    )
    
    # 检查 must_do 合规性
    print(f"\n--- 代码合规性检查 ---")
    code = code_output.code
    
    must_do_checks = [
        ("scale_pos_weight", "scale_pos_weight" in code),
        ("class_weight='balanced'", "class_weight='balanced'" not in code),
        ("prepare_for_prediction", "def prepare_for_prediction" in code),
        ("best_model.pkl", "best_model.pkl" in code),
        ("dill", "dill" in code),
    ]
    
    for item, found in must_do_checks:
        status = "✅" if found else "❌"
        print(f"  {status} {item}")
    
    # 保存代码
    output_dir = Path("/tmp/test_plan_output")
    (output_dir / "generated_code.py").write_text(code, encoding='utf-8')
    (output_dir / "coding_plan.txt").write_text(code_output.plan, encoding='utf-8')
    print(f"\n代码已保存到: {output_dir}/generated_code.py")
    print(f"代码长度: {len(code)} 字符")
    
    return code_output


def test_end_to_end():
    """端到端测试 — 使用 BenchmarkEvaluator 跑完整流程"""
    print("\n" + "=" * 70)
    print("测试 3: 端到端 Benchmark 测试")
    print("=" * 70)
    
    from app.models.schemas import LLMConfig
    
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
        eval_id="test_credit_fraud_single"
    )
    
    # 只跑这个任务
    print(f"\n启动 Benchmark，任务目录: {TASK_DIR}")
    print("这将运行完整的 FastEngine 流程（包括产物生成），预计需要 5-15 分钟...")
    print("=" * 70)
    
    # 手动执行单个任务
    tasks = evaluator._discover_tasks()
    if not tasks:
        print("未找到任务!")
        return
    
    task = tasks[0]
    print(f"发现任务: {task.task_name}")
    print(f"  train: {task.train_path}")
    print(f"  test: {task.test_path}")
    print(f"  gt: {task.ground_truth_path}")
    
    # 预跑意图识别（填充缓存）
    desc_path = Path(task.desc_path) if task.desc_path else None
    desc = desc_path.read_text(encoding='utf-8').strip() if desc_path and desc_path.exists() else ""
    intent = evaluator.intent_agent.recognize(
        task_description=desc,
        columns=task.data_profile.get("columns", []) if task.data_profile else [],
        row_count=task.data_profile.get("rowCount", 0) if task.data_profile else 0
    )
    evaluator._intent_cache[task.task_name] = intent
    print(f"\n意图识别: complexity={intent.complexity}, is_time_series={intent.is_time_series}")
    
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
        print(f"  val_auc: {m.val_auc}")
        print(f"  val_accuracy: {m.val_accuracy}")
        print(f"  train_score: {m.train_score}")
        print(f"  overfit_ratio: {m.overfit_ratio}")
    
    if result.test_metrics:
        print(f"\n测试集指标:")
        tm = result.test_metrics
        print(f"  test_auc: {tm.test_auc}")
        print(f"  test_accuracy: {tm.test_accuracy}")
        print(f"  test_f1: {tm.test_f1}")
    
    if result.logs:
        print(f"\n日志 ({len(result.logs)} 条):")
        for log in result.logs[-20:]:  # 最后20条
            print(f"  {log[:150]}")
    
    # 保存完整结果
    output_dir = Path("/tmp/test_plan_output")
    (output_dir / "benchmark_result.json").write_text(
        result.model_dump_json(indent=2), encoding='utf-8'
    )
    print(f"\n完整结果已保存到: {output_dir}/benchmark_result.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true", help="只测试 PlanAgent")
    parser.add_argument("--code-only", action="store_true", help="测试 PlanAgent + CodingAgent")
    parser.add_argument("--e2e", action="store_true", help="端到端完整测试")
    parser.add_argument("--all", action="store_true", help="全部测试")
    args = parser.parse_args()
    
    if args.all or (not args.plan_only and not args.code_only and not args.e2e):
        args.plan_only = True
        args.code_only = True
        args.e2e = True
    
    tc = None
    plan_result = None
    formatted_plan = None
    
    if args.plan_only:
        tc, plan_result, formatted_plan = test_plan_agent_only()
    
    if args.code_only:
        if tc is None:
            tc, plan_result, formatted_plan = test_plan_agent_only()
        test_coding_agent(tc, formatted_plan)
    
    if args.e2e:
        test_end_to_end()

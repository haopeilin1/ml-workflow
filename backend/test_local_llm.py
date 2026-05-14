#!/usr/bin/env python3
"""
本地 LLM 端到端测试脚本
记录从意图识别到最终产物的完整指标

配置：Plan/Evaluation/Intent → 本地 VLLM (qwen3.6-27b)
      Coding → DeepSeek 官方 (deepseek-v4-pro)
      Judge → 阿里云 (qwen3.5-flash)
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# ============ 配置本地 VLLM ============
os.environ["EVAL_PLAN_BASE_URL"] = "http://localhost:8000/v1"
os.environ["EVAL_PLAN_API_KEY"] = "not-needed"
os.environ["EVAL_PLAN_MODEL"] = "qwen3.6-27b"
os.environ["EVAL_PLAN_PROVIDER"] = "openai"
os.environ["EVAL_PLAN_EXTRA_BODY"] = "{}"

os.environ["EVAL_EVALUATION_BASE_URL"] = "http://localhost:8000/v1"
os.environ["EVAL_EVALUATION_API_KEY"] = "not-needed"
os.environ["EVAL_EVALUATION_MODEL"] = "qwen3.6-27b"
os.environ["EVAL_EVALUATION_PROVIDER"] = "openai"
os.environ["EVAL_EVALUATION_EXTRA_BODY"] = "{}"

os.environ["EVAL_INTENT_BASE_URL"] = "http://localhost:8000/v1"
os.environ["EVAL_INTENT_API_KEY"] = "not-needed"
os.environ["EVAL_INTENT_MODEL"] = "qwen3.6-27b"
os.environ["EVAL_INTENT_PROVIDER"] = "openai"
os.environ["EVAL_INTENT_EXTRA_BODY"] = "{}"

# 保持 Coding/Judge 不变（由 .env 控制）

# 导入必须在设置环境变量之后
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings, build_eval_llm_config
from app.core.evaluator import BenchmarkEvaluator
from app.models.schemas import LLMConfig
from app.agents.intent_recognition import IntentRecognitionAgent
from app.agents.base import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_llm_config_from_dict(cfg: dict) -> LLMConfig:
    """将 build_eval_llm_config 返回的字典转为 LLMConfig"""
    return LLMConfig(
        provider=cfg.get("provider", "openai"),
        base_url=cfg.get("base_url", ""),
        api_key=cfg.get("api_key", ""),
        model=cfg.get("model", ""),
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 4096),
        extra_body=cfg.get("extra_body")
    )


def measure_llm_latency(agent_name: str, system_prompt: str, user_prompt: str, llm_config: dict) -> dict:
    """测量单个 LLM 调用的延迟和首 token 时间"""
    client = LLMClient(
        provider=llm_config.get("provider", "openai"),
        base_url=llm_config.get("base_url", ""),
        api_key=llm_config.get("api_key", ""),
        model=llm_config.get("model", ""),
        temperature=llm_config.get("temperature", 0.3),
        max_tokens=llm_config.get("max_tokens", 4096),
        extra_body=llm_config.get("extra_body")
    )
    
    start = time.time()
    try:
        content, usage = client.chat_completion(system_prompt, user_prompt, max_retries=1)
        elapsed = time.time() - start
        return {
            "agent": agent_name,
            "latency_seconds": round(elapsed, 2),
            "success": True,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "model": usage.model,
            "provider": usage.provider,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "agent": agent_name,
            "latency_seconds": round(elapsed, 2),
            "success": False,
            "error": str(e)[:200]
        }


def run_benchmark_test(task_dir: str, num_runs: int = 1) -> dict:
    """运行端到端 benchmark 测试"""
    
    # 获取当前配置
    plan_cfg = build_eval_llm_config("plan")
    coding_cfg = build_eval_llm_config("coding")
    unified_cfg = build_eval_llm_config("unified")
    evaluation_cfg = build_eval_llm_config("evaluation")
    judge_cfg = build_eval_llm_config("judge")
    intent_cfg = build_eval_llm_config("intent")
    
    logger.info("=" * 60)
    logger.info("开始本地 LLM 端到端测试")
    logger.info("=" * 60)
    logger.info(f"Task: {task_dir}")
    logger.info(f"Plan:      {plan_cfg['model']} @ {plan_cfg['base_url']}")
    logger.info(f"Coding:    {coding_cfg['model']} @ {coding_cfg['base_url']}")
    logger.info(f"Eval:      {evaluation_cfg['model']} @ {evaluation_cfg['base_url']}")
    logger.info(f"Intent:    {intent_cfg['model']} @ {intent_cfg['base_url']}")
    logger.info(f"Judge:     {judge_cfg['model']} @ {judge_cfg['base_url']}")
    logger.info(f"num_runs:  {num_runs}")
    
    # 测量各 Agent 的连接延迟（轻量级 ping）
    logger.info("[Pre-test] 测量各 Agent LLM 连接延迟...")
    latency_results = []
    for name, cfg in [("plan", plan_cfg), ("coding", coding_cfg), 
                       ("evaluation", evaluation_cfg), ("intent", intent_cfg)]:
        r = measure_llm_latency(
            name, 
            "你是一个助手", 
            "回复一个单词 'ok'", 
            cfg
        )
        latency_results.append(r)
        logger.info(f"[Pre-test] {name}: latency={r.get('latency_seconds')}s, success={r.get('success')}")
    
    # 运行端到端 benchmark
    logger.info("[Benchmark] 启动端到端测试...")
    total_start = time.time()
    
    evaluator = BenchmarkEvaluator(
        benchmark_dir=task_dir,
        num_runs=num_runs,
        plan_llm_config=build_llm_config_from_dict(plan_cfg),
        coding_llm_config=build_llm_config_from_dict(coding_cfg),
        unified_llm_config=build_llm_config_from_dict(unified_cfg),
        evaluation_llm_config=build_llm_config_from_dict(evaluation_cfg),
        judge_llm_config=build_llm_config_from_dict(judge_cfg),
        max_wait_seconds=1800,  # 30分钟总超时
    )
    
    report = evaluator.run_benchmark()
    total_elapsed = time.time() - total_start
    
    logger.info(f"[Benchmark] 完成，总耗时: {total_elapsed:.1f}s")
    
    # 提取详细指标（含所有中间信息）
    results = []
    for round_result in report.round_results:
        for task_result in round_result.task_results:
            results.append({
                "task_name": task_result.task_name,
                "run_index": task_result.run_index,
                "success": task_result.success,
                "judge_accepted": task_result.judge_accepted,
                "best_score": task_result.best_score,
                "phase": task_result.phase,
                "duration_seconds": round(task_result.duration_seconds, 2),
                "error_message": task_result.error_message,
                "test_metrics": task_result.test_metrics.model_dump() if task_result.test_metrics else None,
                "artifacts": {
                    "completeness": getattr(task_result.artifacts, 'completeness', None) if task_result.artifacts else None,
                    "files": [f.name for f in task_result.artifacts.files] if task_result.artifacts and hasattr(task_result.artifacts, 'files') else []
                },
                "timing": task_result.timing.model_dump() if task_result.timing else None,
                "token_usage": task_result.token_usage.model_dump() if task_result.token_usage else None,
                "dimension_scores": task_result.dimension_scores,
                "judge_analysis": task_result.judge_analysis,
                "judge_reason": task_result.judge_reason,
                # 【新增】各环节实际使用的 LLM（含 fallback）
                "llm_usage_trace": task_result.llm_usage_trace,
            })
    
    # 构建产物总结
    artifact_summaries = []
    for r in results:
        artifacts = r.get("artifacts", {})
        artifact_summaries.append({
            "task_name": r["task_name"],
            "run_index": r["run_index"],
            "completeness": artifacts.get("completeness", "unknown"),
            "generated_files": artifacts.get("files", []),
        })
    
    # 汇总报告
    summary = {
        "test_timestamp": datetime.utcnow().isoformat(),
        "task_dir": task_dir,
        "num_runs": num_runs,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "overall_success_rate": report.overall_success_rate,
        "total_accepted": report.total_accepted,
        
        # LLM 配置
        "llm_configs": {
            "plan": {k: v for k, v in plan_cfg.items() if k != "api_key"},
            "coding": {k: v for k, v in coding_cfg.items() if k != "api_key"},
            "evaluation": {k: v for k, v in evaluation_cfg.items() if k != "api_key"},
            "intent": {k: v for k, v in intent_cfg.items() if k != "api_key"},
            "judge": {k: v for k, v in judge_cfg.items() if k != "api_key"},
        },
        
        # 预测试延迟
        "pre_test": {
            "llm_connection_latencies": latency_results,
        },
        
        # 详细结果
        "results": results,
        
        # 【新增】产物总结
        "artifact_summaries": artifact_summaries,
        
        # 全局统计
        "global": {
            "overall_avg_duration": report.overall_avg_duration_seconds,
            "overall_avg_tokens": report.overall_avg_total_tokens,
            "overall_score_std": report.overall_score_std,
            "overall_duration_std": report.overall_duration_std,
        }
    }
    
    # 保存报告
    report_path = Path("outputs") / f"local_llm_test_{report.eval_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[Done] 报告保存至: {report_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="test_data/睡眠障碍预测", help="测试任务目录")
    parser.add_argument("--runs", type=int, default=1, help="运行次数")
    args = parser.parse_args()
    
    summary = run_benchmark_test(args.task, args.runs)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("测试摘要")
    print("=" * 60)
    print(f"任务: {summary['task_dir']}")
    print(f"运行次数: {summary['num_runs']}")
    print(f"总耗时: {summary['total_elapsed_seconds']:.1f}s")
    print(f"成功率: {summary['overall_success_rate']:.1%}")
    print(f"Judge通过: {summary['total_accepted']}/{summary['num_runs']}")
    
    for r in summary["results"]:
        print(f"\n  Run #{r['run_index']}:")
        print(f"    success={r['success']}, judge={r['judge_accepted']}, score={r['best_score']}")
        print(f"    duration={r['duration_seconds']:.1f}s")
        if r['timing']:
            t = r['timing']
            print(f"    timing: code_gen={t.get('code_generation_seconds',0):.1f}s, sandbox={t.get('sandbox_execution_seconds',0):.1f}s, eval={t.get('evaluation_seconds',0):.1f}s, artifact={t.get('artifact_generation_seconds',0):.1f}s")
        if r['token_usage']:
            u = r['token_usage']
            print(f"    tokens: total={u.get('total_tokens',0)}, plan_coding={u.get('plan_coding_total_tokens',0)}, eval={u.get('evaluation_total_tokens',0)}")
        if r['test_metrics']:
            print(f"    test_metrics: {r['test_metrics']}")
        # 产物总结
        if r.get('artifacts'):
            arts = r['artifacts']
            print(f"    artifacts: completeness={arts.get('completeness', 'unknown')}, files={arts.get('files', [])}")
        # 【新增】各环节实际使用的 LLM
        if r.get('llm_usage_trace'):
            print(f"    LLM使用追踪:")
            for agent, info in r['llm_usage_trace'].items():
                if isinstance(info, dict) and 'primary_model' in info:
                    print(f"      {agent}: model={info['primary_model']}, provider={info['primary_provider']}, calls={info['total_calls']}, tokens={info['total_tokens']}")
                    # 如果有 fallback 使用情况，显示明细
                    if info.get('model_breakdown') and len(info['model_breakdown']) > 1:
                        for md in info['model_breakdown']:
                            print(f"        - {md['provider']}/{md['model']}: {md['calls']}次, {md['tokens']}tokens")
        if r['error_message']:
            print(f"    ERROR: {r['error_message'][:100]}")
    
    # 产物总结报告
    print("\n" + "=" * 60)
    print("产物生成总结")
    print("=" * 60)
    for art_sum in summary.get("artifact_summaries", []):
        print(f"\n  {art_sum['task_name']} Run #{art_sum['run_index']}:")
        print(f"    completeness: {art_sum['completeness']}")
        if art_sum['generated_files']:
            print(f"    generated files: {', '.join(art_sum['generated_files'])}")
        else:
            print(f"    generated files: (none)")
    
    # 【新增】LLM 配置与 Fallback 配置
    print("\n" + "=" * 60)
    print("LLM 配置与 Fallback 策略")
    print("=" * 60)
    for name, cfg in summary["llm_configs"].items():
        print(f"  {name}: {cfg['model']} @ {cfg['base_url']}")
    print(f"\n  Fallback策略:")
    print(f"    connect_timeout={settings.EXTERNAL_API_CONNECT_TIMEOUT}s")
    print(f"    read_timeout={settings.EXTERNAL_API_READ_TIMEOUT}s")
    print(f"    fast_retries={settings.EXTERNAL_API_FAST_RETRIES}")
    print(f"    fallback1: {settings.FALLBACK_LLM_MODEL or '未配置'} @ {settings.FALLBACK_LLM_BASE_URL or 'N/A'}")
    print(f"    fallback2: {settings.FALLBACK_LLM2_MODEL or '未配置'} @ {settings.FALLBACK_LLM2_BASE_URL or 'N/A'}")
    print(f"    cycle_enabled={settings.FALLBACK_CYCLE_ENABLED}")
    
    print(f"\n预测试延迟:")
    for l in summary["pre_test"]["llm_connection_latencies"]:
        print(f"  {l['agent']}: {l.get('latency_seconds')}s (success={l.get('success')})")

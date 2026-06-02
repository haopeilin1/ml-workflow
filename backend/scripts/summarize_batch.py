"""
批次测评汇总脚本
用法: python -m scripts.summarize_batch <eval_id> <output_base_dir>
"""
import json
import sys
from pathlib import Path
from datetime import datetime


def main():
    # Windows GBK 编码修复
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    if len(sys.argv) < 3:
        print("Usage: python -m scripts.summarize_batch <eval_id> <output_base_dir>")
        sys.exit(1)

    eval_id = sys.argv[1]
    output_base = Path(sys.argv[2])

    if not output_base.exists():
        print(f"目录不存在: {output_base}")
        sys.exit(1)

    results = []
    for task_dir in sorted(output_base.iterdir()):
        if not task_dir.is_dir():
            continue
        result_file = task_dir / "run_1" / "task_result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            results.append({
                "task_name": data.get("task_name", task_dir.name),
                "success": data.get("success", False),
                "judge_accepted": data.get("judge_accepted", False),
                "best_score": data.get("best_score"),
                "duration_seconds": data.get("duration_seconds", 0),
                "artifacts_completeness": data.get("artifacts", {}).get("completeness", "none"),
                "error_message": data.get("error_message"),
                "prediction_strategy": data.get("prediction_strategy"),
                "test_metrics": data.get("test_metrics", {}),
            })
        except Exception as e:
            print(f"读取 {result_file} 失败: {e}")

    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    judge_count = sum(1 for r in results if r["judge_accepted"])
    full_artifact_count = sum(1 for r in results if r["artifacts_completeness"] == "full")
    partial_artifact_count = sum(1 for r in results if r["artifacts_completeness"] == "partial")
    simplified_artifact_count = sum(1 for r in results if r["artifacts_completeness"] == "simplified")

    summary = {
        "eval_id": eval_id,
        "timestamp": datetime.now().isoformat(),
        "total_tasks": total,
        "success_count": success_count,
        "success_rate": success_count / total if total > 0 else 0,
        "judge_count": judge_count,
        "judge_rate": judge_count / total if total > 0 else 0,
        "full_artifact_count": full_artifact_count,
        "partial_artifact_count": partial_artifact_count,
        "simplified_artifact_count": simplified_artifact_count,
        "details": results,
    }

    # 保存 JSON 汇总
    summary_file = output_base / "batch_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存 CSV 明细
    csv_file = output_base / "batch_results.csv"
    lines = ["task_name,success,judge_accepted,best_score,duration_seconds,artifacts_completeness"]
    for r in results:
        lines.append(
            f"{r['task_name']},{r['success']},{r['judge_accepted']},"
            f"{r['best_score'] if r['best_score'] is not None else ''},"
            f"{r['duration_seconds']:.1f},{r['artifacts_completeness']}"
        )
    csv_file.write_text("\n".join(lines), encoding="utf-8")

    # 打印汇总
    print("\n" + "=" * 60)
    print(f"批次汇总报告: {eval_id}")
    print("=" * 60)
    print(f"总任务数: {total}")
    print(f"跑通数: {success_count} ({summary['success_rate']:.1%})")
    print(f"Judge 通过数: {judge_count} ({summary['judge_rate']:.1%})")
    print(f"产物完整度: full={full_artifact_count}, partial={partial_artifact_count}, simplified={simplified_artifact_count}")
    print("-" * 60)
    for r in results:
        status = "✅" if r["success"] else "❌"
        judge = "✅" if r["judge_accepted"] else "❌"
        print(f"{status} {r['task_name']} | 跑通={status} Judge={judge} score={r['best_score'] or 'N/A'} duration={r['duration_seconds']:.0f}s")
    print("=" * 60)
    print(f"汇总文件: {summary_file}")
    print(f"CSV 文件: {csv_file}")


if __name__ == "__main__":
    main()

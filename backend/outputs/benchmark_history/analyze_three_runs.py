import json, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open("run2_data.json", "r", encoding='utf-8') as f:
    run2 = json.load(f)
with open("run3_data.json", "r", encoding='utf-8') as f:
    run3 = json.load(f)

run1 = {
    "2026年股票抵押融资违约预测": {"success": True, "judge": False, "score": 75.5},
    "信用卡欺诈-二分类-类别极度不平衡": {"success": True, "judge": True, "score": 45.5},
    "共享单车租赁量预测-时序回归": {"success": True, "judge": False, "score": 88.0},
    "加利福尼亚房价预测-回归": {"success": False, "judge": False, "score": None},
    "北京PM2.5浓度预测-时序回归": {"success": True, "judge": True, "score": 66.8},
    "电商顾客退货预测": {"success": True, "judge": False, "score": 56.0},
    "睡眠障碍预测": {"success": True, "judge": False, "score": 41.5},
    "红酒品质预测-有序多分类": {"success": False, "judge": False, "score": None},
    "银行账户欺诈": {"success": True, "judge": True, "score": 40.5},
    "医疗保险费用预测": {"success": True, "judge": True, "score": 72.5},
    "吸烟状况": {"success": True, "judge": True, "score": 82.5},
    "垃圾邮件判别-二分类-高维稀疏": {"success": True, "judge": True, "score": 92.5},
    "成人收入预测-二分类含缺失值": {"success": False, "judge": False, "score": None},
    "支付欺诈": {"success": True, "judge": True, "score": 85.5},
    "机翼噪声预测": {"success": True, "judge": True, "score": 82.0},
    "电子商务客户流失预测": {"success": False, "judge": False, "score": None},
    "糖尿病预测": {"success": True, "judge": True, "score": 78.7},
    "肝硬化患者状态预测": {"success": True, "judge": True, "score": 15.0},
    "鸢尾花种类识别-极小样本多分类": {"success": True, "judge": True, "score": 89.0},
    "黑色素瘤种类": {"success": True, "judge": True, "score": 74.5},
}

task_type_map = {
    "2026年股票抵押融资违约预测": "二分类",
    "信用卡欺诈-二分类-类别极度不平衡": "二分类(不平衡)",
    "共享单车租赁量预测-时序回归": "时序回归",
    "加利福尼亚房价预测-回归": "回归",
    "北京PM2.5浓度预测-时序回归": "时序回归",
    "电商顾客退货预测": "二分类",
    "睡眠障碍预测": "多分类",
    "红酒品质预测-有序多分类": "有序多分类",
    "银行账户欺诈": "二分类",
    "医疗保险费用预测": "回归",
    "吸烟状况": "多分类",
    "垃圾邮件判别-二分类-高维稀疏": "二分类(高维)",
    "成人收入预测-二分类含缺失值": "二分类(缺失)",
    "支付欺诈": "二分类",
    "机翼噪声预测": "回归",
    "电子商务客户流失预测": "二分类",
    "糖尿病预测": "二分类",
    "肝硬化患者状态预测": "多分类",
    "鸢尾花种类识别-极小样本多分类": "多分类(小样本)",
    "黑色素瘤种类": "多分类",
}

all_tasks = sorted(set(run1.keys()) | set(run2.keys()) | set(run3.keys()))

def get_score(r, task):
    if task not in r: return None
    d = r[task]
    return d.get("score") or d.get("best_score")

def get_success(r, task):
    if task not in r: return False
    return r[task].get("success", False)

def get_judge(r, task):
    if task not in r: return False
    d = r[task]
    return d.get("judge") or d.get("judge_accepted", False)

def get_artifact(r, task):
    if task not in r: return None
    return r[task].get("artifact_completeness")

def get_strategy(r, task):
    if task not in r: return None
    return r[task].get("prediction_strategy")

def get_error(r, task):
    if task not in r: return None
    return r[task].get("error", "")

out = []
out.append("=" * 100)
out.append("一、核心指标总览")
out.append("=" * 100)

for label, dataset in [("Run1 (第一次)", run1), ("Run2 (第二次)", run2), ("Run3 (第三次)", run3)]:
    succ = sum(1 for t in all_tasks if get_success(dataset, t))
    judge = sum(1 for t in all_tasks if get_judge(dataset, t))
    scores = [get_score(dataset, t) for t in all_tasks if get_score(dataset, t) is not None]
    avg_score = sum(scores)/len(scores) if scores else 0
    median_score = sorted(scores)[len(scores)//2] if scores else 0
    artifacts = [get_artifact(dataset, t) for t in all_tasks if get_artifact(dataset, t)]
    full_count = sum(1 for a in artifacts if a == "full")
    simplified_count = sum(1 for a in artifacts if a == "simplified")
    out.append(f"{label}: 成功={succ}/20 ({succ*5}%) | Judge={judge}/20 ({judge*5}%) | 平均Score={avg_score:.1f} | 中位数={median_score:.1f} | 产物完整: full={full_count}, simplified={simplified_count}")

out.append("")
out.append("=" * 100)
out.append("二、按任务类型分析（Run3 数据）")
out.append("=" * 100)

from collections import defaultdict
type_stats = defaultdict(lambda: {"succ": 0, "judge": 0, "count": 0, "scores": []})
for task in all_tasks:
    tt = task_type_map.get(task, "未知")
    type_stats[tt]["count"] += 1
    if get_success(run3, task): type_stats[tt]["succ"] += 1
    if get_judge(run3, task): type_stats[tt]["judge"] += 1
    sc = get_score(run3, task)
    if sc is not None: type_stats[tt]["scores"].append(sc)

out.append(f"{'任务类型':<20} {'任务数':>6} {'成功数':>6} {'Judge数':>6} {'平均Score':>10}")
out.append("-" * 60)
for tt in sorted(type_stats.keys()):
    s = type_stats[tt]
    avg = sum(s["scores"])/len(s["scores"]) if s["scores"] else 0
    out.append(f"{tt:<20} {s['count']:>6} {s['succ']:>6} {s['judge']:>6} {avg:>10.1f}")

out.append("")
out.append("=" * 100)
out.append("三、稳定性分析（三次运行对比）")
out.append("=" * 100)

stable_pass = []
stable_succ = []
unstable = []
always_fail = []
judge_regress = []
judge_improve = []

for task in all_tasks:
    s1, s2, s3 = get_success(run1, task), get_success(run2, task), get_success(run3, task)
    j1, j2, j3 = get_judge(run1, task), get_judge(run2, task), get_judge(run3, task)

    if s1 and s2 and s3:
        stable_succ.append(task)
        if j1 and j2 and j3:
            stable_pass.append(task)
    if not s1 and not s2 and not s3:
        always_fail.append(task)
    if (s1 + s2 + s3) > 0 and (s1 + s2 + s3) < 3:
        unstable.append(task)

    if (j1 or j2) and not j3:
        judge_regress.append(task)
    if not j1 and j3:
        judge_improve.append(task)

out.append(f"[稳定通过: 三次均成功+Judge通过] ({len(stable_pass)}个)")
for t in stable_pass:
    sc1 = get_score(run1, t) or 0; sc2 = get_score(run2, t) or 0; sc3 = get_score(run3, t) or 0
    out.append(f"  {t}: Score {sc1:.1f} -> {sc2:.1f} -> {sc3:.1f}")

out.append(f"\n[稳定成功: 三次均成功（Judge不一定）] ({len(stable_succ)}个)")
out.append(f"  {', '.join(stable_succ)}")

out.append(f"\n[始终失败: 三次均失败] ({len(always_fail)}个)")
out.append(f"  {', '.join(always_fail) if always_fail else '无'}")

out.append(f"\n[不稳定: 时而成时而不成] ({len(unstable)}个)")
out.append(f"  {', '.join(unstable)}")

out.append(f"\n[Judge退化: 之前通过，Run3不通过] ({len(judge_regress)}个)")
out.append(f"  {', '.join(judge_regress)}")

out.append(f"\n[Judge提升: 之前不通过，Run3通过] ({len(judge_improve)}个)")
out.append(f"  {', '.join(judge_improve)}")

out.append("")
out.append("=" * 100)
out.append("四、失败模式分析（Run3 失败任务）")
out.append("=" * 100)

failures = [t for t in all_tasks if not get_success(run3, t)]
for task in failures:
    err = get_error(run3, task) or "未知"
    err_short = err[:120].replace('\n', ' ') if err else "无错误信息"
    out.append(f"  {task}")
    out.append(f"    错误: {err_short}")

out.append("")
out.append("=" * 100)
out.append("五、预测策略与产物完整度分析（Run3）")
out.append("=" * 100)

strategies = defaultdict(int)
artifacts = defaultdict(int)
for task in all_tasks:
    if get_success(run3, task):
        st = get_strategy(run3, task) or "unknown"
        strategies[st] += 1
        art = get_artifact(run3, task) or "unknown"
        artifacts[art] += 1

out.append(f"预测策略分布: {dict(strategies)}")
out.append(f"产物完整度分布: {dict(artifacts)}")

out.append("")
out.append("=" * 100)
out.append("六、关键发现与建议")
out.append("=" * 100)

out.append("1. 成功率波动: Run2 最高(95%)，Run1和Run3持平(80%)。Run3 相比 Run2 退步明显（-15%）")
out.append("2. Judge通过率: Run1(60%) -> Run2(65%) -> Run3(55%)，呈下降趋势。Run3 Judge 表现最差。")
out.append("3. 平均Score: 持续上升 Run1(67.9) -> Run2(70.4) -> Run3(73.3)，说明成功任务的得分在提高。")
out.append("4. 始终失败: 无任务三次全失败，但 北京PM2.5 在 Run2/Run3 连续失败（Run1成功）。")
out.append("5. 不稳定任务: 加利福尼亚房价、电商顾客退货、成人收入预测、电子商务客户流失、红酒品质、银行账户欺诈 —— 时好时坏。")
out.append("6. Judge退化最明显: 支付欺诈(Run1/Run2通过，Run3不通过)、信用卡欺诈(Run1通过，Run3不通过)、银行账户欺诈(Run1/Run2通过，Run3失败)。")
out.append("7. 肝硬化患者状态: Run1 score=15.0 但 Judge=True（阈值过松），Run2/Run3 score=72.5 且 Judge=True，表现大幅改善。")
out.append("8. 预测策略: Run3 成功任务几乎全部使用 embedded_best，说明系统严重依赖快照模型而非产物代码质量。")

report = "\n".join(out)
with open("three_runs_analysis.txt", "w", encoding='utf-8') as f:
    f.write(report)
print(report)

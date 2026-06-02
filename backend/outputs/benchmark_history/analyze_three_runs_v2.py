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

def get_completeness(r, task):
    if task not in r: return None
    art = r[task].get("artifacts") or {}
    return art.get("completeness")

def get_strategy(r, task):
    if task not in r: return None
    return r[task].get("prediction_strategy")

def get_error_msg(r, task):
    if task not in r: return None
    return r[task].get("error_message", "") or ""

def get_duration(r, task):
    if task not in r: return None
    return r[task].get("duration_seconds")

def get_test_metrics(r, task):
    if task not in r: return {}
    return r[task].get("test_metrics", {}) or {}

def get_val_metrics(r, task):
    if task not in r: return {}
    return r[task].get("val_metrics", {}) or {}

out = []
out.append("=" * 110)
out.append("一、核心指标总览")
out.append("=" * 110)

for label, dataset in [("Run1 (第一次)", run1), ("Run2 (第二次)", run2), ("Run3 (第三次)", run3)]:
    succ = sum(1 for t in all_tasks if get_success(dataset, t))
    judge = sum(1 for t in all_tasks if get_judge(dataset, t))
    scores = [get_score(dataset, t) for t in all_tasks if get_score(dataset, t) is not None]
    avg_score = sum(scores)/len(scores) if scores else 0
    median_score = sorted(scores)[len(scores)//2] if scores else 0
    full_count = sum(1 for t in all_tasks if get_completeness(dataset, t) == "full")
    simplified_count = sum(1 for t in all_tasks if get_completeness(dataset, t) == "simplified")
    partial_count = sum(1 for t in all_tasks if get_completeness(dataset, t) == "partial")
    out.append(f"{label}: 成功={succ}/20 ({succ*5}%) | Judge={judge}/20 ({judge*5}%) | 平均Score={avg_score:.1f} | 中位数={median_score:.1f} | 产物完整: full={full_count}, simplified={simplified_count}, partial={partial_count}")

out.append("")
out.append("=" * 110)
out.append("二、按任务类型分析（三次对比）")
out.append("=" * 110)

from collections import defaultdict

type_tasks = defaultdict(list)
for task in all_tasks:
    tt = task_type_map.get(task, "未知")
    type_tasks[tt].append(task)

out.append(f"{'任务类型':<18} {'任务数':>4} {'Run1成功':>8} {'Run2成功':>8} {'Run3成功':>8} {'Run1Judge':>9} {'Run2Judge':>9} {'Run3Judge':>9} {'Run3均分':>8}")
out.append("-" * 90)
for tt in sorted(type_tasks.keys()):
    tasks = type_tasks[tt]
    cnt = len(tasks)
    r1s = sum(1 for t in tasks if get_success(run1, t))
    r2s = sum(1 for t in tasks if get_success(run2, t))
    r3s = sum(1 for t in tasks if get_success(run3, t))
    r1j = sum(1 for t in tasks if get_judge(run1, t))
    r2j = sum(1 for t in tasks if get_judge(run2, t))
    r3j = sum(1 for t in tasks if get_judge(run3, t))
    scores = [get_score(run3, t) for t in tasks if get_score(run3, t) is not None]
    avg = sum(scores)/len(scores) if scores else 0
    out.append(f"{tt:<18} {cnt:>4} {r1s:>8} {r2s:>8} {r3s:>8} {r1j:>9} {r2j:>9} {r3j:>9} {avg:>8.1f}")

out.append("")
out.append("=" * 110)
out.append("三、稳定性分析")
out.append("=" * 110)

stable_pass = []
stable_succ = []
unstable = []
always_fail = []
judge_regress = []
judge_improve = []
score_volatile = []

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

    scores = [get_score(run1, task), get_score(run2, task), get_score(run3, task)]
    scores_valid = [s for s in scores if s is not None]
    if len(scores_valid) >= 2 and (max(scores_valid) - min(scores_valid)) > 20:
        score_volatile.append((task, scores_valid))

out.append(f"[稳定通过: 三次均成功+Judge通过] ({len(stable_pass)}个)")
for t in stable_pass:
    sc1 = get_score(run1, t) or 0; sc2 = get_score(run2, t) or 0; sc3 = get_score(run3, t) or 0
    out.append(f"  {t}: {sc1:.1f} -> {sc2:.1f} -> {sc3:.1f}")

out.append(f"\n[稳定成功: 三次均成功] ({len(stable_succ)}个)")
out.append(f"  {', '.join(stable_succ)}")

out.append(f"\n[始终失败: 三次均失败] ({len(always_fail)}个)")
out.append(f"  {', '.join(always_fail) if always_fail else '无'}")

out.append(f"\n[不稳定: 时而成时而不成] ({len(unstable)}个)")
for t in unstable:
    s1 = "Y" if get_success(run1, t) else "N"
    s2 = "Y" if get_success(run2, t) else "N"
    s3 = "Y" if get_success(run3, t) else "N"
    out.append(f"  {t}: Run1={s1} Run2={s2} Run3={s3}")

out.append(f"\n[Judge退化: Run3相比之前不通过] ({len(judge_regress)}个)")
for t in judge_regress:
    j1 = "Y" if get_judge(run1, t) else "N"
    j2 = "Y" if get_judge(run2, t) else "N"
    j3 = "Y" if get_judge(run3, t) else "N"
    out.append(f"  {t}: Run1={j1} Run2={j2} Run3={j3}")

out.append(f"\n[Judge提升: Run3相比之前通过] ({len(judge_improve)}个)")
for t in judge_improve:
    j1 = "Y" if get_judge(run1, t) else "N"
    j2 = "Y" if get_judge(run2, t) else "N"
    j3 = "Y" if get_judge(run3, t) else "N"
    out.append(f"  {t}: Run1={j1} Run2={j2} Run3={j3}")

out.append(f"\n[Score波动大 (>20分)] ({len(score_volatile)}个)")
for t, scores in score_volatile:
    out.append(f"  {t}: min={min(scores):.1f} max={max(scores):.1f} ({scores})")

out.append("")
out.append("=" * 110)
out.append("四、失败模式分析（Run3 失败任务详情）")
out.append("=" * 110)

failures = [t for t in all_tasks if not get_success(run3, t)]
for task in failures:
    err = get_error_msg(run3, task) or "无错误信息"
    err_short = err[:150].replace('\n', ' ')
    strategy = get_strategy(run3, task) or "N/A"
    phase = run3.get(task, {}).get("phase", "N/A")
    out.append(f"  {task}")
    out.append(f"    失败阶段: {phase}")
    out.append(f"    错误摘要: {err_short}")
    out.append(f"    预测策略: {strategy}")

out.append("")
out.append("=" * 110)
out.append("五、产物完整度与预测策略（Run3）")
out.append("=" * 110)

strategies = defaultdict(int)
completeness = defaultdict(int)
for task in all_tasks:
    if get_success(run3, task):
        st = get_strategy(run3, task) or "unknown"
        strategies[st] += 1
        comp = get_completeness(run3, task) or "unknown"
        completeness[comp] += 1

out.append(f"预测策略分布: {dict(strategies)}")
out.append(f"产物完整度分布: {dict(completeness)}")

strategies2 = defaultdict(int)
completeness2 = defaultdict(int)
for task in all_tasks:
    if get_success(run2, task):
        st = get_strategy(run2, task) or "unknown"
        strategies2[st] += 1
        comp = get_completeness(run2, task) or "unknown"
        completeness2[comp] += 1

out.append(f"Run2 预测策略分布: {dict(strategies2)}")
out.append(f"Run2 产物完整度分布: {dict(completeness2)}")

out.append("")
out.append("=" * 110)
out.append("六、Score 变化详细分析")
out.append("=" * 110)

out.append("[Run3 vs Run2 Score 提升明显的任务]")
for task in all_tasks:
    s2 = get_score(run2, task)
    s3 = get_score(run3, task)
    if s2 is not None and s3 is not None and s3 - s2 > 10:
        out.append(f"  {task}: {s2:.1f} -> {s3:.1f} (+{s3-s2:.1f})")

out.append("\n[Run3 vs Run2 Score 下降明显的任务]")
for task in all_tasks:
    s2 = get_score(run2, task)
    s3 = get_score(run3, task)
    if s2 is not None and s3 is not None and s2 - s3 > 10:
        out.append(f"  {task}: {s2:.1f} -> {s3:.1f} ({s3-s2:.1f})")

out.append("")
out.append("=" * 110)
out.append("七、关键发现与建议")
out.append("=" * 110)

out.append("1. 成功率波动: Run2 最高(95%)，Run1和Run3持平(80%)。Run3 相比 Run2 退步明显（-15%）")
out.append("2. Judge通过率: Run1(60%) -> Run2(65%) -> Run3(55%)，呈下降趋势。Run3 Judge 表现最差。")
out.append("3. 平均Score: 持续上升 Run1(67.9) -> Run2(70.4) -> Run3(73.3)，成功任务的质量在提高。")
out.append("4. 始终失败: 无任务三次全失败，但 北京PM2.5 在 Run2/Run3 连续失败（Run1成功），属于明显退化。")
out.append("5. 不稳定任务(7个): 加利福尼亚房价、北京PM2.5、成人收入预测、电商顾客退货、电子商务客户流失、红酒品质、银行账户欺诈。")
out.append("6. Judge退化(5个): 信用卡欺诈、北京PM2.5、成人收入预测、支付欺诈、银行账户欺诈 —— Run3 Judge 标准变严格或模型变差。")
out.append("7. Judge提升(3个): 股票违约、加州房价、电商客户流失 —— 从Run1不通过到Run3通过。")
out.append("8. Score波动大: 肝硬化患者状态(15->72.5) 改善最明显；共享单车(88->39.5->88.5) 波动剧烈。")
out.append("9. 肝硬化患者状态: Run1 score=15.0 但 Judge=True（阈值过松，已知Issue），Run2/Run3 score=72.5 表现正常。")
out.append("10. 预测策略: Run3 成功任务全部使用 embedded_best，系统严重依赖快照模型而非产物代码质量。")
out.append("11. 产物完整度: Run3 中 full=7, simplified=3（从batch summary得知），说明产物代码生成仍有问题。")

report = "\n".join(out)
with open("three_runs_analysis.txt", "w", encoding='utf-8') as f:
    f.write(report)
print(report)

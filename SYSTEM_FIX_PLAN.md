# ML-Workflow 系统级修复方案

> 基于 20-task benchmark 结果（跑通率 80%，Judge 通过率 50%）
> 本方案只分析问题根因和修复思路，**暂不执行代码修改**。

---

## 一、版本状态

- **本地 Commit**: `5452453`
- **Tag**: `v0.9.0-benchmark-20tasks`
- **GitHub Push**: ⚠️ 当前网络不通，commit 和 tag 已保存在本地，网络恢复后可执行 `git push origin main && git push origin --tags`

---

## 二、问题分类与修复方案

### Issue 1：Feature Engineering 序列化 Gap（鸢尾花、红酒品质 Judge 0 分）

**现象**：
- 训练时 `feature_engineering()` 在 `model.fit()` 外部执行，验证集表现极好（鸢尾花 val_accuracy=0.9444）
- 保存时 `dill.dump(model, f)` 只保存了 model 对象，feature engineering 步骤丢失
- Judge 直接加载 `best_model.pkl` 对原始数据预测，缺少额外特征 → Accuracy=0.0000

**根因**：系统架构层面**没有强制要求 feature engineering 嵌入到可序列化的 Pipeline 中**。

**修复方案**：
1. **强制 Pipeline 化**：修改 `code_best.py` 模板，要求 LLM 生成的 `build_model()` 必须返回 `sklearn.pipeline.Pipeline`，且第一步必须是 `FunctionTransformer(feature_engineering)` 或自定义 Transformer
2. **序列化完整性检查**：在 sandbox 执行成功后，系统主动调用 `model.predict(X_test_raw)` 验证——如果传入原始数据（未经过 feature engineering）能跑出结果，说明 feature engineering 已被嵌入 Pipeline；否则触发 REPLAN
3. **Predict.py 与训练代码绑定**：不再让 PlanCodingAgent 独立生成 `predict.py`，而是**从 `code_best.py` 自动提取** `feature_engineering`、`preprocess`、`build_model` 等函数，组装成 `predict.py`。确保 predict 阶段的特征工程与训练阶段 100% 一致
4. **模型格式统一**：保存时统一包装为 `{'pipeline': model, 'feature_engineering_src': source_code, 'preprocess_src': source_code}`，predict.py 严格按此格式加载

---

### Issue 2：目标变量变换在测试预测阶段丢失（共享单车 RMSE 暴涨）

**现象**：
- LLM 在 `preprocess()` 中对 `cnt` 做了 `log1p`
- `evaluate_model()` 中正确做了 `expm1` 还原 → 验证集 RMSE=59
- 但 `code_best.py` **末尾的测试预测兜底代码**直接保存 `model.predict()` 结果，未做 `expm1` → 测试集预测值 5.6 vs ground truth 200-400
- Judge 误判为"时序泄漏"

**根因**：系统模板中的测试预测兜底代码**假设了"预测值可直接保存"**，对目标变换一无所知。

**修复方案**：
1. **目标变换元数据追踪**：在 `preprocess()` 中如果检测到对目标列做了数值变换（如 log1p、sqrt、标准化），将变换类型和参数写入 `PREPROCESS_STATE` 或独立的 `transform_meta.json`
2. **测试预测代码模板化**：不再用固定兜底代码，而是根据 `transform_meta` 动态生成测试预测代码：
   ```python
   test_preds = model.predict(X_test_fe)
   if target_transform == 'log1p':
       test_preds = np.expm1(test_preds)
   ```
3. **统一 inverse_transform 入口**：要求 LLM 如果定义了目标变换，必须同时定义 `inverse_transform_target(preds)` 函数。系统兜底代码优先调用此函数，不存在时才走默认逻辑
4. **预测值合理性检查**：保存 `test_predictions.csv` 前，系统检查预测值的分布是否与训练集目标列分布处于同一数量级。如果偏差超过 10 倍，触发警告或 REPLAN

---

### Issue 3：Predict.py 独立生成导致格式不匹配（鸢尾花 predict.py 直接报错）

**现象**：
- `code_best.py` 保存的是单个 model 对象
- LLM 独立生成的 `predict.py` 期望加载 `{'pipeline': ..., 'label_encoder': ...}` 的 dict
- 两者格式不匹配，predict.py 实际上无法运行

**根因**：PlanCodingAgent 生成 predict.py 时**没有参考实际保存的 model 格式**，也没有参考训练代码中的预处理逻辑。

**修复方案**：
1. **Predict.py 自动生成，禁止 LLM 自由发挥**：在 `_save_intermediate_results()` 中，系统从 `code_best.py` 提取以下信息自动生成 predict.py：
   - `preprocess()` 函数源码
   - `feature_engineering()` 函数源码
   - `inverse_transform_target()` 函数源码（如有）
   - model 的实际保存格式（单对象 vs dict vs pipeline）
2. **模型保存格式标准化**：统一规定 `best_model.pkl` 的 schema：
   ```python
   {
       'model': model,  # 原始模型
       'preprocess_fn': preprocess,  # dill 序列化的函数
       'feature_engineering_fn': feature_engineering,
       'inverse_transform_fn': inverse_transform_target,
       'feature_names': list(X_train_fe.columns),
       'task_type': task_type,
   }
   ```
    predict.py 按此固定 schema 加载，不再依赖 LLM 猜测格式
3. **Predict.py 执行预检**：在最终保存前，系统在 sandbox 中执行一次 `python predict.py data/test.csv /tmp/verify.csv`，如果失败，直接触发 REPLAN 或降级为 embedded 预测策略

---

### Issue 4：EvaluationAgent 异常后 Fallback 机制不完善（肝硬化产物全部丢失）

**现象**：
- sandbox 执行成功（val_accuracy=0.84125）
- EvaluationAgent 耗时 0.0s，疑似 API 调用失败或异常中断
- 系统 fallback 到 `strategy=embedded`，声称"成功提取最佳模型"但 `artifacts=none`

**根因**：FastEngine 的 fallback 路径**绕过了正常的 `_save_intermediate_results()` 流程**，写了文件但没有更新 artifact 元数据。

**修复方案**：
1. **EvaluationAgent 异常隔离**：将 EvaluationAgent 调用包装在独立的 try-catch 中，异常时不应中断整个流程，而是记录错误并继续使用 sandbox 中的最佳结果
2. **Fallback 产物完整性保证**：无论走哪条 fallback 路径，最终必须通过统一的 `save_artifacts()` 函数保存产物，确保 `task_result.json`、`metrics.json`、`best_model.pkl`、`test_predictions.csv` 的元数据一致性
3. **Eval 超时/重试机制**：EvaluationAgent 调用增加 3 次重试（指数退避），单次超时从当前值适当放宽。避免因瞬态网络问题导致整个任务失败
4. **Windows GBK 编码根治**：stdout 编码问题导致日志写入失败可能是 EvaluationAgent 中断的诱因之一。在 `run_single_task.py` 和 FastEngine 初始化时强制设置 `sys.stdout.reconfigure(encoding='utf-8')`，或移除所有 emoji 输出

---

### Issue 5：验证集评估与 Judge 测试集评估标准不一致（多个任务高分被 Judge 拒）

**现象**：
- 鸢尾花 EvaluationAgent 给 88.0 分（验证集），Judge 测试集 Accuracy=0
- 共享单车 EvaluationAgent 给 82.0 分（验证集），Judge 测试集 R²=-1.21
- 电商顾客退货 EvaluationAgent 给 58.8 分，Judge 拒绝（AUC=0.58）

**根因**：EvaluationAgent 只看**验证集**表现，Judge 看**测试集**表现。两者评估逻辑和数据路径不一致。

**修复方案**：
1. **引入 Test-set Sanity Check**：在 sandbox 执行成功后，系统用**训练好的模型**对 `test.csv` 做一次"盲测"（走与训练相同的 feature engineering + predict 路径），计算测试集指标的粗略估计。如果测试集指标与验证集差距过大（如相对差距 >50%），在提交给 Judge 前触发内部 REPLAN
2. **EvaluationAgent 评估维度扩充**：除了验证集指标，增加以下维度到 EvaluationAgent 的 prompt：
   - 测试集预测分布 vs 训练集目标分布的 KL 散度/均值差异
   - 测试集预测值是否在合理范围（如分类任务预测类别是否覆盖了全部类别）
   - 特征工程在 test 上的可复现性（列名、列数是否与 train 一致）
3. **Judge 标准前置**：在 EvaluationAgent 的评分权重中，增加"测试集泛化风险"项（权重 0.2-0.3）。让 EvaluationAgent 在优化阶段就意识到"不能只看 validation，要看 test 泛化"
4. **测试集切分一致性校验**：对于时序任务，EvaluationAgent 明确检查 `data_splitter` 是否使用了时间切分（非随机切分）。如果不是，直接扣分并提示时序泄漏风险

---

## 三、优先级建议

| 优先级 | Issue | 预期收益 | 改动面 |
|--------|-------|----------|--------|
| P0 | Issue 1（Feature Engineering Pipeline 化） | Judge 通过率 +20%~30%（鸢尾花、红酒品质等直接修复） | `code_best.py` 模板、`sandbox_executor`、序列化逻辑 |
| P0 | Issue 2（目标变换逆变换） | 共享单车、时序/回归类任务稳定性大幅提升 | `code_best.py` 模板、测试预测生成逻辑 |
| P1 | Issue 3（Predict.py 自动生成） | 消除 predict.py 与训练代码的格式 mismatch | `plan_coding_agent`、`_save_intermediate_results` |
| P1 | Issue 4（Fallback 鲁棒性） | 肝硬化类任务不再丢失产物 | `fast_engine.py`、异常处理层 |
| P2 | Issue 5（Eval-Judge 标准对齐） | 减少"高分被拒"的误判，提升优化效率 | `evaluation_agent` prompt、评分权重 |

---

## 四、不修复的部分（LLM 纯不稳定）

以下问题**不通过系统架构修复**，建议通过更强模型或更多 DEBUG 轮次解决：

- 4 个完全失败任务的语法错误（`unexpected indent` 等）→ 建议 DEBUG 轮次从 3 次提升到 5 次，或 coding model 升级到更强的版本
- LLM 在 `evaluate_model()` 和测试预测代码之间的不一致性 → 系统模板修复后（Issue 2），此问题可被兜底代码覆盖

---

*文档生成时间：2026-05-26*
*对应基准版本：v0.9.0-benchmark-20tasks*

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

**根因**：EvaluationAgent 只看**验证集**表现，Judge 看**测试集**表现。由于测试集 ground truth 在训练阶段不可见，EvaluationAgent 无法直接评估测试集泛化能力。两者的信息不对等导致优化方向偏离。

**修复方案**：
1. **预测分布合理性检查（无需 ground truth）**：EvaluationAgent 读取 `test_predictions.csv`，检查以下明显异常（无需测试集标签即可发现）：
   - 预测值是否全为同一常量（如全 0 或全 1）→ 极可能是 feature engineering 不一致
   - 分类任务预测的类别集合是否是训练集类别集合的子集 → 若不是，说明标签映射出错
   - 回归任务预测值的数量级是否与训练集目标列相差 10 倍以上 → 极可能是目标变换未逆变换
   - 预测列是否存在大量 NaN
2. **特征工程可复现性检查（无需 ground truth）**：在 sandbox 中，系统在保存 model 前，用 `feature_engineering()` 分别处理 train 和 test 的原始数据，对比输出特征的列名和列数。如果不一致，说明 test 上的特征工程与 train 不同（如 test 多了/少了某些列），立即标记为高风险
3. **时序切分合规性检查（从代码推断）**：EvaluationAgent 检查 LLM 生成的代码中是否存在 `shuffle=True`、`train_test_split(..., shuffle=True)`、`random_state` 等关键词。时序任务中若存在这些关键词，说明可能使用了随机切分，直接扣分并提示时序泄漏风险
4. **评估权重调整**：在 EvaluationAgent 的评分权重中，增加"代码鲁棒性"维度（权重 0.15-0.2），包括：
   - 是否正确使用 sklearn Pipeline（而非手动分步执行）
   - 是否正确处理类别不平衡（class_weight / sample_weight）
   - 是否避免了数据泄漏（如时序任务中的未来信息）
   让 EvaluationAgent 不只关注验证集数字，也关注代码结构上的抗风险能力

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

## 五、安全实施约束（防止改坏已跑通任务）

**核心原则：所有修复必须保留旧路径作为 fallback，先验证、再推广。**

### 5.1 增量验证流程

每实施一个修复后，必须跑一轮 **"回归验证"**：
- 选取当前已跑通的 16 个任务中的 **8 个代表性任务**（覆盖二分类、多分类、回归、时序）
- 对比修复前后的 `success` 率和 `judge_accepted` 率
- **如果原有通过的任务有任何一项失败或 Judge 被降级，立即回滚该修复**

### 5.2 各 Issue 的兼容策略

| Issue | 高风险点 | 安全实施方式 |
|-------|----------|-------------|
| **Issue 1** FE Pipeline 化 | 强制要求 LLM 写 Pipeline 可能导致原来能跑的任务因 Pipeline 结构错误而失败 | **双路径兼容**：<br>① 新增 `AutoPipelineWrapper`：系统侧自动检测 `feature_engineering()` 是否存在，如果存在且不在 Pipeline 中，系统在保存前自动包装为 `Pipeline([('fe', FunctionTransformer(fe_fn)), ('clf', model)])`<br>② 不修改 `code_best.py` 模板对 LLM 的要求，LLM 仍可按旧方式写代码<br>③ 只有系统侧的包装逻辑失败时，才fallback到旧路径 |
| **Issue 2** 目标变换逆变换 | 启发式检测可能误判，给不需要逆变换的任务也加了变换 | **元数据显式优先**：<br>① 优先读取 LLM 在 `PREPROCESS_STATE` 中显式声明的 `target_transform`（如 `log1p`）<br>② 如果没有显式声明，系统不做任何猜测/启发式检测，保持现有兜底代码不变<br>③ 逆变换只在 `target_transform` 字段存在时生效 |
| **Issue 3** predict.py 自动生成 | 自动提取函数可能因函数名变化、全局变量依赖而失败 | **生成 + 预检 + fallback**：<br>① 系统尝试从 `code_best.py` AST 提取函数自动生成 predict.py<br>② 生成后在 sandbox 中执行 `python predict.py test.csv /tmp/verify.csv` 预检<br>③ 预检失败则 **静默回退**到现有 PlanCodingAgent 生成逻辑，不中断流程 |
| **Issue 4** Fallback 鲁棒性 | 增加 try-catch 和重试可能掩盖真正需要暴露的代码错误 | **异常粒度控制**：<br>① 只对 EvaluationAgent **网络/IO 类异常**做重试（HTTPError、Timeout）<br>② 对代码执行错误（SyntaxError、ValueError）不重试，保持现有行为<br>③ fallback 产物保存走与正常路径**相同的 `save_artifacts()` 函数**，确保元数据一致性 |
| **Issue 5** Eval-Judge 对齐 | 预测分布检查可能误报（如任务本身预测值就应高度集中） | **分层处理**：<br>① **预测全常量 / 存在 NaN / 类别越界**：这些是客观规则，误报率极低，可直接触发 warning<br>② **特征工程列数不一致**：同样是客观规则，train/test 列数不一致必定有问题，可直接标记高风险<br>③ **时序切分关键词检查**：可能因注释/字符串匹配误报，只作为 soft 提示，不扣分<br>④ **代码鲁棒性评分权重**：纯 prompt 层面调整，不影响代码执行逻辑，风险为 0 |

### 5.3 实施顺序建议

**第一阶段（低风险、高回报）**：
1. Issue 4：增加 EvaluationAgent 网络异常重试 + fallback 统一 save（只改异常处理层，不动代码生成逻辑，最安全）
2. Issue 2：支持 `PREPROCESS_STATE['target_transform']` 的显式逆变换（只在 LLM 声明时生效，无声明则行为不变）

**第二阶段（验证后推广）**：
3. Issue 3：predict.py 自动生成 + 预检 + fallback（需要 AST 提取，有一定复杂度，但 fallback 保证安全）
4. Issue 1：系统侧 `AutoPipelineWrapper`（双路径兼容，LLM 无感知）

**第三阶段（数据驱动）**：
5. Issue 5：收集足够数据后，将 sanity check 从 warning 升级为 soft REPLAN 条件

---

## 六、实施状态

| Issue | 状态 | 对应 Commit |
|-------|------|-------------|
| Issue 1（AutoPipelineWrapper） | ✅ 已实施 | `7d3ce4f` |
| Issue 2（目标变换逆变换） | ✅ 已实施 | `7d3ce4f` |
| Issue 3（predict.py 自动生成） | ✅ 已实施 | `7d3ce4f` |
| Issue 4（Fallback 鲁棒性） | ✅ 已实施 | `7d3ce4f` |
| Issue 5（Eval-Judge 对齐） | ⏸️ 待实施 | — |

---

*文档生成时间：2026-05-26*
*对应基准版本：v0.9.0-benchmark-20tasks*

"""
直接用已保存的 plan 测试 CodingAgent
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path

from app.agents.coding_agent import CodingAgent
from app.agents.base import LLMClient
from app.models.schemas import TaskConfig, ExtractedSlots, UploadedFile, FileRole, TaskType

llm_config = {
    "provider": "openai",
    "base_url": "http://localhost:8000/v1",
    "api_key": "not-needed",
    "model": "qwen3.6-27b",
    "temperature": 0.3,
    "max_tokens": 4096,
}

TASK_DIR = "/home/hpl/ml-workflow/test_data/信用卡欺诈-二分类-类别极度不平衡"

# 读取已保存的 plan
plan_formatted = Path("/tmp/test_plan_output/plan_formatted.txt").read_text(encoding='utf-8')

# 构建 TaskConfig（复用之前的数据）
train_path = Path(TASK_DIR) / "用于建模" / "train.csv"
desc_path = Path(TASK_DIR) / "用于建模" / "任务描述-信用卡欺诈.txt"
desc = desc_path.read_text(encoding='utf-8').strip() if desc_path.exists() else ""

# 构建 data profile（简化版）
import pandas as pd
train_df = pd.read_csv(train_path)
cols = []
for col in train_df.columns:
    s = train_df[col]
    cols.append({
        "name": col,
        "type": "numeric" if s.dtype.kind in 'ifc' else "categorical",
        "missingCount": int(s.isnull().sum()),
        "uniqueCount": int(s.nunique()),
        "isLikelyId": col == "id" or (s.dtype.kind in 'ifc' and s.nunique() > len(train_df) * 0.95),
    })

profile = {
    "rowCount": len(train_df),
    "colCount": len(train_df.columns),
    "columns": cols,
    "targetStats": {
        "isImbalanced": True,
        "minorityRatio": float(train_df['IsFraud'].mean()),
    }
}

tc = TaskConfig(
    extracted_slots=ExtractedSlots(
        target_column="IsFraud",
        task_type=TaskType.BINARY_CLASSIFICATION,
        eval_metric="AUC",
        complexity="complex",
        complexity_reason="极度不平衡",
        is_time_series=False,
        feature_constraints=[],
        user_modeling_suggestions=None,
    ),
    uploaded_files=[
        UploadedFile(name="train.csv", path=str(train_path), role=FileRole.TRAIN),
    ],
    user_description=desc,
    data_profile=profile,
)

print("=" * 70)
print("调用 CodingAgent 生成代码...")
print("=" * 70)

llm = LLMClient(**llm_config)
coding_agent = CodingAgent(llm_client=llm)

code_output = coding_agent.generate(
    task_config=tc,
    structured_plan=plan_formatted,
    run_state="INIT",
    context_payload="",
    previous_code=""
)

code = code_output.code

# 合规性检查
print("\n--- 代码合规性检查 ---")
checks = [
    ("scale_pos_weight", "scale_pos_weight" in code),
    ("class_weight='balanced' 不存在", "class_weight='balanced'" not in code),
    ("class_weight=\"balanced\" 不存在", 'class_weight="balanced"' not in code),
    ("prepare_for_prediction 函数", "def prepare_for_prediction" in code),
    ("best_model.pkl 保存", "best_model.pkl" in code),
    ("dill 导入", "dill" in code),
    ("log1p 变换 (Amount)", "log1p" in code.lower() or "np.log" in code),
    ("AP (average_precision)", "average_precision" in code.lower() or "ap_score" in code.lower() or "average precision" in code.lower()),
    ("阈值搜索", "threshold" in code.lower()),
    ("验证集不重新切分", "train_test_split" not in code),
    ("Pipeline 包装", "Pipeline" in code),
]

for item, found in checks:
    status = "✅" if found else "❌"
    print(f"  {status} {item}")

# 保存代码
output_dir = Path("/tmp/test_plan_output")
(output_dir / "generated_code.py").write_text(code, encoding='utf-8')
(output_dir / "coding_plan.txt").write_text(code_output.plan, encoding='utf-8')

print(f"\n代码已保存到: {output_dir}/generated_code.py")
print(f"代码长度: {len(code)} 字符")
print(f"\n{'='*70}")
print("生成代码前 100 行预览:")
print("=" * 70)
for i, line in enumerate(code.split('\n')[:100], 1):
    print(f"{i:3d}| {line}")

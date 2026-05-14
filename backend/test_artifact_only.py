#!/usr/bin/env python3
"""
产物生成阶段独立测试
跳过训练/评估，直接从产物生成开始验证
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings, build_eval_llm_config
from app.models.schemas import TaskConfig, ExtractedSlots, UploadedFile, FileRole, LLMConfig
from app.agents.plan_coding import PlanCodingAgent
from app.agents.base import LLMClient
from app.sandbox.executor import SandboxExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_llm_config_from_dict(cfg: dict) -> LLMConfig:
    return LLMConfig(
        provider=cfg.get("provider", "openai"),
        base_url=cfg.get("base_url", ""),
        api_key=cfg.get("api_key", ""),
        model=cfg.get("model", ""),
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 4096),
        extra_body=cfg.get("extra_body")
    )


def test_artifact_generation():
    """测试产物生成阶段"""
    
    # 1. 准备数据来源
    result_dir = Path("/home/hpl/ml-workflow/backend/outputs/eval_e6c567256df6/睡眠障碍预测/run_1")
    data_source = Path("/home/hpl/ml-workflow/test_data/睡眠障碍预测/建模")
    
    # 2. 创建测试数据目录
    test_data_dir = Path("outputs/test_artifact_data")
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制数据文件
    shutil.copy2(data_source / "train_split.csv", test_data_dir / "train.csv")
    shutil.copy2(result_dir / "code_best.py", test_data_dir / "code_best.py")
    shutil.copy2(result_dir / "best_model.pkl", test_data_dir / "best_model.pkl")
    
    # 如果有测试集也复制
    test_csv = data_source / "test.csv"
    if test_csv.exists():
        shutil.copy2(test_csv, test_data_dir / "test.csv")
    
    # 3. 读取 best_code
    best_code = (test_data_dir / "code_best.py").read_text(encoding='utf-8')
    logger.info(f"[Test] best_code 长度={len(best_code)}")
    
    # 4. 创建 TaskConfig
    tc = TaskConfig(
        extracted_slots=ExtractedSlots(
            target_column="Sleep Disorder",
            task_type="multiclass_classification",
            eval_metric="F1-macro",
            complexity="complex",
            is_time_series=False,
            feature_constraints=[],
            user_modeling_suggestions=None
        ),
        uploaded_files=[
            UploadedFile(name="train.csv", path=str(test_data_dir / "train.csv"), role=FileRole.TRAIN),
        ],
        user_description="睡眠障碍预测",
        agent_llm_configs={}
    )
    
    # 5. 创建 PlanCodingAgent（使用 coding_llm 生成产物）
    plan_cfg = build_eval_llm_config("plan")
    coding_cfg = build_eval_llm_config("coding")
    
    agent = PlanCodingAgent(
        llm_client=LLMClient(
            provider=plan_cfg["provider"],
            base_url=plan_cfg["base_url"],
            api_key=plan_cfg["api_key"],
            model=plan_cfg["model"],
            temperature=plan_cfg["temperature"],
            max_tokens=plan_cfg["max_tokens"],
            extra_body=plan_cfg.get("extra_body")
        ),
        plan_llm_client=LLMClient(
            provider=plan_cfg["provider"],
            base_url=plan_cfg["base_url"],
            api_key=plan_cfg["api_key"],
            model=plan_cfg["model"],
            temperature=plan_cfg["temperature"],
            max_tokens=plan_cfg["max_tokens"],
            extra_body=plan_cfg.get("extra_body")
        ),
        coding_llm_client=LLMClient(
            provider=coding_cfg["provider"],
            base_url=coding_cfg["base_url"],
            api_key=coding_cfg["api_key"],
            model=coding_cfg["model"],
            temperature=coding_cfg["temperature"],
            max_tokens=coding_cfg["max_tokens"],
            extra_body=coding_cfg.get("extra_body")
        ),
        unified_llm_client=LLMClient(
            provider=plan_cfg["provider"],
            base_url=plan_cfg["base_url"],
            api_key=plan_cfg["api_key"],
            model=plan_cfg["model"],
            temperature=plan_cfg["temperature"],
            max_tokens=plan_cfg["max_tokens"],
            extra_body=plan_cfg.get("extra_body")
        ),
    )
    
    # 6. 生成 predict.py
    logger.info("=" * 60)
    logger.info("[Test] 开始生成 predict.py")
    logger.info("=" * 60)
    
    predict_start = datetime.now()
    predict_result = agent.generate_predict_script(
        task_config=tc,
        best_code=best_code,
        data_dir=str(test_data_dir)
    )
    predict_elapsed = (datetime.now() - predict_start).total_seconds()
    
    if predict_result and predict_result.code:
        logger.info(f"[Test] predict.py 生成成功: 长度={len(predict_result.code)}, 耗时={predict_elapsed:.1f}s")
        predict_path = test_data_dir / "predict.py"
        predict_path.write_text(predict_result.code, encoding='utf-8')
    else:
        logger.error(f"[Test] predict.py 生成失败")
        return False
    
    # 7. 生成产物代码
    logger.info("=" * 60)
    logger.info("[Test] 开始生成产物代码")
    logger.info("=" * 60)
    
    artifact_start = datetime.now()
    artifact_result = agent.generate_artifacts(
        task_config=tc,
        best_code=best_code,
        has_test_set=test_csv.exists(),
        data_dir=str(test_data_dir)
    )
    artifact_elapsed = (datetime.now() - artifact_start).total_seconds()
    
    if not artifact_result or not artifact_result.code:
        logger.error(f"[Test] 产物代码生成失败")
        return False
    
    logger.info(f"[Test] 产物代码生成成功: 长度={len(artifact_result.code)}, 耗时={artifact_elapsed:.1f}s")
    
    # 保存产物代码用于调试
    artifact_script_path = test_data_dir / "artifact_script.py"
    artifact_script_path.write_text(artifact_result.code, encoding='utf-8')
    
    # 8. 沙箱执行产物代码
    logger.info("=" * 60)
    logger.info("[Test] 开始沙箱执行产物代码")
    logger.info("=" * 60)
    
    artifact_dir = test_data_dir / "output"
    artifact_dir.mkdir(exist_ok=True)
    
    sandbox = SandboxExecutor(timeout=300)
    sandbox_start = datetime.now()
    result = sandbox.execute(
        code=artifact_result.code,
        data_dir=test_data_dir,
        task_type="multiclass_classification",
        artifact_mode=True,
        artifact_output_dir=artifact_dir
    )
    sandbox_elapsed = (datetime.now() - sandbox_start).total_seconds()
    
    logger.info(f"[Test] 沙箱执行: success={result.success}, 耗时={sandbox_elapsed:.1f}s")
    if not result.success:
        logger.error(f"[Test] 沙箱执行失败: {result.error_message}")
        logger.error(f"[Test] stderr: {result.stderr[:500]}")
        return False
    
    # 9. 检查产物文件
    logger.info("=" * 60)
    logger.info("[Test] 产物文件检测")
    logger.info("=" * 60)
    
    expected_files = {
        "model.pkl": "模型文件",
        "feature_importance.csv": "特征重要性数据",
        "feature_importance.png": "特征重要性图",
        "report.html": "HTML报告",
    }
    if test_csv.exists():
        expected_files["test_predictions.csv"] = "测试集预测"
    
    found = []
    missing = []
    for fname, desc in expected_files.items():
        fpath = artifact_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            size = fpath.stat().st_size
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
            found.append(f"  ✓ {fname} ({desc}, {size_str})")
        else:
            missing.append(f"  ✗ {fname} ({desc})")
    
    for f in found:
        logger.info(f)
    for m in missing:
        logger.warning(m)
    
    # 10. 汇总报告
    logger.info("=" * 60)
    logger.info("[Test] 产物生成测试报告")
    logger.info("=" * 60)
    logger.info(f"predict.py 生成: 成功, {predict_elapsed:.1f}s")
    logger.info(f"产物代码生成: 成功, {artifact_elapsed:.1f}s")
    logger.info(f"沙箱执行: {'成功' if result.success else '失败'}, {sandbox_elapsed:.1f}s")
    logger.info(f"产物文件: {len(found)}/{len(expected_files)}")
    logger.info(f"产物目录: {artifact_dir}")
    
    return result.success and len(missing) == 0


if __name__ == "__main__":
    success = test_artifact_generation()
    print(f"\n{'='*60}")
    print(f"测试{'通过' if success else '失败'}")
    print(f"{'='*60}")
    sys.exit(0 if success else 1)

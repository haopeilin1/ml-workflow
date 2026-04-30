"""
快速模式任务编排引擎 (Fast Engine)

状态机驱动的建模流程控制：
IDLE → PLANNING → CODING → RUNNING → EVALUATING → (OPTIMIZING → CODING → RUNNING → EVALUATING) × ≤3
                                    ↓ YIELD_TO_USER
                                 PRESENTING → WAITING_FEEDBACK
                                    ↓ 满意
                                 COMPLETED
                                    ↓ 不满意
                                 (OPTIMIZING → CODING → RUNNING → EVALUATING) × ≤3

RUNNING 失败 → DEBUG → CODING → RUNNING（最多5次）
"""

import json
import logging
import os
import shutil
import threading
import time
from typing import Optional
from pathlib import Path

from app.config import settings
from app.models.schemas import (
    TaskConfig, FastTaskPhase, DecisionType,
    ExecutionMetrics, EvaluationResult, CodeOutput, LLMConfig,
    ArtifactInfo, ArtifactFile
)
from app.core.state import task_manager
from app.core.data_splitter import DataSplitter
from app.agents.plan_coding import PlanCodingAgent
from app.agents.evaluation import EvaluationAgent
from app.sandbox.executor import SandboxExecutor, SandboxResult

logger = logging.getLogger(__name__)


class FastEngine:
    """
    快速模式引擎
    
    将 PlanCodingAgent、EvaluationAgent、SandboxExecutor、DataSplitter
    串联成完整的状态机流程。
    
    执行在独立后台线程中进行，前端通过 task_manager 轮询状态。
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.state = task_manager.get_task(task_id)
        if not self.state:
            raise ValueError(f"任务 {task_id} 不存在")
        
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        
        # 根据 task_config 中的 llm_config 创建 LLM 客户端
        # 如果前端传入了配置，优先使用；否则使用后端默认配置
        tc = self.state.task_config
        global_llm_config = tc.llm_config
        
        # 支持按阶段独立配置 LLM（agent_llm_configs 供开发/测试使用）
        # 例如：{"plan_coding": LLMConfig(...), "evaluation": LLMConfig(...)}
        # 若某阶段未单独配置，则回退到全局 llm_config
        agent_configs = tc.agent_llm_configs or {}
        plan_llm_config = agent_configs.get('plan_coding') or global_llm_config
        eval_llm_config = agent_configs.get('evaluation') or global_llm_config
        
        self.plan_coding_agent = PlanCodingAgent(llm_client=self._build_llm_client(plan_llm_config))
        self.evaluation_agent = EvaluationAgent(llm_client=self._build_llm_client(eval_llm_config))
        self.sandbox = SandboxExecutor(timeout=settings.SANDBOX_TIMEOUT)
        self.data_splitter = DataSplitter(settings.UPLOAD_DIR, settings.OUTPUT_DIR)
        
        # 数据集路径（由 _prepare_data 填充）
        self.datasets: Optional[dict] = None
    
    def _build_llm_client(self, llm_config: Optional[LLMConfig]):
        """根据配置构建 LLM 客户端"""
        from app.agents.base import LLMClient
        if llm_config:
            return LLMClient(
                provider=llm_config.provider,
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
            )
        return LLMClient.from_settings()
    
    def _append_log(self, message: str):
        """将日志追加到状态日志中"""
        if message:
            self.state.logs.append(message)
            task_manager.update_task(self.task_id, logs=self.state.logs)
    
    # ========== 启动入口 ==========
    
    def start(self):
        """启动快速模式流程（在后台线程中运行）"""
        if self._thread and self._thread.is_alive():
            logger.warning(f"[FastEngine] 任务 {self.task_id} 已在运行中")
            return
        
        self._stopped = False
        self._thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._thread.start()
        logger.info(f"[FastEngine] 任务 {self.task_id} 已启动")
    
    def stop(self):
        """停止任务"""
        self._stopped = True
        logger.info(f"[FastEngine] 任务 {self.task_id} 收到停止信号")
    
    # ========== 主流程 ==========
    
    def _run_pipeline(self):
        """主流程线程"""
        try:
            tc = self.state.task_config
            
            # 1. 数据准备
            self._prepare_data(tc)
            if self._stopped:
                return
            
            # 2. 初始代码生成 (INIT)
            self._generate_init_code(tc)
            if self._stopped:
                return
            
            # 3. 执行-评估循环
            self._execute_evaluate_loop(tc)
            if self._stopped:
                return
            
            # 4. 循环结束后的状态处理
            if self.state.phase == FastTaskPhase.PRESENTING:
                logger.info(f"[FastEngine] 任务 {self.task_id} 进入等待用户反馈阶段")
                # 此时线程结束，等待用户通过 API 调用 continue_with_feedback
            elif self.state.phase == FastTaskPhase.FAILED:
                logger.error(f"[FastEngine] 任务 {self.task_id} 失败: {self.state.execution_error}")
            
        except Exception as e:
            logger.exception(f"[FastEngine] 任务 {self.task_id} 发生未捕获异常")
            self._set_phase(FastTaskPhase.FAILED)
            task_manager.update_task(
                self.task_id,
                execution_error=f"引擎异常: {str(e)}"
            )
    
    # ========== 用户反馈处理 ==========
    
    def continue_with_feedback(self, satisfied: bool, suggestion: str = ""):
        """
        用户提交反馈后继续流程
        
        Args:
            satisfied: 用户是否满意
            suggestion: 用户的不满意建议
        """
        if self._thread and self._thread.is_alive():
            logger.warning(f"[FastEngine] 任务 {self.task_id} 仍在运行中，忽略反馈")
            return
        
        self._stopped = False
        self._thread = threading.Thread(
            target=self._handle_feedback_pipeline,
            args=(satisfied, suggestion),
            daemon=True
        )
        self._thread.start()
    
    def _handle_feedback_pipeline(self, satisfied: bool, suggestion: str):
        """处理用户反馈的后台线程"""
        try:
            tc = self.state.task_config
            
            if satisfied:
                # 用户满意 → 生成最终产物（产物就绪后再设置 COMPLETED）
                logger.info(f"[FastEngine] 任务 {self.task_id} 用户确认满意")
                self._append_log("正在生成可视化报告...")
                if self.state.has_test_set:
                    self._append_log("正在对测试集进行预测...")
                self._generate_artifacts(tc)
                return
            
            # 用户不满意
            self.state.user_feedback_round += 1
            
            # 根据用户反馈优化代码
            self._set_phase(FastTaskPhase.OPTIMIZING)
            code_output = self.plan_coding_agent.generate(
                task_config=tc,
                run_state="OPTIMIZE",
                context_payload=suggestion or "用户未填写具体建议",
                previous_code=(self.state.best_code or self.state.code)
            )
            self.state.code = code_output.code
            self.state.code_history.append({
                "round": self.state.optimize_round + self.state.user_feedback_round,
                "code": code_output.code,
                "type": "user_feedback",
                "suggestion": suggestion
            })
            
            # 记录 LLM 原始响应到日志
            self._append_log("[Plan & Coding Agent] 根据用户反馈调整代码")
            if code_output.raw_response:
                self._append_log(code_output.raw_response)
            
            # 重新走执行-评估流程
            self._execute_evaluate_loop(tc)
            
        except Exception as e:
            logger.exception(f"[FastEngine] 用户反馈处理异常")
            self._set_phase(FastTaskPhase.FAILED)
            task_manager.update_task(
                self.task_id,
                execution_error=f"反馈处理异常: {str(e)}"
            )
    
    # ========== 产物生成 ==========
    
    def _generate_artifacts(self, tc: TaskConfig):
        """
        生成最终产物
        
        产物包括：
        - model.pkl（模型文件）
        - test_predictions.csv（测试集预测，如有）
        - feature_importance.csv（特征重要性）
        - feature_importance.png（特征重要性图）
        - report.html（可视化报告）
        
        注意：无论成功/失败/超时，finally 中都会将任务标记为 COMPLETED，
        确保前端能收到终态信号。若发生严重异常，外层 _handle_feedback_pipeline
        会将其覆盖为 FAILED。
        """
        try:
            best_code = self.state.best_code or self.state.code
            if not best_code:
                logger.warning(f"[FastEngine] 无可用代码，生成简化产物")
                self._append_log("[WARN] 无可用代码，生成简化产物")
                self._generate_fallback_artifacts(tc)
                return
            
            # 1. 生成产物代码（带线程级超时，防止 LLM 调用无限挂起）
            self._append_log("[Plan & Coding Agent] 正在调用 LLM 生成产物代码...")
            logger.info(f"[FastEngine] 开始生成产物代码, best_code长度={len(best_code)}")
            
            llm_result = [None]
            llm_error = [None]
            
            def _call_llm_worker():
                try:
                    llm_result[0] = self.plan_coding_agent.generate_artifacts(
                        task_config=tc,
                        best_code=best_code,
                        has_test_set=self.state.has_test_set
                    )
                except Exception as e:
                    llm_error[0] = e
            
            llm_thread = threading.Thread(target=_call_llm_worker)
            llm_thread.daemon = True
            llm_thread.start()
            llm_thread.join(timeout=600)  # 最多等待 600 秒（10分钟）
            
            if llm_thread.is_alive():
                logger.error("[FastEngine] LLM 生成产物代码超时（600秒）")
                self._append_log("[WARN] LLM 生成产物代码超时，将使用简化产物")
                self._generate_fallback_artifacts(tc, reason="timeout")
                return
            
            if llm_error[0]:
                logger.error(f"[FastEngine] LLM 生成产物代码失败: {llm_error[0]}")
                self._append_log(f"[WARN] LLM 生成产物代码失败: {llm_error[0]}")
                self._generate_fallback_artifacts(tc, reason="error")
                return
            
            code_output = llm_result[0]
            self._append_log(f"[Plan & Coding Agent] 产物代码生成完成, 长度={len(code_output.code)}")
            logger.info(f"[FastEngine] 产物代码生成完成, 长度={len(code_output.code)}")
            
            # 2. 沙箱执行产物代码（允许文件写入），失败时自动修复最多5次
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            artifact_dir = settings.OUTPUT_DIR / self.task_id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            debug_round = 0
            max_debug_rounds = 5
            
            while True:
                self._append_log("[FastEngine] 正在沙箱中执行产物代码（允许文件写入）...")
                logger.info(f"[FastEngine] 开始沙箱执行产物代码, data_dir={data_dir}")
                
                result = self.sandbox.execute(
                    code=code_output.code,
                    data_dir=data_dir,
                    task_type=tc.extracted_slots.task_type or "binary_classification",
                    artifact_mode=True,
                    artifact_output_dir=artifact_dir
                )
                
                if result.success:
                    break
                
                debug_round += 1
                if debug_round > max_debug_rounds:
                    logger.warning(f"[FastEngine] 产物代码调试达到上限({max_debug_rounds})，降级为简化产物")
                    self._append_log(f"[WARN] 产物代码调试达到上限，降级为简化产物")
                    self._generate_fallback_artifacts(tc, reason="debug_max")
                    return
                
                # 自动修复产物代码
                error_detail = result.error_message or result.stderr or "Unknown error"
                logger.info(f"[FastEngine] 产物代码执行失败，第 {debug_round} 次自动修复...")
                self._append_log(f"[WARN] 产物代码执行失败: {error_detail[:200]}")
                self._append_log(f"[FastEngine] 正在第 {debug_round} 次修复产物代码...")
                
                fix_result = [None]
                fix_error = [None]
                
                def _call_fix_worker():
                    try:
                        fix_result[0] = self.plan_coding_agent.generate_artifacts(
                            task_config=tc,
                            best_code=best_code,
                            has_test_set=self.state.has_test_set,
                            error_message=error_detail
                        )
                    except Exception as e:
                        fix_error[0] = e
                
                fix_thread = threading.Thread(target=_call_fix_worker)
                fix_thread.daemon = True
                fix_thread.start()
                fix_thread.join(timeout=120)  # 每次修复最多等待 120 秒
                
                if fix_thread.is_alive():
                    logger.error(f"[FastEngine] 产物代码第 {debug_round} 次修复超时")
                    self._append_log(f"[WARN] 产物代码修复超时，将使用简化产物")
                    self._generate_fallback_artifacts(tc, reason="fix_timeout")
                    return
                
                if fix_error[0]:
                    logger.error(f"[FastEngine] 产物代码修复失败: {fix_error[0]}")
                    self._append_log(f"[WARN] 产物代码修复失败: {fix_error[0]}")
                    self._generate_fallback_artifacts(tc, reason="fix_error")
                    return
                
                code_output = fix_result[0]
                self._append_log(f"[Plan & Coding Agent] 产物代码修复完成, 长度={len(code_output.code)}")
                logger.info(f"[FastEngine] 产物代码修复完成, 长度={len(code_output.code)}")
            
            self._append_log("[FastEngine] 产物代码执行成功，正在解析产物...")
            logger.info(f"[FastEngine] 产物代码执行成功, stdout长度={len(result.stdout)}")
            
            # 3. 解析产物文件
            artifacts = ArtifactInfo()
            files = []
            
            # 扫描产物目录
            if artifact_dir.exists():
                for f in artifact_dir.iterdir():
                    if f.is_file():
                        size_bytes = f.stat().st_size
                        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.1f} MB"
                        
                        file_type = "file"
                        desc = ""
                        if f.suffix == '.pkl':
                            file_type = "model"
                            desc = "训练好的模型文件"
                        elif f.suffix == '.csv':
                            file_type = "data"
                            if 'test' in f.name:
                                desc = "测试集预测结果"
                            elif 'feature' in f.name:
                                desc = "特征重要性数据"
                            else:
                                desc = "数据文件"
                        elif f.suffix == '.png':
                            file_type = "image"
                            desc = "特征重要性可视化图"
                        elif f.suffix == '.html':
                            file_type = "report"
                            desc = "可视化评估报告"
                        elif f.suffix == '.py':
                            file_type = "code"
                            desc = "Pipeline 代码"
                        
                        # 生成可供前端直接访问的 URL 路径
                        file_url = f"/artifacts/{self.task_id}/artifacts/{f.name}"
                        files.append(ArtifactFile(
                            name=f.name,
                            path=file_url,
                            type=file_type,
                            size=size_str,
                            desc=desc
                        ))
            
            artifacts.files = files
            
            # 4. 读取测试集预测
            test_pred_path = artifact_dir / "test_predictions.csv"
            if test_pred_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(test_pred_path)
                    # 取前 20 条作为预览
                    preview = df.head(20)
                    predictions = []
                    for idx, row in preview.iterrows():
                        pred_dict = {"id": idx}
                        # 尝试找到预测列
                        if 'prediction' in row:
                            pred_dict["pred"] = row['prediction']
                        # 尝试找到概率列
                        prob_cols = [c for c in df.columns if 'prob' in c.lower() or 'proba' in c.lower() or 'score' in c.lower()]
                        if prob_cols:
                            pred_dict["prob"] = round(float(row[prob_cols[0]]), 4)
                        predictions.append(pred_dict)
                    artifacts.test_predictions = predictions
                except Exception as e:
                    logger.warning(f"[FastEngine] 读取测试集预测失败: {e}")
            
            # 5. 读取特征重要性
            fi_path = artifact_dir / "feature_importance.csv"
            if fi_path.exists():
                try:
                    import pandas as pd
                    fi_df = pd.read_csv(fi_path)
                    fi_list = []
                    for _, row in fi_df.iterrows():
                        fi_list.append({
                            "name": str(row.get('name', row.iloc[0])),
                            "importance": round(float(row.get('importance', row.iloc[1])), 4)
                        })
                    artifacts.feature_importance = fi_list
                except Exception as e:
                    logger.warning(f"[FastEngine] 读取特征重要性失败: {e}")
            
            # 6. 报告路径
            report_path = artifact_dir / "report.html"
            if report_path.exists():
                artifacts.report_path = str(report_path)
            
            # 7. 更新状态
            task_manager.update_task(self.task_id, artifacts=artifacts)
            self._append_log("[FastEngine] 产物生成完成")
            self._append_log(f"产物文件: {[f.name for f in files]}")
            
            logger.info(f"[FastEngine] 产物生成完成: {len(files)} 个文件")
            
        except Exception as e:
            logger.exception(f"[FastEngine] 产物生成异常")
            self._append_log(f"[WARN] 产物生成异常: {str(e)}")
            self._generate_fallback_artifacts(tc)
        finally:
            # 确保产物阶段结束后标记为 COMPLETED；若后续外层捕获到严重异常，
            # 会被覆盖为 FAILED，因此这里先设为 COMPLETED 是安全的。
            self._set_phase(FastTaskPhase.COMPLETED)
            self._append_log("[FastEngine] 任务已完成")
    
    def _generate_fallback_artifacts(self, tc: TaskConfig, reason: str = "timeout"):
        """
        生成简化产物（当 LLM 调用失败或超时时使用）
        
        基于已有的 metrics 和 evaluation 数据生成简单的 HTML 报告，
        不依赖 LLM，不重新训练模型。
        """
        try:
            self._append_log("[FastEngine] 正在生成简化产物...")
            logger.info(f"[FastEngine] 开始生成简化产物")
            
            artifact_dir = settings.OUTPUT_DIR / self.task_id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            metrics = self.state.metrics
            evaluation = self.state.evaluation
            task_type = tc.extracted_slots.task_type or 'unknown'
            
            # 根据任务类型生成对应的指标 HTML
            if task_type == 'regression':
                primary_metrics_html = f"""<div class="metric">
    <div class="metric-label">验证集 RMSE</div>
    <div class="metric-value">{getattr(metrics, 'val_rmse', 'N/A') if metrics else 'N/A'}</div>
</div>
<div class="metric">
    <div class="metric-label">训练集 Score</div>
    <div class="metric-value">{getattr(metrics, 'train_score', 'N/A') if metrics else 'N/A'}</div>
</div>"""
            else:
                primary_metrics_html = f"""<div class="metric">
    <div class="metric-label">验证集 AUC</div>
    <div class="metric-value">{getattr(metrics, 'val_auc', 'N/A') if metrics else 'N/A'}</div>
</div>
<div class="metric">
    <div class="metric-label">验证集准确率</div>
    <div class="metric-value">{getattr(metrics, 'val_accuracy', 'N/A') if metrics else 'N/A'}</div>
</div>"""
            
            reason_text = "LLM 调用超时（600秒）" if reason == "timeout" else "LLM 调用失败"
            
            # 生成简化 HTML 报告
            html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>模型评估报告</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
.metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }}
.metric-label {{ color: #666; font-size: 14px; }}
.metric-value {{ color: #333; font-size: 24px; font-weight: bold; }}
.warning {{ color: #ff9800; }}
.error {{ color: #f44336; }}
</style>
</head>
<body>
<h1>🤖 模型评估报告</h1>
<div class="metric">
    <div class="metric-label">任务类型</div>
    <div class="metric-value">{task_type}</div>
</div>
<div class="metric">
    <div class="metric-label">目标列</div>
    <div class="metric-value">{tc.extracted_slots.target_column or 'unknown'}</div>
</div>
{primary_metrics_html}
<div class="metric">
    <div class="metric-label">过拟合比</div>
    <div class="metric-value {'warning' if metrics and getattr(metrics, 'overfit_ratio', 0) and getattr(metrics, 'overfit_ratio', 0) > 1.05 else ''}">{getattr(metrics, 'overfit_ratio', 'N/A') if metrics else 'N/A'}</div>
</div>
<div class="metric">
    <div class="metric-label">评估得分</div>
    <div class="metric-value">{getattr(evaluation, 'score', 'N/A') if evaluation else 'N/A'}/100</div>
</div>
<p style="color: #999; margin-top: 30px;">注：由于 {reason_text}，本报告为简化版本。如需完整产物（测试集预测、特征重要性图、模型文件），请稍后重试。</p>
</body>
</html>"""
            
            report_path = artifact_dir / "report.html"
            report_path.write_text(html_content, encoding='utf-8')
            
            files = [ArtifactFile(
                name="report.html",
                path=f"/artifacts/{self.task_id}/artifacts/report.html",
                type="report",
                size=f"{len(html_content) / 1024:.1f} KB",
                desc="简化版评估报告（LLM 服务暂时不可用）"
            )]
            
            # 同时将已有的最佳代码也保存为产物，确保用户至少能拿到代码
            best_code = self.state.best_code or self.state.code
            if best_code:
                code_path = artifact_dir / "pipeline.py"
                code_path.write_text(best_code, encoding='utf-8')
                files.append(ArtifactFile(
                    name="pipeline.py",
                    path=f"/artifacts/{self.task_id}/artifacts/pipeline.py",
                    type="code",
                    size=f"{len(best_code) / 1024:.1f} KB",
                    desc="建模 Pipeline 代码（最佳版本）"
                ))
            
            artifacts = ArtifactInfo(
                files=files,
                report_path=str(report_path)
            )
            
            task_manager.update_task(self.task_id, artifacts=artifacts)
            self._append_log("[FastEngine] 简化产物生成完成")
            logger.info(f"[FastEngine] 简化产物生成完成: {len(files)} 个文件")
            
        except Exception as e:
            logger.exception(f"[FastEngine] 简化产物生成也失败了")
            self._append_log(f"[WARN] 简化产物生成失败: {str(e)}")
    
    # ========== 数据准备 ==========
    
    def _prepare_data(self, tc: TaskConfig):
        """准备数据集"""
        self._set_phase(FastTaskPhase.PLANNING)
        
        datasets = self.data_splitter.prepare_datasets(
            files=[f.model_dump() for f in tc.uploaded_files],
            target_column=tc.extracted_slots.target_column or "target",
            task_type=tc.extracted_slots.task_type or "binary_classification",
            task_id=self.task_id
        )
        self.datasets = datasets
        has_test = datasets.get('test') is not None
        task_manager.update_task(self.task_id, has_test_set=has_test)
        
        logger.info(
            f"[FastEngine] 数据集准备完成: train={datasets['train']}, "
            f"validation={datasets['validation']}, test={datasets.get('test')}"
        )
    
    # ========== 初始代码生成 ==========
    
    def _generate_init_code(self, tc: TaskConfig):
        """生成初始基线代码"""
        self._set_phase(FastTaskPhase.CODING)
        
        code_output = self.plan_coding_agent.generate(
            task_config=tc,
            run_state="INIT",
            context_payload="",
            previous_code=""
        )
        
        self.state.plan = code_output.plan
        self.state.code = code_output.code
        self.state.code_history.append({
            "round": 0,
            "code": code_output.code,
            "type": "init"
        })
        
        # 记录初始代码生成日志到终端
        self._append_log("[Plan & Coding Agent] 生成初始基线代码")
        if code_output.plan:
            self._append_log(f"=== 建模计划 ===\n{code_output.plan}")
        if code_output.raw_response:
            self._append_log(f"=== LLM 原始响应 ===\n{code_output.raw_response}")
        
        logger.info(f"[FastEngine] 初始代码生成完成, code长度={len(code_output.code)}")
    
    # ========== 执行-评估循环 ==========
    
    def _execute_evaluate_loop(self, tc: TaskConfig):
        """
        执行-评估循环
        
        循环体：
        1. 沙箱执行代码
        2. 若失败 → Debug 闭环（最多3次）
        3. 若成功 → Evaluation Agent 评估
        4. 若 AUTO_OPTIMIZE 且轮数 < 3 → 生成优化代码，继续循环
        5. 若 YIELD_TO_USER → 进入 PRESENTING，break
        """
        while True:
            if self._stopped:
                return
            
            # --- RUNNING ---
            self._set_phase(FastTaskPhase.RUNNING)
            
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            result = self.sandbox.execute(
                code=self.state.code,
                data_dir=data_dir,
                task_type=tc.extracted_slots.task_type or "binary_classification"
            )
            
            # 执行失败 → Debug 闭环
            if not result.success:
                if not self._debug_loop(result, tc):
                    return  # Debug 3次都失败，任务结束
                continue  # Debug 成功，重新执行
            
            # 执行成功，保存结果
            self.state.execution_output = result.stdout
            self.state.metrics = result.metrics
            logger.info(
                f"[FastEngine] 沙箱执行成功, val_auc={result.metrics.val_auc if result.metrics else 'N/A'}"
            )
            
            # --- EVALUATING ---
            if self._stopped:
                return
            
            self._set_phase(FastTaskPhase.EVALUATING)
            
            evaluation = self.evaluation_agent.evaluate(
                task_target=f"{tc.extracted_slots.task_type.value} - target={tc.extracted_slots.target_column}",
                metrics=result.metrics,
                optimize_round=self.state.optimize_round,
                max_optimize_rounds=settings.FAST_MAX_OPTIMIZE_ROUNDS,
                execution_output=result.stdout
            )
            self.state.evaluation = evaluation
            
            # 记录 Evaluation Agent 原始响应到日志
            self._append_log("[Evaluation Agent] 评估结果")
            if evaluation.raw_response:
                self._append_log(evaluation.raw_response)
            
            logger.info(f"[FastEngine] 评估决策: {evaluation.decision.value}, score={evaluation.score}")
            
            # --- 更新最佳代码（评分最高者）---
            current_score = evaluation.score or 0
            if current_score > (self.state.best_score or 0):
                self.state.best_code = self.state.code
                self.state.best_score = current_score
                self.state.best_metrics = self.state.metrics
                self.state.best_evaluation = self.state.evaluation
                task_manager.update_task(
                    self.task_id,
                    best_code=self.state.best_code,
                    best_score=self.state.best_score,
                    best_metrics=self.state.best_metrics,
                    best_evaluation=self.state.best_evaluation
                )
                logger.info(f"[FastEngine] 发现更优代码，score={current_score}，已更新 best_code")
            
            # --- 决策分支 ---
            if evaluation.decision == DecisionType.AUTO_OPTIMIZE:
                # 检查是否已达到用户反馈次数上限（若已达上限，强制 presenting）
                if self.state.user_feedback_round >= settings.FAST_MAX_USER_FEEDBACK_ROUNDS:
                    logger.warning(
                        f"[FastEngine] 用户反馈次数已达上限 ({settings.FAST_MAX_USER_FEEDBACK_ROUNDS})，强制结束"
                    )
                    evaluation.decision = DecisionType.YIELD_TO_USER
                    evaluation.report_to_user = (
                        f"已达到最大反馈次数（{settings.FAST_MAX_USER_FEEDBACK_ROUNDS} 轮），"
                        f"将当前最优结果交由您确认。"
                    )
                    evaluation.suggestions_for_coding_agent = None
                # 检查自动优化次数上限
                elif self.state.optimize_round >= settings.FAST_MAX_OPTIMIZE_ROUNDS:
                    logger.warning(
                        f"[FastEngine] 自动优化次数已达上限 ({settings.FAST_MAX_OPTIMIZE_ROUNDS})"
                    )
                    # 强制改为 YIELD_TO_USER
                    evaluation.decision = DecisionType.YIELD_TO_USER
                    evaluation.report_to_user = (
                        f"已达到最大自动优化次数（{settings.FAST_MAX_OPTIMIZE_ROUNDS} 轮），"
                        f"将当前最优结果交由您确认。"
                    )
                    evaluation.suggestions_for_coding_agent = None
                else:
                    # 继续自动优化
                    self.state.optimize_round += 1
                    self._set_phase(FastTaskPhase.OPTIMIZING)
                    
                    logger.info(
                        f"[FastEngine] 开始第 {self.state.optimize_round} 轮自动优化"
                    )
                    
                    code_output = self.plan_coding_agent.generate(
                        task_config=tc,
                        run_state="OPTIMIZE",
                        context_payload=evaluation.suggestions_for_coding_agent or "",
                        previous_code=(self.state.best_code or self.state.code)
                    )
                    self.state.code = code_output.code
                    self.state.code_history.append({
                        "round": self.state.optimize_round,
                        "code": code_output.code,
                        "type": "optimize"
                    })
                    
                    # 记录 LLM 原始响应到日志
                    self._append_log(f"[Plan & Coding Agent] 第 {self.state.optimize_round} 轮优化代码")
                    if code_output.raw_response:
                        self._append_log(code_output.raw_response)
                    
                    # 继续循环
                    continue
            
            # YIELD_TO_USER → 进入 PRESENTING，结束循环
            if evaluation.decision == DecisionType.YIELD_TO_USER:
                self._set_phase(FastTaskPhase.PRESENTING)
                break
    
    # ========== Debug 闭环 ==========
    
    def _debug_loop(self, result: SandboxResult, tc: TaskConfig) -> bool:
        """
        Debug 闭环
        
        Returns:
            True: Debug 成功，代码已修复
            False: 3次都失败，任务标记为 FAILED
        """
        # 累积所有历史错误信息，防止 LLM 修复了上一个错误又引入上上次的错误
        debug_history = []
        # 记录本次进入 debug 的起始轮次，局部计数从 1 开始显示
        start_debug_round = self.state.debug_round
        
        while self.state.debug_round < settings.FAST_MAX_DEBUG_ROUNDS:
            self.state.debug_round += 1
            # 局部轮次：每次进入 _debug_loop 都从 1 开始计数，避免 optimize 后显示错乱
            local_round = self.state.debug_round - start_debug_round
            
            # 记录本次错误
            current_error = result.error_message or result.stderr or "未知错误"
            debug_history.append(f"第 {local_round} 次执行错误:\n{current_error}")
            
            logger.warning(
                f"[FastEngine] 代码执行失败，开始第 {local_round} 次自动修复"
            )
            
            self._set_phase(FastTaskPhase.CODING)
            
            # 将所有历史错误信息合并传给 LLM
            all_errors = "\n\n".join(debug_history)
            
            code_output = self.plan_coding_agent.generate(
                task_config=tc,
                run_state="DEBUG",
                context_payload=all_errors,
                previous_code=(self.state.best_code or self.state.code)
            )
            # Debug 修复的代码不更新 best_code（未经评估的代码不参与评分比较）
            self.state.code = code_output.code
            self.state.code_history.append({
                "round": local_round,
                "code": code_output.code,
                "type": "debug"
            })
            
            # 同步最新代码到 task_manager，让前端轮询能看到 code 变化
            task_manager.update_task(self.task_id, code=code_output.code)
            
            # 记录 LLM 原始响应到日志
            self._append_log(f"[Plan & Coding Agent] 第 {local_round} 次 Debug 修复")
            if code_output.raw_response:
                self._append_log(code_output.raw_response)
            
            # 重新执行验证
            self._set_phase(FastTaskPhase.RUNNING)
            
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            result = self.sandbox.execute(
                code=self.state.code,
                data_dir=data_dir,
                task_type=tc.extracted_slots.task_type or "binary_classification"
            )
            
            if result.success:
                # 修复成功
                self.state.execution_output = result.stdout
                self.state.metrics = result.metrics
                logger.info(f"[FastEngine] 第 {local_round} 次修复成功")
                return True
            
            # 继续下一轮 debug
            logger.warning(f"[FastEngine] 第 {local_round} 次修复仍失败")
        
        # 5次都失败
        error_msg = (
            f"代码运行失败，经过 {settings.FAST_MAX_DEBUG_ROUNDS} 次自动修复仍未解决。"
            f"建议切换至深度模式进行更深入的探索。"
        )
        self.state.execution_error = error_msg
        self._set_phase(FastTaskPhase.FAILED)
        logger.error(f"[FastEngine] {error_msg}")
        return False
    
    # ========== 工具方法 ==========
    
    def _set_phase(self, phase: FastTaskPhase):
        """更新任务阶段"""
        self.state.phase = phase
        task_manager.update_task(self.task_id, phase=phase)
        logger.info(f"[FastEngine] 任务 {self.task_id} 阶段切换: {phase.value}")


# ========== 全局引擎管理 ==========

_fast_engines: dict = {}
_lock = threading.Lock()


def get_or_create_engine(task_id: str) -> FastEngine:
    """获取或创建引擎实例"""
    with _lock:
        if task_id not in _fast_engines:
            _fast_engines[task_id] = FastEngine(task_id)
        return _fast_engines[task_id]


def remove_engine(task_id: str):
    """移除引擎实例"""
    with _lock:
        _fast_engines.pop(task_id, None)

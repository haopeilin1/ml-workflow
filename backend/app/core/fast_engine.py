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
import re
import shutil
import textwrap
import threading
import time
from typing import Optional
from pathlib import Path

import pandas as pd

from app.config import settings
from app.models.schemas import (
    TaskConfig, FastTaskPhase, DecisionType,
    ExecutionMetrics, EvaluationResult, CodeOutput, LLMConfig,
    ArtifactInfo, ArtifactFile, OptimizationRecord, BestCodeSnapshot
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
    
    def __init__(self, task_id: str, max_wait_seconds: Optional[int] = None):
        self.task_id = task_id
        self.state = task_manager.get_task(task_id)
        if not self.state:
            raise ValueError(f"任务 {task_id} 不存在")
        
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        # 【C方案】外部传入的最大等待时间，用于动态时间预算
        self.max_wait_seconds = max_wait_seconds
        
        # 根据 task_config 中的 llm_config 创建 LLM 客户端
        # 如果前端传入了配置，优先使用；否则使用后端默认配置
        tc = self.state.task_config
        global_llm_config = tc.llm_config
        
        # 支持按阶段独立配置 LLM（agent_llm_configs 供开发/测试使用）
        # 例如：{"plan": LLMConfig(...), "coding": LLMConfig(...), "simple": LLMConfig(...), "evaluation": LLMConfig(...)}
        # 若某阶段未单独配置，则回退到全局 llm_config
        agent_configs = tc.agent_llm_configs or {}
        
        # 【新增】Plan/Coding 分离后的独立 LLM 配置
        # 优先级：agent_llm_configs > .env 配置 > 全局 llm_config
        from app.config import build_eval_llm_config
        
        # Plan Agent（复杂任务计划生成）
        plan_llm_config = (
            agent_configs.get('plan') 
            or self._build_llm_config_from_dict(build_eval_llm_config('plan'))
            or global_llm_config
        )
        # Coding Agent（复杂任务代码生成）
        coding_llm_config = (
            agent_configs.get('coding')
            or self._build_llm_config_from_dict(build_eval_llm_config('coding'))
            or global_llm_config
        )
        # Unified Agent（简单任务单步 PlanCoding）
        unified_llm_config = (
            agent_configs.get('unified')
            or self._build_llm_config_from_dict(build_eval_llm_config('unified'))
            or global_llm_config
        )
        # 评估 Agent（EvaluationAgent：代码评审、优化决策、Debug根因分析）
        eval_llm_config = (
            agent_configs.get('evaluation')
            or self._build_llm_config_from_dict(build_eval_llm_config('evaluation'))
            or global_llm_config
        )
        
        self.plan_coding_agent = PlanCodingAgent(
            llm_client=self._build_llm_client(plan_llm_config),
            plan_llm_client=self._build_llm_client(plan_llm_config),
            coding_llm_client=self._build_llm_client(coding_llm_config),
            unified_llm_client=self._build_llm_client(unified_llm_config),
        )
        self.evaluation_agent = EvaluationAgent(llm_client=self._build_llm_client(eval_llm_config))
        self.sandbox = SandboxExecutor(timeout=settings.SANDBOX_TIMEOUT)
        self.data_splitter = DataSplitter(settings.UPLOAD_DIR, settings.OUTPUT_DIR)
        
        # 数据集路径（由 _prepare_data 填充）
        self.datasets: Optional[dict] = None
        
        # 各阶段耗时记录（供评测系统使用）
        self.timings: Dict[str, float] = {
            "code_generation_seconds": 0.0,
            "sandbox_execution_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "artifact_generation_seconds": 0.0,
        }
        self._timing_stack: List[tuple] = []  # 嵌套计时栈
        
        # 【新增】评估历史记录，用于避免重复犯错，传给 EvaluationAgent
        self._evaluation_history: List[Dict[str, Any]] = []
        
        # 【新增】优化/调试历史记录，传给 PlanAgent 避雷
        # 从持久化状态加载，确保任务重启后历史不丢失
        self._optimization_history: List[Dict[str, Any]] = []
        if self.state and self.state.optimization_history:
            for rec in self.state.optimization_history:
                if isinstance(rec, dict):
                    self._optimization_history.append(rec)
                elif hasattr(rec, 'model_dump'):
                    self._optimization_history.append(rec.model_dump())
                else:
                    self._optimization_history.append(dict(rec))
    
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
                max_tokens=llm_config.max_tokens,
                extra_body=llm_config.extra_body
            )
        return LLMClient.from_settings()
    
    def _build_llm_config_from_dict(self, config_dict: dict) -> Optional[LLMConfig]:
        """将 build_eval_llm_config 返回的字典转为 LLMConfig"""
        if not config_dict:
            return None
        return LLMConfig(
            provider=config_dict.get("provider", "openai"),
            base_url=config_dict.get("base_url", ""),
            api_key=config_dict.get("api_key", ""),
            model=config_dict.get("model", ""),
            temperature=config_dict.get("temperature", 0.3),
            max_tokens=config_dict.get("max_tokens", 4096),
            extra_body=config_dict.get("extra_body")
        )
    
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
            
            # 【C方案】计算执行-评估循环的 deadline（基于外部传入的 max_wait_seconds）
            import time as _time
            pipeline_start = _time.time()
            deadline = None
            if self.max_wait_seconds:
                deadline = pipeline_start + self.max_wait_seconds
                logger.info(f"[FastEngine] 任务 {self.task_id} 时间预算: max_wait={self.max_wait_seconds}s, deadline={deadline:.0f}")
            
            # 3. 执行-评估循环
            self._execute_evaluate_loop(tc, deadline=pipeline_start + self.max_wait_seconds if self.max_wait_seconds else None)
            if self._stopped:
                return
            
            # 4. 循环结束后的状态处理
            if self.state.phase == FastTaskPhase.PRESENTING:
                logger.info(f"[FastEngine] 任务 {self.task_id} 进入等待用户反馈阶段")
                # 生成最终产物
                self._generate_artifacts(tc)
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
                # 【修复】has_test_set 会在 _generate_artifacts 中根据实际文件存在性修正
                # 先假设可能有测试集，在产物阶段确认
                self._append_log("正在对测试集进行预测（如存在测试集）...")
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
                previous_code=(self.state.best_code or self.state.code),
                evaluation_history=self._evaluation_history,
                optimization_history=self._optimization_history
            )
            
            # 【框架对齐】用户反馈优化：将函数合并到 best_code 本身（不再注入系统骨架）
            base_code = self.state.best_code or self.state.code
            merged_code = self._replace_functions_in_code(base_code, code_output.code)
            try:
                validated_code = self._validate_llm_code(merged_code, "用户反馈优化代码")
            except ValueError as ve:
                logger.error(f"[FastEngine] 用户反馈优化代码验证失败: {ve}")
                self._append_log(f"[ERROR] 用户反馈优化代码验证失败: {ve}")
                self.state.error_message = f"用户反馈优化代码验证失败: {ve}"
                self._set_phase(FastTaskPhase.ERROR)
                return
            full_code = validated_code
            
            self.state.code = full_code
            self.state.code_history.append({
                "round": self.state.optimize_round + self.state.user_feedback_round,
                "code": full_code,
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
        self._start_timing("artifact_generation_seconds")
        try:
            best_code = self.state.best_code or self.state.code
            if not best_code:
                logger.warning(f"[FastEngine] 无可用代码，生成简化产物")
                self._append_log("[WARN] 无可用代码，生成简化产物")
                self._generate_fallback_artifacts(tc)
                return
            
            # 确定 data_dir
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            artifact_dir = settings.OUTPUT_DIR / self.task_id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            # ========== 【新增】Step 0: 单独生成 predict.py ==========
            self._append_log("[Plan & Coding Agent] 正在单独生成配套预测脚本 predict.py...")
            logger.info(f"[FastEngine] 开始生成 predict.py")
            
            predict_result = [None]
            predict_error = [None]
            
            def _call_predict_worker():
                try:
                    predict_result[0] = self.plan_coding_agent.generate_predict_script(
                        task_config=tc,
                        best_code=best_code,
                        data_dir=str(data_dir)
                    )
                except Exception as e:
                    predict_error[0] = e
            
            predict_thread = threading.Thread(target=_call_predict_worker)
            predict_thread.daemon = True
            predict_thread.start()
            predict_thread.join(timeout=120)  # 最多等待 120 秒
            
            if predict_thread.is_alive():
                logger.warning("[FastEngine] predict.py 生成超时，将跳过")
                self._append_log("[WARN] predict.py 生成超时，跳过")
            elif predict_error[0]:
                logger.warning(f"[FastEngine] predict.py 生成失败: {predict_error[0]}")
                self._append_log(f"[WARN] predict.py 生成失败: {predict_error[0]}")
            elif predict_result[0] and predict_result[0].code:
                # 保存 predict.py 到 artifact_dir
                predict_py_path = artifact_dir / "predict.py"
                predict_py_path.write_text(predict_result[0].code, encoding='utf-8')
                self._append_log(f"[Plan & Coding Agent] predict.py 生成完成, 长度={len(predict_result[0].code)}")
                logger.info(f"[FastEngine] predict.py 已保存到 {predict_py_path}")
            
            # 确认测试集状态（test.csv 在训练阶段即已可用）
            actual_has_test_set = (data_dir / "test.csv").exists()
            if actual_has_test_set != self.state.has_test_set:
                logger.info(f"[FastEngine] 修正 has_test_set: {self.state.has_test_set} -> {actual_has_test_set} (data_dir/test.csv exists={actual_has_test_set})")
                self.state.has_test_set = actual_has_test_set
                task_manager.update_task(self.task_id, has_test_set=actual_has_test_set)
            
            # 1. 生成其他产物代码（带线程级超时，防止 LLM 调用无限挂起）
            self._append_log("[Plan & Coding Agent] 正在调用 LLM 生成产物代码...")
            logger.info(f"[FastEngine] 开始生成产物代码, best_code长度={len(best_code)}, has_test_set={self.state.has_test_set}")
            
            llm_result = [None]
            llm_error = [None]
            
            def _call_llm_worker():
                try:
                    llm_result[0] = self.plan_coding_agent.generate_artifacts(
                        task_config=tc,
                        best_code=best_code,
                        has_test_set=self.state.has_test_set,
                        data_dir=str(data_dir)
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
            
            # 【兜底1】产物代码为空或极短时，尝试重新生成一次
            if not code_output.code or len(code_output.code.strip()) < 50:
                logger.warning(f"[FastEngine] 产物代码为空或极短(长度={len(code_output.code)}), 尝试重新生成")
                self._append_log("[WARN] 产物代码为空，尝试重新生成...")
                retry_result = [None]
                retry_error = [None]
                
                def _retry_worker():
                    try:
                        retry_result[0] = self.plan_coding_agent.generate_artifacts(
                            task_config=tc,
                            best_code=best_code,
                            has_test_set=self.state.has_test_set,
                            data_dir=str(data_dir)
                        )
                    except Exception as e:
                        retry_error[0] = e
                
                retry_thread = threading.Thread(target=_retry_worker)
                retry_thread.daemon = True
                retry_thread.start()
                retry_thread.join(timeout=120)
                
                if retry_thread.is_alive():
                    logger.error("[FastEngine] 产物代码重试生成超时")
                    self._generate_fallback_artifacts(tc, reason="error")
                    return
                if retry_error[0]:
                    logger.error(f"[FastEngine] 产物代码重试生成失败: {retry_error[0]}")
                    self._generate_fallback_artifacts(tc, reason="error")
                    return
                
                code_output = retry_result[0]
                self._append_log(f"[Plan & Coding Agent] 产物代码重试生成完成, 长度={len(code_output.code)}")
                logger.info(f"[FastEngine] 产物代码重试生成完成, 长度={len(code_output.code)}")
                
                if not code_output.code or len(code_output.code.strip()) < 50:
                    logger.warning("[FastEngine] 产物代码重试后仍为空，降级为简化产物")
                    self._generate_fallback_artifacts(tc, reason="error")
                    return
            
            # 2. 沙箱执行产物代码（允许文件写入），失败时自动修复最多5次
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            artifact_dir = settings.OUTPUT_DIR / self.task_id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            debug_round = 0
            max_debug_rounds = 3  # 【约束】产物代码自动修复最多3次
            
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
                logger.error(f"[FastEngine] 产物代码执行失败，第 {debug_round} 次自动修复... 错误: {error_detail}")
                self._append_log(f"[ERROR] 产物代码执行失败 (第{debug_round}次):\n{error_detail}")
                self._append_log(f"[FastEngine] 正在第 {debug_round} 次修复产物代码...")
                
                fix_result = [None]
                fix_error = [None]
                
                def _call_fix_worker():
                    try:
                        fix_result[0] = self.plan_coding_agent.generate_artifacts(
                            task_config=tc,
                            best_code=best_code,
                            has_test_set=self.state.has_test_set,
                            error_message=error_detail,
                            data_dir=str(data_dir)
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
            
            # 【加固】产物代码未生成 test_predictions.csv 时，从训练阶段 fallback
            test_pred_path = artifact_dir / "test_predictions.csv"
            if not test_pred_path.exists():
                best_pred_src = data_dir / "best_test_predictions.csv"
                pred_src = data_dir / "test_predictions.csv"
                if best_pred_src.exists():
                    import shutil
                    shutil.copy2(best_pred_src, test_pred_path)
                    logger.info(f"[FastEngine] 产物代码未生成 test_predictions.csv，从 best_test_predictions.csv fallback")
                    self._append_log("[FastEngine] 产物未生成测试预测，已使用训练阶段最佳预测")
                elif pred_src.exists():
                    import shutil
                    shutil.copy2(pred_src, test_pred_path)
                    logger.info(f"[FastEngine] 产物代码未生成 test_predictions.csv，从 test_predictions.csv fallback")
                    self._append_log("[FastEngine] 产物未生成测试预测，已使用训练阶段预测")
            
            # 4. 读取测试集预测（含格式验证）
            if test_pred_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(test_pred_path)
                    
                    # 【格式验证】记录基本信息到日志
                    n_rows = len(df)
                    cols = list(df.columns)
                    has_pred = 'prediction' in cols
                    has_prob = any(c.lower() in ('probability', 'prob', 'proba', 'score') for c in cols)
                    pred_range = ""
                    if has_pred and n_rows > 0:
                        try:
                            pred_min = float(df['prediction'].min())
                            pred_max = float(df['prediction'].max())
                            pred_range = f"prediction range=[{pred_min:.4f}, {pred_max:.4f}]"
                        except (ValueError, TypeError):
                            # 字符串标签（如多分类的 'C'/'CL'/'D'）
                            uniq_vals = df['prediction'].dropna().unique()[:5]
                            pred_range = f"prediction unique={list(uniq_vals)}..."
                    logger.info(f"[FastEngine] 测试集预测文件验证: rows={n_rows}, cols={cols}, has_pred={has_pred}, has_prob={has_prob}, {pred_range}")
                    self._append_log(f"[FastEngine] 测试预测文件: {n_rows} 行, 列={cols}")
                    
                    if not has_pred:
                        logger.warning(f"[FastEngine] test_predictions.csv 缺少 prediction 列，列名={cols}")
                        self._append_log(f"[WARN] 测试预测文件缺少 prediction 列")
                    
                    # 取前 50 条作为预览
                    preview = df.head(50)
                    predictions = []
                    for idx, row in preview.iterrows():
                        pred_dict = {"id": idx}
                        # 尝试找到预测列（可能是概率值 0~1，也可能是 0/1 标签）
                        if 'prediction' in row:
                            try:
                                raw_pred = float(row['prediction'])
                                # 如果是概率值（0~1 之间），四舍五入为 0/1
                                if 0 <= raw_pred <= 1:
                                    pred_dict["pred"] = round(raw_pred)
                                    pred_dict["prob"] = round(raw_pred, 4)
                                else:
                                    pred_dict["pred"] = int(raw_pred)
                            except (ValueError, TypeError):
                                # 字符串标签（多分类）
                                pred_dict["pred"] = str(row['prediction'])
                        # 如果存在独立的概率列（proba/prob/score），也一并读取
                        prob_cols = [c for c in df.columns if c.lower() != 'prediction' and ('prob' in c.lower() or 'proba' in c.lower() or 'score' in c.lower())]
                        if prob_cols:
                            pred_dict["prob"] = round(float(row[prob_cols[0]]), 4)
                        predictions.append(pred_dict)
                    artifacts.test_predictions = predictions
                except Exception as e:
                    logger.warning(f"[FastEngine] 读取测试集预测失败: {e}")
            
            # 5. 读取特征重要性（过滤掉目标列，防止数据泄露）
            fi_path = artifact_dir / "feature_importance.csv"
            if fi_path.exists():
                try:
                    import pandas as pd
                    fi_df = pd.read_csv(fi_path)
                    target_col = tc.extracted_slots.target_column
                    # 归一化目标列名用于匹配（忽略空格/下划线差异）
                    def _norm_name(n):
                        return str(n).lower().replace('_', ' ').replace('-', ' ').strip()
                    norm_target = _norm_name(target_col) if target_col else ''
                    fi_list = []
                    for _, row in fi_df.iterrows():
                        name = str(row.get('name', row.iloc[0]))
                        # 过滤掉目标列（支持空格/下划线/横线差异）
                        if norm_target and _norm_name(name) == norm_target:
                            continue
                        fi_list.append({
                            "name": name,
                            "importance": round(float(row.get('importance', row.iloc[1])), 4)
                        })
                    artifacts.feature_importance = fi_list
                except Exception as e:
                    logger.warning(f"[FastEngine] 读取特征重要性失败: {e}")
            
            # 6. 报告路径
            report_path = artifact_dir / "report.html"
            if report_path.exists():
                artifacts.report_path = f"/artifacts/{self.task_id}/artifacts/report.html"
            
            # 6.5 生成产物说明 notes：检查是否有产物被跳过
            expected_files = {
                'report.html': '可视化评估报告',
                'feature_importance.png': '特征重要性可视化图',
                'feature_importance.csv': '特征重要性数据',
                'test_predictions.csv': '测试集预测结果',
                'model.pkl': '模型文件',
                'pipeline.py': 'Pipeline 代码',
                'predict.py': '配套预测脚本'
            }
            actual_names = {f.name for f in files}
            missing = [desc for name, desc in expected_files.items() if name not in actual_names]
            if missing:
                artifacts.notes = f"本次产物生成已跳过以下项目（数据集较大或生成超时）：{', '.join(missing)}。核心模型与指标已就绪。"
            
            # 7. 更新状态
            task_manager.update_task(self.task_id, artifacts=artifacts)
            self._append_log("[FastEngine] 产物生成完成")
            self._append_log(f"产物文件: {[f.name for f in files]}")
            if artifacts.notes:
                self._append_log(f"[NOTE] {artifacts.notes}")
            
            logger.info(f"[FastEngine] 产物生成完成: {len(files)} 个文件")
            
        except Exception as e:
            logger.exception(f"[FastEngine] 产物生成异常")
            self._append_log(f"[WARN] 产物生成异常: {str(e)}")
            self._generate_fallback_artifacts(tc)
        finally:
            self._end_timing("artifact_generation_seconds")
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
            
            reason_map = {
                "timeout": "LLM 调用超时（600秒）",
                "fix_timeout": "产物代码修复超时",
                "fix_error": "产物代码修复失败",
                "debug_max": "产物代码调试达到上限",
                "error": "LLM 调用失败"
            }
            reason_text = reason_map.get(reason, "LLM 调用失败")
            notes = f"由于 {reason_text}，以下产物被跳过：特征重要性图、测试集预测、完整评估报告、模型文件。已降级为简化产物。"
            
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
.notice {{ background: #fff3cd; border-left: 4px solid #ff9800; padding: 12px 16px; margin: 15px 0; color: #856404; }}
</style>
</head>
<body>
<h1>🤖 模型评估报告</h1>
<div class="notice">
    <strong>📋 产物说明：</strong>{notes}
</div>
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
<p style="color: #999; margin-top: 30px;">注：由于 {reason_text}，本报告为简化版本。如需完整产物，请稍后重试。</p>
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
            
            # 【加固】fallback 时保留训练阶段已有的测试集预测和模型
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            # 优先用 best_test_predictions.csv（最佳模型对应的快照），否则用 test_predictions.csv
            best_pred_src = data_dir / "best_test_predictions.csv"
            pred_src = data_dir / "test_predictions.csv"
            pred_dst = artifact_dir / "test_predictions.csv"
            if not pred_dst.exists():
                if best_pred_src.exists():
                    import shutil
                    shutil.copy2(best_pred_src, pred_dst)
                    files.append(ArtifactFile(
                        name="test_predictions.csv",
                        path=f"/artifacts/{self.task_id}/artifacts/test_predictions.csv",
                        type="data",
                        size=f"{pred_dst.stat().st_size / 1024:.1f} KB",
                        desc="测试集预测结果（来自训练阶段最佳模型）"
                    ))
                    notes += " 测试集预测已保留（训练阶段最佳模型）。"
                    logger.info(f"[FastEngine] fallback 时从 best_test_predictions.csv 复制测试预测")
                elif pred_src.exists():
                    import shutil
                    shutil.copy2(pred_src, pred_dst)
                    files.append(ArtifactFile(
                        name="test_predictions.csv",
                        path=f"/artifacts/{self.task_id}/artifacts/test_predictions.csv",
                        type="data",
                        size=f"{pred_dst.stat().st_size / 1024:.1f} KB",
                        desc="测试集预测结果（来自训练阶段）"
                    ))
                    notes += " 测试集预测已保留（训练阶段）。"
                    logger.info(f"[FastEngine] fallback 时从 test_predictions.csv 复制测试预测")
            
            # 保留最佳模型文件
            model_src = data_dir / "best_model.pkl"
            model_dst = artifact_dir / "model.pkl"
            if not model_dst.exists() and model_src.exists():
                import shutil
                shutil.copy2(model_src, model_dst)
                files.append(ArtifactFile(
                    name="model.pkl",
                    path=f"/artifacts/{self.task_id}/artifacts/model.pkl",
                    type="model",
                    size=f"{model_dst.stat().st_size / 1024:.1f} KB",
                    desc="训练好的模型文件（最佳版本）"
                ))
                logger.info(f"[FastEngine] fallback 时从 best_model.pkl 复制模型")
            
            artifacts = ArtifactInfo(
                files=files,
                report_path=str(report_path),
                notes=notes
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
            task_id=self.task_id,
            is_time_series=tc.extracted_slots.is_time_series or False,
            data_profile=tc.data_profile
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
        """生成初始基线代码（模板化：系统骨架 + LLM 填充函数）"""
        self._set_phase(FastTaskPhase.CODING)
        self._start_timing("code_generation_seconds")
        
        # 1. 调用 LLM 生成填充函数（preprocess / feature_engineering / build_model）
        code_output = self.plan_coding_agent.generate(
            task_config=tc,
            run_state="INIT",
            context_payload="",
            previous_code=""
        )
        self._end_timing("code_generation_seconds")
        
        # 2. 获取系统骨架代码
        skeleton = self._get_train_skeleton(tc)
        
        # 3. 将 LLM 生成的函数注入到骨架中（先验证语义完整性）
        validated_code = self._validate_llm_code(code_output.code, "初始代码")
        full_code = skeleton.replace("{USER_CODE}", validated_code)
        
        self.state.plan = code_output.plan
        self.state.code = full_code
        self.state.code_history.append({
            "round": 0,
            "code": full_code,
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
    
    def _execute_evaluate_loop(self, tc: TaskConfig, deadline: Optional[float] = None):
        """
        执行-评估循环
        
        循环体：
        1. 沙箱执行代码
        2. 若失败 → Debug 闭环（最多3次）
        3. 若成功 → Evaluation Agent 评估
        4. 若 AUTO_OPTIMIZE 且轮数 < 3 → 生成优化代码，继续循环
        5. 若 YIELD_TO_USER → 进入 PRESENTING，break
        
        Args:
            deadline: 时间预算截止时间点（unix timestamp）。若设置，剩余时间不足时
                      将强制结束优化循环，避免 _wait_for_phase 超时。
        """
        import time as _time
        while True:
            if self._stopped:
                return
            
            # --- RUNNING ---
            self._set_phase(FastTaskPhase.RUNNING)
            
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            
            # 执行前备份已有的 best_model.pkl（防止后续轮次覆盖更优模型）
            self._backup_best_model(data_dir)
            
            self._start_timing("sandbox_execution_seconds")
            result = self.sandbox.execute(
                code=self.state.code,
                data_dir=data_dir,
                task_type=tc.extracted_slots.task_type or "binary_classification"
            )
            self._end_timing("sandbox_execution_seconds")
            
            # 执行失败 → Debug 闭环
            if not result.success:
                # 记录详细错误信息到日志
                init_error = result.error_message or result.stderr or "Unknown sandbox error"
                logger.error(f"[FastEngine] 初始代码执行失败: {init_error}")
                self._append_log(f"[ERROR] 初始代码执行失败:\n{init_error}")
                # 执行失败时恢复旧模型（防止Debug过程中产生错误的模型文件覆盖最优模型）
                self._restore_best_model_backup(data_dir)
                if not self._debug_loop(result, tc):
                    return  # Debug 3次都失败，任务结束
                continue  # Debug 成功，重新执行
            
            # 执行成功，保存结果
            self.state.execution_output = result.stdout
            self.state.metrics = result.metrics
            logger.info(
                f"[FastEngine] 沙箱执行成功, val_auc={result.metrics.val_auc if result.metrics else 'N/A'}"
            )
            
            # 【框架对齐】契约检查：关键产物是否存在
            contract_passed, contract_error = self._check_contract(result, Path(data_dir))
            if not contract_passed:
                logger.error(f"[FastEngine] 契约检查失败: {contract_error}")
                self._append_log(f"[ERROR] 契约检查失败: {contract_error}")
                # 构造一个假的 SandboxResult 来触发 debug
                contract_result = SandboxResult(
                    success=False,
                    stdout=result.stdout,
                    stderr="",
                    returncode=1,
                    execution_time=0.0,
                    metrics=result.metrics,
                    error_message=f"契约检查失败: {contract_error}"
                )
                self._restore_best_model_backup(data_dir)
                if not self._debug_loop(contract_result, tc):
                    return
                continue
            
            # 【关键修复】后处理 test_predictions.csv，自动修复格式问题
            self._postprocess_test_predictions(data_dir, tc.extracted_slots.task_type or "binary_classification")
            
            # --- EVALUATING ---
            if self._stopped:
                return
            
            self._set_phase(FastTaskPhase.EVALUATING)
            
            self._start_timing("evaluation_seconds")
            try:
                evaluation = self.evaluation_agent.evaluate(
                    task_target=f"{tc.extracted_slots.task_type.value} - target={tc.extracted_slots.target_column}",
                    metrics=result.metrics,
                    optimize_round=self.state.optimize_round,
                    max_optimize_rounds=settings.FAST_MAX_OPTIMIZE_ROUNDS,
                    execution_output=result.stdout,
                    user_modeling_suggestions=tc.extracted_slots.user_modeling_suggestions,
                    eval_metric=tc.extracted_slots.eval_metric,
                    evaluation_history=self._evaluation_history,
                    current_code=self.state.code or ""
                )
            except Exception as eval_err:
                # 【修复】EvaluationAgent 异常隔离：网络/IO/解析异常不应中断主流程
                logger.error(f"[FastEngine] EvaluationAgent 调用失败: {eval_err}")
                self._append_log(f"[WARN] EvaluationAgent 调用失败: {eval_err}，使用默认评估结果继续")
                evaluation = EvaluationResult(
                    evaluation_analysis=f"EvaluationAgent 调用失败: {eval_err}",
                    decision=DecisionType.YIELD_TO_USER,
                    score=0,
                    suggestions_for_coding_agent="",
                    method_summary="评估阶段异常，建议检查代码正确性",
                    dimension_scores=[]
                )
            self._end_timing("evaluation_seconds")
            self.state.evaluation = evaluation
            
            # 【新增】将本轮评估结果追加到历史记录，供下一轮使用
            self._evaluation_history.append({
                "round": self.state.optimize_round,
                "score": evaluation.score,
                "decision": evaluation.decision.value if evaluation.decision else None,
                "suggestions_for_coding_agent": evaluation.suggestions_for_coding_agent,
                "method_summary": evaluation.method_summary,
                "dimension_scores": [
                    {"name": ds.name, "score": ds.score, "reason": ds.reason}
                    for ds in (evaluation.dimension_scores or [])
                ]
            })
            
            # --- 更新最佳代码（评分最高者）---
            current_score = evaluation.score or 0
            
            # 【框架对齐】记录本轮优化/调试历史
            run_type = "init" if self.state.optimize_round == 0 else "optimize"
            opt_record = {
                "round": self.state.optimize_round,
                "run_type": run_type,
                "code": self.state.code or "",
                "plan": evaluation.method_summary,
                "success": True,
                "metrics": self.state.metrics.model_dump() if self.state.metrics else None,
                "evaluation": evaluation.model_dump() if evaluation else None,
                "error_message": None,
                "score": evaluation.score,
                "is_best": current_score > (self.state.best_score or 0)
            }
            self._optimization_history.append(opt_record)
            # 同步到持久化状态
            self.state.optimization_history = [
                OptimizationRecord(**r) for r in self._optimization_history
            ]
            task_manager.update_task(self.task_id, optimization_history=self.state.optimization_history)
            
            # 记录 Evaluation Agent 原始响应到日志
            self._append_log("[Evaluation Agent] 评估结果")
            if evaluation.raw_response:
                self._append_log(evaluation.raw_response)
            
            logger.info(f"[FastEngine] 评估决策: {evaluation.decision.value}, score={evaluation.score}")
            
            # --- 更新最佳代码（评分最高者）---
            if current_score > (self.state.best_score or 0):
                self.state.best_code = self.state.code
                self.state.best_score = current_score
                self.state.best_metrics = self.state.metrics
                self.state.best_evaluation = self.state.evaluation
                # 【新增】存储 best_snapshot 完整快照，供 PlanAgent optimize 模式使用
                self.state.best_snapshot = BestCodeSnapshot(
                    code=self.state.code,
                    score=current_score,
                    metrics=self.state.metrics,
                    evaluation=self.state.evaluation,
                    optimize_round=self.state.optimize_round,
                    execution_output=self.state.execution_output
                )
                task_manager.update_task(
                    self.task_id,
                    best_code=self.state.best_code,
                    best_score=self.state.best_score,
                    best_metrics=self.state.best_metrics,
                    best_evaluation=self.state.best_evaluation,
                    best_snapshot=self.state.best_snapshot
                )
                logger.info(f"[FastEngine] 发现更优代码，score={current_score}，已更新 best_code 和 best_snapshot")
                # 【新增】快照保存当前轮次的测试集预测（最佳模型对应的最佳预测）
                pred_path = Path(data_dir) / "test_predictions.csv"
                if pred_path.exists():
                    try:
                        best_pred_path = Path(data_dir) / "best_test_predictions.csv"
                        shutil.copy2(pred_path, best_pred_path)
                        logger.info(f"[FastEngine] 已快照保存最佳测试预测: {pred_path}(size={pred_path.stat().st_size}) -> {best_pred_path}")
                    except Exception as e:
                        logger.warning(f"[FastEngine] 快照保存最佳测试预测失败: {e}")
                else:
                    logger.warning(f"[FastEngine][DEBUG] best_score更新时 test_predictions.csv 不存在，无法快照保存 best_test_predictions.csv")
                # 本轮是更优模型，保留新保存的 best_model.pkl（无需恢复）
            else:
                # 本轮得分不优于历史最佳，恢复之前备份的最优模型
                logger.info(f"[FastEngine][DEBUG] 本轮得分({current_score}) <= 最佳({self.state.best_score or 0})，准备恢复备份")
                restored = self._restore_best_model_backup(data_dir)
                if restored:
                    logger.info(f"[FastEngine] 本轮得分({current_score})未超过最佳({self.state.best_score or 0})，已恢复之前保存的最优模型")
                # 【关键修复】同时恢复代码和指标状态到最佳版本，避免下一轮执行低分代码或显示错误指标
                if self.state.best_code:
                    self.state.code = self.state.best_code
                    self.state.metrics = self.state.best_metrics
                    self.state.evaluation = self.state.best_evaluation
                    # 同步恢复 execution_output
                    if self.state.best_snapshot and self.state.best_snapshot.execution_output:
                        self.state.execution_output = self.state.best_snapshot.execution_output
                    # 同步到持久化状态
                    task_manager.update_task(
                        self.task_id,
                        code=self.state.code,
                        metrics=self.state.metrics,
                        evaluation=self.state.evaluation,
                        execution_output=self.state.execution_output
                    )
                    logger.info(f"[FastEngine] 状态已回退到 best_code (score={self.state.best_score or 0})")
            
            # 【关键修复】核心指标缺失检测：optimize 后如果验证指标缺失，说明模型训练完全失败
            # 强制终止优化，避免死循环浪费 token
            if self.state.optimize_round > 0 and evaluation.decision == DecisionType.AUTO_OPTIMIZE:
                metrics = self.state.metrics
                has_core_metric = False
                task_type_val = tc.extracted_slots.task_type
                if task_type_val == "binary_classification":
                    has_core_metric = getattr(metrics, 'val_auc', None) is not None
                elif task_type_val == "multiclass_classification":
                    has_core_metric = getattr(metrics, 'val_f1_macro', None) is not None or getattr(metrics, 'val_accuracy', None) is not None
                elif task_type_val == "regression":
                    has_core_metric = getattr(metrics, 'val_rmse', None) is not None
                
                if not has_core_metric:
                    logger.warning(
                        f"[FastEngine] 第 {self.state.optimize_round} 轮优化后核心验证指标缺失 "
                        f"(val_auc/val_rmse is None)，模型训练失败。强制终止优化。"
                    )
                    self._append_log(
                        f"[WARN] 第 {self.state.optimize_round} 轮优化后核心指标缺失，"
                        f"模型训练完全失败，终止优化循环"
                    )
                    evaluation.decision = DecisionType.YIELD_TO_USER
                    evaluation.report_to_user = (
                        f"第 {self.state.optimize_round} 轮优化后模型未产生有效验证指标，"
                        f"提交当前最优结果（score={self.state.best_score or 0}）。"
                    )
                    evaluation.suggestions_for_coding_agent = None
            
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
                    # 【C方案】动态时间预算：检查剩余时间是否足够完成下一轮
                    if deadline is not None:
                        remaining = deadline - _time.time()
                        round_estimate = getattr(settings, 'FAST_ROUND_ESTIMATE_SECONDS', 300)
                        if remaining < round_estimate:
                            logger.warning(
                                f"[FastEngine] 剩余时间不足（{remaining:.0f}s < {round_estimate}s），"
                                f"强制结束优化循环，当前为第 {self.state.optimize_round} 轮"
                            )
                            self._append_log(
                                f"[WARN] 剩余时间不足（{remaining:.0f}s < 预估{round_estimate}s），"
                                f"强制将当前最优结果呈现给用户"
                            )
                            evaluation.decision = DecisionType.YIELD_TO_USER
                            evaluation.report_to_user = (
                                f"已达到时间预算上限（剩余 {remaining:.0f} 秒不足以完成下一轮优化），"
                                f"将当前最优结果交由您确认。"
                            )
                            evaluation.suggestions_for_coding_agent = None
                        else:
                            logger.info(f"[FastEngine] 时间预算充足（剩余 {remaining:.0f}s），继续第 {self.state.optimize_round + 1} 轮优化")
                    
                    # 若时间预算检查未强制改为 YIELD_TO_USER，则继续自动优化
                    if evaluation.decision == DecisionType.AUTO_OPTIMIZE:
                        self.state.optimize_round += 1
                        self._set_phase(FastTaskPhase.OPTIMIZING)
                        
                        logger.info(
                            f"[FastEngine] 开始第 {self.state.optimize_round} 轮自动优化"
                        )
                    
                    # 【架构变更 v3】PlanAgent 负责生成优化计划，EvaluationAgent 只负责评估+决策
                    # 传入 best_evaluation 和 best_metrics 供 PlanAgent 参考
                    best_eval = self.state.best_evaluation
                    best_metr = self.state.best_metrics
                    
                    # 【框架对齐】OPTIMIZE 阶段传完整的 best_code，让 LLM 基于完整上下文做最小改动
                    optimize_previous_code = self.state.best_code or self.state.code
                    code_output = self.plan_coding_agent.generate(
                        task_config=tc,
                        run_state="OPTIMIZE",
                        context_payload=evaluation.suggestions_for_coding_agent or "",
                        previous_code=optimize_previous_code,
                        evaluation_history=self._evaluation_history,
                        best_evaluation=best_eval,
                        best_metrics=best_metr,
                        optimization_history=self._optimization_history
                    )
                    
                    # 【框架对齐】OPTIMIZE 阶段：将优化后的函数合并到 best_code 本身（不再是注入系统骨架）
                    base_code = self.state.best_code or self.state.code
                    merged_code = self._replace_functions_in_code(base_code, code_output.code)
                    try:
                        validated_code = self._validate_llm_code(merged_code, f"第{self.state.optimize_round}轮优化代码")
                    except ValueError as ve:
                        logger.error(f"[FastEngine] 优化代码验证失败: {ve}")
                        self._append_log(f"[ERROR] 优化代码验证失败: {ve}")
                        # 【框架对齐】记录优化验证失败历史
                        opt_fail_record = {
                            "round": self.state.optimize_round,
                            "run_type": "optimize",
                            "code": code_output.code or "",
                            "plan": None,
                            "success": False,
                            "metrics": None,
                            "evaluation": None,
                            "error_message": f"代码验证失败: {ve}",
                            "score": None,
                            "is_best": False
                        }
                        self._optimization_history.append(opt_fail_record)
                        # 同步到持久化状态
                        self.state.optimization_history = [
                            OptimizationRecord(**r) for r in self._optimization_history
                        ]
                        task_manager.update_task(self.task_id, optimization_history=self.state.optimization_history)
                        continue
                    full_code = validated_code
                    
                    self.state.code = full_code
                    self.state.code_history.append({
                        "round": self.state.optimize_round,
                        "code": full_code,
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
            
            logger.error(f"[FastEngine] 第 {local_round} 次代码执行失败: {current_error}")
            self._append_log(f"[ERROR] 第 {local_round} 次代码执行失败:\n{current_error}")
            
            logger.warning(
                f"[FastEngine] 代码执行失败，开始第 {local_round} 次自动修复"
            )
            
            self._set_phase(FastTaskPhase.CODING)
            
            # 【修复】精简 Debug 上下文：只保留最近2次错误的完整 traceback
            # 之前的错误只保留1句话摘要，避免 LLM 被淹没
            if len(debug_history) <= 2:
                all_errors = "\n\n".join(debug_history)
            else:
                # 早期错误只保留摘要（取第一行作为错误类型）
                summarized = []
                for i, err in enumerate(debug_history[:-2], 1):
                    first_line = err.split('\n')[0] if err else f"第 {i} 次错误"
                    summarized.append(f"[历史错误摘要] {first_line}")
                # 最近2次保留完整 traceback
                all_errors = "\n\n".join(summarized + debug_history[-2:])
            
            self._start_timing("code_generation_seconds")
            # 【框架对齐】DEBUG 阶段传完整的 best_code，让 LLM 从全局视角找 bug
            debug_previous_code = self.state.best_code or self.state.code
            code_output = self.plan_coding_agent.generate(
                task_config=tc,
                run_state="DEBUG",
                context_payload=all_errors,
                previous_code=debug_previous_code
            )
            self._end_timing("code_generation_seconds")
            
            # 【框架对齐】判断 LLM 返回的是完整脚本还是部分函数
            if self._is_full_script(code_output.code):
                # 完整脚本：直接运行，不注入骨架
                logger.info("[FastEngine] DEBUG: LLM 返回完整脚本，直接替换")
                self._append_log("[DEBUG] LLM 返回完整脚本，直接运行")
                try:
                    validated_code = self._validate_llm_code(code_output.code, f"第{local_round}次Debug完整脚本", require_functions=False)
                except ValueError as ve:
                    logger.error(f"[FastEngine] Debug 完整脚本验证失败: {ve}")
                    self._append_log(f"[ERROR] Debug 完整脚本验证失败: {ve}")
                    retry_error = f"完整脚本验证失败: {ve}"
                    logger.error(f"[FastEngine] 第 {local_round} 次修复仍失败: {retry_error}")
                    self._append_log(f"[ERROR] 第 {local_round} 次修复后执行仍失败:\n{retry_error}")
                    continue
                full_code = validated_code
            else:
                # 部分函数：合并到 best_code 本身（不再是注入系统骨架）
                logger.info("[FastEngine] DEBUG: LLM 返回部分函数，合并到 best_code")
                base_code = self.state.best_code or self.state.code
                merged_code = self._replace_functions_in_code(base_code, code_output.code)
                try:
                    validated_code = self._validate_llm_code(merged_code, f"第{local_round}次Debug修复代码")
                except ValueError as ve:
                    logger.error(f"[FastEngine] Debug 代码验证失败: {ve}")
                    self._append_log(f"[ERROR] Debug 代码验证失败: {ve}")
                    retry_error = f"代码验证失败: {ve}"
                    logger.error(f"[FastEngine] 第 {local_round} 次修复仍失败: {retry_error}")
                    self._append_log(f"[ERROR] 第 {local_round} 次修复后执行仍失败:\n{retry_error}")
                    continue
                full_code = validated_code
            
            # Debug 修复的代码不更新 best_code（未经评估的代码不参与评分比较）
            self.state.code = full_code
            self.state.code_history.append({
                "round": local_round,
                "code": full_code,
                "type": "debug"
            })
            
            # 同步最新代码到 task_manager，让前端轮询能看到 code 变化
            task_manager.update_task(self.task_id, code=full_code)
            
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
                # 【框架对齐】记录 debug 成功历史
                debug_ok_record = {
                    "round": local_round,
                    "run_type": "debug",
                    "code": self.state.code or "",
                    "plan": None,
                    "success": True,
                    "metrics": self.state.metrics.model_dump() if self.state.metrics else None,
                    "evaluation": None,
                    "error_message": None,
                    "score": None,
                    "is_best": False
                }
                self._optimization_history.append(debug_ok_record)
                # 同步到持久化状态
                self.state.optimization_history = [
                    OptimizationRecord(**r) for r in self._optimization_history
                ]
                task_manager.update_task(self.task_id, optimization_history=self.state.optimization_history)
                return True
            
            # 继续下一轮 debug
            retry_error = result.error_message or result.stderr or "Unknown sandbox error"
            logger.error(f"[FastEngine] 第 {local_round} 次修复仍失败: {retry_error}")
            self._append_log(f"[ERROR] 第 {local_round} 次修复后执行仍失败:\n{retry_error}")
            # 【框架对齐】记录 debug 失败历史
            debug_record = {
                "round": local_round,
                "run_type": "debug",
                "code": self.state.code or "",
                "plan": None,
                "success": False,
                "metrics": None,
                "evaluation": None,
                "error_message": retry_error,
                "score": None,
                "is_best": False
            }
            self._optimization_history.append(debug_record)
            # 同步到持久化状态
            self.state.optimization_history = [
                OptimizationRecord(**r) for r in self._optimization_history
            ]
            task_manager.update_task(self.task_id, optimization_history=self.state.optimization_history)
        
        # 5次都失败
        # 【兜底2】如果存在 best_code，重置到上一轮可用代码再执行一次
        if self.state.best_code:
            logger.warning(
                f"[FastEngine] DEBUG {settings.FAST_MAX_DEBUG_ROUNDS} 次均失败，尝试回退到 best_code 重新执行"
            )
            self._append_log(
                f"[WARN] DEBUG 达到上限，尝试回退到上一轮最优代码..."
            )
            self.state.code = self.state.best_code
            data_dir = self.datasets["train"].parent if self.datasets else settings.OUTPUT_DIR / self.task_id / "data"
            result = self.sandbox.execute(
                code=self.state.code,
                data_dir=data_dir,
                task_type=tc.extracted_slots.task_type or "binary_classification"
            )
            if result.success:
                self.state.execution_output = result.stdout
                self.state.metrics = result.metrics
                # 【关键修复】回退成功后同步更新 best_snapshot（execution_output 可能已变化）
                if self.state.best_code and self.state.best_snapshot:
                    self.state.best_snapshot = BestCodeSnapshot(
                        code=self.state.best_code,
                        evaluation=self.state.best_evaluation,
                        metrics=self.state.best_metrics,
                        score=self.state.best_score or 0,
                        optimize_round=self.state.best_snapshot.optimize_round,
                        execution_output=result.stdout
                    )
                    task_manager.update_task(self.task_id, best_snapshot=self.state.best_snapshot)
                logger.info("[FastEngine] 回退到 best_code 执行成功")
                self._append_log("[FastEngine] 回退到最优代码执行成功")
                return True
            else:
                fallback_error = result.error_message or result.stderr or "Unknown sandbox error"
                logger.error(f"[FastEngine] 回退到 best_code 执行仍失败: {fallback_error}")
                self._append_log(f"[ERROR] 回退到最优代码执行仍失败:\n{fallback_error}")
        
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
    
    def _start_timing(self, key: str):
        """开始计时某个阶段"""
        self._timing_stack.append((key, time.time()))
    
    def _end_timing(self, key: str):
        """结束计时某个阶段"""
        if self._timing_stack and self._timing_stack[-1][0] == key:
            _, start = self._timing_stack.pop()
            elapsed = time.time() - start
            self.timings[key] = self.timings.get(key, 0.0) + elapsed
            logger.debug(f"[FastEngine] 计时 {key}: +{elapsed:.2f}s, 累计={self.timings[key]:.2f}s")
        else:
            logger.warning(f"[FastEngine] 计时栈不匹配: 期望 {key}, 实际 {self._timing_stack[-1][0] if self._timing_stack else 'empty'}")
    
    def _backup_best_model(self, data_dir) -> bool:
        """
        备份 data_dir 下的 best_model.pkl 和 test_predictions.csv，防止后续轮次覆盖更优模型。
        返回是否成功创建了备份。
        """
        model_path = Path(data_dir) / "best_model.pkl"
        backup_path = Path(data_dir) / "best_model.pkl.bak"
        backed_up = False
        if model_path.exists():
            try:
                shutil.copy2(model_path, backup_path)
                logger.info(f"[FastEngine] 已备份最优模型: {model_path} -> {backup_path} (size={model_path.stat().st_size})")
                backed_up = True
            except Exception as e:
                logger.warning(f"[FastEngine] 备份模型失败: {e}")
        else:
            logger.info(f"[FastEngine][DEBUG] _backup_best_model: best_model.pkl 不存在，无需备份")
        # 【新增】同时备份测试集预测结果
        pred_path = Path(data_dir) / "test_predictions.csv"
        pred_backup = Path(data_dir) / "test_predictions.csv.bak"
        if pred_path.exists():
            try:
                shutil.copy2(pred_path, pred_backup)
                logger.info(f"[FastEngine] 已备份测试预测: {pred_path} -> {pred_backup} (size={pred_path.stat().st_size})")
                backed_up = True
            except Exception as e:
                logger.warning(f"[FastEngine] 备份测试预测失败: {e}")
        else:
            logger.info(f"[FastEngine][DEBUG] _backup_best_model: test_predictions.csv 不存在，无需备份")
        return backed_up
    
    def _restore_best_model_backup(self, data_dir) -> bool:
        """
        从备份恢复 best_model.pkl 和 test_predictions.csv。
        当本轮得分不优于历史最佳时调用，确保 best_model.pkl 始终对应最优模型。
        返回是否成功恢复。
        """
        model_path = Path(data_dir) / "best_model.pkl"
        backup_path = Path(data_dir) / "best_model.pkl.bak"
        restored = False
        if backup_path.exists():
            try:
                shutil.copy2(backup_path, model_path)
                logger.info(f"[FastEngine] 已恢复最优模型备份: {backup_path} -> {model_path} (size={backup_path.stat().st_size})")
                restored = True
            except Exception as e:
                logger.warning(f"[FastEngine] 恢复模型备份失败: {e}")
        else:
            logger.info(f"[FastEngine][DEBUG] _restore_best_model_backup: best_model.pkl.bak 不存在，无法恢复")
        # 【新增】同时恢复测试集预测结果备份
        pred_path = Path(data_dir) / "test_predictions.csv"
        pred_backup = Path(data_dir) / "test_predictions.csv.bak"
        if pred_backup.exists():
            try:
                shutil.copy2(pred_backup, pred_path)
                logger.info(f"[FastEngine] 已恢复测试预测备份: {pred_backup} -> {pred_path} (size={pred_backup.stat().st_size})")
                restored = True
            except Exception as e:
                logger.warning(f"[FastEngine] 恢复测试预测备份失败: {e}")
        else:
            logger.info(f"[FastEngine][DEBUG] _restore_best_model_backup: test_predictions.csv.bak 不存在，无法恢复")
        return restored
    
    def _validate_llm_code(self, code: str, label: str = "LLM代码", require_functions: bool = True) -> str:
        """
        【关键新增】用 AST 检查 LLM 生成的代码是否定义了必需的三个函数。
        如果缺少，抛出 ValueError 让上层捕获并重新生成/重试。
        
        这能提前发现 LLM 漏写函数（如 feature_engineering not defined），
        避免让不完整的代码进入沙箱执行。
        
        Args:
            require_functions: 是否要求必须包含三个函数。DEBUG 完整脚本模式可设为 False。
        """
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"{label} 语法错误: {e}")
        
        if require_functions:
            required_funcs = {"preprocess", "feature_engineering", "build_model"}
            found_funcs = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name in required_funcs:
                        found_funcs.add(node.name)
            
            missing = required_funcs - found_funcs
            if missing:
                raise ValueError(
                    f"{label} 缺少必需函数: {missing}。LLM 必须生成 "
                    f"preprocess(df, mode='train')、feature_engineering(df)、build_model() 三个函数。"
                )
        
        return code
    
    def _has_unsafe_preprocess_state_read(self, func_code: str) -> bool:
        """
        【关键修复】检查 preprocess 函数是否有不安全的 PREPROCESS_STATE 读取。
        
        如果 LLM 在 mode='train' 路径中先读取 PREPROCESS_STATE['xxx'] 再写入，
        会因为字典为空而抛出 KeyError。此检查用于阻止这种不安全的代码覆盖旧代码。
        """
        if 'PREPROCESS_STATE' not in func_code:
            return False
        
        # 如果有 setdefault 或 try/except KeyError，认为是安全的
        if 'setdefault' in func_code or 'KeyError' in func_code:
            return False
        
        # 检查是否有未保护的读取操作
        for line in func_code.split('\n'):
            if 'PREPROCESS_STATE[' in line:
                stripped = line.strip()
                # 排除赋值操作（如 PREPROCESS_STATE['x'] = y）
                if re.search(r"PREPROCESS_STATE\s*\[[^\]]+\]\s*=", stripped):
                    continue
                # 排除 get() 调用
                if '.get(' in stripped:
                    continue
                # 剩下的都是读取操作（如 x = PREPROCESS_STATE['y']），不安全
                return True
        
        return False
    
    def _merge_llm_functions(self, previous_code: str, new_code: str) -> str:
        """
        【关键新增】函数级合并：从 previous_code 中提取缺失的函数，补齐 LLM 返回的不完整代码。
        
        LLM 在 DEBUG/OPTIMIZE 模式下经常只返回修改的函数（如只返回 preprocess），
        此函数自动从 previous_code 中补齐缺失的 feature_engineering 和 build_model，
        避免 DEBUG 因"缺少必需函数"而直接失败。
        """
        import ast
        
        required_funcs = {"preprocess", "feature_engineering", "build_model"}
        
        def extract_functions(code: str):
            """提取代码中的所有顶层函数定义，返回 {name: source_code}"""
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return {}
            
            funcs = {}
            # 只遍历顶层节点（不递归进入类定义或嵌套函数）
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and node.name in required_funcs:
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    lines = code.split('\n')
                    func_source = '\n'.join(lines[start_line:end_line])
                    funcs[node.name] = func_source
            return funcs
        
        previous_funcs = extract_functions(previous_code)
        new_funcs = extract_functions(new_code)
        
        # 【关键修复】安全检查：新代码中的 preprocess 是否有不安全的 PREPROCESS_STATE 读取
        if 'preprocess' in new_funcs:
            new_preprocess = new_funcs['preprocess']
            if self._has_unsafe_preprocess_state_read(new_preprocess):
                logger.warning(
                    f"[FastEngine] 新 preprocess 函数存在不安全的 PREPROCESS_STATE 读取，"
                    f"保留旧版本"
                )
                self._append_log(
                    "[WARN] 新 preprocess 存在不安全的 PREPROCESS_STATE 读取，保留旧版本"
                )
                # 从 new_funcs 中移除 preprocess，强制保留旧版本
                del new_funcs['preprocess']
        
        # 合并：用 new_funcs 覆盖 previous_funcs 中的同名函数
        merged_funcs = {**previous_funcs, **new_funcs}
        
        # 按固定顺序输出函数
        order = ["preprocess", "feature_engineering", "build_model"]
        funcs_code = '\n\n'.join(merged_funcs[name] for name in order if name in merged_funcs)
        
        # 【关键修复】提取 new_code 中的 import 语句，避免合并后丢失依赖
        import_lines = []
        for line in new_code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_lines.append(stripped)
        # 去重并添加
        if import_lines:
            unique_imports = '\n'.join(dict.fromkeys(import_lines))
            merged_code = unique_imports + '\n\n' + funcs_code
        else:
            merged_code = funcs_code
        
        # 日志：报告合并情况
        replaced = [name for name in new_funcs if name in previous_funcs]
        added = [name for name in new_funcs if name not in previous_funcs]
        inherited = [name for name in previous_funcs if name not in new_funcs]
        if replaced:
            logger.info(f"[FastEngine] 函数级合并: 覆盖 {replaced}")
        if inherited:
            logger.info(f"[FastEngine] 函数级合并: 继承 {inherited}")
        if import_lines:
            logger.info(f"[FastEngine] 函数级合并: 保留 {len(dict.fromkeys(import_lines))} 条 import")
        
        return merged_code
    
    def _is_full_script(self, code: str) -> bool:
        """
        判断 LLM 返回的代码是完整脚本还是部分函数。
        
        最可靠的判断标准：
        1. 包含 if __name__ == "__main__": → 完整脚本
        2. AST 解析后，缺少三个必需函数中的任意一个 → 完整脚本（或完全不同的结构）
        3. 否则 → 部分函数（LLM 只返回了修改的函数）
        """
        code_stripped = code.strip()
        
        # 包含 main 块 → 完整脚本
        if '__name__' in code_stripped and '__main__' in code_stripped:
            return True
        
        # 检查是否包含必需的三个函数
        required_funcs = {"preprocess", "feature_engineering", "build_model"}
        found_funcs = set()
        try:
            import ast
            tree = ast.parse(code_stripped)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and node.name in required_funcs:
                    found_funcs.add(node.name)
        except SyntaxError:
            # 语法错误时保守判断为完整脚本（让系统直接运行，失败后再 debug）
            return True
        
        # 如果缺少任意一个必需函数 → 认为是完整脚本
        missing = required_funcs - found_funcs
        if missing:
            return True
        
        return False
    
    def _replace_functions_in_code(self, base_code: str, new_code: str) -> str:
        """
        用 new_code 中的函数替换 base_code 中的同名函数。
        保留 base_code 中的所有其他内容（import、系统逻辑、全局变量、辅助函数）。
        同时把 new_code 中的新 import 补充到 base_code 顶部（去重）。
        """
        import ast
        
        # 空值保护
        if not base_code:
            logger.warning("[FastEngine] _replace_functions_in_code: base_code 为空，返回 new_code")
            return new_code or ""
        if not new_code:
            logger.warning("[FastEngine] _replace_functions_in_code: new_code 为空，返回 base_code")
            return base_code
        
        required_funcs = {"preprocess", "feature_engineering", "build_model"}
        
        # 提取新函数
        new_funcs = {}
        try:
            new_tree = ast.parse(new_code)
            for node in ast.iter_child_nodes(new_tree):
                if isinstance(node, ast.FunctionDef) and node.name in required_funcs:
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    lines = new_code.split('\n')
                    func_lines = lines[start_line:end_line]
                    # 去除函数定义前面的前导空格（防止合并后缩进不一致）
                    first_line = func_lines[0]
                    leading_spaces = len(first_line) - len(first_line.lstrip())
                    if leading_spaces > 0:
                        func_lines = [line[leading_spaces:] if line.startswith(' ' * leading_spaces) else line.lstrip() for line in func_lines]
                    func_source = '\n'.join(func_lines)
                    new_funcs[node.name] = func_source
        except SyntaxError:
            logger.warning("[FastEngine] _replace_functions_in_code: new_code 语法错误，无法提取函数")
            return base_code
        
        if not new_funcs:
            logger.info("[FastEngine] _replace_functions_in_code: new_code 中无目标函数，返回 base_code")
            return base_code
        
        # 安全检查：新 preprocess 是否有不安全的 PREPROCESS_STATE 读取
        if 'preprocess' in new_funcs:
            if self._has_unsafe_preprocess_state_read(new_funcs['preprocess']):
                logger.warning("[FastEngine] 新 preprocess 存在不安全的 PREPROCESS_STATE 读取，保留旧版本")
                self._append_log("[WARN] 新 preprocess 存在不安全的 PREPROCESS_STATE 读取，保留旧版本")
                del new_funcs['preprocess']
        
        # 在 base_code 中找到旧函数的位置并替换
        base_lines = base_code.split('\n')
        replacements = []
        try:
            base_tree = ast.parse(base_code)
            for node in ast.iter_child_nodes(base_tree):
                if isinstance(node, ast.FunctionDef) and node.name in new_funcs:
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    replacements.append((start_line, end_line, node.name))
        except SyntaxError:
            logger.warning("[FastEngine] _replace_functions_in_code: base_code 语法错误，无法替换函数")
            return base_code
        
        # 从后向前替换，避免行号偏移
        replacements.sort(key=lambda x: x[0], reverse=True)
        result_lines = list(base_lines)
        
        replaced_names = set()
        for start, end, name in replacements:
            new_lines = new_funcs[name].split('\n')
            result_lines[start:end] = new_lines
            replaced_names.add(name)
        
        # 对于 base_code 中不存在的函数，追加到末尾
        missing_in_base = set(new_funcs.keys()) - replaced_names
        if missing_in_base:
            result_lines.append('')
            result_lines.append('# 【自动追加】以下函数在 base_code 中未找到，从 new_code 追加')
            for name in missing_in_base:
                result_lines.append('')
                result_lines.extend(new_funcs[name].split('\n'))
            logger.warning(f"[FastEngine] _replace_functions_in_code: base_code 中未找到 {missing_in_base}，已追加到末尾")
        
        # 处理 import：从 new_code 中提取 import，补充到 base_code 顶部（去重）
        new_imports = []
        existing_imports = set()
        for line in base_code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                existing_imports.add(stripped)
        
        for line in new_code.split('\n'):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and stripped not in existing_imports:
                new_imports.append(stripped)
        
        if new_imports:
            # 找到 base_code 中最后一个 import 的位置
            last_import_idx = -1
            for i, line in enumerate(result_lines):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    last_import_idx = i
            
            if last_import_idx >= 0:
                result_lines = (result_lines[:last_import_idx+1] + 
                              [''] + new_imports + 
                              result_lines[last_import_idx+1:])
            else:
                result_lines = new_imports + [''] + result_lines
            
            logger.info(f"[FastEngine] _replace_functions_in_code: 补充 {len(new_imports)} 条新 import")
        
        all_replaced = list(replaced_names | missing_in_base)
        logger.info(f"[FastEngine] 函数级替换: 在 best_code 中处理 {all_replaced}")
        
        return '\n'.join(result_lines)
    
    def _check_contract(self, result: SandboxResult, data_dir: Path) -> tuple[bool, str]:
        """
        契约检查：沙箱执行成功后，检查关键产物是否存在。
        
        契约项：
        1. stdout 中包含 METRICS_JSON_START/END
        2. data/test_predictions.csv 存在
        3. data/best_model.pkl 存在
        
        返回: (是否通过, 失败原因)
        """
        if not result.success:
            return False, "沙箱执行未成功"
        
        stdout = result.stdout or ""
        
        # 检查 METRICS_JSON
        if "METRICS_JSON_START" not in stdout or "METRICS_JSON_END" not in stdout:
            return False, "沙箱输出缺少 METRICS_JSON_START/END 标记，必须在代码末尾添加：\nprint('METRICS_JSON_START')\nprint(json.dumps(metrics))\nprint('METRICS_JSON_END')"
        
        # 检查 test_predictions.csv
        pred_path = data_dir / "test_predictions.csv"
        if not pred_path.exists():
            return False, "未保存 test_predictions.csv，必须在代码中添加：\nresult_df = pd.DataFrame({'id': test['id'], 'prediction': test_preds})\nresult_df.to_csv('data/test_predictions.csv', index=False)"
        
        # 检查 best_model.pkl
        model_path = data_dir / "best_model.pkl"
        if not model_path.exists():
            return False, "未保存 best_model.pkl，必须在代码中添加：\nimport dill\nwith open('data/best_model.pkl', 'wb') as f:\n    dill.dump(model, f)"
        
        return True, ""
    
    def _get_train_skeleton(self, tc: TaskConfig) -> str:
        """
        【模板化改造】生成训练代码骨架。
        
        系统提供完整的数据流、评估、保存、预测输出逻辑。
        LLM 只需要填充 {USER_CODE} 区域（preprocess, feature_engineering, build_model 三个函数）。
        
        Returns:
            骨架代码字符串，包含 {USER_CODE} 占位符
        """
        slots = tc.extracted_slots
        task_type = slots.task_type.value if slots.task_type else "binary_classification"
        target_col = slots.target_column or "target"
        id_col = slots.id_column or "id"
        
        # 根据任务类型选择预测逻辑（测试集预测部分）
        if task_type == "binary_classification":
            test_pred_code = """if hasattr(model, 'predict_proba'):
    test_probs = model.predict_proba(X_test_fe)[:, 1]
else:
    test_probs = model.predict(X_test_fe).astype(float)
test_preds = (test_probs >= 0.5).astype(int)"""
            prob_col_code = "result_df['probability'] = test_probs"
            default_eval_code = """if hasattr(model, 'predict_proba'):
    val_probs = model.predict_proba(X_val_fe)[:, 1]
else:
    val_probs = model.predict(X_val_fe).astype(float)
val_preds = (val_probs >= 0.5).astype(int)
metrics = {
    'val_auc': float(roc_auc_score(y_val, val_probs)),
    'val_accuracy': float(accuracy_score(y_val, val_preds))
}"""
        elif task_type == "multiclass_classification":
            test_pred_code = """try:
    test_probs_all = model.predict_proba(X_test_fe)
except Exception:
    test_probs_all = None
test_preds = model.predict(X_test_fe)
if '_label_encoder' in globals() and _label_encoder is not None:
    test_preds = _label_encoder.inverse_transform(test_preds)"""
            prob_col_code = """if test_probs_all is not None:
    for i, col in enumerate(test_probs_all.T):
        result_df[f'proba_{i}'] = col"""
            default_eval_code = """val_preds = model.predict(X_val_fe)
from sklearn.metrics import f1_score
metrics = {'val_accuracy': float(accuracy_score(y_val, val_preds)), 'val_f1_macro': float(f1_score(y_val, val_preds, average='macro'))}"""
        elif task_type in ("regression", "time_series_forecasting"):
            test_pred_code = """test_preds = model.predict(X_test_fe)
test_probs = test_preds"""
            prob_col_code = "result_df['probability'] = test_probs"
            default_eval_code = """val_preds = model.predict(X_val_fe)
metrics = {
    'val_rmse': float(root_mean_squared_error(y_val, val_preds)),
    'val_mae': float(mean_absolute_error(y_val, val_preds)),
    'val_r2': float(r2_score(y_val, val_preds))
}"""
        else:
            # 默认按二分类处理
            test_pred_code = """test_probs = model.predict_proba(X_test_fe)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)
if '_label_encoder' in globals() and _label_encoder is not None:
    test_preds = _label_encoder.inverse_transform(test_preds)"""
            prob_col_code = "result_df['probability'] = test_probs"
            default_eval_code = """val_probs = model.predict_proba(X_val_fe)[:, 1]
metrics = {'val_auc': float(roc_auc_score(y_val, val_probs))}"""
        
        # 【系统】目标变换逆变换代码：只在 regression 任务中插入，防止分类任务预测值被错误变换
        inverse_transform_code = ""
        if task_type in ("regression", "time_series_forecasting"):
            inverse_transform_code = """
# 【系统】目标变换逆变换（如 LLM 在 PREPROCESS_STATE 中声明了 target_transform）
# 兼容多种常见键名: target_transform, target_log_transformer
_test_tform = PREPROCESS_STATE.get('target_transform') or PREPROCESS_STATE.get('target_log_transformer')
if _test_tform == 'log1p':
    test_preds = np.expm1(test_preds)
elif _test_tform == 'log':
    test_preds = np.exp(test_preds)
elif _test_tform == 'sqrt':
    test_preds = np.square(test_preds)
"""
        
        skeleton = f'''import pandas as pd
import numpy as np
import dill
import json
import re
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error, mean_absolute_error, r2_score, f1_score
warnings.filterwarnings('ignore')

# ========== 全局状态（用于保存预处理参数）==========
PREPROCESS_STATE = {{}}

# ========== LLM 填充区（开始）==========
{{USER_CODE}}
# ========== LLM 填充区（结束）==========

# ========== 数据加载（系统负责）==========
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# 获取目标列和 id 列
target_col = '{target_col}'
id_col = '{id_col}'
if id_col not in test.columns:
    id_col = test.columns[0]

# 检查目标列是否存在
if target_col not in train.columns:
    raise ValueError(f"目标列 '{{target_col}}' 不在训练数据中，可用列: {{list(train.columns)}}")

# ========== 预处理（系统调用 LLM 填充的函数）==========
train_clean = preprocess(train, mode='train')
val_clean = preprocess(val, mode='test')
test_clean = preprocess(test, mode='test')

# 分离特征和目标（兼容 preprocess 是否保留目标列的情况）
if target_col in train_clean.columns:
    y_train = train_clean[target_col]
    X_train = train_clean.drop(columns=[target_col])
else:
    y_train = train[target_col]
    X_train = train_clean

if target_col in val_clean.columns:
    y_val = val_clean[target_col]
    X_val = val_clean.drop(columns=[target_col])
else:
    y_val = val[target_col]
    X_val = val_clean

# 【强制编码】如果原始目标列是字符串/类别，强制从原始数据获取并统一编码
# 这能覆盖 LLM 可能在 preprocess 中对目标列做的任何编码，确保 predict 后可反编码回原始标签
_label_encoder = None
if target_col in train.columns and (train[target_col].dtype == object or str(train[target_col].dtype) == 'category'):
    from sklearn.preprocessing import LabelEncoder
    _label_encoder = LabelEncoder()
    y_train = _label_encoder.fit_transform(train[target_col])
    y_val = _label_encoder.transform(val[target_col])

X_test = test_clean.drop(columns=[target_col], errors='ignore')
if X_test is test_clean:
    X_test = test_clean.copy()

# ========== 特征工程（系统调用 LLM 填充的函数）==========
X_train_fe = feature_engineering(X_train)
if isinstance(X_train_fe, np.ndarray):
    X_train_fe = pd.DataFrame(X_train_fe, index=X_train.index)
X_val_fe = feature_engineering(X_val)
if isinstance(X_val_fe, np.ndarray):
    X_val_fe = pd.DataFrame(X_val_fe, index=X_val.index)
X_test_fe = feature_engineering(X_test)
if isinstance(X_test_fe, np.ndarray):
    X_test_fe = pd.DataFrame(X_test_fe, index=X_test.index)

# ========== 清洗特征名（LGBM/XGBoost 不支持特殊 JSON 字符）==========
for _df in [X_train_fe, X_val_fe, X_test_fe]:
    _df.columns = [re.sub('[^\\\\w]', '_', str(c)) for c in _df.columns]
# 去重列名
for _df in [X_train_fe, X_val_fe, X_test_fe]:
    if _df.columns.duplicated().any():
        _df.columns = [f"{{c}}_{{i}}" if i > 0 else str(c) for i, c in enumerate(_df.columns)]

# ========== 模型训练（系统负责）==========
model = build_model()
# 尝试传入 eval_set（XGBoost/LightGBM 等支持 early stopping 的模型需要）
try:
    model.fit(X_train_fe, y_train, eval_set=[(X_val_fe, y_val)])
except Exception:
    # 第一次 fit 可能因 eval_set 不被支持而失败（如 sklearn 原生模型）
    # 尝试不带 eval_set 的 fit；若仍失败，说明是真正的数据/代码错误，必须抛出
    try:
        model.fit(X_train_fe, y_train)
    except Exception as _fit_err:
        print(f"[FIT_ERROR] {{_fit_err}}")
        raise

# ========== 验证评估（LLM 可覆盖，系统兜底）==========
# 如果 LLM 定义了 evaluate_model()，使用 LLM 的评估逻辑；否则使用系统默认指标
try:
    if 'evaluate_model' in globals():
        metrics = evaluate_model(model, X_val_fe, y_val)
    else:
{textwrap.indent(default_eval_code.strip(), '        ')}
except Exception as e:
    print(f"[EVAL_ERROR] {{e}}")
    metrics = {{}}
    # 尝试最基本的预测来兜底
    try:
        _pred = model.predict(X_val_fe)
        if task_type == "binary_classification" and hasattr(model, 'predict_proba'):
            _prob = model.predict_proba(X_val_fe)[:, 1]
            metrics = {{'val_auc': float(roc_auc_score(y_val, _prob)), 'val_accuracy': float(accuracy_score(y_val, (_prob >= 0.5).astype(int)))}}
        elif task_type == "multiclass_classification":
            metrics = {{'val_accuracy': float(accuracy_score(y_val, _pred)), 'val_f1_macro': float(f1_score(y_val, _pred, average='macro'))}}
        elif task_type in ("regression", "time_series_forecasting"):
            metrics = {{'val_rmse': float(root_mean_squared_error(y_val, _pred)), 'val_mae': float(mean_absolute_error(y_val, _pred)), 'val_r2': float(r2_score(y_val, _pred))}}
    except Exception as e2:
        print(f"[EVAL_FALLBACK_ERROR] {{e2}}")
        metrics = {{}}

# ========== 测试预测（系统保证格式）==========
# 注意：如果前面的代码（特征工程/model.fit）有 bug，这里会抛出异常
# 这是正确的行为——错误应该被暴露，让 DEBUG 循环去修复根因，而不是用假数据掩盖
{test_pred_code.strip()}
{inverse_transform_code}

result_df = pd.DataFrame({{
    'id': test[id_col] if id_col in test.columns else range(len(test_preds)),
    'prediction': test_preds,
}})
{prob_col_code}
result_df.to_csv('data/test_predictions.csv', index=False)

# ========== 模型保存（系统保证可序列化）==========
with open('data/best_model.pkl', 'wb') as f:
    dill.dump(model, f)

# ========== 输出指标（系统抓取）==========
print('METRICS_JSON_START')
print(json.dumps(metrics))
print('METRICS_JSON_END')
'''
        return skeleton
    
    def _postprocess_test_predictions(self, data_dir: Path, task_type: str = "binary_classification") -> bool:
        """
        【关键修复】沙箱执行后对 test_predictions.csv 进行后处理，自动修复格式问题。
        
        处理项：
        1. 文件不存在 → 尝试从 eval_predictions.csv 复制
        2. 缺少 prediction 列 → 从 probability 生成（threshold=0.5）或从 test.csv 推断
        3. 缺少 probability 列 → 设为 None（后续 _compute_test_metrics 会处理）
        4. 缺少 id 列 → 从 test.csv 获取
        5. probability 列为整数 0/1 → 记录警告（可能是标签而非概率）
        
        Returns:
            True: 后处理成功（文件存在且包含 prediction 列）
            False: 后处理失败
        """
        pred_path = Path(data_dir) / "test_predictions.csv"
        test_path = Path(data_dir) / "test.csv"
        
        # 1. 文件不存在，尝试从 eval_predictions.csv 复制
        if not pred_path.exists():
            eval_path = Path(data_dir) / "eval_predictions.csv"
            if eval_path.exists():
                try:
                    shutil.copy2(eval_path, pred_path)
                    logger.info(f"[FastEngine] 从 eval_predictions.csv 复制为 test_predictions.csv")
                except Exception as e:
                    logger.warning(f"[FastEngine] 复制 eval_predictions.csv 失败: {e}")
                    return False
            else:
                logger.warning(f"[FastEngine] test_predictions.csv 不存在，且无可用的替代文件")
                return False
        
        # 读取预测文件
        try:
            pred_df = pd.read_csv(pred_path)
        except Exception as e:
            logger.warning(f"[FastEngine] 读取 test_predictions.csv 失败: {e}")
            return False
        
        # 2. 缺少 id 列 → 从 test.csv 获取
        if 'id' not in pred_df.columns:
            if test_path.exists():
                try:
                    test_df = pd.read_csv(test_path)
                    if 'id' in test_df.columns:
                        pred_df['id'] = test_df['id']
                        logger.info(f"[FastEngine] 从 test.csv 补全 id 列")
                    elif len(test_df) == len(pred_df):
                        pred_df['id'] = test_df.iloc[:, 0]
                        logger.info(f"[FastEngine] 从 test.csv 第一列补全 id 列")
                except Exception as e:
                    logger.warning(f"[FastEngine] 从 test.csv 获取 id 列失败: {e}")
            else:
                logger.warning(f"[FastEngine] test.csv 不存在，无法补全 id 列")
        
        # 3. 缺少 prediction 列 → 从 probability 生成
        if 'prediction' not in pred_df.columns:
            if 'probability' in pred_df.columns:
                pred_df['prediction'] = (pred_df['probability'] >= 0.5).astype(int)
                logger.info(f"[FastEngine] 从 probability 列生成 prediction 列")
            else:
                logger.error(f"[FastEngine] test_predictions.csv 缺少 prediction 和 probability 列，无法修复")
                return False
        
        # 4. 缺少 probability 列 → 设为 None（_compute_test_metrics 会处理）
        if 'probability' not in pred_df.columns:
            try:
                pred_df['probability'] = pred_df['prediction'].astype(float)
            except (ValueError, TypeError):
                # prediction 是字符串标签（多分类），无法转 float
                pred_df['probability'] = None
            logger.warning(f"[FastEngine] test_predictions.csv 缺少 probability 列，用 prediction 填充")
        
        # 5. 检查 probability 列是否为整数 0/1（说明可能是标签而非概率）
        if 'probability' in pred_df.columns:
            unique_vals = pred_df['probability'].dropna().unique()
            if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                logger.warning(f"[FastEngine] probability 列只有 0/1 值，说明是硬标签而非概率，AUC 将用标签计算")
        
        # 保存修复后的文件
        try:
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"[FastEngine] test_predictions.csv 后处理完成: shape={pred_df.shape}, columns={list(pred_df.columns)}")
            return True
        except Exception as e:
            logger.warning(f"[FastEngine] 保存修复后的 test_predictions.csv 失败: {e}")
            return False


# ========== 全局引擎管理 ==========

_fast_engines: dict = {}
_lock = threading.Lock()


def get_or_create_engine(task_id: str, max_wait_seconds: Optional[int] = None) -> FastEngine:
    """获取或创建引擎实例"""
    with _lock:
        if task_id not in _fast_engines:
            _fast_engines[task_id] = FastEngine(task_id, max_wait_seconds=max_wait_seconds)
        elif max_wait_seconds is not None:
            # 更新已存在引擎的时间预算
            _fast_engines[task_id].max_wait_seconds = max_wait_seconds
        return _fast_engines[task_id]


def remove_engine(task_id: str):
    """移除引擎实例"""
    with _lock:
        _fast_engines.pop(task_id, None)

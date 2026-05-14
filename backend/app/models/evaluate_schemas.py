"""
评测系统数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.schemas import ExecutionMetrics, LLMConfig, TaskType


class BenchmarkTaskConfig(BaseModel):
    """单个任务的评测配置"""
    task_name: str
    task_dir: str
    train_path: str
    test_path: str
    desc_path: str
    ground_truth_path: str
    target_column: str
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION
    eval_metric: Optional[str] = None
    id_column: Optional[str] = None  # 从 ground_truth 自动推断
    data_profile: Optional[Dict[str, Any]] = None  # 数据画像（供PlanCodingAgent使用）
    complexity_reason: Optional[str] = None  # 复杂度判定原因


class TestSetMetrics(BaseModel):
    """测试集评估指标（根据任务类型动态填充）"""
    auc: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    f1_macro: Optional[float] = None
    log_loss: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None


# 【修复】从 schemas.py 导入统一的 ArtifactInfo，避免字段不一致
from app.models.schemas import ArtifactInfo


class JudgeResult(BaseModel):
    """LLM Judge 评估结果"""
    accepted: bool
    analysis: str
    reason: str
    raw_response: Optional[str] = None


class TimingBreakdown(BaseModel):
    """各阶段耗时分解"""
    intent_recognition_seconds: float = 0.0  # 意图识别耗时
    code_generation_seconds: float = 0.0
    sandbox_execution_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    artifact_generation_seconds: float = 0.0
    test_prediction_seconds: float = 0.0
    total_seconds: float = 0.0


class TokenUsageSummary(BaseModel):
    """Token 消耗汇总"""
    plan_coding_calls: int = 0
    plan_coding_prompt_tokens: int = 0
    plan_coding_completion_tokens: int = 0
    plan_coding_total_tokens: int = 0
    evaluation_calls: int = 0
    evaluation_prompt_tokens: int = 0
    evaluation_completion_tokens: int = 0
    evaluation_total_tokens: int = 0
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


class BenchmarkTaskResult(BaseModel):
    """单个任务单次运行结果"""
    task_name: str
    run_index: int  # 第几次运行（1-3）
    success: bool  # FastEngine 是否成功完成（到达 COMPLETED）
    task_id: Optional[str] = None
    phase: Optional[str] = None
    best_score: Optional[float] = None
    val_metrics: Optional[ExecutionMetrics] = None
    test_metrics: Optional[TestSetMetrics] = None
    dimension_scores: List[Dict[str, Any]] = Field(default_factory=list)  # 各维度评分
    judge_accepted: bool = False
    judge_analysis: Optional[str] = None
    judge_reason: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timing: TimingBreakdown = Field(default_factory=TimingBreakdown)
    token_usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)
    result_dir: Optional[str] = None  # 中间结果保存目录
    artifacts: ArtifactInfo = Field(default_factory=ArtifactInfo)  # 产物生成情况
    
    # 【新增】各环节实际使用的 LLM 追踪（含 fallback 情况）
    llm_usage_trace: Dict[str, Any] = Field(default_factory=dict)  # {
    #   "intent": {"model": "qwen3.6-27b", "provider": "fallback1-local", "calls": 1},
    #   "plan": {"model": "deepseek-v4-pro", "provider": "openai", "calls": 2},
    #   "coding": {"model": "deepseek-v4-pro", "provider": "openai", "calls": 5},
    #   "evaluation": {"model": "qwen3.6-27b", "provider": "fallback1-local", "calls": 4},
    #   "judge": {"model": "qwen3.5-flash", "provider": "openai", "calls": 1}
    # }


class BenchmarkRoundResult(BaseModel):
    """单轮（同一任务多次运行）结果聚合"""
    round_index: int
    task_results: List[BenchmarkTaskResult] = Field(default_factory=list)
    success_rate: float = 0.0  # judge_accepted / total
    avg_best_score: Optional[float] = None
    success_count: int = 0
    fail_count: int = 0
    # 耗时聚合
    avg_duration_seconds: float = 0.0
    min_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    duration_std: float = 0.0
    # Token 消耗聚合
    avg_total_tokens: int = 0
    avg_plan_coding_tokens: int = 0
    avg_evaluation_tokens: int = 0
    # 稳定性
    score_std: float = 0.0
    score_cv: float = 0.0


class BenchmarkReport(BaseModel):
    """完整评测报告"""
    eval_id: str
    benchmark_dir: str
    num_runs: int
    task_names: List[str] = Field(default_factory=list)
    round_results: List[BenchmarkRoundResult] = Field(default_factory=list)
    overall_success_rate: float = 0.0
    total_tasks: int = 0
    total_runs: int = 0
    total_accepted: int = 0
    # 全局聚合指标
    overall_avg_duration_seconds: float = 0.0
    overall_avg_total_tokens: int = 0
    overall_score_std: float = 0.0
    overall_duration_std: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "running"  # running / completed / failed


# ========== API 请求/响应模型 ==========

class StartBenchmarkRequest(BaseModel):
    """启动评测请求"""
    benchmark_dir: str
    num_runs: int = 3
    plan_coding_llm_config: Optional[LLMConfig] = None
    judge_llm_config: Optional[LLMConfig] = None
    max_wait_seconds: int = 600  # 每个任务最大等待时间
    # 【新增】Plan / Coding / Unified Agent 独立 LLM 配置
    # 若未传入，自动回退到 plan_coding_llm_config
    plan_llm_config: Optional[LLMConfig] = None
    coding_llm_config: Optional[LLMConfig] = None
    unified_llm_config: Optional[LLMConfig] = None


class StartBenchmarkResponse(BaseModel):
    """启动评测响应"""
    eval_id: str
    status: str
    message: str
    task_count: int


class BenchmarkStatusResponse(BaseModel):
    """查询评测状态响应"""
    eval_id: str
    status: str
    current_task: Optional[str] = None
    current_run: int = 0
    total_tasks: int = 0
    total_runs: int = 0
    completed_runs: int = 0
    progress_percent: float = 0.0

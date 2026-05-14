"""
Pydantic 数据模型
定义前后端交互的数据结构
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from enum import Enum
from datetime import datetime


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class FileRole(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class FastTaskPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    CODING = "coding"
    RUNNING = "running"
    EVALUATING = "evaluating"
    OPTIMIZING = "optimizing"
    PRESENTING = "presenting"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionType(str, Enum):
    AUTO_OPTIMIZE = "AUTO_OPTIMIZE"
    YIELD_TO_USER = "YIELD_TO_USER"


class UploadedFile(BaseModel):
    """上传文件信息"""
    name: str
    path: str
    role: FileRole = FileRole.UNKNOWN
    size: Optional[int] = None


class ExtractedSlots(BaseModel):
    """意图澄清提取的槽位"""
    target_column: Optional[str] = None
    task_type: Optional[TaskType] = None
    eval_metric: Optional[str] = None
    complexity: Optional[str] = "simple"  # "simple" 或 "complex"，由 Intent Agent 判定
    complexity_reason: Optional[str] = None  # 复杂度判定原因说明
    is_time_series: bool = False  # 是否为时序任务，影响验证集切分方式
    feature_constraints: List[str] = Field(default_factory=list)
    user_modeling_suggestions: Optional[str] = None  # 用户在建模描述中附带的建模建议/偏好


class LLMConfig(BaseModel):
    """LLM 配置（前端传入，支持各 Agent 独立配置预留）"""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4096
    extra_body: Optional[Dict[str, Any]] = None  # 传递给 API 的额外参数（如 enable_thinking）


class TaskConfig(BaseModel):
    """任务配置"""
    extracted_slots: ExtractedSlots
    uploaded_files: List[UploadedFile]
    user_description: Optional[str] = None
    data_profile: Optional[Dict[str, Any]] = None
    llm_config: Optional[LLMConfig] = None
    # 按阶段独立配置 LLM（供开发/测试使用，前端正常使用时无需填写）
    agent_llm_configs: Optional[Dict[str, LLMConfig]] = None


class ExecutionMetrics(BaseModel):
    """沙箱执行返回的评估指标"""
    metric_name: str = ""
    val_score: Optional[float] = None
    val_auc: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_rmse: Optional[float] = None
    train_auc: Optional[float] = None
    train_score: Optional[float] = None
    overfit_ratio: Optional[float] = None
    overfit_severe: bool = False


class DimensionScore(BaseModel):
    """评估维度评分"""
    name: str
    score: float  # 0-100
    weight: float  # 权重，如 0.30
    reason: str


class EvaluationResult(BaseModel):
    """评估Agent输出"""
    evaluation_analysis: str
    decision: DecisionType
    suggestions_for_coding_agent: Optional[str] = None
    report_to_user: Optional[str] = None
    raw_response: Optional[str] = None  # LLM 原始完整响应
    score: Optional[float] = None  # 综合评分 0-100（加权总分）
    dimension_scores: List[DimensionScore] = Field(default_factory=list)  # 各维度评分
    # 【新增】方法总结：对本轮代码使用的方法、策略、模型的结构化总结
    method_summary: Optional[str] = None
    # 【新增】重新规划输出：当 decision=AUTO_OPTIMIZE 时，直接输出结构化计划，跳过 PlanAgent
    replan_output: Optional[str] = None


class LLMUsageInfo(BaseModel):
    """单次 LLM 调用的 Token 消耗信息"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""
    latency_seconds: float = 0.0  # 请求到响应的耗时


class ArtifactFile(BaseModel):
    """产物文件信息"""
    name: str
    path: str
    type: str = "file"  # model, code, data, report, image
    size: str = ""
    desc: str = ""


class ArtifactInfo(BaseModel):
    """产物信息"""
    files: List[ArtifactFile] = Field(default_factory=list)
    test_predictions: Optional[List[Dict[str, Any]]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    report_path: Optional[str] = None
    notes: Optional[str] = None  # 产物生成说明（如跳过哪些报告）
    
    # 【新增】产物完整性评估
    completeness: Optional[str] = None  # full / simplified / partial / minimal / none
    generated_files: List[str] = Field(default_factory=list)  # 实际生成的文件名列表
    
    # 各产物文件存在性标志
    prediction_file: bool = False
    model_file: bool = False
    feature_importance_csv: bool = False
    feature_importance_png: bool = False
    report_html: bool = False
    report_fig_png: bool = False
    predict_script: bool = False
    pipeline_py: bool = False
    residual_png: bool = False
    cluster_png: bool = False
    ts_forecast_png: bool = False


class FastTaskState(BaseModel):
    """快速模式任务状态"""
    task_id: str
    phase: FastTaskPhase = FastTaskPhase.IDLE
    task_config: TaskConfig
    plan: Optional[str] = None
    code: Optional[str] = None
    code_history: List[Dict[str, Any]] = Field(default_factory=list)
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None
    metrics: Optional[ExecutionMetrics] = None
    evaluation: Optional[EvaluationResult] = None
    optimize_round: int = 0
    debug_round: int = 0
    user_feedback_round: int = 0
    logs: List[str] = Field(default_factory=list)  # Agent 过程日志
    best_code: Optional[str] = None  # 历史最高分代码
    best_score: Optional[float] = None  # 历史最高评分
    best_metrics: Optional[ExecutionMetrics] = None  # 历史最佳评估指标
    best_evaluation: Optional[EvaluationResult] = None  # 历史最佳评估结果
    artifacts: Optional[ArtifactInfo] = None  # 最终产物
    has_test_set: bool = False  # 是否有测试集
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserFeedbackRequest(BaseModel):
    """用户反馈请求"""
    satisfied: bool
    suggestion: Optional[str] = None


class StartFastTaskRequest(BaseModel):
    """启动快速模式任务请求"""
    task_config: TaskConfig


class StartFastTaskResponse(BaseModel):
    """启动快速模式任务响应"""
    task_id: str
    phase: FastTaskPhase
    message: str


class TaskStatusResponse(BaseModel):
    """查询任务状态响应"""
    task_id: str
    phase: FastTaskPhase
    metrics: Optional[ExecutionMetrics] = None
    evaluation: Optional[EvaluationResult] = None
    optimize_round: int = 0
    debug_round: int = 0
    user_feedback_round: int = 0
    execution_error: Optional[str] = None
    code: Optional[str] = None
    plan: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    best_code: Optional[str] = None
    artifacts: Optional[ArtifactInfo] = None
    has_test_set: bool = False


class CodeOutput(BaseModel):
    """代码生成产物"""
    plan: str
    code: str
    raw_response: Optional[str] = None  # LLM 原始完整响应


class FileUploadResponse(BaseModel):
    """文件上传响应"""
    file_id: str
    name: str
    role: FileRole
    size: int
    path: str
    message: str

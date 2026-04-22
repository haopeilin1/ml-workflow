/**
 * 全局状态管理
 * 集中管理应用的所有状态，为后续快速/深度模式提供统一的数据入口
 */

const AppState = {
    // 当前阶段
    phase: 'welcome', // welcome | profiling | intent_clarifying | intent_confirmed | mode_selecting | fast_mode | depth_mode

    // 用户输入
    userDescription: '',

    // 上传的文件列表（支持多文件）
    uploadedFiles: [], // { file: File, name, size, type, id }
    activeFileIndex: 0, // 当前主数据集索引

    // 数据画像结果
    dataProfile: null, // { rowCount, colCount, columns[], targetCandidates[], qualityScore, fileSize }

    // 意图澄清
    dialogueHistory: [], // { role: 'user'|'agent', content, timestamp }
    clarificationRound: 0, // 当前澄清轮次 0-based
    maxClarificationRounds: 3,

    // 提取的槽位
    extractedSlots: {
        target_column: null,
        task_type: null,
        eval_metric: null,
        feature_constraints: []
    },

    // 是否可开始建模
    isReadyToBuild: false,

    // 任务是否已确认
    taskConfirmed: false,

    // 当前运行中的进程（用于停止）
    currentProcess: null,

    // 思考中状态
    _thinkingId: null,

    // LLM 配置
    llmConfig: {
        enabled: false,
        provider: localStorage.getItem('mlworkflow_provider') || 'openai', // openai | ollama | local-openai
        apiKey: localStorage.getItem('mlworkflow_api_key') || '',
        baseUrl: localStorage.getItem('mlworkflow_base_url') || 'https://api.openai.com/v1',
        model: localStorage.getItem('mlworkflow_model') || 'gpt-4o-mini'
    },

    // 重置状态（用于重新开始）
    reset() {
        this.phase = 'welcome';
        this.userDescription = '';
        this.uploadedFiles = [];
        this.activeFileIndex = 0;
        this.dataProfile = null;
        this.dialogueHistory = [];
        this.clarificationRound = 0;
        this.extractedSlots = { target_column: null, task_type: null, eval_metric: null, feature_constraints: [] };
        this.isReadyToBuild = false;
        this.taskConfirmed = false;
        this.currentProcess = null;
        this._thinkingId = null;
    },

    // 生成标准化的 taskConfig（供快速/深度模式使用，匹配后端 TaskConfig 格式）
    generateTaskConfig() {
        return {
            extracted_slots: {
                target_column: this.extractedSlots.target_column,
                task_type: this.extractedSlots.task_type,
                eval_metric: this.extractedSlots.eval_metric,
                feature_constraints: this.extractedSlots.feature_constraints || []
            },
            uploaded_files: this.uploadedFiles.map(f => ({
                name: f.name,
                path: f.name,
                role: f.role || 'unknown',
                size: f.size
            })),
            user_description: this.userDescription || '',
            data_profile: this.dataProfile,
            llm_config: {
                provider: this.llmConfig.provider,
                base_url: this.llmConfig.baseUrl,
                api_key: this.llmConfig.apiKey,
                model: this.llmConfig.model,
                temperature: 0.3,
                max_tokens: 4096
            }
        };
    }
};

// 暴露到全局，方便调试和后续模块访问
window.AppState = AppState;

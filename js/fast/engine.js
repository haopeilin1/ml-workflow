/**
 * 快速模式引擎 (Fast Engine)
 * 后端 API 轮询驱动的建模流程控制
 */

const FastEngine = {
    // ========== 配置 ==========
    API_BASE: 'http://localhost:8002',
    POLL_INTERVAL: 1500,

    // ========== 状态 ==========
    state: {
        phase: 'idle',
        taskConfig: null,
        plan: null,
        code: null,
        codeHistory: [],
        executionOutput: null,
        executionError: null,
        metrics: null,
        evaluation: null,
        optimizeRound: 0,
        debugRound: 0,
        userFeedbackRound: 0,
        files: [],
        error: null,
        taskId: null,
        pollingTimer: null,
        // 用于终端去重输出
        _lastLoggedPlan: null,
        _lastLoggedCode: null,
        _lastLoggedEvaluation: null,
        _lastLogIndex: 0  // 后端 logs 数组的已输出位置
    },

    // 常量
    MAX_DEBUG_ROUNDS: 5,
    MAX_OPTIMIZE_ROUNDS: 3,
    MAX_USER_FEEDBACK_ROUNDS: 3,

    // ========== 启动入口 ==========
    async start(taskConfig) {
        Terminal.info('[FastEngine] start() called');
        this._resetState();
        this.state.taskConfig = taskConfig;
        this.state.phase = 'planning';

        AppState.phase = 'fast_mode';
        console.log('[FastEngine] Started:', taskConfig);

        // UI 初始化
        FastUI.onEngineStart();
        Terminal.separator();
        Terminal.system('FastEngine: Starting Fast-Modeling-Pipeline...');
        Terminal.info(`Task: ${taskConfig.extracted_slots?.task_type || 'unknown'}`);
        Terminal.info(`Target: ${taskConfig.extracted_slots?.target_column || 'unknown'}`);
        Terminal.info(`Metric: ${taskConfig.extracted_slots?.eval_metric || 'unknown'}`);

        try {
            // 1. 上传文件到后端
            Terminal.info('Uploading files to backend...');
            const uploadedFiles = await this._uploadFiles(AppState.uploadedFiles);
            Terminal.success('Files uploaded successfully.');

            // 更新 taskConfig 中的 uploaded_files（使用后端返回的路径和 file_id）
            taskConfig.uploaded_files = uploadedFiles;

            // 2. 启动任务
            Terminal.info('Starting backend task...');
            const startRes = await fetch(`${this.API_BASE}/api/tasks/fast/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_config: taskConfig })
            });

            if (!startRes.ok) {
                const errData = await startRes.json();
                throw new Error(errData.detail || `Start failed: ${startRes.status}`);
            }

            const startData = await startRes.json();
            this.state.taskId = startData.task_id;
            Terminal.success(`Task created: ${this.state.taskId}`);

            // 3. 开始轮询状态
            this._startPolling();

        } catch (e) {
            console.error('[FastEngine] Start failed:', e);
            Terminal.error(`Start failed: ${e.message}`);
            this.state.error = e.message;
            FastUI.onFailed(`启动失败: ${e.message}`);
        }
    },

    // ========== 文件上传 ==========
    async _uploadFiles(files) {
        const formData = new FormData();
        files.forEach(f => {
            if (f.file) {
                formData.append('files', f.file, f.name);
            }
        });

        const res = await fetch(`${this.API_BASE}/api/files/upload`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'File upload failed');
        }

        const data = await res.json();
        // 后端返回: [{ file_id, name, role, size, path, message }]
        // 转换为后端 task_config 需要的格式: [{ name, path, role, size }]
        return data.map(f => ({
            name: f.name,
            path: f.path,
            role: f.role,
            size: f.size
        }));
    },

    // ========== 轮询控制 ==========
    _startPolling() {
        this._stopPolling();
        this.state.pollingTimer = setInterval(() => this._poll(), this.POLL_INTERVAL);
    },

    _stopPolling() {
        if (this.state.pollingTimer) {
            clearInterval(this.state.pollingTimer);
            this.state.pollingTimer = null;
        }
    },

    // ========== 状态查询 ==========
    async _poll() {
        if (!this.state.taskId || this._stopped) return;

        try {
            const res = await fetch(`${this.API_BASE}/api/tasks/fast/${this.state.taskId}/status`);
            if (!res.ok) {
                console.warn(`[FastEngine] Poll failed: ${res.status}`);
                return;
            }
            const status = await res.json();
            this._onStatusUpdate(status);
        } catch (e) {
            console.warn('[FastEngine] Poll error:', e);
        }
    },

    // ========== 状态更新处理 ==========
    _onStatusUpdate(status) {
        const phase = status.phase;
        const prevPhase = this.state.phase;
        const prevArtifacts = this.state.artifacts;

        // 更新本地状态
        this.state.phase = phase;
        this.state.metrics = status.metrics;
        this.state.evaluation = status.evaluation;
        this.state.optimizeRound = status.optimize_round;
        this.state.debugRound = status.debug_round;
        this.state.userFeedbackRound = status.user_feedback_round;
        this.state.executionError = status.execution_error;
        this.state.code = status.code || this.state.code;
        this.state.plan = status.plan || this.state.plan;
        this.state.artifacts = status.artifacts || this.state.artifacts;
        this.state.bestCode = status.best_code || this.state.bestCode;

        // 将 Agent 输出打印到右侧终端（去重，避免重复输出）
        this._logAgentOutputs(status);

        // coding 阶段即使 phase 没变，如果 code 变化了也要更新左侧代码卡片（避免 debug 轮次卡片丢失）
        if (phase === 'coding' && status.code && status.code !== this.state._lastSeenCode) {
            this.state._lastSeenCode = status.code;
            FastUI.showCode(status.code, {
                round: this.state.debugRound || this.state.optimizeRound || 0,
                type: this.state.debugRound > 0 ? 'debug' : (this.state.optimizeRound > 0 ? 'optimize' : 'init')
            });
        }

        // debug 开始时的提示已移至 setPhase 动态卡片统一展示，此处不再重复输出

        // 阶段变更时才更新 UI（避免频繁重绘）
        // 对于 completed 阶段，如果产物数据从无到有也要触发渲染，
        // 因为产物可能是在 phase 已变为 completed 之后才写入的。
        let shouldRender = (phase !== prevPhase || phase === 'presenting');
        if (phase === 'completed' && !shouldRender) {
            const hadArtifacts = prevArtifacts && (prevArtifacts.files || []).length > 0;
            const nowHasArtifacts = status.artifacts && (status.artifacts.files || []).length > 0;
            if (!hadArtifacts && nowHasArtifacts) {
                shouldRender = true;
            }
        }
        // 点击满意后正在等待产物生成时，若 phase 未变化则跳过 presenting 重绘，
        // 防止反馈按钮和评估报告反复出现，但日志仍继续更新。
        if (this.state._waitingForArtifacts && phase === prevPhase && phase === 'presenting') {
            shouldRender = false;
        }
        if (shouldRender) {
            this._renderPhase(phase, status);
        }

        // 终态处理
        // presenting 阶段：若正在等待产物生成（点击满意后），保持轮询；否则停止
        if (phase === 'presenting') {
            if (!this.state._waitingForArtifacts) {
                this._stopPolling();
            }
        }
        if (phase === 'failed') {
            this._stopPolling();
        }
        // completed 阶段：产物就绪后才停止轮询
        if (phase === 'completed') {
            const hasArtifacts = status.artifacts && (status.artifacts.files || []).length > 0;
            if (hasArtifacts) {
                this._stopPolling();
            }
            // 产物尚未就绪时保持轮询
        }
    },

    // ========== Agent 输出日志（右侧终端） ==========
    // ========== Agent 输出日志（右侧终端） ==========
    _logAgentOutputs(status) {
        // 1. 输出后端传来的 logs（LLM 原始完整响应）
        if (status.logs && status.logs.length > this.state._lastLogIndex) {
            for (let i = this.state._lastLogIndex; i < status.logs.length; i++) {
                const log = status.logs[i];
                if (log.startsWith('[Plan & Coding Agent]') || log.startsWith('[Evaluation Agent]')) {
                    Terminal.separator();
                    Terminal.system(log);
                } else {
                    Terminal.output(log);
                }
            }
            this.state._lastLogIndex = status.logs.length;
        }

        // 2. 输出沙箱执行错误（如果有）
        if (status.execution_error && status.execution_error !== this.state.executionError) {
            this.state.executionError = status.execution_error;
            Terminal.separator();
            Terminal.error('[Sandbox] 执行错误');
            Terminal.output(status.execution_error);
        }

        // 3. 输出沙箱 stdout（执行输出）
        if (status.execution_output && status.execution_output !== this.state.executionOutput) {
            this.state.executionOutput = status.execution_output;
            Terminal.separator();
            Terminal.system('[Sandbox] 执行输出');
            Terminal.output(status.execution_output);
        }
    },

    _renderPhase(phase, status) {
        switch (phase) {
            case 'planning':
                FastUI.setPhase('planning', { message: 'Plan&Coding Agent 正在分析任务并制定建模计划...' });
                break;

            case 'coding':
                FastUI.setPhase('coding', {
                    message: this.state.debugRound > 0
                        ? '正在修复代码...'
                        : '正在生成完整的 Pipeline 代码...',
                    isDebug: this.state.debugRound > 0,
                    error: this.state.executionError
                });
                // 避免与 _onStatusUpdate 中的 code 变化检测重复调用 showCode
                if (status.code && status.code !== this.state._lastSeenCode) {
                    this.state._lastSeenCode = status.code;
                    FastUI.showCode(status.code, {
                        round: this.state.debugRound || this.state.optimizeRound || 0,
                        type: this.state.debugRound > 0 ? 'debug' : (this.state.optimizeRound > 0 ? 'optimize' : 'init')
                    });
                }
                break;

            case 'running':
                FastUI.setPhase('running', {
                    message: this.state.debugRound > 0
                        ? '修复后的代码重新运行中...'
                        : '代码正在沙盒中运行，训练模型并评估...'
                });
                break;

            case 'evaluating':
                FastUI.setPhase('evaluating', { message: 'Evaluation Agent 正在评估模型效果...' });
                break;

            case 'optimizing':
                if (this.state._lastFeedbackType === 'user_feedback') {
                    FastUI.setPhase('optimizing', {
                        message: `根据您的反馈进行第 ${this.state.userFeedbackRound} 轮调整...`,
                        round: this.state.userFeedbackRound,
                        userSuggestion: this.state._lastFeedbackSuggestion || ''
                    });
                } else {
                    FastUI.setPhase('optimizing', {
                        message: `第 ${this.state.optimizeRound} 轮自动优化中...`,
                        round: this.state.optimizeRound
                    });
                }
                break;

            case 'presenting':
                this._onPresenting(status);
                break;

            case 'completed':
                this._onCompleted(status);
                break;

            case 'failed':
                FastUI.onFailed(status.execution_error || '建模失败');
                break;
        }
    },

    _onPresenting(status) {
        FastUI.setPhase('presenting', { message: '模型已训练完成，请查看结果并确认是否满意' });

        // 展示最佳代码（评分最高者），而非最后一次产生的代码
        const codeToShow = status.best_code || status.code;
        if (codeToShow) {
            FastUI.showCode(codeToShow, { round: this.state.optimizeRound, type: 'present' });
        }

        // 展示评估结果
        if (status.evaluation) {
            FastUI.showEvaluation(status.evaluation);
        }

        // 展示运行输出
        if (this.state.executionOutput) {
            FastUI.showExecutionOutput(this.state.executionOutput, status.metrics);
        }

        // 展示结果面板（presenting 阶段只显示指标和评估报告，不显示测试集预测和 mock 特征重要性）
        FastUI.presentResults({
            metrics: status.metrics,
            evaluation: status.evaluation,
            files: FastMockData.getFiles(),
            featureImportance: [],
            testPredictions: null,
            userFeedbackRound: this.state.userFeedbackRound,
            maxUserFeedbackRounds: this.MAX_USER_FEEDBACK_ROUNDS
        });
        // presenting 后重置反馈类型标记
        this.state._lastFeedbackType = null;
        this.state._lastFeedbackSuggestion = null;
    },

    _onCompleted(status) {
        this.state._waitingForArtifacts = false;
        Terminal.system('User confirmed satisfaction. Finalizing deliverables...');
        const artifacts = status.artifacts || {};
        FastUI.onCompleted({
            code: status.code,
            metrics: status.metrics,
            files: artifacts.files || [],
            featureImportance: artifacts.feature_importance || [],
            testPredictions: artifacts.test_predictions || null,
            reportPath: artifacts.report_path || null
        });
    },

    // ========== 用户反馈处理 ==========
    async handleUserFeedback(type, suggestion) {
        if (!this.state.taskId) return;

        if (type === 'satisfied') {
            Terminal.info('正在生成可视化报告，请稍候...');
            if (this.state.taskConfig?.uploaded_files?.some(f => f.role === 'test')) {
                Terminal.info('正在对测试集进行预测...');
            }
            try {
                const res = await fetch(`${this.API_BASE}/api/tasks/fast/${this.state.taskId}/feedback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ satisfied: true, suggestion: '' })
                });
                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.detail || `Feedback failed (${res.status})`);
                }

                // 不提前设置 completed，让后端真实状态驱动 UI；
                // 继续轮询直到后端产物就绪并返回 completed。
                this.state._waitingForArtifacts = true;
                this._startPolling();
            } catch (e) {
                Terminal.error(`Feedback error: ${e.message}`);
            }
            return;
        }

        // 用户不满意
        this.state.userFeedbackRound++;
        this.state._lastFeedbackType = 'user_feedback';
        this.state._lastFeedbackSuggestion = suggestion;

        // 若已达反馈次数上限，不再发送请求到后端，直接标记失败
        if (this.state.userFeedbackRound > this.MAX_USER_FEEDBACK_ROUNDS) {
            const msg = `用户连续 ${this.MAX_USER_FEEDBACK_ROUNDS} 次不满意。建议切换深度模式。`;
            this.state.error = msg;
            FastUI.onFailed(msg);
            return;
        }

        try {
            const res = await fetch(`${this.API_BASE}/api/tasks/fast/${this.state.taskId}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ satisfied: false, suggestion: suggestion || '用户未填写具体建议' })
            });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Feedback failed (${res.status})`);
            }

            FastUI.setPhase('optimizing', {
                message: `根据您的反馈进行第 ${this.state.userFeedbackRound} 轮调整...`,
                round: this.state.userFeedbackRound,
                userSuggestion: suggestion
            });

            // 轮询等待后续状态更新
            this._startPolling();
        } catch (e) {
            Terminal.error(`Feedback error: ${e.message}`);
        }
    },

    // ========== 重置状态 ==========
    _resetState() {
        this._stopped = false;
        this._stopPolling();
        this.state = {
            phase: 'idle',
            taskConfig: null,
            plan: null,
            code: null,
            codeHistory: [],
            executionOutput: null,
            executionError: null,
            metrics: null,
            evaluation: null,
            optimizeRound: 0,
            debugRound: 0,
            userFeedbackRound: 0,
            files: [],
            error: null,
            taskId: null,
            pollingTimer: null,
            _lastLoggedPlan: null,
            _lastLoggedCode: null,
            _lastLoggedEvaluation: null,
            _lastLogIndex: 0,
            _waitingForArtifacts: false,
            bestCode: null,
            _lastSeenCode: null,
            _lastFeedbackType: null,
            _lastNotifiedDebugRound: 0,
            artifacts: null
        };
    },

    // ========== 停止引擎 ==========
    stop() {
        this._stopped = true;
        this._stopPolling();

        if (this.state.taskId) {
            fetch(`${this.API_BASE}/api/tasks/fast/${this.state.taskId}/stop`, { method: 'POST' })
                .catch(e => console.warn('[FastEngine] Stop request failed:', e));
        }

        this.state.phase = 'idle';
        Terminal.warn('FastEngine: Stopped by user.');
        FastUI.onStoppedByUser();
    }
};

window.FastEngine = FastEngine;

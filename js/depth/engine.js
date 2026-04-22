/**
 * 深度模式引擎 (Depth Engine)
 * MCTS 树搜索前端状态机
 * 负责：倒计时管理、节点探索编排、最优代码跟踪、用户反馈循环
 */

const DepthEngine = {
    // ========== 状态 ==========
    state: {
        phase: 'idle',              // idle | initializing | exploring | time_up | waiting_feedback | continuing | completed | stopped
        taskConfig: null,
        timeLimitMinutes: 60,
        remainingMinutes: 60,
        elapsedMinutes: 0,
        nodeIndex: 0,
        totalNodes: 0,
        currentNode: null,
        bestCode: null,
        bestMetrics: null,
        bestNodeId: null,
        nodeHistory: [],
        extendCount: 0,             // 已使用的时间次数（含初始）
        countdownTimer: null,
        exploreTimer: null,
        _stopped: false,
        _nodeSequence: null
    },

    // 探索节奏配置
    EXPLORE_INTERVAL_MS: 2000,    // 每2秒探索一个节点（演示用）
    MAX_EXTEND_COUNT: 4,          // 总共可设置4次等待时间（初始1次 + 最多延长3次）

    // ========== 启动入口 ==========
    async start(taskConfig, timeLimitMinutes) {
        this._resetState();
        this.state.taskConfig = taskConfig;
        this.state.timeLimitMinutes = timeLimitMinutes;
        this.state.remainingMinutes = timeLimitMinutes;
        this.state.extendCount = 1;  // 初始启动算第1次
        this.state._stopped = false;

        AppState.phase = 'depth_mode';
        console.log('[DepthEngine] Started:', taskConfig, timeLimitMinutes + 'min');

        // 计算总节点数（与时间成正比，但有上下限）
        this.state.totalNodes = Math.max(6, Math.min(25, Math.floor(timeLimitMinutes / 5)));

        // 生成节点序列
        const seq = DepthMockData.getNodeSequence(taskConfig, this.state.totalNodes);
        this.state._nodeSequence = seq.nodes;
        this.state.bestNodeId = seq.bestNodeId;

        // UI 初始化
        DepthUI.onEngineStart(timeLimitMinutes, this.state.extendCount, this.MAX_EXTEND_COUNT);
        Terminal.separator();
        Terminal.system('DepthEngine: Initializing MCTS Tree Search...');
        Terminal.info(`Time budget: ${timeLimitMinutes} minutes`);
        Terminal.info(`Expected nodes: ${this.state.totalNodes}`);
        Terminal.info(`Task: ${taskConfig.extractedSlots?.task_type || 'unknown'}`);
        Terminal.info(`Target: ${taskConfig.extractedSlots?.target_column || 'unknown'}`);

        // 阶段：初始化
        this.state.phase = 'initializing';
        await Utils.sleep(1200);

        if (this.state._stopped) return;

        Terminal.system('MCTS environment ready. Starting exploration...');
        this.state.phase = 'exploring';

        // 启动倒计时和探索
        this._startCountdown();
        this._scheduleNextNode();
    },

    // ========== 重置状态 ==========
    _resetState() {
        if (this.state.countdownTimer) {
            clearInterval(this.state.countdownTimer);
        }
        if (this.state.exploreTimer) {
            clearTimeout(this.state.exploreTimer);
        }
        this.state = {
            phase: 'idle',
            taskConfig: null,
            timeLimitMinutes: 60,
            remainingMinutes: 60,
            elapsedMinutes: 0,
            nodeIndex: 0,
            totalNodes: 0,
            currentNode: null,
            bestCode: null,
            bestMetrics: null,
            bestNodeId: null,
            nodeHistory: [],
            extendCount: 0,
            countdownTimer: null,
            exploreTimer: null,
            _stopped: false,
            _nodeSequence: null
        };
    },

    // ========== 倒计时 ==========
    _startCountdown() {
        const stepMinutes = this.state.timeLimitMinutes / this.state.totalNodes;
        this.state.countdownTimer = setInterval(() => {
            if (this.state._stopped) return;

            this.state.elapsedMinutes += stepMinutes;
            this.state.remainingMinutes = Math.max(0, this.state.timeLimitMinutes - this.state.elapsedMinutes);

            DepthUI.updateCountdown(this.state.remainingMinutes, this.state.elapsedMinutes);

            if (this.state.remainingMinutes <= 0) {
                this._onTimeUp();
            }
        }, this.EXPLORE_INTERVAL_MS);
    },

    // ========== 节点探索调度 ==========
    _scheduleNextNode() {
        if (this.state._stopped) return;
        if (this.state.nodeIndex >= this.state.totalNodes) {
            // 所有节点探索完毕，直接触发时间到
            this._onTimeUp();
            return;
        }

        this.state.exploreTimer = setTimeout(() => {
            this._exploreNextNode();
        }, this.EXPLORE_INTERVAL_MS);
    },

    _exploreNextNode() {
        if (this.state._stopped) return;
        if (this.state.phase === 'time_up' || this.state.phase === 'waiting_feedback') return;

        const node = this.state._nodeSequence[this.state.nodeIndex];
        this.state.nodeIndex++;
        this.state.currentNode = node;
        this.state.nodeHistory.push(node);

        // 终端输出
        const logs = DepthMockData.getTerminalLog(node);
        logs.forEach(line => {
            if (line.includes('❌')) Terminal.error(line);
            else if (line.includes('✅')) Terminal.success(line);
            else Terminal.info(line);
        });

        // 更新最优代码（如果是新最优且成功）
        let isBestUpdate = false;
        if (node.status === 'success' && node.isNewBest) {
            this.state.bestCode = DepthMockData.getBestCode(this.state.taskConfig, node.codeLevel);
            this.state.bestMetrics = node.metrics;
            this.state.bestNodeId = node.id;
            isBestUpdate = true;

            // 自动更新右侧代码面板
            DepthUI.updateBestCode(this.state.bestCode, node.metrics, {
                nodeId: node.id,
                nodeType: node.type,
                totalNodes: this.state.totalNodes
            });
        }

        // UI 更新左侧动态状态卡片
        DepthUI.updateExplorationStatus({
            nodeIndex: this.state.nodeIndex,
            totalNodes: this.state.totalNodes,
            bestNodeId: this.state.bestNodeId,
            bestMetrics: this.state.bestMetrics,
            remainingMinutes: this.state.remainingMinutes,
            elapsedMinutes: this.state.elapsedMinutes,
            currentNode: node
        });

        // 继续下一个
        this._scheduleNextNode();
    },

    // ========== 时间到 ==========
    _onTimeUp() {
        if (this.state.phase === 'time_up' || this.state.phase === 'waiting_feedback') return;

        // 停止计时器
        if (this.state.countdownTimer) {
            clearInterval(this.state.countdownTimer);
            this.state.countdownTimer = null;
        }
        if (this.state.exploreTimer) {
            clearTimeout(this.state.exploreTimer);
            this.state.exploreTimer = null;
        }

        this.state.phase = 'time_up';
        this.state.remainingMinutes = 0;

        Terminal.separator();
        Terminal.system('DepthEngine: Time budget exhausted.');
        Terminal.info(`Explored ${this.state.nodeIndex} / ${this.state.totalNodes} nodes.`);
        if (this.state.bestNodeId) {
            Terminal.success(`Best node: #${this.state.bestNodeId} | val_auc=${this.state.bestMetrics?.val_auc || 'N/A'}`);
        } else {
            Terminal.warn('No successful node found during exploration.');
        }

        // 如果没有最优代码，生成一个基础版本
        if (!this.state.bestCode) {
            this.state.bestCode = DepthMockData.getBestCode(this.state.taskConfig, 1);
            this.state.bestMetrics = DepthMockData._makeMetrics(0.82, 'binary_classification');
        }

        // 确保代码面板有内容
        DepthUI.updateBestCode(this.state.bestCode, this.state.bestMetrics, {
            nodeId: this.state.bestNodeId || 0,
            nodeType: 'final',
            totalNodes: this.state.totalNodes,
            isFinal: true
        });

        // 显示时间到评估
        const stats = {
            totalNodes: this.state.nodeIndex,
            successNodes: this.state.nodeHistory.filter(n => n.status === 'success').length,
            failNodes: this.state.nodeHistory.filter(n => n.status === 'failed').length,
            bestNodeId: this.state.bestNodeId
        };
        DepthUI.showTimeUp(this.state.bestMetrics, stats);

        // 进入等待反馈阶段
        this.state.phase = 'waiting_feedback';
        DepthUI.showFeedbackForm(this.state.extendCount, this.MAX_EXTEND_COUNT);
    },

    // ========== 用户反馈处理 ==========
    async handleUserFeedback(type, newTimeMinutes) {
        if (type === 'satisfied') {
            this.state.phase = 'completed';
            Terminal.system('User confirmed satisfaction. Generating deliverables...');

            const data = {
                code: this.state.bestCode,
                metrics: this.state.bestMetrics,
                files: DepthMockData.getFiles(),
                featureImportance: DepthMockData.getFeatureImportance(),
                testPredictions: DepthMockData.getTestPredictions(),
                report: DepthMockData.getFinalReport(
                    this.state.bestMetrics,
                    this.state.nodeIndex,
                    this.state.nodeHistory.filter(n => n.status === 'success').length
                )
            };

            DepthUI.onCompleted(data);
            return;
        }

        // 用户不满意，检查是否还能延长时间
        if (this.state.extendCount >= this.MAX_EXTEND_COUNT) {
            this.state.phase = 'completed'; // 强制结束
            const msg = `已达到最大延长次数 (${this.MAX_EXTEND_COUNT} 次)。将基于当前最优结果输出产物。`;
            Terminal.warn(msg);

            const data = {
                code: this.state.bestCode,
                metrics: this.state.bestMetrics,
                files: DepthMockData.getFiles(),
                featureImportance: DepthMockData.getFeatureImportance(),
                testPredictions: DepthMockData.getTestPredictions(),
                report: DepthMockData.getFinalReport(
                    this.state.bestMetrics,
                    this.state.nodeIndex,
                    this.state.nodeHistory.filter(n => n.status === 'success').length
                ),
                forcedEnd: true,
                forcedMessage: msg
            };
            DepthUI.onCompleted(data);
            return;
        }

        // 继续探索
        this.state.extendCount++;
        const extendTime = newTimeMinutes || 30;
        this.state.timeLimitMinutes = extendTime;
        this.state.remainingMinutes = extendTime;
        this.state.elapsedMinutes = 0;
        this.state.totalNodes = Math.max(4, Math.min(15, Math.floor(extendTime / 6)));

        // 生成新的节点序列（基于当前最优继续）
        const seq = DepthMockData.getNodeSequence(this.state.taskConfig, this.state.totalNodes);
        this.state._nodeSequence = seq.nodes;
        this.state.nodeIndex = 0;

        Terminal.separator();
        Terminal.system(`DepthEngine: Continuing exploration (extend ${this.state.extendCount - 1}/${this.MAX_EXTEND_COUNT - 1}, +${extendTime}min)`);

        this.state.phase = 'continuing';
        DepthUI.onContinuing(extendTime, this.state.extendCount, this.MAX_EXTEND_COUNT);

        // 重新启动倒计时和探索
        this._startCountdown();
        this._scheduleNextNode();
    },

    // ========== 停止引擎 ==========
    stop() {
        if (this.state._stopped) return;
        this.state._stopped = true;

        if (this.state.countdownTimer) {
            clearInterval(this.state.countdownTimer);
            this.state.countdownTimer = null;
        }
        if (this.state.exploreTimer) {
            clearTimeout(this.state.exploreTimer);
            this.state.exploreTimer = null;
        }

        this.state.phase = 'stopped';
        Terminal.warn('DepthEngine: Stopped by user.');
        DepthUI.onStoppedByUser(this.state.bestMetrics, {
            totalNodes: this.state.nodeIndex,
            bestNodeId: this.state.bestNodeId
        });
    }
};

window.DepthEngine = DepthEngine;

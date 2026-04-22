/**
 * 意图澄清流程控制
 *  orchestrate 完整流程：数据画像 → 意图识别 → 多轮澄清 → 任务确认 → 模式选择
 */

const IntentFlow = {

    // ===================== 阶段一：数据画像（后台静默执行） =====================
    async startDataProfiling() {
        AppState.phase = 'profiling';
        Renderer.updateStage('stage-profile', 'active');
        
        const activeFile = AppState.uploadedFiles[AppState.activeFileIndex];
        
        try {
            // 后台解析并画像，不向用户展示中间过程
            const profiles = await DataProfiler.profileFiles([{ file: activeFile.file, id: activeFile.id }]);
            const profile = profiles[0];
            
            if (profile.error) {
                throw new Error(profile.error);
            }
            
            AppState.dataProfile = profile;
            Renderer.updateStage('stage-profile', 'done');
            
            // 直接进入意图识别，不展示画像结果
            this.startIntentRecognition();
            
        } catch (err) {
            Renderer.renderAgentMessage({
                thought_process: '',
                extracted_slots: { target_column: null, task_type: null, eval_metric: null, feature_constraints: [] },
                is_ready_to_build: false,
                reply_to_user: `数据解析时出错了：${err.message}。请检查文件格式是否正确，或尝试重新上传。`
            }, 0);
        }
    },

    // ===================== 阶段二：意图识别 =====================
    async startIntentRecognition() {
        AppState.phase = 'intent_clarifying';
        Renderer.updateStage('stage-intent', 'active');
        
        // 调用 Agent 首轮分析
        const agentResponse = await IntentAgent.callIntentAgent({
            dataProfile: AppState.dataProfile,
            dialogueHistory: AppState.dialogueHistory,
            userInput: AppState.userDescription
        });
        
        // 更新状态
        AppState.extractedSlots = { ...agentResponse.extracted_slots };
        AppState.isReadyToBuild = agentResponse.is_ready_to_build;
        AppState.dialogueHistory.push({
            role: 'agent',
            content: agentResponse.reply_to_user,
            timestamp: Date.now()
        });
        
        // 渲染
        Renderer.renderAgentMessage(agentResponse, 0);
        
        Terminal.system('IntentAgent: Analyzing user intent...');
        Terminal.info(`Target candidate: ${agentResponse.extracted_slots.target_column || 'unknown (needs clarification)'}`);
        Terminal.info(`Task type guess: ${agentResponse.extracted_slots.task_type || 'pending'}`);
        Terminal.info(`Confidence: ${agentResponse.is_ready_to_build ? 'HIGH' : 'LOW - clarification needed'}`);
        
        if (agentResponse.is_ready_to_build) {
            setTimeout(() => this.showTaskConfirmation(), 800);
        }
    },

    // ===================== 多轮澄清 =====================
    async handleUserSend() {
        const inputEl = document.getElementById('bottom-input');
        const text = inputEl.value.trim();
        if (!text) return;
        
        if (AppState.phase !== 'intent_clarifying') return;
        
        // 渲染用户消息
        Renderer.renderUserReply(text);
        AppState.dialogueHistory.push({
            role: 'user',
            content: text,
            timestamp: Date.now()
        });
        inputEl.value = '';
        
        // 显示思考中
        Renderer.showAgentThinking();
        
        // 延迟模拟网络请求
        AppState.currentProcess = setTimeout(async () => {
            AppState.clarificationRound++;
            
            const agentResponse = await IntentAgent.callIntentAgent({
                dataProfile: AppState.dataProfile,
                dialogueHistory: AppState.dialogueHistory,
                userInput: text
            });
            
            AppState.extractedSlots = { ...agentResponse.extracted_slots };
            AppState.isReadyToBuild = agentResponse.is_ready_to_build;
            AppState.dialogueHistory.push({
                role: 'agent',
                content: agentResponse.reply_to_user,
                timestamp: Date.now()
            });
            
            Renderer.removeAgentThinking();
            Renderer.renderAgentMessage(agentResponse, AppState.clarificationRound);
            
            if (agentResponse.is_ready_to_build) {
                Renderer.updateStage('stage-intent', 'done');
                setTimeout(() => this.showTaskConfirmation(), 600);
            } else if (AppState.clarificationRound >= AppState.maxClarificationRounds - 1) {
                // 已达最大轮次但未 ready，下一轮强制闭环（由 Agent 逻辑处理）
            }
        }, 1200);
    },

    /**
     * 快捷回复发送
     */
    sendQuickReply(text) {
        document.getElementById('bottom-input').value = text;
        this.handleUserSend();
    },

    // ===================== 阶段三：任务确认 =====================
    showTaskConfirmation() {
        AppState.phase = 'intent_confirmed';
        Renderer.updateStage('stage-confirm', 'active');
        Renderer.renderTaskConfirmation(AppState.extractedSlots);
        
        Terminal.separator();
        Terminal.system('IntentAgent: Task clarification complete.');
        Terminal.info(`Target: ${AppState.extractedSlots.target_column} | Task: ${AppState.extractedSlots.task_type} | Metric: ${AppState.extractedSlots.eval_metric}`);
        Terminal.warn('Waiting for user confirmation...');
    },

    confirmTask() {
        AppState.taskConfirmed = true;
        AppState.phase = 'mode_selecting';
        Renderer.updateStage('stage-confirm', 'done');
        Renderer.updateStage('stage-mode', 'active');
        
        // 生成 taskConfig
        window.taskConfig = AppState.generateTaskConfig();
        
        // 渲染模式选择
        Renderer.renderModeSelection(AppState.isReadyToBuild);
        
        Terminal.separator();
        Terminal.success('Task confirmed. Ready for modeling.');
        Terminal.info('Please select Fast Mode or Depth Mode to continue.');
    },

    modifyTask() {
        const html = `
            <div class="self-start bg-amber-50 border border-amber-100 rounded-2xl p-4 text-amber-900 text-sm max-w-[85%] shadow-sm animate-fade-in">
                <div class="font-bold mb-1 flex items-center gap-2">
                    <i class="ph-fill ph-pencil-simple text-amber-500"></i> 请补充说明
                </div>
                <p class="text-amber-700/80">请告诉我需要修改的地方，我会重新分析。</p>
            </div>
        `;
        document.getElementById('system-messages').insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
        
        AppState.phase = 'intent_clarifying';
        AppState.isReadyToBuild = false;
        document.getElementById('bottom-input').placeholder = '请说明需要修改的内容...';
        document.getElementById('bottom-input').focus();
    },

    // ===================== 停止当前进程 =====================
    stop() {
        if (AppState.currentProcess) {
            clearTimeout(AppState.currentProcess);
            AppState.currentProcess = null;
        }
        Renderer.removeAgentThinking();
        Terminal.error('[HALTED] Process terminated by user.');
    }
};

window.IntentFlow = IntentFlow;

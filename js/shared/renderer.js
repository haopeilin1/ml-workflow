/**
 * 通用 UI 渲染组件
 * 左侧对话区的所有卡片渲染函数
 */

const Renderer = {
    // 获取容器
    get chatContainer() { return document.getElementById('chat-container'); },
    get userMessages() { return document.getElementById('user-messages'); },
    get systemMessages() { return document.getElementById('system-messages'); },

    scrollToBottom() {
        const el = this.chatContainer;
        if (el) setTimeout(() => el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }), 50);
    },

    // ===== 用户初始消息 =====
    renderUserInitial(files, description) {
        const container = this.userMessages;
        const fileHtml = files.map((f, i) => `
            <div class="self-end flex items-center bg-[#F3F4F6] rounded-xl overflow-hidden shadow-sm max-w-[85%] animate-fade-in border border-gray-100 ${i > 0 ? 'mt-2' : ''}">
                <div class="bg-[#8CB4FF] text-blue-950 p-3 px-4 text-xs font-bold leading-tight text-center">
                    ${f.name.replace(/\.[^.]+$/, '')}<br>数据
                </div>
                <div class="p-3 px-4 text-xs text-gray-500 leading-tight bg-white">
                    ${f.name}<br>${f.size}
                </div>
            </div>
        `).join('');

        container.innerHTML = fileHtml + `
            <div class="self-end bg-[#F3F4F6] rounded-2xl p-4 text-gray-700 text-sm max-w-[85%] leading-relaxed shadow-sm border border-gray-100 animate-fade-in mt-2">
                ${description.replace(/\n/g, '<br>')}
            </div>
        `;
    },

    // ===== 用户回复消息 =====
    renderUserReply(text) {
        const div = document.createElement('div');
        div.className = 'self-end bg-[#F3F4F6] rounded-2xl p-4 text-gray-700 text-sm max-w-[85%] leading-relaxed shadow-sm border border-gray-100 animate-fade-in';
        div.innerHTML = text.replace(/\n/g, '<br>');
        this.systemMessages.appendChild(div);
        this.scrollToBottom();
    },

    // ===== Loading 卡片 =====
    renderLoading(text, label) {
        const html = `
            <div class="self-start bg-blue-50/40 border border-blue-100 rounded-2xl p-5 text-gray-800 text-sm max-w-[95%] shadow-sm animate-fade-in">
                <div class="font-bold mb-3 flex items-center gap-2 text-blue-900">
                    <i class="ph-fill ph-spinner animate-spin-slow text-blue-500 text-xl"></i>
                    ${label}
                </div>
                <div class="text-gray-500 text-xs">${text}</div>
            </div>
        `;
        this.systemMessages.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    },

    // ===== 数据画像卡片 =====
    renderDataProfile(profile) {
        const numCols = profile.columns.filter(c => c.type === 'numeric').length;
        const catCols = profile.columns.filter(c => c.type === 'categorical').length;
        const textCols = profile.columns.filter(c => c.type === 'text').length;
        const missingCols = profile.columns.filter(c => c.missingRate > 0);
        const targetCols = profile.targetCandidates || [];

        const html = `
            <div class="self-start profile-card border border-blue-100 rounded-2xl p-5 text-gray-800 text-sm max-w-[95%] shadow-sm animate-fade-in">
                <div class="font-bold mb-3 flex items-center gap-2 text-blue-900">
                    <i class="ph-fill ph-chart-bar text-blue-500 text-xl"></i>
                    数据画像
                    <span class="ml-auto text-[10px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full">质量分 ${profile.qualityScore || '--'}</span>
                </div>
                <div class="bg-white/80 backdrop-blur-sm p-4 rounded-xl border border-blue-50 text-xs text-gray-700 space-y-3 shadow-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-400">数据集</span>
                        <span class="font-medium">${profile.rowCount} 行 × ${profile.colCount} 列 · ${profile.fileSize || '--'}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">字段类型</span>
                        <span class="font-medium">数值型 ${numCols} · 分类型 ${catCols} · 文本型 ${textCols}</span>
                    </div>
                    ${missingCols.length > 0 ? `
                    <div class="flex justify-between">
                        <span class="text-gray-400">缺失情况</span>
                        <span class="font-medium">${missingCols.length} 列有缺失</span>
                    </div>` : ''}
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">目标猜测</span>
                        <div class="flex gap-1.5">
                            ${targetCols.slice(0, 3).map(c => `<span class="column-tag target"><i class="ph-fill ph-bullseye"></i>${c}</span>`).join('') || '<span class="text-gray-400">暂不明确</span>'}
                        </div>
                    </div>
                    <div class="pt-2 border-t border-gray-100">
                        <div class="text-gray-400 mb-1.5">全部字段</div>
                        <div class="flex flex-wrap gap-1">
                            ${profile.columns.map(c => {
                                const isTarget = targetCols.includes(c.name);
                                return `<span class="column-tag ${isTarget ? 'target' : ''}" title="类型: ${c.type}, 缺失: ${(c.missingRate * 100).toFixed(1)}%, 唯一值: ${c.uniqueCount}">${c.name}</span>`;
                            }).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.systemMessages.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    },

    // ===== Agent 消息卡片 =====
    renderAgentMessage(agentResponse, round) {
        const isReady = agentResponse.is_ready_to_build;
        const reply = agentResponse.reply_to_user;
        const slots = agentResponse.extracted_slots;

        // 槽位标签
        const slotTags = [];
        if (slots.target_column) slotTags.push(`<span class="slot-badge target"><i class="ph-fill ph-bullseye"></i>${slots.target_column}</span>`);
        if (slots.task_type) slotTags.push(`<span class="slot-badge task"><i class="ph-fill ph-funnel"></i>${Utils.formatTaskType(slots.task_type)}</span>`);
        if (slots.eval_metric) slotTags.push(`<span class="slot-badge metric"><i class="ph-fill ph-chart-line-up"></i>${slots.eval_metric}</span>`);
        if (slots.feature_constraints && slots.feature_constraints.length > 0) {
            slotTags.push(`<span class="slot-badge constraints"><i class="ph-fill ph-x-circle"></i>排除 ${slots.feature_constraints.length} 列</span>`);
        }

        const roundLabel = round > 0 ? `<span class="text-[10px] bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full ml-2">第${round + 1}轮澄清</span>` : '';

        // 快捷回复按钮
        let quickActions = '';
        if (!isReady) {
            const actions = [];
            if (reply.includes('可以吗') || reply.includes('可以吗') || reply.includes('您看')) {
                actions.push(`<button onclick="IntentFlow.sendQuickReply('可以')" class="px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-xs font-medium transition-colors border border-blue-100">可以</button>`);
                actions.push(`<button onclick="IntentFlow.sendQuickReply('我想换一个目标')" class="px-3 py-1.5 bg-gray-50 hover:bg-gray-100 text-gray-600 rounded-lg text-xs font-medium transition-colors border border-gray-100">换一个</button>`);
            }
            if (reply.includes('排除') || reply.includes('不用') || reply.includes('哪些')) {
                actions.push(`<button onclick="IntentFlow.sendQuickReply('没有要排除的')" class="px-3 py-1.5 bg-gray-50 hover:bg-gray-100 text-gray-600 rounded-lg text-xs font-medium transition-colors border border-gray-100">没有排除</button>`);
            }
            if (reply.includes('选择') && reply.includes('列')) {
                const colMatches = reply.match(/`([^`]+)`/g);
                if (colMatches) {
                    colMatches.slice(0, 3).forEach(m => {
                        const col = m.replace(/`/g, '');
                        actions.push(`<button onclick="IntentFlow.sendQuickReply('用${col}作为目标')" class="px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-xs font-medium transition-colors border border-blue-100">${col}</button>`);
                    });
                }
            }
            if (actions.length > 0) {
                quickActions = `<div class="flex flex-wrap gap-2 mt-3 pt-3 border-t border-gray-100">${actions.join('')}</div>`;
            }
        }

        const html = `
            <div class="self-start bg-white border border-blue-100 rounded-2xl p-5 text-gray-800 text-sm max-w-[95%] shadow-sm animate-fade-in">
                <div class="flex items-center gap-2 mb-3">
                    <div class="agent-avatar"><i class="ph-fill ph-sparkle"></i></div>
                    <div class="font-bold text-blue-900">AI 建模助手</div>
                    ${roundLabel}
                    ${isReady ? '<span class="text-[10px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full ml-auto">已理解</span>' : '<span class="text-[10px] bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full ml-auto">待澄清</span>'}
                </div>
                <div class="text-gray-700 leading-relaxed mb-3">${reply.replace(/\n/g, '<br>')}</div>
                ${slotTags.length > 0 ? `<div class="flex flex-wrap gap-2 pt-3 border-t border-gray-100">${slotTags.join('')}</div>` : ''}
                ${quickActions}
            </div>
        `;

        this.systemMessages.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    },

    // ===== Agent 思考中 =====
    showAgentThinking() {
        const id = 'thinking-' + Date.now();
        const html = `
            <div id="${id}" class="self-start bg-blue-50/40 border border-blue-100 rounded-2xl p-4 text-gray-500 text-sm max-w-[60%] shadow-sm animate-fade-in">
                <div class="flex items-center gap-3">
                    <div class="agent-avatar" style="width:24px;height:24px;font-size:12px;"><i class="ph-fill ph-sparkle"></i></div>
                    <div class="flex gap-1">
                        <div class="w-2 h-2 rounded-full bg-blue-400 typing-dot"></div>
                        <div class="w-2 h-2 rounded-full bg-blue-400 typing-dot"></div>
                        <div class="w-2 h-2 rounded-full bg-blue-400 typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        this.systemMessages.insertAdjacentHTML('beforeend', html);
        AppState._thinkingId = id;
        this.scrollToBottom();
        return id;
    },

    removeAgentThinking() {
        if (AppState._thinkingId) {
            const el = document.getElementById(AppState._thinkingId);
            if (el) el.remove();
            AppState._thinkingId = null;
        }
    },

    // ===== 任务确认卡片 =====
    renderTaskConfirmation(slots) {
        const taskDesc = Utils.formatTaskType(slots.task_type);
        const constraintText = slots.feature_constraints.length > 0
            ? `（已排除 ${slots.feature_constraints.join('、')}）`
            : '';

        const html = `
            <div class="self-start bg-gradient-to-br from-green-50 to-blue-50 border border-green-200 rounded-2xl p-5 text-gray-800 text-sm max-w-[95%] shadow-sm animate-fade-in">
                <div class="font-bold mb-3 flex items-center gap-2 text-green-800">
                    <i class="ph-fill ph-check-circle text-green-500 text-xl"></i>
                    任务确认
                </div>
                <div class="bg-white/80 p-4 rounded-xl border border-green-100 text-xs text-gray-700 space-y-2.5 shadow-sm">
                    <p class="text-sm text-gray-800 font-medium">我已经完全理解了您的任务：</p>
                    <div class="flex justify-between py-1.5 border-b border-gray-50">
                        <span class="text-gray-400">预测目标</span>
                        <span class="font-bold text-blue-800">\`${slots.target_column}\`</span>
                    </div>
                    <div class="flex justify-between py-1.5 border-b border-gray-50">
                        <span class="text-gray-400">任务类型</span>
                        <span class="font-medium">${taskDesc}</span>
                    </div>
                    <div class="flex justify-between py-1.5 border-b border-gray-50">
                        <span class="text-gray-400">评估指标</span>
                        <span class="font-medium">${slots.eval_metric}</span>
                    </div>
                    <div class="flex justify-between py-1.5 ${slots.feature_constraints.length ? 'border-b border-gray-50' : ''}">
                        <span class="text-gray-400">使用特征</span>
                        <span class="font-medium">全部字段 ${constraintText}</span>
                    </div>
                    ${slots.feature_constraints.length ? `
                    <div class="flex justify-between py-1.5">
                        <span class="text-gray-400">排除特征</span>
                        <span class="font-medium text-red-600">${slots.feature_constraints.join('、')}</span>
                    </div>` : ''}
                </div>
                <div class="mt-4 flex gap-2.5">
                    <button onclick="IntentFlow.confirmTask()" class="flex-1 py-2.5 bg-[#8CB4FF] text-blue-950 rounded-xl text-sm font-bold shadow-sm hover:bg-blue-400 hover:shadow-md transition-all active:scale-[0.98]">
                        <i class="ph-fill ph-check mr-1"></i>确认并开始
                    </button>
                    <button onclick="IntentFlow.modifyTask()" class="px-4 py-2.5 bg-white border border-gray-200 text-gray-600 rounded-xl text-sm font-medium hover:bg-gray-50 transition-all">
                        修改
                    </button>
                </div>
            </div>
        `;
        this.systemMessages.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    },

    // ===== 模式选择卡片 =====
    renderModeSelection(isReady) {
        const html = `
            <div id="mode-selection-card" class="self-start bg-blue-50/40 border border-blue-100 rounded-2xl p-5 text-gray-800 text-sm max-w-[95%] shadow-sm animate-fade-in mt-2">
                <div class="font-bold mb-3 flex items-center gap-2 text-blue-900">
                    <i class="ph-fill ph-rocket-launch text-blue-500 text-xl"></i>
                    选择执行模式
                </div>
                <div class="text-gray-500 text-xs mb-4">请选择您希望开始的建模方式</div>

                <div class="flex flex-col gap-2.5">
                    <button id="btn-mode-fast" onclick="ModeInterface.selectMode('fast')" class="mode-card w-full p-4 bg-white border border-blue-100 hover:border-blue-400 hover:shadow-md rounded-2xl text-left flex items-center gap-4 group ${isReady ? '' : 'mode-btn-disabled'}">
                        <div class="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                            <i class="ph ph-lightning text-xl font-bold"></i>
                        </div>
                        <div>
                            <div class="text-sm font-bold text-gray-800 tracking-tight">快速模式</div>
                            <div class="text-[11px] text-gray-400 mt-0.5">快速生成基础 Baseline 代码</div>
                        </div>
                        ${isReady ? '<span class="ml-auto text-[10px] bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">推荐</span>' : '<span class="ml-auto text-[10px] bg-gray-100 text-gray-400 px-2 py-0.5 rounded-full">需先确认任务</span>'}
                    </button>
                    <button id="btn-mode-depth" onclick="ModeInterface.showDepthConfig()" class="mode-card w-full p-4 bg-white border border-blue-100 hover:border-blue-400 hover:shadow-md rounded-2xl text-left flex items-center gap-4 group ${isReady ? '' : 'mode-btn-disabled'}">
                        <div class="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                            <i class="ph ph-selection-all text-xl font-bold"></i>
                        </div>
                        <div>
                            <div class="text-sm font-bold text-gray-800 tracking-tight">深度模式</div>
                            <div class="text-[11px] text-gray-400 mt-0.5">自动特征工程与模型调优</div>
                        </div>
                    </button>
                </div>

                <div id="depth-config" class="mt-4 pt-4 border-t border-blue-100 hidden animate-fade-in">
                    <div class="flex justify-between text-[11px] text-gray-500 mb-3 font-bold uppercase tracking-wider">
                        <span>优化时长上限</span>
                        <span id="time-display" class="text-blue-600 bg-blue-100 px-2 py-0.5 rounded">60 分钟</span>
                    </div>
                    <input type="range" id="time-slider" min="15" max="120" value="60"
                        class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        oninput="document.getElementById('time-display').innerText = this.value + ' 分钟'">
                    <button onclick="ModeInterface.selectMode('depth')" class="w-full mt-5 py-3 bg-[#8CB4FF] text-blue-950 rounded-xl text-sm font-bold shadow-sm hover:bg-blue-400 hover:shadow-md transition-all active:scale-[0.98]">
                        启动深度优化
                    </button>
                </div>
            </div>
        `;
        this.systemMessages.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    },

    // ===== 阶段指示器更新 =====
    updateStage(id, state) {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.remove('active', 'done');
        if (state) el.classList.add(state);
    },

    // ===== 清空系统消息 =====
    clearSystemMessages() {
        const el = document.getElementById('system-messages');
        if (el) el.innerHTML = '';
    }
};

window.Renderer = Renderer;

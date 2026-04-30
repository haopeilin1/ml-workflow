/**
 * 快速模式 UI 渲染器
 * 负责：阶段状态卡片、代码展示、结果面板、文件列表、用户反馈组件
 */

const FastUI = {
    // 代码历史记录，用于追溯修复及迭代过程
    _codeHistory: [],

    // ========== 引擎启动 ==========
    onEngineStart() {
        // 清空代码历史
        this._codeHistory = [];
        // 隐藏模式选择按钮
        const modeCard = document.getElementById('mode-selection-card');
        if (modeCard) modeCard.remove();

        // 禁用底部输入框，切换为停止按钮
        const bottomInput = document.getElementById('bottom-input');
        const btnSend = document.getElementById('btn-send');
        const btnStop = document.getElementById('btn-stop');
        if (bottomInput) {
            bottomInput.placeholder = '快速模式运行中，请稍候...';
            bottomInput.disabled = true;
            bottomInput.classList.add('opacity-50');
        }
        if (btnSend) {
            btnSend.classList.add('btn-hidden');
            btnSend.disabled = true;
        }
        if (btnStop) {
            btnStop.classList.remove('btn-hidden');
        }

        // 添加快速模式标识到左侧
        const sysContainer = document.getElementById('system-messages');
        const html = `
            <div id="fast-mode-banner" class="self-start bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-2 animate-fade-in">
                <div class="font-bold mb-2 text-blue-900 flex items-center gap-2">
                    <i class="ph-fill ph-lightning text-yellow-500"></i>
                    快速模式已启动
                </div>
                <p class="text-gray-600 text-xs">快速基线引擎将自动完成数据清洗 → 特征工程 → 模型训练 → 验证评估的全流程。</p>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
    },

    // ========== 阶段状态切换 ==========
    setPhase(phase, meta = {}) {
        const sysContainer = document.getElementById('system-messages');
        const existingCard = document.getElementById('fast-phase-card');
        if (existingCard) existingCard.remove();

        // 运行过程中（非 presenting/completed）清空右侧预测结果面板，避免显示过期的 mock 数据
        if (phase !== 'presenting' && phase !== 'completed') {
            this._resetResultsPanel();
        }

        const { message, round, isDebug, error, userSuggestion } = meta;
        const phaseConfig = this._getPhaseConfig(phase);

        let extraContent = '';
        if (isDebug && error) {
            extraContent = `<div class="mt-2 p-2 bg-red-50 border border-red-100 rounded-lg text-xs text-red-700 font-mono overflow-x-auto">${error.substring(0, 200)}${error.length > 200 ? '...' : ''}</div>`;
        }
        if (userSuggestion) {
            extraContent = `<div class="mt-2 p-2 bg-amber-50 border border-amber-100 rounded-lg text-xs text-amber-700">💬 用户反馈：${userSuggestion}</div>`;
        }
        if (round) {
            extraContent += `<div class="mt-1 text-xs text-blue-500 font-medium">第 ${round} 轮</div>`;
        }

        const html = `
            <div id="fast-phase-card" class="self-start bg-white border border-gray-100 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-3 animate-fade-in">
                <div class="flex items-center gap-3 mb-2">
                    <div class="w-8 h-8 rounded-full ${phaseConfig.bg} flex items-center justify-center">
                        <i class="ph ${phaseConfig.icon} ${phaseConfig.iconColor} text-lg"></i>
                    </div>
                    <div>
                        <div class="font-bold text-gray-800">${phaseConfig.label}</div>
                        <div class="text-xs text-gray-400">${message || phaseConfig.defaultMessage}</div>
                    </div>
                    ${phase !== 'presenting' && phase !== 'completed' && phase !== 'failed' ? `
                    <div class="ml-auto">
                        <div class="w-5 h-5 border-2 border-blue-200 border-t-blue-500 rounded-full animate-spin"></div>
                    </div>` : ''}
                </div>
                ${extraContent}
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
    },

    _getPhaseConfig(phase) {
        const configs = {
            planning: { label: '规划阶段', icon: 'ph-strategy', iconColor: 'text-purple-500', bg: 'bg-purple-50', defaultMessage: 'Plan&Coding Agent 正在制定建模计划...' },
            coding: { label: '编码阶段', icon: 'ph-code', iconColor: 'text-blue-500', bg: 'bg-blue-50', defaultMessage: '正在生成 Pipeline 代码...' },
            running: { label: '运行阶段', icon: 'ph-play-circle', iconColor: 'text-green-500', bg: 'bg-green-50', defaultMessage: '代码在沙盒中执行...' },
            evaluating: { label: '评估阶段', icon: 'ph-chart-bar', iconColor: 'text-orange-500', bg: 'bg-orange-50', defaultMessage: 'Evaluation Agent 正在评估...' },
            optimizing: { label: '优化阶段', icon: 'ph-arrows-clockwise', iconColor: 'text-indigo-500', bg: 'bg-indigo-50', defaultMessage: '根据反馈优化模型...' },
            presenting: { label: '结果呈现', icon: 'ph-presentation-chart', iconColor: 'text-teal-500', bg: 'bg-teal-50', defaultMessage: '模型训练完成' },
            completed: { label: '已完成', icon: 'ph-check-circle', iconColor: 'text-green-600', bg: 'bg-green-50', defaultMessage: '用户已确认满意' },
            failed: { label: '建模失败', icon: 'ph-x-circle', iconColor: 'text-red-500', bg: 'bg-red-50', defaultMessage: '建模未成功完成' }
        };
        return configs[phase] || configs.planning;
    },

    // ========== 展示建模计划 ==========
    showPlan(planText) {
        const sysContainer = document.getElementById('system-messages');
        const html = `
            <div class="self-start bg-gray-50 border border-gray-100 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-2 animate-fade-in">
                <div class="font-bold mb-2 text-gray-800 flex items-center gap-2">
                    <i class="ph-fill ph-strategy text-purple-500"></i> 建模计划
                </div>
                <pre class="text-xs text-gray-600 whitespace-pre-wrap font-mono bg-white p-3 rounded-xl border border-gray-100">${this._escapeHtml(planText)}</pre>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
    },

    // ========== 展示代码 ==========
    showCode(code, meta = {}) {
        // 存入全局，供右侧面板查看
        window._fastCurrentCode = code;
        window._fastCurrentCodeMeta = meta;

        const { type } = meta;
        // Debug 过程中的中间代码不显示在模型代码面板，只保留在终端日志中
        if (type === 'debug') {
            // 仅添加左侧摘要卡片，不渲染到代码面板
        } else {
            // 记录到代码历史（追加，不覆盖）
            this._codeHistory.push({ code, meta, timestamp: new Date().toLocaleTimeString() });
            // 渲染代码（渲染全部历史）
            this._renderCodePanel();
        }

        // 自动切换到代码标签页（仅在非 debug 时，避免修复过程中频繁跳转）
        if (typeof switchTab === 'function' && type !== 'debug') {
            switchTab('code');
        }

        // 移除编码阶段的动态状态卡片，避免与后续状态卡片重复
        const sysContainer = document.getElementById('system-messages');
        const existingPhaseCard = document.getElementById('fast-phase-card');
        if (existingPhaseCard) existingPhaseCard.remove();

        // 仅在非 debug 时左侧显示代码生成摘要（修复代码不显示，避免刷屏）
        if (type !== 'debug') {
            const { round } = meta;
            let label = 'Pipeline 代码已生成';
            if (type === 'optimize') label = `第 ${round} 轮优化后的代码`;
            if (type === 'user_feedback') label = '根据您的反馈调整后的代码';

            const html = `
                <div class="self-start bg-blue-50/50 border border-blue-100 rounded-2xl p-3 text-gray-600 text-xs max-w-[90%] shadow-sm mt-2 animate-fade-in">
                    <div class="flex items-center gap-2">
                        <i class="ph-fill ph-check-circle text-blue-500"></i>
                        <span>${label}</span>
                        <span class="text-gray-400">(${code.split('\n').length} 行)</span>
                    </div>
                </div>
            `;
            sysContainer.insertAdjacentHTML('beforeend', html);
            Renderer.scrollToBottom();
        }
    },

    _renderCodePanel() {
        const panel = document.getElementById('tab-panel-code');
        if (!panel) return;

        const emptyState = document.getElementById('code-empty-state');
        const contentState = document.getElementById('code-content-state');
        if (emptyState) emptyState.classList.add('hidden');
        if (contentState) contentState.classList.remove('hidden');

        if (this._codeHistory.length === 0) {
            contentState.innerHTML = '<div class="p-4 text-xs text-gray-400">暂无代码历史</div>';
            return;
        }

        // 倒序渲染：最新的在顶部，默认展开；旧的折叠
        const blocks = [...this._codeHistory].reverse().map((item, idx) => {
            const { code, meta, timestamp } = item;
            const isLatest = idx === 0;
            const { round, type } = meta;

            let badgeClass = 'bg-blue-100 text-blue-700';
            let label = '初始版本';
            if (type === 'optimize') { badgeClass = 'bg-purple-100 text-purple-700'; label = `第 ${round} 轮优化`; }
            if (type === 'debug') { badgeClass = 'bg-red-100 text-red-700'; label = `Debug ${round}`; }
            if (type === 'user_feedback') { badgeClass = 'bg-amber-100 text-amber-700'; label = '用户反馈调整'; }
            if (type === 'present') { badgeClass = 'bg-green-100 text-green-700'; label = '最终版本'; }

            const containerId = `code-block-${this._codeHistory.length - 1 - idx}`;

            return `
                <div class="mb-3 border border-gray-200 rounded-xl overflow-hidden">
                    <div class="flex items-center justify-between px-3 py-2 bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
                         onclick="FastUI._toggleCodeBlock('${containerId}')">
                        <div class="flex items-center gap-2">
                            <span class="px-2 py-0.5 ${badgeClass} rounded text-xs font-medium">${label}</span>
                            <span class="text-xs text-gray-400">${timestamp}</span>
                            <span class="text-xs text-gray-400">(${code.split('\n').length} 行)</span>
                        </div>
                        <i class="ph ph-caret-${isLatest ? 'down' : 'up'} text-gray-400 text-sm" id="icon-${containerId}"></i>
                    </div>
                    <div id="${containerId}" class="${isLatest ? '' : 'hidden'}">
                        <div class="code-display-${containerId} p-3 overflow-x-auto"></div>
                    </div>
                </div>
            `;
        }).join('');

        contentState.innerHTML = `
            <div class="p-3">
                <div class="text-xs text-gray-400 mb-2 flex items-center justify-between">
                    <span>Python · Pipeline · 共 ${this._codeHistory.length} 个版本</span>
                    <span class="text-gray-300">点击标题可展开/折叠</span>
                </div>
                ${blocks}
            </div>
        `;

        // 逐个渲染代码块
        setTimeout(() => {
            this._codeHistory.forEach((item, idx) => {
                const containerId = `code-block-${idx}`;
                const container = contentState.querySelector(`.code-display-${containerId}`);
                if (container && window.CodeEditor) {
                    // 为每个代码块创建临时容器 ID
                    container.id = `tmp-code-${containerId}`;
                    CodeEditor.render(`tmp-code-${containerId}`, item.code);
                }
            });
        }, 10);
    },

    showDebugStart(round) {
        const sysContainer = document.getElementById('system-messages');
        const html = `
            <div class="self-start bg-amber-50/50 border border-amber-100 rounded-2xl p-3 text-gray-600 text-xs max-w-[90%] shadow-sm mt-2 animate-fade-in">
                <div class="flex items-center gap-2">
                    <i class="ph-fill ph-wrench text-amber-500"></i>
                    <span>第 ${round} 次代码修复</span>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
    },

    // 根据任务类型和实际存在的指标字段，生成左侧结果卡片要显示的指标项
    _getResultMetricItems(metrics, taskType) {
        const fmt = (v) => {
            if (v === null || v === undefined) return 'N/A';
            const n = Number(v);
            if (isNaN(n)) return String(v);
            if (Math.abs(n) >= 10000) return n.toFixed(2);
            if (Math.abs(n) >= 1) return n.toFixed(4).replace(/\.?0+$/, '');
            return n.toFixed(4);
        };
        const items = [];
        if (taskType === 'regression') {
            if (metrics?.val_rmse !== null && metrics?.val_rmse !== undefined) {
                items.push({ value: fmt(metrics.val_rmse), raw: metrics.val_rmse, label: '验证集 RMSE', color: 'text-blue-600' });
            }
            if (metrics?.train_score !== null && metrics?.train_score !== undefined) {
                items.push({ value: fmt(metrics.train_score), raw: metrics.train_score, label: '训练集 Score', color: 'text-purple-600' });
            }
        } else {
            if (metrics?.val_auc !== null && metrics?.val_auc !== undefined) {
                items.push({ value: fmt(metrics.val_auc), raw: metrics.val_auc, label: '验证集 AUC', color: 'text-blue-600' });
            }
            if (metrics?.val_accuracy !== null && metrics?.val_accuracy !== undefined) {
                items.push({ value: fmt(metrics.val_accuracy), raw: metrics.val_accuracy, label: '准确率', color: 'text-green-600' });
            }
        }
        if (metrics?.overfit_ratio !== null && metrics?.overfit_ratio !== undefined) {
            const color = metrics.overfit_ratio > 1.05 ? 'text-red-500' : 'text-green-600';
            items.push({ value: fmt(metrics.overfit_ratio), raw: metrics.overfit_ratio, label: '过拟合比', color });
        }
        return items;
    },

    // 根据任务类型和实际存在的指标字段，生成右侧结果面板要显示的指标项
    _getPanelMetricItems(metrics, taskType) {
        const fmtMetric = (v) => {
            if (v === null || v === undefined || v === '') return '--';
            const n = Number(v);
            if (isNaN(n)) return String(v);
            if (Math.abs(n) >= 10000) return n.toExponential(2);
            if (Math.abs(n) >= 1) return n.toFixed(4).replace(/\.?0+$/, '');
            return n.toFixed(4);
        };
        const items = [];
        if (taskType === 'regression') {
            if (metrics?.val_rmse !== null && metrics?.val_rmse !== undefined) {
                items.push({ value: fmtMetric(metrics.val_rmse), raw: metrics.val_rmse, label: '验证集 RMSE', color: 'text-blue-400' });
            }
            if (metrics?.val_score !== null && metrics?.val_score !== undefined) {
                items.push({ value: fmtMetric(metrics.val_score), raw: metrics.val_score, label: '验证集 Score', color: 'text-cyan-400' });
            }
            if (metrics?.train_score !== null && metrics?.train_score !== undefined) {
                items.push({ value: fmtMetric(metrics.train_score), raw: metrics.train_score, label: '训练集 Score', color: 'text-purple-400' });
            }
        } else {
            if (metrics?.val_auc !== null && metrics?.val_auc !== undefined) {
                items.push({ value: fmtMetric(metrics.val_auc), raw: metrics.val_auc, label: '验证集 AUC', color: 'text-blue-400' });
            }
            if (metrics?.val_accuracy !== null && metrics?.val_accuracy !== undefined) {
                items.push({ value: fmtMetric(metrics.val_accuracy), raw: metrics.val_accuracy, label: '准确率', color: 'text-green-400' });
            }
            if (metrics?.train_auc !== null && metrics?.train_auc !== undefined) {
                items.push({ value: fmtMetric(metrics.train_auc), raw: metrics.train_auc, label: '训练集 AUC', color: 'text-purple-400' });
            }
        }
        if (metrics?.overfit_ratio !== null && metrics?.overfit_ratio !== undefined) {
            const color = metrics.overfit_severe ? 'text-red-400' : 'text-green-400';
            items.push({ value: fmtMetric(metrics.overfit_ratio), raw: metrics.overfit_ratio, label: '过拟合比', color });
        }
        return items;
    },

    _toggleCodeBlock(containerId) {
        const el = document.getElementById(containerId);
        const icon = document.getElementById(`icon-${containerId}`);
        if (el) {
            el.classList.toggle('hidden');
            if (icon) {
                icon.classList.toggle('ph-caret-down');
                icon.classList.toggle('ph-caret-up');
            }
        }
    },

    // ========== 展示运行输出 ==========
    showExecutionOutput(output, metrics) {
        // 运行输出同时显示在终端
        Terminal.output(output);

        // 数值格式化辅助函数
        const fmt = (v) => {
            if (v === null || v === undefined) return 'N/A';
            const n = Number(v);
            if (isNaN(n)) return String(v);
            if (Math.abs(n) >= 10000) return n.toFixed(2);
            if (Math.abs(n) >= 1) return n.toFixed(4).replace(/\.?0+$/, '');
            return n.toFixed(4);
        };

        // 左侧添加运行摘要
        const sysContainer = document.getElementById('system-messages');
        const scoreVal = metrics?.val_auc ?? metrics?.val_score ?? metrics?.val_accuracy ?? metrics?.val_rmse;
        const metricStr = metrics
            ? `验证集 ${metrics.metric_name || 'Score'}: ${fmt(scoreVal)}`
            : '运行完成';

        const html = `
            <div class="self-start bg-green-50/50 border border-green-100 rounded-2xl p-3 text-gray-600 text-xs max-w-[90%] shadow-sm mt-2 animate-fade-in">
                <div class="flex items-center gap-2">
                    <i class="ph-fill ph-check-circle text-green-500"></i>
                    <span>沙盒运行成功</span>
                    <span class="text-green-600 font-medium ml-1">${metricStr}</span>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
    },

    // ========== 展示评估结果 ==========
    showEvaluation(evaluation) {
        const sysContainer = document.getElementById('system-messages');

        if (evaluation.decision === 'AUTO_OPTIMIZE') {
            const html = `
                <div class="self-start bg-indigo-50 border border-indigo-100 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-2 animate-fade-in">
                    <div class="font-bold mb-2 text-indigo-900 flex items-center gap-2">
                        <i class="ph-fill ph-arrows-clockwise text-indigo-500"></i> 评估结论：需要自动优化
                    </div>
                    <p class="text-xs text-gray-600 mb-2">${this._escapeHtml(evaluation.evaluation_analysis)}</p>
                    <div class="bg-white border border-indigo-100 rounded-xl p-3">
                        <div class="text-xs font-medium text-indigo-700 mb-1">优化建议：</div>
                        <pre class="text-xs text-gray-600 whitespace-pre-wrap font-mono">${this._escapeHtml(evaluation.suggestions_for_coding_agent)}</pre>
                    </div>
                </div>
            `;
            sysContainer.insertAdjacentHTML('beforeend', html);
        } else {
            // YIELD_TO_USER
            const html = `
                <div class="self-start bg-teal-50 border border-teal-100 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-2 animate-fade-in">
                    <div class="font-bold mb-2 text-teal-900 flex items-center gap-2">
                        <i class="ph-fill ph-chart-bar text-teal-500"></i> 评估结论：效果达标
                    </div>
                    <pre class="text-xs text-gray-600 whitespace-pre-wrap font-mono bg-white p-3 rounded-xl border border-teal-100">${this._escapeHtml(evaluation.evaluation_analysis)}</pre>
                </div>
            `;
            sysContainer.insertAdjacentHTML('beforeend', html);
        }
        Renderer.scrollToBottom();
    },

    // ========== 结果呈现 ==========
    presentResults(data) {
        const { metrics, evaluation, files, featureImportance, testPredictions, userFeedbackRound, maxUserFeedbackRounds } = data;

        // 判断是否已达到反馈次数上限
        const isLastChance = (userFeedbackRound || 0) >= (maxUserFeedbackRounds || 3);

        // 1. 左侧：评估汇报 + 反馈组件
        const sysContainer = document.getElementById('system-messages');
        let reportHtml = '';

        if (evaluation && evaluation.report_to_user) {
            reportHtml = `<pre class="text-xs text-gray-600 whitespace-pre-wrap font-mono bg-white p-3 rounded-xl border border-teal-100 mb-3">${this._escapeHtml(evaluation.report_to_user)}</pre>`;
        }

        // 数值格式化：防止过长溢出
        const fmt = (v) => {
            if (v === null || v === undefined) return 'N/A';
            const n = Number(v);
            if (isNaN(n)) return String(v);
            if (Math.abs(n) >= 10000) return n.toFixed(2);
            if (Math.abs(n) >= 1) return n.toFixed(4).replace(/\.?0+$/, '');
            return n.toFixed(4);
        };

        const html = `
            <div id="fast-result-card" class="self-start bg-white border border-gray-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-3 text-gray-800 flex items-center gap-2">
                    <i class="ph-fill ph-trophy text-yellow-500"></i> 模型训练完成
                </div>
                ${reportHtml}
                ${(() => {
                    const taskType = FastEngine.state.taskConfig?.extracted_slots?.task_type || 'classification';
                    const items = this._getResultMetricItems(metrics, taskType);
                    if (items.length === 0) return '';
                    const cols = items.length === 1 ? 'grid-cols-1' : items.length === 2 ? 'grid-cols-2' : 'grid-cols-3';
                    return `
                        <div class="bg-gray-50 rounded-xl p-3 mb-3">
                            <div class="grid ${cols} gap-2 text-center">
                                ${items.map(item => `
                                    <div>
                                        <div class="text-lg font-bold ${item.color} truncate" title="${item.raw}">${item.value}</div>
                                        <div class="text-xs text-gray-400">${item.label}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                })()}
                <div class="flex gap-2" id="fast-feedback-buttons">
                    <button onclick="FastUI.onSatisfied()" class="flex-1 py-2.5 bg-green-500 hover:bg-green-600 text-white rounded-xl text-sm font-medium transition-all active:scale-[0.98] flex items-center justify-center gap-1.5">
                        <i class="ph-fill ph-check"></i> 满意，生成产物
                    </button>
                    ${isLastChance ? `
                    <button onclick="FastUI.onGiveUp()" class="flex-1 py-2.5 bg-white border border-red-200 hover:bg-red-50 text-red-600 rounded-xl text-sm font-medium transition-all active:scale-[0.98] flex items-center justify-center gap-1.5">
                        <i class="ph-fill ph-x-circle"></i> 不满意，放弃任务
                    </button>
                    ` : `
                    <button onclick="FastUI.showFeedbackInput()" class="flex-1 py-2.5 bg-white border border-gray-200 hover:bg-gray-50 text-gray-600 rounded-xl text-sm font-medium transition-all active:scale-[0.98] flex items-center justify-center gap-1.5">
                        <i class="ph-fill ph-arrow-counter-clockwise"></i> 不满意，继续优化
                    </button>
                    `}
                </div>
                ${!isLastChance ? `
                <div id="fast-feedback-input" class="hidden mt-3">
                    <textarea id="fast-feedback-text" class="w-full bg-gray-50 border border-gray-200 rounded-xl p-3 text-sm text-gray-700 focus:outline-none focus:border-blue-300 resize-none" rows="2" placeholder="请描述不满意的地方或改进建议..."></textarea>
                    <div class="flex gap-2 mt-2">
                        <button onclick="FastUI.onUnsatisfied()" class="flex-1 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-xl text-sm font-medium transition-all">
                            提交反馈
                        </button>
                        <button onclick="FastUI.hideFeedbackInput()" class="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-500 rounded-xl text-sm transition-all">
                            取消
                        </button>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();

        // 2. 右侧：自动切换到结果标签页
        this._renderResultsPanel(data);
    },

    // ========== 重置右侧结果面板为空白状态 ==========
    _resetResultsPanel() {
        const emptyState = document.getElementById('results-empty-state');
        const contentState = document.getElementById('results-content-state');
        if (emptyState) emptyState.classList.remove('hidden');
        if (contentState) {
            contentState.classList.add('hidden');
            contentState.innerHTML = '';
        }
    },

    // ========== 渲染右侧结果面板 ==========
    _renderResultsPanel(data) {
        const { metrics, featureImportance, testPredictions, files } = data;

        const panel = document.getElementById('tab-panel-results');
        if (!panel) return;

        const emptyState = document.getElementById('results-empty-state');
        const contentState = document.getElementById('results-content-state');
        if (emptyState) emptyState.classList.add('hidden');
        if (contentState) contentState.classList.remove('hidden');

        const display = contentState;

        // 构建特征重要性图表（纯 CSS 条形图）
        let featureChartHtml = '';
        if (featureImportance && featureImportance.length > 0) {
            const maxImp = Math.max(...featureImportance.map(f => f.importance));
            const rows = featureImportance.slice(0, 8).map(f => {
                const pct = (f.importance / maxImp * 100).toFixed(1);
                return `
                    <div class="flex items-center gap-2 text-xs mb-1">
                        <div class="w-24 text-gray-400 truncate text-right">${f.name}</div>
                        <div class="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
                            <div class="bg-blue-500 h-full rounded-full" style="width: ${pct}%"></div>
                        </div>
                        <div class="w-16 text-gray-300 text-right truncate" title="${(f.importance * 100).toFixed(1)}%">${(f.importance * 100).toFixed(1)}%</div>
                    </div>
                `;
            }).join('');
            featureChartHtml = `
                <div class="bg-gray-800 rounded-xl p-4 mb-4">
                    <div class="text-sm font-medium text-gray-200 mb-3">特征重要性 TOP 8</div>
                    ${rows}
                </div>
            `;
        } else {
            featureChartHtml = `
                <div class="bg-gray-800 rounded-xl p-4 mb-4">
                    <div class="text-sm font-medium text-gray-200 mb-2">特征重要性</div>
                    <div class="text-xs text-gray-500">特征重要性将在模型训练完成后生成</div>
                </div>
            `;
        }

        // 测试集预测表格
        let predictionTableHtml = '';
        if (testPredictions && testPredictions.length > 0) {
            const rows = testPredictions.map(p => `
                <tr class="border-b border-gray-800">
                    <td class="py-1.5 text-gray-400">${p.id}</td>
                    <td class="py-1.5 text-blue-400 font-mono">${p.prob.toFixed(4)}</td>
                    <td class="py-1.5">
                        <span class="px-2 py-0.5 rounded text-xs ${p.pred === 1 ? 'bg-blue-500/20 text-blue-400' : 'bg-gray-700 text-gray-400'}">
                            ${p.pred === 1 ? '正类' : '负类'}
                        </span>
                    </td>
                </tr>
            `).join('');
            predictionTableHtml = `
                <div class="bg-gray-800 rounded-xl p-4 mb-4">
                    <div class="text-sm font-medium text-gray-200 mb-3">测试集预测预览（前10条）</div>
                    <table class="w-full text-xs">
                        <thead><tr class="text-gray-500 border-b border-gray-700">
                            <th class="text-left py-1">ID</th><th class="text-left py-1">概率</th><th class="text-left py-1">预测</th>
                        </tr></thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>
            `;
        }

        // 数值格式化辅助函数
        const fmtMetric = (v) => {
            if (v === null || v === undefined || v === '') return '--';
            const n = Number(v);
            if (isNaN(n)) return String(v);
            if (Math.abs(n) >= 10000) return n.toExponential(2);
            if (Math.abs(n) >= 1) return n.toFixed(4).replace(/\.?0+$/, '');
            return n.toFixed(4);
        };

        // 指标卡片（根据任务类型动态渲染）
        const taskType = FastEngine.state.taskConfig?.extracted_slots?.task_type || 'classification';
        const panelItems = this._getPanelMetricItems(metrics, taskType);
        const panelGridCols = panelItems.length <= 2 ? `grid-cols-${Math.max(panelItems.length, 1)}` : panelItems.length === 3 ? 'grid-cols-3' : 'grid-cols-4';
        const metricCards = panelItems.length > 0 ? `
            <div class="grid ${panelGridCols} gap-3 mb-4">
                ${panelItems.map(item => `
                    <div class="bg-gray-800 rounded-xl p-3 text-center">
                        <div class="text-xl font-bold ${item.color} truncate" title="${item.raw}">${item.value}</div>
                        <div class="text-xs text-gray-500 mt-1">${item.label}</div>
                    </div>
                `).join('')}
            </div>
        ` : '<div class="text-xs text-gray-500 mb-4">暂无指标数据</div>';

        display.innerHTML = `
            <div class="mb-4">
                <div class="text-xs text-gray-400 font-medium mb-3">快速模式 · 基线结果</div>
                ${metricCards}
                ${featureChartHtml}
                ${predictionTableHtml}
            </div>
        `;
    },

    // ========== 渲染右侧文件面板 ==========
    _renderFilesPanel(files) {
        const panel = document.getElementById('tab-panel-files');
        if (!panel) return;

        const emptyState = document.getElementById('files-empty-state');
        const contentState = document.getElementById('files-content-state');
        if (emptyState) emptyState.classList.add('hidden');
        if (contentState) contentState.classList.remove('hidden');

        const display = contentState;

        const fileTypeIcons = {
            model: 'ph-cube',
            code: 'ph-file-code',
            data: 'ph-table',
            report: 'ph-file-html'
        };
        const fileTypeColors = {
            model: 'text-purple-400',
            code: 'text-blue-400',
            data: 'text-green-400',
            report: 'text-orange-400'
        };

        const apiBase = window.FastEngine?.API_BASE || '';
        const fileRows = files.map(f => {
            const fileUrl = apiBase + f.path;
            return `
            <div class="flex items-center justify-between p-3 bg-gray-800 rounded-xl hover:bg-gray-750 transition-colors group">
                <a href="${fileUrl}" target="_blank" class="flex items-center gap-3 flex-1 min-w-0">
                    <i class="ph-fill ${fileTypeIcons[f.type] || 'ph-file'} ${fileTypeColors[f.type] || 'text-gray-400'} text-xl"></i>
                    <div class="min-w-0">
                        <div class="text-sm text-gray-200 hover:text-blue-400 transition-colors truncate">${f.name}</div>
                        <div class="text-xs text-gray-500 truncate">${f.desc}</div>
                    </div>
                </a>
                <div class="flex items-center gap-3 flex-shrink-0 ml-3">
                    <span class="text-xs text-gray-500 whitespace-nowrap">${f.size}</span>
                    <a href="${fileUrl}" target="_blank" class="p-1.5 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors" title="下载/预览">
                        <i class="ph-fill ph-download-simple"></i>
                    </a>
                </div>
            </div>
        `;
        }).join('');

        display.innerHTML = `
            <div class="text-xs text-gray-400 font-medium mb-3">生成文件列表</div>
            <div class="space-y-2">${fileRows}</div>
        `;
    },

    // ========== 用户交互：满意 ==========
    onSatisfied() {
        const btnContainer = document.getElementById('fast-feedback-buttons');
        const inputContainer = document.getElementById('fast-feedback-input');
        if (btnContainer) btnContainer.remove();
        if (inputContainer) inputContainer.remove();

        // 显示产物生成中提示
        const sysContainer = document.getElementById('system-messages');
        const loadingHtml = `
            <div id="fast-artifact-loading" class="self-start bg-green-50 border border-green-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-2 text-green-900 flex items-center gap-2">
                    <div class="w-4 h-4 border-2 border-green-200 border-t-green-500 rounded-full animate-spin"></div>
                    正在生成最终产物...
                </div>
                <p class="text-xs text-gray-600">正在生成可视化报告、特征重要性分析、模型文件等产物，请稍候。</p>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', loadingHtml);
        Renderer.scrollToBottom();

        // 调用引擎完成
        FastEngine.handleUserFeedback('satisfied', '');
    },

    // ========== 用户交互：不满意 ==========
    onUnsatisfied() {
        const textarea = document.getElementById('fast-feedback-text');
        const suggestion = textarea ? textarea.value.trim() : '';

        const btnContainer = document.getElementById('fast-feedback-buttons');
        const inputContainer = document.getElementById('fast-feedback-input');
        if (btnContainer) btnContainer.remove();
        if (inputContainer) inputContainer.remove();

        // 调用引擎继续优化
        FastEngine.handleUserFeedback('unsatisfied', suggestion || '用户未填写具体建议');
    },

    // ========== 用户交互：放弃任务（已达反馈上限） ==========
    onGiveUp() {
        const btnContainer = document.getElementById('fast-feedback-buttons');
        const inputContainer = document.getElementById('fast-feedback-input');
        if (btnContainer) btnContainer.remove();
        if (inputContainer) inputContainer.remove();

        const msg = `用户连续 ${FastEngine.MAX_USER_FEEDBACK_ROUNDS || 3} 次不满意。建议切换深度模式。`;
        FastEngine.state.error = msg;
        this.onFailed(msg);
    },

    // ========== 重新生成产物（针对简化产物或超时情况） ==========
    onRegenerateArtifacts() {
        const sysContainer = document.getElementById('system-messages');
        const loadingHtml = `
            <div id="fast-artifact-loading" class="self-start bg-amber-50 border border-amber-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-2 text-amber-900 flex items-center gap-2">
                    <div class="w-4 h-4 border-2 border-amber-200 border-t-amber-500 rounded-full animate-spin"></div>
                    正在重新生成产物...
                </div>
                <p class="text-xs text-gray-600">已延长 LLM 超时时间至 10 分钟，正在重新调用 LLM 生成完整产物，请稍候。</p>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', loadingHtml);
        Renderer.scrollToBottom();

        FastEngine.handleUserFeedback('satisfied', '');
    },

    showFeedbackInput() {
        const el = document.getElementById('fast-feedback-input');
        if (el) el.classList.remove('hidden');
    },

    hideFeedbackInput() {
        const el = document.getElementById('fast-feedback-input');
        if (el) el.classList.add('hidden');
    },

    // ========== 完成回调 ==========
    onCompleted(data) {
        this._restoreBottomControls();

        // 移除产物生成中的 loading 卡片
        const loadingCard = document.getElementById('fast-artifact-loading');
        if (loadingCard) loadingCard.remove();

        const sysContainer = document.getElementById('system-messages');
        let reportBtn = '';
        if (data.reportPath) {
            const reportUrl = (window.FastEngine?.API_BASE || '') + data.reportPath;
            reportBtn = `
                <button onclick="window.open('${reportUrl}', '_blank')" class="px-4 py-2 bg-white border border-green-200 rounded-xl text-xs text-green-700 hover:bg-green-50 transition-all">
                    <i class="ph-fill ph-file-html"></i> 查看可视化报告
                </button>
            `;
        }
        const html = `
            <div class="self-start bg-green-50 border border-green-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-2 text-green-900 flex items-center gap-2">
                    <i class="ph-fill ph-check-circle text-green-600"></i> 任务完成
                </div>
                <p class="text-xs text-gray-600 mb-3">所有产物已生成，您可以在右侧「模型代码」「预测结果」「文件」标签页查看和下载。</p>
                <div class="flex gap-2 flex-wrap">
                    <button onclick="switchTab('code')" class="px-4 py-2 bg-white border border-green-200 rounded-xl text-xs text-green-700 hover:bg-green-50 transition-all">
                        <i class="ph-fill ph-code"></i> 查看代码
                    </button>
                    <button onclick="switchTab('files')" class="px-4 py-2 bg-white border border-green-200 rounded-xl text-xs text-green-700 hover:bg-green-50 transition-all">
                        <i class="ph-fill ph-download-simple"></i> 下载文件
                    </button>
                    ${reportBtn}
                    <button onclick="FastUI.onRegenerateArtifacts()" class="px-4 py-2 bg-white border border-amber-200 rounded-xl text-xs text-amber-700 hover:bg-amber-50 transition-all">
                        <i class="ph-fill ph-arrow-counter-clockwise"></i> 重新生成产物
                    </button>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();

        // 更新阶段卡片
        this.setPhase('completed', { message: '用户已确认满意，任务完成' });

        // 渲染文件面板
        if (data.files && data.files.length > 0) {
            this._renderFilesPanel(data.files);
        }

        // 用户满意后，重新渲染结果面板（显示测试集预测和特征重要性）
        this._renderResultsPanel({
            metrics: data.metrics,
            featureImportance: data.featureImportance || [],
            testPredictions: data.testPredictions,
            files: data.files
        });
    },

    // ========== 失败回调 ==========
    onFailed(error) {
        this._restoreBottomControls();

        const sysContainer = document.getElementById('system-messages');
        const html = `
            <div class="self-start bg-red-50 border border-red-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-2 text-red-900 flex items-center gap-2">
                    <i class="ph-fill ph-warning-circle text-red-500"></i> 建模失败
                </div>
                <p class="text-xs text-gray-600 mb-3">${this._escapeHtml(error)}</p>
                <div class="flex gap-2">
                    <button onclick="location.reload()" class="px-4 py-2 bg-white border border-red-200 rounded-xl text-xs text-red-700 hover:bg-red-50 transition-all">
                        <i class="ph-fill ph-arrow-counter-clockwise"></i> 重新开始
                    </button>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();

        this.setPhase('failed', { message: error });
    },

    // ========== 用户手动停止 ==========
    onStoppedByUser() {
        const sysContainer = document.getElementById('system-messages');
        const html = `
            <div class="self-start bg-gray-50 border border-gray-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-3 animate-fade-in">
                <div class="font-bold mb-1 text-gray-800 flex items-center gap-2">
                    <i class="ph-fill ph-hand-palm text-gray-500"></i> 任务已终止
                </div>
                <p class="text-xs text-gray-500">您已手动停止快速模式。已生成的中间结果仍可在右侧标签页查看。</p>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();

        // 恢复阶段卡片为停止状态
        this.setPhase('failed', { message: '用户手动终止' });

        // 恢复底部控制按钮
        this._restoreBottomControls();
    },

    // ========== 恢复底部控制按钮 ==========
    _restoreBottomControls() {
        const bottomInput = document.getElementById('bottom-input');
        const btnSend = document.getElementById('btn-send');
        const btnStop = document.getElementById('btn-stop');
        if (bottomInput) {
            bottomInput.placeholder = '输入更多指令...';
            bottomInput.disabled = false;
            bottomInput.classList.remove('opacity-50');
        }
        if (btnSend) {
            btnSend.classList.remove('btn-hidden');
            btnSend.disabled = false;
        }
        if (btnStop) {
            btnStop.classList.add('btn-hidden');
        }
    },

    // ========== 标签页恢复 ==========
    restoreTab(tab) {
        if (tab === 'code') {
            if (this._codeHistory.length > 0 && window.CodeEditor) {
                this._renderCodePanel();
            }
        } else if (tab === 'results') {
            // 只在 presenting 或 completed 阶段才渲染预测结果，避免运行中显示过期/mock数据
            const phase = FastEngine.state.phase;
            const artifacts = FastEngine.state.artifacts;
            if ((phase === 'presenting' || phase === 'completed') && FastEngine.state.metrics) {
                this._renderResultsPanel({
                    metrics: FastEngine.state.metrics,
                    featureImportance: artifacts?.feature_importance || [],
                    testPredictions: artifacts?.test_predictions || null,
                    files: artifacts?.files || []
                });
            } else {
                this._resetResultsPanel();
            }
        } else if (tab === 'files') {
            const artifacts = FastEngine.state.artifacts;
            if (artifacts?.files && artifacts.files.length > 0) {
                this._renderFilesPanel(artifacts.files);
            }
        }
    },

    // ========== 文件下载 ==========
    async _downloadFile(url, filename) {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const blob = await response.blob();
            const blobUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = blobUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(blobUrl);
        } catch (err) {
            console.error('[FastUI] 下载失败:', err);
            alert('下载失败: ' + err.message);
        }
    },

    _escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
};

window.FastUI = FastUI;

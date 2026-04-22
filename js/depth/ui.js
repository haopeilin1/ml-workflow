/**
 * 深度模式 UI 渲染器
 * 负责：MCTS 探索节点卡片、倒计时、最优代码展示、时间到评估、用户反馈闭环
 */

const DepthUI = {
    // ========== 引擎启动 ==========
    onEngineStart(timeLimit, extendCount, maxExtend) {
        // 隐藏模式选择卡片
        const modeCard = document.getElementById('mode-selection-card');
        if (modeCard) modeCard.remove();

        // 禁用底部输入框，切换为停止按钮
        const bottomInput = document.getElementById('bottom-input');
        const btnSend = document.getElementById('btn-send');
        const btnStop = document.getElementById('btn-stop');
        if (bottomInput) {
            bottomInput.placeholder = '深度模式运行中，请稍候...';
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

        const sysContainer = document.getElementById('system-messages');

        // 深度模式 Banner
        const extendInfo = extendCount > 1 ? `（第 ${extendCount - 1} 次延长）` : '';
        const bannerHtml = `
            <div id="depth-mode-banner" class="self-start bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-2 animate-fade-in">
                <div class="font-bold mb-2 text-indigo-900 flex items-center gap-2">
                    <i class="ph-fill ph-selection-all text-purple-500"></i>
                    深度模式已启动 ${extendInfo}
                </div>
                <p class="text-gray-600 text-xs">MCTS 树搜索将自动探索多种建模策略，根据 UCT 值动态选择最优探索路径。</p>
                <div class="mt-3 flex items-center gap-3">
                    <div class="flex items-center gap-1.5 text-xs">
                        <span class="w-2 h-2 rounded-full bg-blue-400"></span>
                        <span class="text-gray-500">Draft</span>
                    </div>
                    <div class="flex items-center gap-1.5 text-xs">
                        <span class="w-2 h-2 rounded-full bg-purple-400"></span>
                        <span class="text-gray-500">Improve</span>
                    </div>
                    <div class="flex items-center gap-1.5 text-xs">
                        <span class="w-2 h-2 rounded-full bg-red-400"></span>
                        <span class="text-gray-500">Debug</span>
                    </div>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', bannerHtml);

        // 动态状态卡片（实时更新）
        const statusHtml = `
            <div id="depth-status-card" class="self-start bg-white border border-gray-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] shadow-sm mt-3 animate-fade-in">
                <div class="flex items-center justify-between mb-2">
                    <div class="font-bold text-gray-800 flex items-center gap-2">
                        <i class="ph-fill ph-hourglass text-indigo-500 animate-pulse"></i>
                        深度探索中
                    </div>
                    <div id="depth-countdown-display" class="text-xs font-mono bg-gray-100 px-2 py-1 rounded text-gray-600">
                        剩余 ${timeLimit} 分钟
                    </div>
                </div>
                <div class="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
                    <div id="depth-progress-bar" class="bg-gradient-to-r from-indigo-400 to-purple-400 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
                <div class="mt-3 grid grid-cols-2 gap-2">
                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-[10px] text-gray-400">已探索节点</div>
                        <div id="depth-node-counter" class="font-mono font-bold text-gray-700 text-xs">0 / --</div>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-[10px] text-gray-400">最优代码状态</div>
                        <div id="depth-best-status" class="font-bold text-gray-700 text-xs">暂无</div>
                    </div>
                </div>
                <div class="mt-2 text-[10px] text-gray-400 text-right">
                    已运行 <span id="depth-elapsed">0.0</span> 分钟
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', statusHtml);
        Renderer.scrollToBottom();
    },

    // ========== 动态状态更新（替代逐节点卡片）==========
    updateExplorationStatus({ nodeIndex, totalNodes, bestNodeId, bestMetrics, remainingMinutes, elapsedMinutes }) {
        const display = document.getElementById('depth-countdown-display');
        const progressBar = document.getElementById('depth-progress-bar');
        const counter = document.getElementById('depth-node-counter');
        const bestStatus = document.getElementById('depth-best-status');
        const elapsedEl = document.getElementById('depth-elapsed');

        if (display) {
            display.textContent = `剩余 ${remainingMinutes.toFixed(1)} 分钟`;
        }
        if (progressBar) {
            const total = remainingMinutes + elapsedMinutes;
            const pct = total > 0 ? (elapsedMinutes / total * 100) : 100;
            progressBar.style.width = pct + '%';
        }
        if (counter) {
            counter.textContent = `${nodeIndex} / ${totalNodes}`;
        }
        if (bestStatus) {
            if (bestNodeId) {
                const auc = bestMetrics?.val_auc || '--';
                bestStatus.innerHTML = `<span class="text-green-600">✓ 可提交</span> <span class="text-gray-400 font-normal">AUC ${auc}</span>`;
            } else {
                bestStatus.textContent = '暂无';
            }
        }
        if (elapsedEl) {
            elapsedEl.textContent = elapsedMinutes.toFixed(1);
        }
    },

    // ========== 更新最优代码（右侧模型代码标签页） ==========
    updateBestCode(code, metrics, nodeInfo) {
        // 存入全局供标签恢复
        window._depthCurrentCode = code;
        window._depthCurrentCodeMeta = { ...nodeInfo, metrics };

        // 自动切换到代码标签页（只在第一次或用户未主动切换时）
        // 为了不打断用户查看其他标签页，这里不做强制切换，只在代码标签页激活时渲染
        this._renderCodePanel(code, nodeInfo, metrics);

        // 不再在左侧逐条显示最优代码更新（实际运行可能并行产生大量节点）
        // 最优代码状态统一在 depth-status-card 中展示
    },

    _renderCodePanel(code, nodeInfo, metrics) {
        const panel = document.getElementById('tab-panel-code');
        if (!panel) return;

        const emptyState = document.getElementById('code-empty-state');
        const contentState = document.getElementById('code-content-state');
        if (emptyState) emptyState.classList.add('hidden');
        if (contentState) contentState.classList.remove('hidden');

        const { nodeId, nodeType, totalNodes, isFinal } = nodeInfo || {};
        const typeLabel = DepthMockData.nodeTypeConfig[nodeType]?.label || 'Optimal';
        const badge = isFinal
            ? '<span class="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded text-xs">最终版本</span>'
            : `<span class="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs">${typeLabel} #${nodeId}</span>`;

        const metricBadge = metrics?.val_auc
            ? `<span class="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs">AUC ${metrics.val_auc}</span>`
            : '';

        contentState.innerHTML = `
            <div class="p-4">
                <div class="flex items-center justify-between mb-2">
                    <div class="text-xs text-gray-400">Python · 最优 Pipeline</div>
                    <div class="flex items-center gap-2">${badge}${metricBadge}</div>
                </div>
                <div id="depth-code-display-container" class="h-full overflow-y-auto no-scrollbar"></div>
            </div>
        `;

        setTimeout(() => {
            const container = document.getElementById('depth-code-display-container');
            if (container && window.CodeEditor) {
                CodeEditor.render('depth-code-display-container', code);
            }
        }, 10);
    },

    // ========== 时间到评估 ==========
    showTimeUp(bestMetrics, stats) {
        const sysContainer = document.getElementById('system-messages');

        // 更新动态状态卡片为完成状态
        const statusCard = document.getElementById('depth-status-card');
        if (statusCard) {
            statusCard.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <div class="font-bold text-gray-800 flex items-center gap-2">
                        <i class="ph-fill ph-check-circle text-green-500"></i>
                        探索完成
                    </div>
                    <div class="text-xs font-mono bg-green-100 px-2 py-1 rounded text-green-700">
                        时间已用尽
                    </div>
                </div>
                <div class="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
                    <div class="bg-green-400 h-full rounded-full" style="width: 100%"></div>
                </div>
                <div class="mt-3 grid grid-cols-2 gap-2">
                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-[10px] text-gray-400">已探索节点</div>
                        <div class="font-mono font-bold text-gray-700 text-xs">${stats.totalNodes} / ${stats.totalNodes}</div>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-[10px] text-gray-400">最优代码状态</div>
                        <div class="font-bold text-green-600 text-xs">✓ 可提交</div>
                    </div>
                </div>
                <div class="mt-2 text-[10px] text-gray-400 text-right">
                    成功 ${stats.successNodes} · 失败 ${stats.failNodes}
                </div>
            `;
        }

        // 评估结果卡片
        const { val_auc, val_accuracy, overfit_ratio, train_auc } = bestMetrics || {};
        const html = `
            <div id="depth-result-card" class="self-start bg-white border border-gray-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-3 text-gray-800 flex items-center gap-2">
                    <i class="ph-fill ph-trophy text-yellow-500"></i>
                    深度探索最优结果
                </div>
                <div class="bg-gray-50 rounded-xl p-3 mb-3">
                    <div class="grid grid-cols-3 gap-2 text-center">
                        <div>
                            <div class="text-lg font-bold text-purple-600">${val_auc || 'N/A'}</div>
                            <div class="text-xs text-gray-400">验证集 AUC</div>
                        </div>
                        <div>
                            <div class="text-lg font-bold text-green-600">${val_accuracy || 'N/A'}</div>
                            <div class="text-xs text-gray-400">准确率</div>
                        </div>
                        <div>
                            <div class="text-lg font-bold ${(overfit_ratio > 1.05) ? 'text-red-500' : 'text-green-600'}">${overfit_ratio || 'N/A'}</div>
                            <div class="text-xs text-gray-400">过拟合比</div>
                        </div>
                    </div>
                </div>
                <div class="text-xs text-gray-500 mb-3">
                    探索统计：共探索 <span class="font-mono text-gray-700">${stats.totalNodes}</span> 个节点，
                    成功 <span class="font-mono text-green-600">${stats.successNodes}</span> 个，
                    最优来自节点 <span class="font-mono text-purple-600">#${stats.bestNodeId || 'N/A'}</span>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
    },

    // ========== 用户反馈表单 ==========
    showFeedbackForm(extendCount, maxExtend) {
        const sysContainer = document.getElementById('system-messages');
        const remainingExtends = maxExtend - extendCount;

        const html = `
            <div id="depth-feedback-card" class="self-start bg-white border border-gray-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-3 text-gray-800 flex items-center gap-2">
                    <i class="ph-fill ph-user-check text-blue-500"></i>
                    请确认是否满意
                </div>
                <p class="text-xs text-gray-500 mb-3">
                    对当前最优结果是否满意？不满意可延长探索时间（还可延长 ${remainingExtends} 次）。
                </p>
                <div class="flex gap-2" id="depth-feedback-buttons">
                    <button onclick="DepthUI.onSatisfied()" class="flex-1 py-2.5 bg-green-500 hover:bg-green-600 text-white rounded-xl text-sm font-medium transition-all active:scale-[0.98] flex items-center justify-center gap-1.5">
                        <i class="ph-fill ph-check"></i> 满意，生成产物
                    </button>
                    ${remainingExtends > 0 ? `
                    <button onclick="DepthUI.showExtendInput()" class="flex-1 py-2.5 bg-white border border-gray-200 hover:bg-gray-50 text-gray-600 rounded-xl text-sm font-medium transition-all active:scale-[0.98] flex items-center justify-center gap-1.5">
                        <i class="ph-fill ph-clock-counter-clockwise"></i> 不满意，延长时间
                    </button>
                    ` : ''}
                </div>
                ${remainingExtends > 0 ? `
                <div id="depth-extend-input" class="hidden mt-3">
                    <div class="flex items-center justify-between text-[11px] text-gray-500 mb-2">
                        <span>延长时长</span>
                        <span id="depth-extend-display" class="text-blue-600 bg-blue-100 px-2 py-0.5 rounded">30 分钟</span>
                    </div>
                    <input type="range" id="depth-extend-slider" min="15" max="120" value="30"
                        class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        oninput="document.getElementById('depth-extend-display').innerText = this.value + ' 分钟'">
                    <div class="flex gap-2 mt-3">
                        <button onclick="DepthUI.onUnsatisfied()" class="flex-1 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl text-sm font-medium transition-all">
                            继续探索
                        </button>
                        <button onclick="DepthUI.hideExtendInput()" class="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-500 rounded-xl text-sm transition-all">
                            取消
                        </button>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();
    },

    showExtendInput() {
        const el = document.getElementById('depth-extend-input');
        if (el) el.classList.remove('hidden');
    },

    hideExtendInput() {
        const el = document.getElementById('depth-extend-input');
        if (el) el.classList.add('hidden');
    },

    // ========== 满意 ==========
    onSatisfied() {
        const btnContainer = document.getElementById('depth-feedback-buttons');
        const inputContainer = document.getElementById('depth-extend-input');
        if (btnContainer) btnContainer.remove();
        if (inputContainer) inputContainer.remove();

        DepthEngine.handleUserFeedback('satisfied');
    },

    // ========== 不满意（延长时间） ==========
    onUnsatisfied() {
        const slider = document.getElementById('depth-extend-slider');
        const newTime = slider ? parseInt(slider.value) : 30;

        const btnContainer = document.getElementById('depth-feedback-buttons');
        const inputContainer = document.getElementById('depth-extend-input');
        if (btnContainer) btnContainer.remove();
        if (inputContainer) inputContainer.remove();

        DepthEngine.handleUserFeedback('unsatisfied', newTime);
    },

    // ========== 继续探索 UI ==========
    onContinuing(newTime, extendCount, maxExtend) {
        const sysContainer = document.getElementById('system-messages');

        // 移除旧的反馈卡片
        const oldFeedback = document.getElementById('depth-feedback-card');
        if (oldFeedback) oldFeedback.remove();

        // 移除旧的动态状态卡片
        const oldStatus = document.getElementById('depth-status-card');
        if (oldStatus) oldStatus.remove();

        const html = `
            <div class="self-start bg-indigo-50 border border-indigo-100 rounded-2xl p-3 text-gray-700 text-xs max-w-[90%] shadow-sm mt-2 animate-fade-in">
                <div class="flex items-center gap-2">
                    <i class="ph-fill ph-arrows-clockwise text-indigo-500"></i>
                    <span>继续探索：增加 ${newTime} 分钟（${extendCount - 1} / ${maxExtend - 1}）</span>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);

        // 重新插入动态状态卡片
        const statusHtml = `
            <div id="depth-status-card" class="self-start bg-white border border-gray-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] shadow-sm mt-3 animate-fade-in">
                <div class="flex items-center justify-between mb-2">
                    <div class="font-bold text-gray-800 flex items-center gap-2">
                        <i class="ph-fill ph-hourglass text-indigo-500 animate-pulse"></i>
                        继续探索中
                    </div>
                    <div id="depth-countdown-display" class="text-xs font-mono bg-gray-100 px-2 py-1 rounded text-gray-600">
                        剩余 ${newTime} 分钟
                    </div>
                </div>
                <div class="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
                    <div id="depth-progress-bar" class="bg-gradient-to-r from-indigo-400 to-purple-400 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
                <div class="mt-3 grid grid-cols-2 gap-2">
                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-[10px] text-gray-400">已探索节点</div>
                        <div id="depth-node-counter" class="font-mono font-bold text-gray-700 text-xs">0 / --</div>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-[10px] text-gray-400">最优代码状态</div>
                        <div id="depth-best-status" class="font-bold text-gray-700 text-xs">
                            ${DepthEngine.state.bestNodeId ? `<span class="text-green-600">✓ 可提交</span> <span class="text-gray-400 font-normal">AUC ${DepthEngine.state.bestMetrics?.val_auc || '--'}</span>` : '暂无'}
                        </div>
                    </div>
                </div>
                <div class="mt-2 text-[10px] text-gray-400 text-right">
                    已运行 <span id="depth-elapsed">0.0</span> 分钟
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', statusHtml);
        Renderer.scrollToBottom();
    },

    // ========== 完成回调 ==========
    onCompleted(data) {
        this._restoreBottomControls();

        const sysContainer = document.getElementById('system-messages');
        const { metrics, files, report, forcedMessage } = data;

        let forcedHtml = '';
        if (forcedMessage) {
            forcedHtml = `<div class="mb-3 p-2 bg-amber-50 border border-amber-100 rounded-lg text-xs text-amber-700">${this._escapeHtml(forcedMessage)}</div>`;
        }

        const html = `
            <div class="self-start bg-green-50 border border-green-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-md mt-3 animate-fade-in">
                <div class="font-bold mb-2 text-green-900 flex items-center gap-2">
                    <i class="ph-fill ph-check-circle text-green-600"></i>
                    深度建模完成
                </div>
                ${forcedHtml}
                <pre class="text-xs text-gray-600 whitespace-pre-wrap font-mono bg-white p-3 rounded-xl border border-green-100 mb-3">${this._escapeHtml(report || '')}</pre>
                <div class="flex gap-2">
                    <button onclick="switchTab('code')" class="px-4 py-2 bg-white border border-green-200 rounded-xl text-xs text-green-700 hover:bg-green-50 transition-all">
                        <i class="ph-fill ph-code"></i> 查看代码
                    </button>
                    <button onclick="switchTab('results')" class="px-4 py-2 bg-white border border-green-200 rounded-xl text-xs text-green-700 hover:bg-green-50 transition-all">
                        <i class="ph-fill ph-chart-bar"></i> 查看结果
                    </button>
                    <button onclick="switchTab('files')" class="px-4 py-2 bg-white border border-green-200 rounded-xl text-xs text-green-700 hover:bg-green-50 transition-all">
                        <i class="ph-fill ph-download-simple"></i> 下载文件
                    </button>
                </div>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();

        // 渲染右侧结果面板
        this._renderResultsPanel(data);
        // 渲染右侧文件面板
        this._renderFilesPanel(files);
    },

    // ========== 用户手动停止 ==========
    onStoppedByUser(bestMetrics, stats) {
        this._restoreBottomControls();

        const sysContainer = document.getElementById('system-messages');
        const html = `
            <div class="self-start bg-gray-50 border border-gray-200 rounded-2xl p-4 text-gray-700 text-sm max-w-[90%] leading-relaxed shadow-sm mt-3 animate-fade-in">
                <div class="font-bold mb-1 text-gray-800 flex items-center gap-2">
                    <i class="ph-fill ph-hand-palm text-gray-500"></i>
                    深度探索已终止
                </div>
                <p class="text-xs text-gray-500 mb-2">您已手动停止 MCTS 探索。当前最优结果仍可在右侧标签页查看。</p>
                ${bestMetrics?.val_auc ? `
                <div class="bg-white border border-gray-100 rounded-xl p-2 text-xs">
                    <span class="text-gray-500">当前最优 AUC:</span>
                    <span class="font-bold text-purple-600 ml-1">${bestMetrics.val_auc}</span>
                    <span class="text-gray-400 ml-2">(节点 #${stats.bestNodeId || 'N/A'})</span>
                </div>
                ` : ''}
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();

        // 确保代码面板有内容
        if (window._depthCurrentCode) {
            this._renderCodePanel(window._depthCurrentCode, { nodeType: 'stopped', isFinal: true }, bestMetrics);
        }
    },

    // ========== 恢复底部控制 ==========
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

    // ========== 渲染右侧结果面板 ==========
    _renderResultsPanel(data) {
        const { metrics, featureImportance, testPredictions } = data;
        const panel = document.getElementById('tab-panel-results');
        if (!panel) return;

        const emptyState = document.getElementById('results-empty-state');
        const contentState = document.getElementById('results-content-state');
        if (emptyState) emptyState.classList.add('hidden');
        if (contentState) contentState.classList.remove('hidden');

        // 特征重要性图
        let featureChartHtml = '';
        if (featureImportance && featureImportance.length > 0) {
            const maxImp = Math.max(...featureImportance.map(f => f.importance));
            const rows = featureImportance.map(f => {
                const pct = (f.importance / maxImp * 100).toFixed(1);
                return `
                    <div class="flex items-center gap-2 text-xs mb-1">
                        <div class="w-24 text-gray-400 truncate text-right">${f.name}</div>
                        <div class="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
                            <div class="bg-purple-500 h-full rounded-full" style="width: ${pct}%"></div>
                        </div>
                        <div class="w-10 text-gray-300 text-right">${(f.importance * 100).toFixed(1)}%</div>
                    </div>
                `;
            }).join('');
            featureChartHtml = `
                <div class="bg-gray-800 rounded-xl p-4 mb-4">
                    <div class="text-sm font-medium text-gray-200 mb-3">特征重要性 TOP 8</div>
                    ${rows}
                </div>
            `;
        }

        // 预测表格（仅在用户满意后才生成）
        let predictionTableHtml = '';
        if (testPredictions === null) {
            predictionTableHtml = `
                <div class="bg-gray-800 rounded-xl p-4 mb-4">
                    <div class="text-sm font-medium text-gray-200 mb-2">测试集预测</div>
                    <div class="text-xs text-gray-400 flex items-center gap-2">
                        <i class="ph-fill ph-lock-key text-gray-500"></i>
                        将在确认满意后生成测试集预测结果
                    </div>
                </div>
            `;
        } else if (testPredictions && testPredictions.length > 0) {
            const rows = testPredictions.map(p => `
                <tr class="border-b border-gray-800">
                    <td class="py-1.5 text-gray-400">${p.id}</td>
                    <td class="py-1.5 text-purple-400 font-mono">${p.prob.toFixed(4)}</td>
                    <td class="py-1.5">
                        <span class="px-2 py-0.5 rounded text-xs ${p.pred === 1 ? 'bg-purple-500/20 text-purple-400' : 'bg-gray-700 text-gray-400'}">
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

        // 指标卡片
        const metricCards = `
            <div class="grid grid-cols-4 gap-3 mb-4">
                <div class="bg-gray-800 rounded-xl p-3 text-center">
                    <div class="text-xl font-bold text-purple-400">${metrics?.val_auc || '--'}</div>
                    <div class="text-xs text-gray-500 mt-1">验证集 AUC</div>
                </div>
                <div class="bg-gray-800 rounded-xl p-3 text-center">
                    <div class="text-xl font-bold text-green-400">${metrics?.val_accuracy || '--'}</div>
                    <div class="text-xs text-gray-500 mt-1">准确率</div>
                </div>
                <div class="bg-gray-800 rounded-xl p-3 text-center">
                    <div class="text-xl font-bold ${metrics?.overfit_severe ? 'text-red-400' : 'text-green-400'}">${metrics?.overfit_ratio || '--'}</div>
                    <div class="text-xs text-gray-500 mt-1">过拟合比</div>
                </div>
                <div class="bg-gray-800 rounded-xl p-3 text-center">
                    <div class="text-xl font-bold text-blue-400">${metrics?.train_auc || '--'}</div>
                    <div class="text-xs text-gray-500 mt-1">训练集 AUC</div>
                </div>
            </div>
        `;

        contentState.innerHTML = `
            <div class="mb-4">
                <div class="text-xs text-gray-400 font-medium mb-3">深度模式 · MCTS 最优结果</div>
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

        const fileRows = files.map(f => `
            <div class="flex items-center justify-between p-3 bg-gray-800 rounded-xl hover:bg-gray-750 transition-colors group">
                <div class="flex items-center gap-3">
                    <i class="ph-fill ${fileTypeIcons[f.type] || 'ph-file'} ${fileTypeColors[f.type] || 'text-gray-400'} text-xl"></i>
                    <div>
                        <div class="text-sm text-gray-200">${f.name}</div>
                        <div class="text-xs text-gray-500">${f.desc}</div>
                    </div>
                </div>
                <div class="flex items-center gap-3">
                    <span class="text-xs text-gray-500">${f.size}</span>
                    <button class="p-1.5 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors" title="下载">
                        <i class="ph-fill ph-download-simple"></i>
                    </button>
                </div>
            </div>
        `).join('');

        contentState.innerHTML = `
            <div class="text-xs text-gray-400 font-medium mb-3">深度模式 · 生成文件列表</div>
            <div class="space-y-2">${fileRows}</div>
        `;
    },

    // ========== 标签页恢复 ==========
    restoreTab(tab) {
        if (tab === 'code') {
            if (window._depthCurrentCode && window.CodeEditor) {
                this._renderCodePanel(window._depthCurrentCode, window._depthCurrentCodeMeta || {}, null);
            }
        } else if (tab === 'results') {
            if (DepthEngine.state.bestMetrics) {
                const hasTestPredictions = DepthEngine.state.phase === 'completed';
                this._renderResultsPanel({
                    metrics: DepthEngine.state.bestMetrics,
                    featureImportance: DepthMockData.getFeatureImportance(),
                    testPredictions: hasTestPredictions ? DepthMockData.getTestPredictions() : null
                });
            }
        } else if (tab === 'files') {
            if (DepthEngine.state.phase === 'completed' || DepthEngine.state.phase === 'stopped') {
                this._renderFilesPanel(DepthMockData.getFiles());
            }
        }
    },

    _escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
};

window.DepthUI = DepthUI;

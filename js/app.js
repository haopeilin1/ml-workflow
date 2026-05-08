/**
 * 应用入口
 * 负责页面初始化、事件绑定、流程启动
 */

const App = {
    init() {
        this.bindEvents();
        Terminal.init();
        this.loadLLMConfig();
        console.log('[ML Agent] App initialized');
    },

    bindEvents() {
        // 文件输入
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        // 拖拽上传（欢迎页）
        const dropZone = document.getElementById('drop-zone');
        if (dropZone) {
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                }, false);
            });
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
            });
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
            });
            dropZone.addEventListener('drop', (e) => this.handleFiles(e.dataTransfer.files), false);
        }

        // 底部输入框回车发送
        const bottomInput = document.getElementById('bottom-input');
        if (bottomInput) {
            bottomInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    IntentFlow.handleUserSend();
                }
            });
        }
    },

    // ===================== 文件上传处理 =====================
    // ===================== LLM 配置 =====================
    loadLLMConfig() {
        const saved = AppState.llmConfig;
        document.getElementById('llm-api-key').value = saved.apiKey;
        document.getElementById('llm-base-url').value = saved.baseUrl;
        document.getElementById('llm-model').value = saved.model;
        document.getElementById('llm-enabled').checked = saved.enabled;
        document.getElementById('llm-provider').value = saved.provider;
        document.getElementById('llm-separate-configs').checked = saved.useSeparateConfigs;
        this.updateProviderUI(saved.provider);
        this.updateLLMStatusDot();
        this.toggleSeparateConfigs();
        // 加载独立配置
        if (saved.intent) {
            document.getElementById('intent-provider').value = saved.intent.provider;
            document.getElementById('intent-base-url').value = saved.intent.baseUrl;
            document.getElementById('intent-api-key').value = saved.intent.apiKey;
            document.getElementById('intent-model').value = saved.intent.model;
        }
        if (saved.planCoding) {
            document.getElementById('plan-provider').value = saved.planCoding.provider;
            document.getElementById('plan-base-url').value = saved.planCoding.baseUrl;
            document.getElementById('plan-api-key').value = saved.planCoding.apiKey;
            document.getElementById('plan-model').value = saved.planCoding.model;
        }
        if (saved.evaluation) {
            document.getElementById('eval-provider').value = saved.evaluation.provider;
            document.getElementById('eval-base-url').value = saved.evaluation.baseUrl;
            document.getElementById('eval-api-key').value = saved.evaluation.apiKey;
            document.getElementById('eval-model').value = saved.evaluation.model;
        }
    },

    toggleSeparateConfigs() {
        const checked = document.getElementById('llm-separate-configs').checked;
        const panel = document.getElementById('separate-configs-panel');
        if (checked) {
            panel.classList.remove('hidden');
        } else {
            panel.classList.add('hidden');
        }
    },

    onProviderChange() {
        const provider = document.getElementById('llm-provider').value;
        this.updateProviderUI(provider);
    },

    updateProviderUI(provider) {
        const baseUrlInput = document.getElementById('llm-base-url');
        const apiKeyInput = document.getElementById('llm-api-key');
        const hintText = document.getElementById('provider-hint-text');

        const presets = {
            'openai': {
                baseUrl: 'https://api.openai.com/v1',
                model: 'gpt-4o-mini',
                hint: '支持 OpenAI、Azure、Moonshot、DeepSeek 等所有兼容 OpenAI 格式的云端 API。',
                needsKey: true
            },
            'local-openai': {
                baseUrl: 'http://localhost:1234/v1',
                model: 'local-model',
                hint: '适用于 LM Studio、vLLM、text-generation-webui 等本地服务。启动本地服务后填入对应的端口地址即可。',
                needsKey: false
            },
            'ollama': {
                baseUrl: 'http://localhost:11434',
                model: 'llama3',
                hint: '适用于 Ollama 本地部署。请确保已在本地运行 `ollama run 模型名`，并填入正确的端口号（默认 11434）。',
                needsKey: false
            }
        };

        const preset = presets[provider];
        if (preset) {
            if (!baseUrlInput.value || baseUrlInput.value === presets['openai'].baseUrl || baseUrlInput.value === presets['local-openai'].baseUrl || baseUrlInput.value === presets['ollama'].baseUrl) {
                baseUrlInput.value = preset.baseUrl;
            }
            if (!document.getElementById('llm-model').value || document.getElementById('llm-model').value === 'gpt-4o-mini' || document.getElementById('llm-model').value === 'local-model' || document.getElementById('llm-model').value === 'llama3') {
                document.getElementById('llm-model').value = preset.model;
            }
            hintText.textContent = preset.hint;
            if (!preset.needsKey) {
                apiKeyInput.placeholder = '本地部署无需填写';
            } else {
                apiKeyInput.placeholder = 'sk-...';
            }
        }
    },

    openLLMConfig() {
        const modal = document.getElementById('llm-config-modal');
        modal.classList.remove('hidden');
        modal.classList.add('flex');
        this.loadLLMConfig();
    },

    closeLLMConfig() {
        const modal = document.getElementById('llm-config-modal');
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    },

    toggleLLM() {
        const enabled = document.getElementById('llm-enabled').checked;
        AppState.llmConfig.enabled = enabled;
        this.updateLLMStatusDot();
    },

    saveLLMConfig() {
        const provider = document.getElementById('llm-provider').value;
        const apiKey = document.getElementById('llm-api-key').value.trim();
        const baseUrl = document.getElementById('llm-base-url').value.trim();
        const model = document.getElementById('llm-model').value.trim();
        const enabled = document.getElementById('llm-enabled').checked;
        const useSeparateConfigs = document.getElementById('llm-separate-configs').checked;

        const intent = {
            provider: document.getElementById('intent-provider').value,
            baseUrl: document.getElementById('intent-base-url').value.trim(),
            apiKey: document.getElementById('intent-api-key').value.trim(),
            model: document.getElementById('intent-model').value.trim()
        };
        const planCoding = {
            provider: document.getElementById('plan-provider').value,
            baseUrl: document.getElementById('plan-base-url').value.trim(),
            apiKey: document.getElementById('plan-api-key').value.trim(),
            model: document.getElementById('plan-model').value.trim()
        };
        const evaluation = {
            provider: document.getElementById('eval-provider').value,
            baseUrl: document.getElementById('eval-base-url').value.trim(),
            apiKey: document.getElementById('eval-api-key').value.trim(),
            model: document.getElementById('eval-model').value.trim()
        };

        AppState.llmConfig = { enabled, provider, apiKey, baseUrl, model, useSeparateConfigs, intent, planCoding, evaluation };
        localStorage.setItem('mlworkflow_provider', provider);
        localStorage.setItem('mlworkflow_api_key', apiKey);
        localStorage.setItem('mlworkflow_base_url', baseUrl);
        localStorage.setItem('mlworkflow_model', model);
        localStorage.setItem('mlworkflow_separate', useSeparateConfigs);
        localStorage.setItem('mlworkflow_intent_provider', intent.provider);
        localStorage.setItem('mlworkflow_intent_api_key', intent.apiKey);
        localStorage.setItem('mlworkflow_intent_base_url', intent.baseUrl);
        localStorage.setItem('mlworkflow_intent_model', intent.model);
        localStorage.setItem('mlworkflow_plan_provider', planCoding.provider);
        localStorage.setItem('mlworkflow_plan_api_key', planCoding.apiKey);
        localStorage.setItem('mlworkflow_plan_base_url', planCoding.baseUrl);
        localStorage.setItem('mlworkflow_plan_model', planCoding.model);
        localStorage.setItem('mlworkflow_eval_provider', evaluation.provider);
        localStorage.setItem('mlworkflow_eval_api_key', evaluation.apiKey);
        localStorage.setItem('mlworkflow_eval_base_url', evaluation.baseUrl);
        localStorage.setItem('mlworkflow_eval_model', evaluation.model);

        this.updateLLMStatusDot();
        this.closeLLMConfig();

        const providerName = { 'openai': '云端 API', 'local-openai': '本地 OpenAI', 'ollama': 'Ollama' }[provider];
        const modeText = useSeparateConfigs ? '（独立配置）' : '（统一配置）';
        Terminal.info(`LLM 配置已保存：${providerName} ${enabled ? '（已启用）' : '（已禁用）'}${modeText}`);
    },

    async testLLMConfig() {
        const provider = document.getElementById('llm-provider').value;
        const apiKey = document.getElementById('llm-api-key').value.trim();
        const baseUrl = document.getElementById('llm-base-url').value.trim();
        const model = document.getElementById('llm-model').value.trim();

        if (provider === 'openai' && !apiKey) {
            alert('云端 API 需要输入 API Key');
            return;
        }

        try {
            let response;
            if (provider === 'ollama') {
                response = await fetch(`${baseUrl}/api/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        messages: [{ role: 'user', content: 'Hello' }],
                        stream: false
                    })
                });
            } else {
                const headers = { 'Content-Type': 'application/json' };
                if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
                response = await fetch(`${baseUrl}/chat/completions`, {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({
                        model: model,
                        messages: [{ role: 'user', content: 'Hello' }],
                        max_tokens: 5
                    })
                });
            }

            if (response.ok) {
                const data = await response.json();
                const content = data.message?.content || data.choices?.[0]?.message?.content;
                alert(`连接成功！模型返回："${content?.substring(0, 50)}..."`);
            } else {
                const err = await response.text();
                alert(`连接失败：HTTP ${response.status}\n${err.substring(0, 200)}`);
            }
        } catch (err) {
            let hint = '';
            if (provider === 'ollama') {
                hint = '【Ollama 常见原因】\n1. Ollama 未启动：在终端运行 `ollama serve`\n2. CORS 跨域限制：运行 `OLLAMA_ORIGINS=* ollama serve` 允许浏览器访问\n3. 模型未下载：运行 `ollama pull ' + model + '`\n4. 端口被占用：检查是否有其他服务占用了 11434 端口';
            } else if (provider === 'local-openai') {
                hint = '【本地服务常见原因】\n1. 服务未启动：请启动 LM Studio / vLLM / text-generation-webui\n2. 端口不对：检查服务实际监听的端口（LM Studio 默认 1234，vLLM 默认 8000）\n3. 防火墙/代理：检查是否被系统防火墙或代理软件拦截';
            } else {
                hint = '【云端 API 常见原因】\n1. 网络问题：检查网络连接或代理设置\n2. Key 无效：确认 API Key 是否正确\n3. Base URL 错误：确认接口地址是否完整（需包含 /v1）';
            }
            alert(`连接失败：${err.message}\n\n${hint}`);
        }
    },

    updateLLMStatusDot() {
        const enabled = AppState.llmConfig.enabled && (AppState.llmConfig.apiKey || AppState.llmConfig.provider !== 'openai');
        
        // 工作台状态点
        const dot = document.getElementById('llm-status-dot');
        if (dot) {
            dot.className = `absolute top-0.5 right-0.5 w-2 h-2 rounded-full ${enabled ? 'bg-green-500' : 'bg-gray-300'}`;
        }
        
        // 欢迎页状态点
        const welcomeDot = document.getElementById('welcome-llm-status');
        if (welcomeDot) {
            welcomeDot.className = `w-2 h-2 rounded-full ${enabled ? 'bg-green-500' : 'bg-gray-300'}`;
        }
    },

    handleFileSelect(e) {
        this.handleFiles(e.target.files);
    },

    handleFiles(fileList) {
        const files = Array.from(fileList);
        const validTypes = ['csv', 'xlsx', 'xls'];
        const maxSize = 500 * 1024 * 1024; // 500MB
        const maxCount = 5;

        // 校验数量
        if (files.length > maxCount) {
            alert(`一次最多上传 ${maxCount} 个文件，您选择了 ${files.length} 个。`);
            return;
        }

        const validFiles = [];
        const errors = [];

        for (const file of files) {
            const ext = file.name.split('.').pop().toLowerCase();

            if (!validTypes.includes(ext)) {
                errors.push(`${file.name}: 不支持的格式（仅支持 CSV, XLSX, XLS）`);
                continue;
            }

            if (file.size > maxSize) {
                errors.push(`${file.name}: 文件过大（最大 500MB）`);
                continue;
            }

            validFiles.push(file);
        }

        if (errors.length > 0) {
            alert(errors.join('\n'));
        }

        if (validFiles.length === 0) return;

        // 添加到状态
        validFiles.forEach(file => {
            AppState.uploadedFiles.push({
                file: file,
                name: file.name,
                size: Utils.formatFileSize(file.size),
                type: file.name.split('.').pop().toLowerCase(),
                id: Utils.uid(),
                role: 'unknown'
            });
        });

        // 自动识别训练集/测试集
        this._detectFileRoles();
        this.updateFileDisplay();
    },

    /**
     * 自动识别文件角色（训练集/测试集/验证集）
     * 优先将包含 train 关键字的文件设为主数据集
     */
    _detectFileRoles() {
        const files = AppState.uploadedFiles;
        if (files.length === 0) return;

        const trainKeywords = ['train', 'training', '训练', '学习', 'learn'];
        const testKeywords = ['test', 'testing', '预测', 'predict', 'submission'];
        const valKeywords = ['val', 'validation', 'valid', '验证', 'dev'];

        // 先给每个文件打上角色标签
        files.forEach(f => {
            const lower = f.name.toLowerCase();
            if (trainKeywords.some(k => lower.includes(k))) {
                f.role = 'train';
            } else if (testKeywords.some(k => lower.includes(k))) {
                f.role = 'test';
            } else if (valKeywords.some(k => lower.includes(k))) {
                f.role = 'validation';
            } else {
                f.role = 'unknown';
            }
        });

        // 如果没有明确的 train 文件，将第一个 unknown 视为 train
        const hasTrain = files.some(f => f.role === 'train');
        if (!hasTrain) {
            const firstUnknown = files.find(f => f.role === 'unknown');
            if (firstUnknown) firstUnknown.role = 'train';
        }

        // 将第一个 train 文件设为主数据集（activeFileIndex）
        const trainIndex = files.findIndex(f => f.role === 'train');
        if (trainIndex !== -1) {
            AppState.activeFileIndex = trainIndex;
        }
    },

    updateFileDisplay() {
        const badge = document.getElementById('file-badge');
        const nameEl = document.getElementById('file-name');
        const sizeEl = document.getElementById('file-size');
        const listEl = document.getElementById('file-list');

        if (AppState.uploadedFiles.length === 0) {
            badge.classList.add('hidden');
            if (listEl) listEl.innerHTML = '';
            return;
        }

        badge.classList.remove('hidden');
        const active = AppState.uploadedFiles[AppState.activeFileIndex];
        nameEl.textContent = active.name;

        if (AppState.uploadedFiles.length === 1) {
            sizeEl.textContent = active.size;
            sizeEl.classList.remove('hidden');
        } else {
            sizeEl.textContent = `共${AppState.uploadedFiles.length}个文件`;
            sizeEl.classList.remove('hidden');
        }

        // 多文件列表展示
        if (listEl) {
            const roleLabels = { train: '训练集', test: '测试集', validation: '验证集', unknown: '' };
            const roleColors = { train: 'bg-green-100 text-green-700', test: 'bg-blue-100 text-blue-700', validation: 'bg-amber-100 text-amber-700' };
            listEl.innerHTML = AppState.uploadedFiles.map((f, i) => `
                <div class="file-chip ${i === AppState.activeFileIndex ? 'active' : ''}" onclick="App.setActiveFile(${i})" title="${f.role === 'train' ? '主数据集（用于建模）' : f.role === 'test' ? '测试集（用于预测）' : '点击设为主数据集'}">
                    <i class="ph-fill ph-file-${f.type === 'csv' ? 'csv' : 'xls'}"></i>
                    ${f.name}
                    ${f.role !== 'unknown' ? `<span class="text-[10px] px-1.5 py-0.5 rounded ${roleColors[f.role] || ''} ml-1">${roleLabels[f.role]}</span>` : ''}
                    <span class="remove text-gray-400 hover:text-red-500 cursor-pointer" onclick="event.stopPropagation(); App.removeFile(${i})">
                        <i class="ph-fill ph-x-circle"></i>
                    </span>
                </div>
            `).join('');
        }
    },

    setActiveFile(index) {
        // 用户手动切换主数据集时，更新文件角色
        AppState.uploadedFiles.forEach(f => { if (f.role === 'train') f.role = 'unknown'; });
        AppState.uploadedFiles[index].role = 'train';
        AppState.activeFileIndex = index;
        this.updateFileDisplay();
    },

    removeFile(index) {
        const removed = AppState.uploadedFiles.splice(index, 1)[0];
        if (AppState.activeFileIndex >= AppState.uploadedFiles.length) {
            AppState.activeFileIndex = Math.max(0, AppState.uploadedFiles.length - 1);
        }
        // 如果删除的是训练集，重新自动检测角色
        if (removed && removed.role === 'train') {
            this._detectFileRoles();
        }
        this.updateFileDisplay();
    },

    // ===================== 停止任务 =====================
    onStopClicked() {
        if (AppState.phase === 'fast_mode' && window.FastEngine) {
            FastEngine.stop();
            // 恢复底部按钮
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
        } else if (AppState.phase === 'depth_mode' && window.DepthEngine) {
            DepthEngine.stop();
        } else if (window.IntentFlow) {
            IntentFlow.stop();
        }
    },

    // ===================== 页面切换 =====================
    transitionToWorkspace() {
        const desc = document.getElementById('main-input').value.trim();
        if (!desc) {
            alert('请先描述您的任务');
            return;
        }

        if (AppState.uploadedFiles.length === 0) {
            alert('请先上传至少一个数据文件（CSV / XLSX / XLS）');
            return;
        }

        AppState.userDescription = desc;

        document.getElementById('screen-1').classList.add('opacity-0', 'pointer-events-none');
        setTimeout(() => {
            const s2 = document.getElementById('screen-2');
            s2.classList.remove('opacity-0', 'pointer-events-none');
            s2.classList.add('opacity-100');

            // 渲染用户初始消息
            Renderer.renderUserInitial(AppState.uploadedFiles, desc);
            // 开始数据画像
            IntentFlow.startDataProfiling();
        }, 300);
    },

    // ===================== 返回欢迎页 =====================
    returnToWelcome() {
        try {
            console.log('[App] returnToWelcome called');
            // 1. 停止任何运行中的引擎
            if (AppState.phase === 'fast_mode' && window.FastEngine) {
                FastEngine.stop();
            } else if (AppState.phase === 'depth_mode' && window.DepthEngine) {
                DepthEngine.stop();
            } else if (window.IntentFlow) {
                IntentFlow.stop();
            }

            // 2. 重置全局状态
            AppState.reset();
            if (window.FastEngine) {
                FastEngine._resetState();
            }

            // 3. 清空左侧对话区
            Renderer.clearSystemMessages();
            const userMessages = document.getElementById('user-messages');
            if (userMessages) userMessages.innerHTML = '';

            // 4. 重置右侧标签页与面板
            switchTab('terminal');
            if (window.FastUI) {
                FastUI._codeHistory = [];
                FastUI._resetResultsPanel();
            }
            const codeEmpty = document.getElementById('code-empty-state');
            const codeContent = document.getElementById('code-content-state');
            if (codeEmpty) codeEmpty.classList.remove('hidden');
            if (codeContent) { codeContent.classList.add('hidden'); codeContent.innerHTML = ''; }

            const filesEmpty = document.getElementById('files-empty-state');
            const filesContent = document.getElementById('files-content-state');
            if (filesEmpty) filesEmpty.classList.remove('hidden');
            if (filesContent) { filesContent.classList.add('hidden'); filesContent.innerHTML = ''; }

            // 5. 重置终端
            if (window.Terminal) {
                Terminal.init();
            }

            // 6. 重置文件上传 UI
            const fileBadge = document.getElementById('file-badge');
            const fileList = document.getElementById('file-list');
            const mainInput = document.getElementById('main-input');
            const fileName = document.getElementById('file-name');
            const fileSize = document.getElementById('file-size');
            if (fileBadge) fileBadge.classList.add('hidden');
            if (fileList) fileList.innerHTML = '';
            if (mainInput) mainInput.value = '';
            if (fileName) fileName.textContent = '--';
            if (fileSize) fileSize.textContent = '--';

            // 7. 重置阶段指示器
            ['stage-profile', 'stage-intent', 'stage-confirm', 'stage-mode'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active', 'done');
            });

            // 8. 切换回欢迎页（使用 style.display 确保可靠）
            const s2 = document.getElementById('screen-2');
            const s1 = document.getElementById('screen-1');
            if (s2) {
                s2.classList.add('opacity-0', 'pointer-events-none');
                s2.classList.remove('opacity-100');
                s2.style.display = 'none';
            }
            if (s1) {
                s1.classList.remove('opacity-0', 'pointer-events-none');
                s1.style.display = 'flex';
            }
            console.log('[App] returnToWelcome done');
        } catch (e) {
            console.error('[App] returnToWelcome error:', e);
            alert('返回失败: ' + e.message);
        }
    }
};

// ===================== 全局暴露 =====================
window.App = App;

// ===================== 标签切换 =====================
window.switchTab = function(tab) {
    // 切换按钮样式
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.dataset.tab === tab) {
            btn.classList.remove('bg-gray-100', 'text-gray-500');
            btn.classList.add('bg-[#8CB4FF]', 'text-blue-950');
        } else {
            btn.classList.remove('bg-[#8CB4FF]', 'text-blue-950');
            btn.classList.add('bg-gray-100', 'text-gray-500');
        }
    });

    // 切换面板显示
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.add('hidden');
    });
    const activePanel = document.getElementById('tab-panel-' + tab);
    if (activePanel) {
        activePanel.classList.remove('hidden');
    }

    // 快速模式下恢复对应标签页内容
    if (AppState.phase === 'fast_mode' && window.FastUI) {
        FastUI.restoreTab(tab);
    }
    // 深度模式下恢复对应标签页内容
    if (AppState.phase === 'depth_mode' && window.DepthUI) {
        DepthUI.restoreTab(tab);
    }
};

// ===================== 初始化 =====================
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

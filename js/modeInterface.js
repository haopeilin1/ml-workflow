/**
 * 模式接口层
 * 快速模式 / 深度模式的入口与预留接口
 * 意图澄清完成后，通过此模块进入具体建模流程
 */

const ModeInterface = {
    /**
     * 选择模式并启动
     */
    async selectMode(mode) {
        if (!AppState.isReadyToBuild || !AppState.taskConfirmed) {
            alert('请先完成意图澄清并确认任务');
            return;
        }

        const time = mode === 'depth'
            ? parseInt(document.getElementById('time-slider')?.value || 60)
            : null;

        // 禁用模式按钮防止重复点击
        const fastBtn = document.getElementById('btn-mode-fast');
        const depthBtn = document.getElementById('btn-mode-depth');
        if (fastBtn) fastBtn.classList.add('mode-btn-disabled');
        if (depthBtn) depthBtn.classList.add('mode-btn-disabled');

        // 渲染执行确认消息
        const sysContainer = document.getElementById('system-messages');
        const html = `
            <div class="self-start bg-blue-50/50 border border-blue-100 rounded-2xl p-4 text-gray-700 text-sm max-w-[85%] leading-relaxed shadow-sm mt-2 animate-fade-in">
                <div class="font-bold mb-1 text-blue-900 flex items-center gap-2">
                    <i class="ph-fill ph-check-circle text-blue-500"></i> 执行确认
                </div>
                <p class="text-gray-600">${mode === 'fast' ? '已启动快速模式。正在为您编写 Baseline 脚本并加载数据...' : `已启动深度模式（设定时长：${time}分钟）。正在初始化自动化特征工程环境...`}</p>
            </div>
        `;
        sysContainer.insertAdjacentHTML('beforeend', html);
        Renderer.scrollToBottom();

        // 生成最终配置
        const taskConfig = AppState.generateTaskConfig();

        if (mode === 'fast') {
            await this._startFastMode(taskConfig);
        } else {
            await this._startDepthMode(taskConfig, time);
        }
    },

    /**
     * 显示深度模式配置（时长滑块）
     */
    showDepthConfig() {
        if (!AppState.isReadyToBuild || !AppState.taskConfirmed) return;
        const config = document.getElementById('depth-config');
        if (config) config.classList.remove('hidden');
    },

    // ===================== 快速模式 =====================
    async _startFastMode(taskConfig) {
        try {
            Terminal.info('[ModeInterface] 启动快速模式...');
            await FastEngine.start(taskConfig);
        } catch (e) {
            console.error('[ModeInterface] Fast mode start failed:', e);
            Terminal.error(`快速模式启动失败: ${e.message}`);
        }
    },

    // ===================== 深度模式 =====================
    async _startDepthMode(taskConfig, timeLimitMinutes) {
        try {
            await DepthEngine.start(taskConfig, timeLimitMinutes);
        } catch (e) {
            console.error('[ModeInterface] Depth mode start failed:', e);
            Terminal.error(`深度模式启动失败: ${e.message}`);
        }
    }
};

window.ModeInterface = ModeInterface;

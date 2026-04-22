import re

with open('../js/fast/engine.js', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''    // ========== Agent 输出日志（右侧终端） ==========
    _logAgentOutputs(status) {
        // Plan & Coding Agent 的 Plan 输出
        if (status.plan && status.plan !== this.state._lastLoggedPlan) {
            this.state._lastLoggedPlan = status.plan;
            Terminal.separator();
            Terminal.system('[Plan & Coding Agent] 生成建模计划');
            Terminal.output(status.plan);
        }

        // Plan & Coding Agent 的 Code 输出
        if (status.code && status.code !== this.state._lastLoggedCode) {
            this.state._lastLoggedCode = status.code;
            const round = status.debug_round || status.optimize_round || 0;
            let label = '初始代码';
            if (status.debug_round > 0) label = `Debug 修复 (第 ${status.debug_round} 次)`;
            else if (status.optimize_round > 0) label = `优化代码 (第 ${status.optimize_round} 轮)`;

            Terminal.separator();
            Terminal.system(`[Plan & Coding Agent] 生成 ${label}`);
            Terminal.output(`\\n${status.code}\\n`);
        }

        // Evaluation Agent 的输出
        if (status.evaluation && JSON.stringify(status.evaluation) !== this.state._lastLoggedEvaluation) {
            this.state._lastLoggedEvaluation = JSON.stringify(status.evaluation);
            const ev = status.evaluation;
            Terminal.separator();
            Terminal.system('[Evaluation Agent] 评估结果');
            Terminal.info(`决策: ${ev.decision}`);
            if (ev.evaluation_analysis) {
                Terminal.output(ev.evaluation_analysis);
            }
            if (ev.suggestions_for_coding_agent) {
                Terminal.info('优化建议:');
                Terminal.output(ev.suggestions_for_coding_agent);
            }
            if (ev.report_to_user) {
                Terminal.info('用户汇报:');
                Terminal.output(ev.report_to_user);
            }
        }
    },'''

new = '''    // ========== Agent 输出日志（右侧终端） ==========
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
    },'''

if old in content:
    content = content.replace(old, new)
    with open('../js/fast/engine.js', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Patched frontend successfully')
else:
    print('ERROR: old text not found')

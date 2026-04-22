with open('../js/fast/engine.js', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start and end of _logAgentOutputs method
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '_logAgentOutputs(status)' in line:
        start_idx = i
    if start_idx is not None and end_idx is None:
        # Look for the end of the method (next method starts)
        if i > start_idx and lines[i].strip().startswith('_renderPhase'):
            end_idx = i
            break

if start_idx is None or end_idx is None:
    print(f'ERROR: Could not find method boundaries. start={start_idx}, end={end_idx}')
    exit(1)

new_method = '''    // ========== Agent 输出日志（右侧终端） ==========
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

'''

new_lines = lines[:start_idx] + [new_method] + lines[end_idx:]
with open('../js/fast/engine.js', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f'Patched lines {start_idx}-{end_idx}')

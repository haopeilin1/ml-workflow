import re

with open('app/core/fast_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Patch user feedback section
old = '''            self.state.code_history.append({
                "round": self.state.optimize_round + self.state.user_feedback_round,
                "code": code_output.code,
                "type": "user_feedback",
                "suggestion": suggestion
            })
            
            # '''

new = '''            self.state.code_history.append({
                "round": self.state.optimize_round + self.state.user_feedback_round,
                "code": code_output.code,
                "type": "user_feedback",
                "suggestion": suggestion
            })
            
            # 记录 LLM 原始响应到日志
            self._append_log("[Plan & Coding Agent] 根据用户反馈调整代码")
            if code_output.raw_response:
                self._append_log(code_output.raw_response)
            
            # '''

if old in content:
    content = content.replace(old, new, 1)
    print('Patched user_feedback section')
else:
    print('ERROR: user_feedback section not found')

with open('app/core/fast_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')

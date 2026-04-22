/**
 * 轻量级代码展示组件
 * 纯 JS/CSS 实现 Python 语法高亮，无外部依赖
 */

const CodeEditor = {
    // Python 关键字
    keywords: [
        'import', 'from', 'as', 'def', 'class', 'return', 'if', 'elif', 'else',
        'for', 'while', 'in', 'not', 'and', 'or', 'try', 'except', 'finally',
        'with', 'lambda', 'yield', 'raise', 'assert', 'break', 'continue',
        'pass', 'global', 'nonlocal', 'del', 'is', 'None', 'True', 'False',
        'print', 'json', 'pd', 'np', 'plt', 'sns', 'lgb', 'xgb'
    ],

    // 渲染代码到指定容器
    render(containerId, code, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const { showLineNumbers = true, foldable = true } = options;

        // 清空容器（保留原有类并追加编辑器类）
        container.innerHTML = '';
        container.className = (container.className || '') + ' code-editor-container';

        // 代码头部：文件名/操作
        const header = document.createElement('div');
        header.className = 'code-editor-header';
        header.innerHTML = `
            <div class="code-editor-lang">Python</div>
            <button class="code-editor-copy" onclick="CodeEditor.copyCode('${containerId}')">
                <i class="ph ph-copy"></i> 复制
            </button>
        `;
        container.appendChild(header);

        // 代码主体
        const body = document.createElement('div');
        body.className = 'code-editor-body';

        const lines = code.split('\n');
        const lineCount = lines.length;
        const lineNumWidth = String(lineCount).length * 10 + 20;

        lines.forEach((line, idx) => {
            const lineEl = document.createElement('div');
            lineEl.className = 'code-line';
            lineEl.style.display = 'flex';

            // 行号
            if (showLineNumbers) {
                const numEl = document.createElement('span');
                numEl.className = 'code-line-num';
                numEl.textContent = idx + 1;
                numEl.style.minWidth = `${lineNumWidth}px`;
                lineEl.appendChild(numEl);
            }

            // 代码内容（高亮）
            const contentEl = document.createElement('span');
            contentEl.className = 'code-line-content';
            contentEl.innerHTML = this._highlightLine(line);
            lineEl.appendChild(contentEl);

            body.appendChild(lineEl);
        });

        container.appendChild(body);
    },

    // 单行长亮
    _highlightLine(line) {
        if (!line.trim()) return '&nbsp;';

        let html = this._escapeHtml(line);

        // 注释（优先级最高）
        html = html.replace(/^(.*?)(#.*)$/, (m, code, comment) => {
            return this._highlightCode(code) + `<span class="code-comment">${comment}</span>`;
        });

        // 如果没有注释，对整个代码部分高亮
        if (!html.includes('code-comment')) {
            html = this._highlightCode(html);
        }

        return html;
    },

    // 代码 token 高亮
    _highlightCode(code) {
        // 字符串
        code = code.replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span class="code-string">$1</span>');

        // f-string
        code = code.replace(/(f"(?:[^"\\]|\\.)*")/g, '<span class="code-string">$1</span>');

        // 数字
        code = code.replace(/\b(\d+\.?\d*)\b/g, '<span class="code-number">$1</span>');

        // 关键字（使用单词边界避免部分匹配）
        this.keywords.forEach(kw => {
            const regex = new RegExp(`\\b(${kw})\\b`, 'g');
            code = code.replace(regex, '<span class="code-keyword">$1</span>');
        });

        // 函数调用
        code = code.replace(/(\w+)(\s*\()/g, '<span class="code-function">$1</span>$2');

        // 装饰器
        code = code.replace(/(@\w+)/g, '<span class="code-decorator">$1</span>');

        return code;
    },

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },

    // 复制代码
    async copyCode(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const lines = container.querySelectorAll('.code-line-content');
        const code = Array.from(lines).map(el => el.textContent).join('\n');

        try {
            await navigator.clipboard.writeText(code);
            const btn = container.querySelector('.code-editor-copy');
            const original = btn.innerHTML;
            btn.innerHTML = '<i class="ph ph-check"></i> 已复制';
            setTimeout(() => btn.innerHTML = original, 2000);
        } catch (e) {
            console.error('Copy failed:', e);
        }
    }
};

window.CodeEditor = CodeEditor;

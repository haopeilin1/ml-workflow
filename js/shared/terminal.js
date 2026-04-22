/**
 * 终端输出管理
 * 右侧工作区终端的统一输出接口
 */

const Terminal = {
    element: null,
    _buffer: '',

    init() {
        this.element = document.getElementById('terminal-content');
    },

    clear() {
        this._buffer = '';
        if (this.element) {
            this.element.innerHTML = '<span class="text-blue-500/70">waiting for instructions...</span><span class="cursor"></span>';
        }
    },

    append(html) {
        if (!this.element) this.init();
        // 移除 cursor
        const cursor = this.element.querySelector('.cursor');
        if (cursor) cursor.remove();
        
        this.element.innerHTML += html;
        this._buffer = this.element.innerHTML;
        // 重新添加 cursor
        this.element.innerHTML += '<span class="cursor"></span>';
        this.element.scrollTop = this.element.scrollHeight;
    },

    log(level, message) {
        const colors = {
            info: 'blue-400',
            warn: 'yellow-400',
            error: 'red-400',
            success: 'green-500',
            system: 'gray-100'
        };
        const color = colors[level] || 'gray-400';
        const prefix = level === 'system' ? '' : `[${level.toUpperCase()}]`;
        this.append(`<span class="text-${color}">${prefix}</span> ${message}<br>`);
    },

    info(msg) { this.log('info', msg); },
    warn(msg) { this.log('warn', msg); },
    error(msg) { this.log('error', msg); },
    success(msg) { this.log('success', msg); },
    system(msg) { this.append(`<span class="text-green-500">➜</span> <span class="text-gray-100">System:</span> ${msg}<br>`); },
    output(msg) { this.append(`<span class="text-gray-300">${msg}</span><br>`); },

    separator() {
        this.append('<br>');
    }
};

window.Terminal = Terminal;

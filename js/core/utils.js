/**
 * 通用工具函数
 */

const Utils = {
    /**
     * 格式化文件大小
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0B';
        const k = 1024;
        const sizes = ['B', 'K', 'M', 'G'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + sizes[i];
    },

    /**
     * 格式化任务类型为中文
     */
    formatTaskType(type) {
        const map = {
            'binary_classification': '二分类',
            'multiclass_classification': '多分类',
            'regression': '回归',
            'clustering': '聚类',
            'forecasting': '时序预测'
        };
        return map[type] || type;
    },

    /**
     * 推断任务类型
     */
    inferTaskType(column) {
        if (!column) return null;
        if (column.uniqueCount === 2) return 'binary_classification';
        if (column.type === 'numeric' && column.uniqueCount > 10) return 'regression';
        if (column.uniqueCount <= 10) return 'multiclass_classification';
        return null;
    },

    /**
     * 推断默认评估指标
     */
    inferEvalMetric(taskType, userInput = '') {
        if (!taskType) return null;
        const input = userInput.toLowerCase();
        if (taskType.includes('classification')) {
            if (input.includes('召回') || input.includes('不能放过') || input.includes('宁可') || input.includes('抓错') || input.includes('recall')) return 'Recall';
            if (input.includes('精确') || input.includes('准确') || input.includes('precision')) return 'Precision';
            if (input.includes('f1')) return 'F1';
            return 'AUC';
        }
        if (taskType === 'regression') {
            if (input.includes('r2') || input.includes('决定系数')) return 'R2';
            return 'RMSE';
        }
        return null;
    },

    /**
     * 检测列名是否像目标列
     */
    isTargetLikeColumn(name) {
        const keywords = ['label', 'target', 'status', 'survived', 'churn', 'class', 'category', 'y'];
        const lower = name.toLowerCase();
        // 完全匹配或前缀/后缀匹配
        if (keywords.includes(lower)) return true;
        if (lower.startsWith('is_') || lower.startsWith('has_')) return true;
        if (lower.endsWith('_label') || lower.endsWith('_target') || lower.endsWith('_y')) return true;
        return false;
    },

    /**
     * 从用户输入中提取被排除的列名
     */
    extractConstraints(userInput, allColumns) {
        const constraints = [];
        const excludePatterns = ['不要', '排除', '不用', '去掉', '忽略', '跳过', '别用', 'no need', 'exclude', 'remove', 'drop', '删除'];
        allColumns.forEach(col => {
            if (userInput.includes(col)) {
                const idx = userInput.indexOf(col);
                const before = userInput.substring(Math.max(0, idx - 25), idx);
                const after = userInput.substring(idx + col.length, Math.min(userInput.length, idx + col.length + 10));
                const context = before + ' ' + after;
                if (excludePatterns.some(p => context.includes(p) || before.includes(p))) {
                    constraints.push(col);
                }
            }
        });
        return [...new Set(constraints)];
    },

    /**
     * 检测用户输入是否为确认
     */
    isConfirmation(input) {
        const confirms = ['可以', '是的', '对', 'ok', '好', '行', '没错', '正确', '确认', '同意', '没问题'];
        return confirms.some(c => input.toLowerCase().includes(c.toLowerCase()));
    },

    /**
     * 检测用户输入是否是否定
     */
    isRejection(input) {
        const rejects = ['不行', '不对', '不是', '不要', '换个', '换一个', '重新', '否', 'no', '不对'];
        return rejects.some(r => input.toLowerCase().includes(r.toLowerCase()));
    },

    /**
     * 防抖函数
     */
    debounce(fn, delay) {
        let timer;
        return function(...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    },

    /**
     * 生成唯一ID
     */
    uid() {
        return 'id_' + Math.random().toString(36).substr(2, 9);
    },

    /**
     * 睡眠等待
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};

window.Utils = Utils;

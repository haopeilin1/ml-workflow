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
     * 从用户输入中提取建模建议/偏好
     * 识别用户提到的算法、方法、评估侧重、预处理要求等
     */
    extractModelingSuggestions(userInput) {
        if (!userInput) return null;
        const input = userInput.trim();
        // 建模建议关键词模式
        const suggestionPatterns = [
            // 算法/模型偏好
            /(?:请?用|使用|采用|尝试|试试|推荐|建议).{0,20}(?:XGBoost|LightGBM|Random Forest|随机森林|梯度提升|GBDT|SVM|神经网络|深度学习|逻辑回归|线性回归|决策树|KNN|贝叶斯|聚类|KMeans)/i,
            // 评估侧重
            /(?:重点关注|侧重|优先|看重|重视|尽量).{0,15}(?:召回率|Recall|精确率|Precision|准确率|Accuracy|F1|AUC|RMSE|R2|误差|损失)/i,
            /(?:宁可|宁愿|不能|避免).{0,20}(?:漏掉|放过|误杀|错杀|假阴性|假阳性)/i,
            // 预处理方法
            /(?:不要|不用|跳过|忽略).{0,10}(?:归一化|标准化|缩放|Scaling|Normalization)/i,
            /(?:需要|要|做|进行).{0,10}(?:归一化|标准化|缩放|Scaling|Normalization|PCA|降维|特征选择|编码|OneHot|独热)/i,
            // 特征工程
            /(?:创建|构造|生成|增加|添加).{0,15}(?:特征|变量|交互项|多项式|交叉)/i,
            /(?:做|进行|先).{0,10}(?:EDA|探索性分析|分布分析|相关性分析|可视化)/i,
            // 数据清洗
            /(?:删除|去掉|移除).{0,10}(?:缺失值|异常值|离群点|重复)/i,
            /(?:填充|补全|插补).{0,10}(?:缺失值|缺失)/i,
            // 训练/验证策略
            /(?:交叉验证|K折|分层采样|Stratified|时间序列划分|留出法)/i,
            // 其他建模偏好
            /(?:正则化|L1|L2|早停|Early Stopping| Bagging|Boosting|集成|Ensemble)/i,
            /(?:调参|调优|网格搜索|Grid Search|随机搜索|贝叶斯优化)/i,
            /(?:类别不平衡|不平衡|过采样|欠采样|SMOTE)/i
        ];
        const matches = [];
        for (const pattern of suggestionPatterns) {
            const match = input.match(pattern);
            if (match) {
                // 扩展匹配上下文，取前后各15个字符
                const start = Math.max(0, match.index - 15);
                const end = Math.min(input.length, match.index + match[0].length + 15);
                matches.push(input.substring(start, end).trim());
            }
        }
        if (matches.length === 0) return null;
        // 去重并合并
        const unique = [...new Set(matches)];
        return unique.join('；');
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

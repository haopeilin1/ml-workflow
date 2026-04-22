/**
 * 数据解析与画像生成
 * 支持 CSV / Excel 文件的前端解析
 * 为后续快速/深度模式提供标准化的数据画像
 */

const DataProfiler = {
    // 解析时取样的最大行数（避免前端解析过大文件导致卡顿）
    SAMPLE_SIZE: 5000,

    /**
     * 解析单个文件
     * @param {File} file
     * @returns {Promise<{headers: string[], rows: any[][], rowCount: number}>}
     */
    async parseFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();

        if (ext === 'csv') {
            return this._parseCSV(file);
        } else if (['xlsx', 'xls'].includes(ext)) {
            return this._parseExcel(file);
        } else {
            throw new Error('不支持的文件格式: ' + ext);
        }
    },

    /**
     * 使用 PapaParse 解析 CSV
     */
    _parseCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: false,
                skipEmptyLines: true,
                preview: this.SAMPLE_SIZE,
                complete: (results) => {
                    if (!results.data || results.data.length === 0) {
                        reject(new Error('CSV 文件为空'));
                        return;
                    }
                    const headers = results.data[0].map(h => String(h).trim());
                    const rows = results.data.slice(1);
                    resolve({ headers, rows, rowCount: rows.length });
                },
                error: (err) => reject(err)
            });
        });
    },

    /**
     * 使用 SheetJS 解析 Excel
     */
    async _parseExcel(file) {
        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer, { type: 'array' });
        const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
        const data = XLSX.utils.sheet_to_json(firstSheet, { header: 1, defval: '' });

        if (!data || data.length === 0) {
            throw new Error('Excel 文件为空');
        }

        // 限制采样行数
        const sampled = data.slice(0, this.SAMPLE_SIZE + 1);
        const headers = sampled[0].map(h => String(h).trim());
        const rows = sampled.slice(1);

        return { headers, rows, rowCount: rows.length };
    },

    /**
     * 分析单列特征
     */
    _analyzeColumn(name, values) {
        const nonEmpty = values.filter(v => v !== '' && v !== null && v !== undefined);
        const total = values.length;
        const missingRate = total === 0 ? 0 : (total - nonEmpty.length) / total;

        // 尝试解析为数字
        const numericAttempts = nonEmpty.map(v => {
            const n = parseFloat(v);
            return isNaN(n) ? null : n;
        }).filter(n => n !== null);

        const numericRatio = nonEmpty.length === 0 ? 0 : numericAttempts.length / nonEmpty.length;

        let type = 'text';
        let uniqueCount = 0;
        let stats = {};

        if (numericRatio > 0.8) {
            // 数值型
            type = 'numeric';
            const sorted = numericAttempts.sort((a, b) => a - b);
            uniqueCount = new Set(numericAttempts.map(n => n.toFixed(6))).size;
            stats = {
                min: sorted[0],
                max: sorted[sorted.length - 1],
                mean: numericAttempts.reduce((a, b) => a + b, 0) / numericAttempts.length
            };
        } else {
            // 分类型或文本型
            const strValues = nonEmpty.map(String);
            uniqueCount = new Set(strValues).size;
            if (uniqueCount <= 20 && nonEmpty.length > 0) {
                type = 'categorical';
            } else {
                type = 'text';
            }
        }

        return { name, type, missingRate, uniqueCount, stats };
    },

    /**
     * 生成数据画像
     */
    generateProfile(parsedData, fileMeta) {
        const { headers, rows, rowCount } = parsedData;
        const colCount = headers.length;

        // 转置：按列组织数据
        const columnsData = headers.map((_, colIdx) => rows.map(row => row[colIdx]));

        // 分析每列
        const columns = headers.map((name, idx) => this._analyzeColumn(name, columnsData[idx]));

        // 识别目标候选列
        const targetCandidates = columns
            .filter(col => {
                // 规则1：列名像目标列
                if (Utils.isTargetLikeColumn(col.name)) return true;
                // 规则2：唯一值很少（二元或少数类别）
                if (col.uniqueCount <= 5 && col.uniqueCount >= 2) return true;
                return false;
            })
            .map(col => col.name);

        // 质量评分（简单规则）
        const avgMissing = columns.reduce((sum, c) => sum + c.missingRate, 0) / columns.length;
        const hasIdCol = columns.some(c => c.uniqueCount === rowCount && rowCount > 10);
        let qualityScore = Math.round(100 - avgMissing * 100);
        if (hasIdCol) qualityScore = Math.min(qualityScore + 5, 100); // 有ID列通常是结构化数据
        if (targetCandidates.length === 0) qualityScore -= 10;
        qualityScore = Math.max(0, Math.min(100, qualityScore));

        // 识别重复行（采样内）
        const rowStrings = rows.map(r => JSON.stringify(r));
        const uniqueRows = new Set(rowStrings).size;
        const duplicateRows = rowCount - uniqueRows;

        return {
            fileName: fileMeta.name,
            fileSize: fileMeta.size,
            rowCount,
            colCount,
            columns,
            targetCandidates,
            qualityScore,
            duplicateRows
        };
    },

    /**
     * 解析并画像多个文件
     * @param {Array<{file: File, id: string}>} fileList
     * @returns {Promise<Array>}
     */
    async profileFiles(fileList) {
        const profiles = [];
        for (const item of fileList) {
            try {
                const parsed = await this.parseFile(item.file);
                const profile = this.generateProfile(parsed, {
                    name: item.file.name,
                    size: Utils.formatFileSize(item.file.size)
                });
                profiles.push(profile);
            } catch (err) {
                console.error('解析文件失败:', item.file.name, err);
                profiles.push({
                    fileName: item.file.name,
                    error: err.message,
                    rowCount: 0,
                    colCount: 0,
                    columns: [],
                    targetCandidates: [],
                    qualityScore: 0
                });
            }
        }
        return profiles;
    }
};

window.DataProfiler = DataProfiler;

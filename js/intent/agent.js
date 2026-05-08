/**
 * Intent Agent - 意图识别与澄清
 *
 * 标准接口：async callIntentAgent({ dataProfile, dialogueHistory, userInput })
 * 严格输出 System Prompt 要求的 JSON 格式
 *
 * 当 LLM 配置启用时，直接调用真实大模型 API（OpenAI 兼容格式）。
 * 当未配置或调用失败时，回退到前端规则模拟。
 */

const IntentAgent = {
    /**
     * 标准接口 - 调用 Intent Agent
     */
    async callIntentAgent({ dataProfile, dialogueHistory, userInput }) {
        // 如果启用了 LLM 配置且 API Key 存在，优先调用真实 LLM
        if (AppState.llmConfig.enabled) {
            try {
                const result = await this._callRealLLM({ dataProfile, dialogueHistory, userInput });
                return result;
            } catch (err) {
                console.warn('[IntentAgent] LLM call failed, falling back to mock:', err);
                Terminal.warn(`LLM 调用失败 (${err.message})，已切换为本地模拟模式。`);
            }
        }

        return this._mockAgent({ dataProfile, dialogueHistory, userInput });
    },

    /**
     * 调用真实 LLM API
     * 支持：OpenAI 兼容格式 / Ollama 本地格式
     */
    async _callRealLLM({ dataProfile, dialogueHistory, userInput }) {
        const config = AppState.llmConfig;
        // 如果启用了独立配置且配置了意图识别 Agent，使用独立配置
        let agentConfig = config;
        if (config.useSeparateConfigs && config.intent) {
            agentConfig = { ...config, ...config.intent };
        }
        const provider = agentConfig.provider || 'openai';
        const systemPrompt = this._buildSystemPrompt();

        // 构建对话历史
        const messages = [
            { role: 'system', content: systemPrompt }
        ];
        dialogueHistory.forEach(h => {
            messages.push({
                role: h.role === 'agent' ? 'assistant' : 'user',
                content: h.content
            });
        });
        const userPayload = this._buildUserPayload(dataProfile, userInput);
        messages.push({ role: 'user', content: userPayload });

        let response, content;

        if (provider === 'ollama') {
            // Ollama 原生格式
            response = await fetch(`${agentConfig.baseUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: agentConfig.model,
                    messages: messages,
                    stream: false,
                    options: { temperature: 0.3 }
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            content = data.message?.content;
        } else {
            // OpenAI 兼容格式（云端 + local-openai）
            const headers = { 'Content-Type': 'application/json' };
            if (agentConfig.apiKey) headers['Authorization'] = `Bearer ${agentConfig.apiKey}`;

            response = await fetch(`${agentConfig.baseUrl}/chat/completions`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    model: agentConfig.model,
                    messages: messages,
                    temperature: 0.3,
                    max_tokens: 1500
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            content = data.choices?.[0]?.message?.content;
        }

        if (!content) {
            throw new Error('LLM returned empty content');
        }

        // 解析 JSON
        const jsonStr = this._extractJson(content);
        const result = JSON.parse(jsonStr);

        // 验证必要字段
        if (!result.thought_process || !result.extracted_slots || result.is_ready_to_build === undefined || !result.reply_to_user) {
            throw new Error('LLM response missing required fields');
        }

        return result;
    },

    /**
     * 从 LLM 输出中提取 JSON
     */
    _extractJson(content) {
        // 尝试匹配 ```json ... ``` 代码块
        const codeBlockMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (codeBlockMatch) {
            return codeBlockMatch[1].trim();
        }
        // 否则直接返回（假设是纯 JSON）
        return content.trim();
    },

    /**
     * 构建 System Prompt（严格遵循产品 PRD）
     */
    _buildSystemPrompt() {
        return `你是一个经验丰富且极具亲和力的"机器学习智能副驾驶 (ML Copilot)"。你的核心目标是通过简短的自然语言对话，帮助非AI专业的实验室科研人员、业务人员明确他们的"数据建模需求"。

Background
用户已经上传了一份结构化数据表格，并准备开始建模。系统已经对数据进行了极速扫描，生成了基础的【数据画像】（包含列名、数据类型、少量统计特征）。
注意：
1. 用户界面上有"快速模式"和"深度模式"供他们点击，你的任务不是代替他们选择模式，而是帮他们理清要让大模型去解决什么核心问题。
2. 用户可能同时上传了多个文件（如训练集和测试集）。训练集包含目标列，用于建模；测试集通常不包含目标列，仅用于最终预测。请仅从训练集的列中识别目标列。

Objective
你需要从用户的输入中，提取或推断出以下 4 个核心信息槽位：
1. target_column (必须获取): 预测目标列。必须是【数据画像】中真实存在的列名。
2. task_type (推断获取): 任务类型（二分类/多分类/回归）。如果没有明确说，请根据目标列的数据特征自动推断。
3. eval_metric (推断获取): 核心评估偏好。如果用户提到"宁可抓错不能放过"等业务倾向，请映射为 Recall 等指标；否则赋予系统默认值(分类默认AUC/回归默认RMSE)。
4. feature_constraints (可选获取): 特征排除。用户明确说"不要用到XX信息"、"排除XX列"时才记录。
5. user_modeling_suggestions (可选获取): 用户的建模建议或偏好。如果用户在描述中提到了具体的建模方法、算法偏好、特征工程思路、评估侧重等（如"请用XGBoost"、"重点关注召回率"、"不要归一化"、"先做个EDA看看分布"等），请将其完整提取并记录。注意：这只是用户的"建议"，不是必须遵守的强制约束。后续规划Agent会参考这些建议，但有权根据数据和任务特点进行取舍。

Rules & Constraints (非常重要)
1. 【禁止技术黑话】：绝对不要询问算法偏好（如XGBoost还是随机森林）、超参数、缺失值填充方式、测试集切分比例等技术细节。
2. 【极简追问】：如果缺失 target_column，请主动追问。结合数据画像，列出 2-3 个最可能是目标的列名（如包含 label、target、status、is_ 等关键字，或末尾的列）供用户选择，绝不干巴巴地问"你想预测什么"。
3. 【推断优先】：对于 task_type 和 eval_metric，只要能根据目标列推断，就不要问用户。只需在总结时用大白话带过即可（例如："看起来您想预测具体金额..."）。
4. 【结束语】：当 target_column 已明确，请用简短的一句话总结任务，并提示用户："我已经完全理解您的任务。请在下方选择『快速模式』或『深度模式』开始建模。"
5. 【控制轮数】：如果在尝试 2 轮追问后用户仍未给出明确目标，请直接指定一个最具可能性的列，并询问："我们是否可以直接将 XX列 作为预测目标开始探索？"

Output Format (Strict JSON)
你必须严格输出如下 JSON 格式，不要包含任何额外的 markdown 标记或解释文字：
{
  "thought_process": "你的内部推理过程：用户说了什么？还缺什么？如何映射到表头？",
  "extracted_slots": {
    "target_column": "列名" 或 null,
    "task_type": "binary_classification" / "multiclass_classification" / "regression" 或 null,
    "eval_metric": "AUC" / "F1" / "Recall" / "Precision" / "RMSE" / "R2" 或 null,
    "feature_constraints": ["要排除的列名1", "要排除的列名2"] 或 [],
    "user_modeling_suggestions": "用户提到的建模建议原文" 或 null
  },
  "is_ready_to_build": true 或 false,
  "reply_to_user": "你要回复给用户的话。语气自然亲和，人话表达，切忌啰嗦。"
}`;
    },

    /**
     * 构建用户输入 Payload（包含数据画像和历史上下文）
     */
    _buildUserPayload(dataProfile, userInput) {
        const columns = dataProfile.columns.map(c => ({
            name: c.name,
            type: c.type,
            missing_rate: c.missingRate,
            unique_count: c.uniqueCount
        }));

        // 收集文件角色信息
        const fileRoles = AppState.uploadedFiles.map(f => ({
            name: f.name,
            role: f.role || 'unknown'
        }));

        const dataProfileJson = JSON.stringify({
            file_name: dataProfile.fileName,
            row_count: dataProfile.rowCount,
            col_count: dataProfile.colCount,
            columns: columns,
            target_candidates: dataProfile.targetCandidates || [],
            file_roles: fileRoles
        }, null, 2);

        return `【数据画像 Data Profile】: \n${dataProfileJson}\n\n【当前用户输入 User Input】: \n${userInput}`;
    },

    /**
     * 前端模拟实现 - 基于规则引擎（LLM 不可用时的 fallback）
     */
    _mockAgent({ dataProfile, dialogueHistory, userInput }) {
        const cols = dataProfile.columns.map(c => c.name);
        const targetCandidates = dataProfile.targetCandidates || [];

        const accumulatedSlots = this._accumulateSlots(dialogueHistory);
        const agentMessages = dialogueHistory.filter(h => h.role === 'agent');
        const round = agentMessages.length;

        const newConstraints = Utils.extractConstraints(userInput, cols);
        const allConstraints = [...new Set([...accumulatedSlots.feature_constraints, ...newConstraints])];

        let targetConfirmed = accumulatedSlots.target_column;

        if (!targetConfirmed && Utils.isConfirmation(userInput)) {
            const lastAgentMsg = agentMessages[agentMessages.length - 1];
            if (lastAgentMsg && lastAgentMsg.content) {
                const backtickMatch = lastAgentMsg.content.match(/`([^`]+)`/g);
                if (backtickMatch && backtickMatch.length > 0) {
                    const suggested = backtickMatch.map(m => m.replace(/`/g, '')).find(c => cols.includes(c));
                    if (suggested) targetConfirmed = suggested;
                }
            }
            if (!targetConfirmed && targetCandidates.length > 0) {
                targetConfirmed = targetCandidates[0];
            }
        }

        if (!targetConfirmed) {
            const sortedCols = [...cols].sort((a, b) => b.length - a.length);
            for (const col of sortedCols) {
                if (userInput.includes(col)) {
                    const idx = userInput.indexOf(col);
                    const context = userInput.substring(Math.max(0, idx - 15), idx + col.length + 15);
                    const targetKeywords = ['目标', '预测', '作为', '用', 'target', 'predict'];
                    if (targetKeywords.some(k => context.includes(k))) {
                        targetConfirmed = col;
                        break;
                    }
                }
            }
        }

        if (!targetConfirmed && agentMessages.length > 0) {
            const lastAgentMsg = agentMessages[agentMessages.length - 1];
            if (lastAgentMsg && /选择|选哪|哪一个/.test(lastAgentMsg.content)) {
                const numMatch = userInput.match(/(\d+)/);
                if (numMatch) {
                    const num = parseInt(numMatch[1]) - 1;
                    if (targetCandidates[num]) targetConfirmed = targetCandidates[num];
                }
            }
        }

        let taskType = accumulatedSlots.task_type;
        if (targetConfirmed && !taskType) {
            const targetCol = dataProfile.columns.find(c => c.name === targetConfirmed);
            taskType = Utils.inferTaskType(targetCol);
        }

        let evalMetric = accumulatedSlots.eval_metric;
        if (taskType && !evalMetric) {
            evalMetric = Utils.inferEvalMetric(taskType, userInput);
        }

        return this._buildResponse({
            round, targetConfirmed, taskType, evalMetric, allConstraints,
            targetCandidates, cols, dataProfile, userInput, accumulatedSlots
        });
    },

    _accumulateSlots(dialogueHistory) {
        const slots = { target_column: null, task_type: null, eval_metric: null, feature_constraints: [] };
        dialogueHistory.forEach(msg => {
            if (msg.role === 'user') {
                slots.feature_constraints.push(...Utils.extractConstraints(msg.content, []));
            }
        });
        slots.feature_constraints = [...new Set(slots.feature_constraints)];
        return slots;
    },

    _buildResponse({ round, targetConfirmed, taskType, evalMetric, allConstraints, targetCandidates, cols, dataProfile, userInput, accumulatedSlots }) {
        let isReady = false;
        let reply = '';
        let thought = '';

        if (targetConfirmed) {
            isReady = true;
            const taskName = Utils.formatTaskType(taskType) || '预测';
            const constraintText = allConstraints.length > 0
                ? `并且我会帮您排除 \`${allConstraints.join('`、`')}\` 避免干扰。`
                : '';
            reply = `好的！目标已锁定为 \`${targetConfirmed}\`，${constraintText}这是一个${taskName}任务，我会用 ${evalMetric} 作为主要评估指标。我已经完全理解您的任务啦！请在下方选择『快速模式』或『深度模式』，我们马上开跑！`;
            thought = `用户已确认或指定目标列为 '${targetConfirmed}'。任务类型推断为 ${taskName}，评估指标 ${evalMetric}。约束列：[${allConstraints.join(', ')}]。信息完整，可以开始建模。`;
        } else {
            if (round >= 2) {
                const forcedTarget = targetCandidates[0] || this._guessBestTarget(cols, dataProfile);
                reply = `我们已经聊了${round + 1}轮啦，为了不耽误您的时间，我建议直接将 \`${forcedTarget}\` 列作为预测目标开始探索，您看可以吗？`;
                thought = `已达最大澄清轮次限制(3轮)。强制推荐最可能目标列 '${forcedTarget}'。`;
            } else if (round === 1) {
                const candidates = targetCandidates.slice(0, 3);
                if (candidates.length > 0) {
                    reply = `收到！为了更准确地帮您建模，请从下面选择最符合您预测目标的列：\n${candidates.map((c, i) => `${i + 1}. \`${c}\``).join('\n')}\n\n或者您也可以直接告诉我应该用哪一列～`;
                    thought = `用户尚未明确 target_column。从数据画像中提取最可能的候选列：[${candidates.join(', ')}]，供用户选择。`;
                } else {
                    const suggested = cols.slice(0, Math.min(5, cols.length));
                    reply = `为了更好地理解您的需求，请问您希望预测哪一列呢？数据中包含以下列：\n${suggested.map((c, i) => `${i + 1}. \`${c}\``).join('\n')}\n\n请告诉我您的预测目标～`;
                    thought = `数据中没有明显的目标列候选，需要用户明确指定。`;
                }
            } else {
                const inferredTarget = targetCandidates[0] || this._guessBestTarget(cols, dataProfile);
                const colInfo = dataProfile.columns.find(c => c.name === inferredTarget);
                let colDesc = '';
                if (colInfo) {
                    if (colInfo.uniqueCount === 2) colDesc = `（取值为 0/1 或两类）`;
                    else if (colInfo.type === 'numeric') colDesc = `（数值型）`;
                    else colDesc = `（${colInfo.uniqueCount} 个不同取值）`;
                }
                reply = `我已经分析了您的数据。看起来您想做一个预测任务，数据中的 \`${inferredTarget}\` 列${colDesc}看起来很适合作为预测目标，我们把它作为目标列可以吗？\n\n另外，您希望排除哪些不相关的列吗（比如 ID、编号、姓名等唯一标识信息）？`;
                thought = `根据用户描述'${userInput}'和数据画像，推断 '${inferredTarget}' 最可能是目标列。需要用户确认 target_column，并收集 feature_constraints。`;
            }
        }

        // 从用户输入中提取建模建议
        const userModelingSuggestions = Utils.extractModelingSuggestions(userInput);

        return {
            thought_process: thought,
            extracted_slots: {
                target_column: targetConfirmed,
                task_type: taskType,
                eval_metric: evalMetric,
                feature_constraints: allConstraints,
                user_modeling_suggestions: userModelingSuggestions
            },
            is_ready_to_build: isReady,
            reply_to_user: reply
        };
    },

    _guessBestTarget(cols, dataProfile) {
        if (dataProfile.targetCandidates && dataProfile.targetCandidates.length > 0) {
            return dataProfile.targetCandidates[0];
        }
        const nonIdCols = dataProfile.columns.filter(c => {
            const isIdLike = c.uniqueCount === dataProfile.rowCount && dataProfile.rowCount > 10;
            return !isIdLike;
        });
        if (nonIdCols.length > 0) {
            return nonIdCols.sort((a, b) => a.uniqueCount - b.uniqueCount)[0].name;
        }
        return cols[cols.length - 1];
    }
};

window.IntentAgent = IntentAgent;

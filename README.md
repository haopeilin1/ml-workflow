# ML Agent — AI 自动化建模助手

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于 **大语言模型（LLM）** 驱动的自动化机器学习建模平台。用户只需上传数据文件，系统即可自动完成数据理解、特征工程、模型训练、评估优化，并生成可下载的建模产物（代码、模型、报告、预测结果）。

---

## 📌 项目能做什么

| 功能 | 说明 |
|------|------|
| **一键建模** | 上传 CSV/Excel 数据，AI 自动完成从数据清洗到模型训练的全流程 |
| **智能迭代优化** | LLM 驱动的代码生成 → 沙箱执行 → 自动评估 → 多轮优化，直到获得满意模型 |
| **人机交互反馈** | 每轮训练后展示评估指标，用户可确认满意或提出改进建议，AI 继续优化 |
| **多模式支持** | **快速模式**：分钟级出 baseline；**深度模式**：（预留）更精细的参数搜索与特征工程 |
| **可视化产物** | 自动生成特征重要性图、测试集预测结果、HTML 报告、可执行 Python 脚本 |
| **模型可交付** | 产物包含 `model.pkl`（训练好的模型）、`pipeline.py`（可复现代码）、`report.html`（可视化报告） |

### 典型使用场景
- 📊 业务人员快速验证数据建模可行性
- 🤖 数据分析师快速生成 baseline 代码
- 🎓 机器学习教学演示 AutoML 流程
- 🏢 企业级数据竞赛快速提交预测结果

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Frontend)                       │
│  HTML5 + Tailwind CSS + Vanilla JS                          │
│  http://localhost:5500/mlworkflow.html                              │
├─────────────────────────────────────────────────────────────┤
│                        后端 (Backend)                        │
│  FastAPI + Python 沙箱 + LLM Agent 编排引擎                  │
│  http://localhost:8000                                      │
├─────────────────────────────────────────────────────────────┤
│  Agent 体系                                                  │
│  ├─ Plan & Coding Agent：生成建模计划与 Python 代码          │
│  ├─ Evaluation Agent：评估模型性能，给出优化/接受建议        │
│  └─ Sandbox Executor：在隔离环境中安全执行生成的代码         │
├─────────────────────────────────────────────────────────────┤
│  LLM 支持：OpenAI / 本地 OpenAI 兼容接口 / Ollama            │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10/11 / macOS / Linux |
| **Python** | 3.10 或更高版本 |
| **内存** | 建议 8GB+（LLM 推理和模型训练需要） |
| **网络** | 需要能访问 LLM API（OpenAI 或本地部署的模型服务） |
| **浏览器** | Chrome / Edge / Firefox 最新版 |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/mlworkflow.git
cd mlworkflow
```

### 2. 配置 LLM API

在 `backend/` 目录下创建 `.env` 文件：

```bash
cd backend
```

**方式一：使用 OpenAI（推荐）**

```ini
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4o-mini
```

**方式二：使用本地/私有化模型（兼容 OpenAI 格式）**

```ini
LLM_PROVIDER=local-openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:14b
```

> 📖 支持的 provider：`openai` | `ollama` | `local-openai`

### 3. 安装并启动后端

**Windows：**
```bash
cd backend
start.bat
```

**macOS / Linux：**
```bash
cd backend
chmod +x start.sh
./start.sh
```

**手动安装（任意平台）：**
```bash
cd backend

# 创建虚拟环境
python -m venv venv

# Windows 激活
venv\Scripts\activate
# macOS/Linux 激活
# source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

后端启动成功后：
- API 服务：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

### 4. 启动前端

在**新的终端窗口**中：

```bash
# 回到项目根目录
cd ..                    # 如果在 backend 目录
# 或
cd mlworkflow              # 如果在其他位置

# 启动静态文件服务
python -m http.server 5500
```

然后在浏览器中访问：

```
http://localhost:5500/mlworkflow.html
```

---

## 📖 使用指南

### 快速模式（Fast Mode）完整流程

1. **上传数据**
   - 在首页拖拽或点击上传 CSV / Excel 文件
   - 系统自动识别训练集、验证集、测试集

2. **意图识别**
   - 系统分析数据内容，自动判断建模目标（分类 / 回归）
   - 展示数据画像（字段类型、分布、缺失值等）

3. **任务确认**
   - 确认建模目标和特征列选择
   - 选择**快速模式**或深度模式

4. **自动建模**
   - AI 自动生成 baseline 代码并在沙箱中执行
   - 实时展示训练日志和终端输出

5. **结果呈现**
   - 展示验证集 AUC、准确率、过拟合比等指标
   - 用户可以：
     - ✅ **接受**：进入产物生成阶段
     - 💬 **反馈**：输入改进建议，AI 继续优化（最多 3 轮）

6. **产物下载**
   - 任务完成后，在右侧标签页可查看：
     - **模型代码**：完整 Python 建模脚本
     - **预测结果**：测试集预测表格 + 特征重要性图
     - **文件**：`model.pkl` + `pipeline.py` + `report.html` + `test_predictions.csv`

---

## 📁 项目结构

```
mlworkflow/
├── mlworkflow.html             # 前端主页面（单页应用）
├── css/
│   └── main.css               # 全局样式
├── js/
│   ├── app.js                 # 前端总控逻辑
│   ├── core/
│   │   ├── state.js           # 全局状态管理
│   │   └── utils.js           # 工具函数
│   ├── fast/                  # 快速模式前端
│   │   ├── engine.js          # 快速模式引擎（轮询驱动）
│   │   ├── ui.js              # 快速模式 UI 渲染
│   │   └── mockData.js        # 演示数据
│   ├── depth/                 # 深度模式前端（预留）
│   │   ├── engine.js
│   │   ├── ui.js
│   │   └── mockData.js
│   ├── intent/                # 意图识别模块
│   │   ├── agent.js
│   │   ├── flow.js
│   │   └── profiler.js
│   ├── shared/                # 共享组件
│   │   ├── codeEditor.js      # 代码编辑器
│   │   ├── renderer.js        # 通用渲染器
│   │   └── terminal.js        # 终端模拟器
│   └── modeInterface.js       # 模式接口抽象
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI 应用入口
│   │   ├── config.py          # 配置管理（支持 .env）
│   │   ├── api/
│   │   │   ├── files.py       # 文件上传/下载 API
│   │   │   └── tasks.py       # 任务管理 API
│   │   ├── core/
│   │   │   ├── fast_engine.py # 快速模式任务编排引擎
│   │   │   ├── data_splitter.py # 数据切分
│   │   │   └── state.py       # 任务状态管理器
│   │   ├── agents/
│   │   │   ├── base.py        # Agent 基类（LLM 调用封装）
│   │   │   ├── plan_coding.py # 计划与代码生成 Agent
│   │   │   └── evaluation.py  # 评估 Agent
│   │   ├── models/
│   │   │   └── schemas.py     # Pydantic 数据模型
│   │   └── sandbox/
│   │       └── executor.py    # Python 沙箱执行器
│   ├── requirements.txt       # Python 依赖
│   ├── start.bat              # Windows 一键启动脚本
│   ├── start.sh               # macOS/Linux 一键启动脚本
│   ├── uploads/               # 用户上传文件持久化目录
│   └── outputs/               # 任务产物目录（按 task_id 分组）
│       └── {task_id}/
│           ├── data/          # 训练/验证/测试数据拆分
│           └── artifacts/     # 模型、代码、报告等产物
└── README.md
```

---

## 🔧 后端配置详解

所有配置项均可通过 `backend/.env` 文件覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_PROVIDER` | `openai` | LLM 提供商 |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | API 基础地址 |
| `LLM_API_KEY` | `""` | API 密钥 |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名称 |
| `LLM_TEMPERATURE` | `0.3` | 采样温度 |
| `LLM_MAX_TOKENS` | `4096` | 最大生成 token 数 |
| `SANDBOX_TIMEOUT` | `300` | 沙箱执行超时（秒） |
| `FAST_MAX_OPTIMIZE_ROUNDS` | `3` | 快速模式最大优化轮数 |
| `FAST_MAX_DEBUG_ROUNDS` | `5` | 代码调试最大重试次数 |

---

## 💾 数据持久化说明

| 数据类型 | 存储位置 | 说明 |
|----------|----------|------|
| 用户上传的原始数据 | `backend/uploads/` | 按 UUID 命名，永久保留 |
| 数据拆分结果 | `backend/outputs/{task_id}/data/` | `train.csv` / `validation.csv` / `test.csv` |
| 建模产物 | `backend/outputs/{task_id}/artifacts/` | `model.pkl` / `pipeline.py` / `report.html` / `feature_importance.*` / `test_predictions.csv` |
| 任务执行日志 | 内存（`task_manager`） | ⚠️ 后端重启后丢失 |

---

## 🐛 常见问题

### Q1: 前端页面空白或报错？
- 确保前端服务已启动（`python -m http.server 5500`）
- 浏览器强制刷新：`Ctrl + F5`
- 检查浏览器控制台（F12 → Console）是否有 CORS 错误

### Q2: 后端启动报错 "端口被占用"？
```bash
# Windows：查找并终止占用 8000 端口的进程
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process

# 或使用其他端口
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Q3: LLM 调用超时或报错？
- 检查 `.env` 中 `LLM_API_KEY` 是否正确
- 检查网络是否能访问 `LLM_BASE_URL`
- 增大超时配置：`SANDBOX_TIMEOUT=600`

### Q4: 产物文件下载无反应？
- 确保产物已生成完成（任务状态为 "completed"）
- 直接访问产物 URL 测试：`http://localhost:8000/artifacts/{task_id}/artifacts/report.html`

---

## 📄 License

[MIT](LICENSE)

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

如有问题，请通过 GitHub Issues 反馈。

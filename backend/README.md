# ML Agent 后端

快速模式后端服务，基于 FastAPI + Python 沙箱 + LLM Agent。

## 快速启动

### Windows
```bash
cd backend
start.bat
```

### macOS / Linux
```bash
cd backend
chmod +x start.sh
./start.sh
```

### 手动启动
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 接口文档

启动后访问：http://localhost:8000/docs

## 环境变量（可选）

创建 `.env` 文件：
```
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
```

"""后端 API 联调测试"""
import json
import urllib.request
import tempfile
import os

BASE = "http://localhost:8000"

def get(url):
    return json.loads(urllib.request.urlopen(url).read().decode())

def post_json(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req).read().decode())

print("=" * 50)
print("后端 API 联调测试")
print("=" * 50)

# 1. Health check
print("\n[1/4] Health Check")
print(get(f"{BASE}/health"))

# 2. File upload
print("\n[2/4] File Upload")
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write("a,b,target\n1,2,0\n3,4,1\n")
    temp_path = f.name

try:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = f"--{boundary}\r\n"
    body += 'Content-Disposition: form-data; name="files"; filename="train.csv"\r\n'
    body += "Content-Type: text/csv\r\n\r\n"
    with open(temp_path, "rb") as f:
        body = body.encode() + f.read()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(f"{BASE}/api/files/upload", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    })
    upload_res = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"  上传成功: {upload_res[0]['name']} -> role={upload_res[0]['role']}")
    uploaded_file = upload_res[0]
finally:
    os.unlink(temp_path)

# 3. Start fast task (without LLM, it will fail at planning stage but API should work)
print("\n[3/4] Start Fast Task")
task_config = {
    "extracted_slots": {
        "target_column": "target",
        "task_type": "binary_classification",
        "eval_metric": "AUC",
        "feature_constraints": []
    },
    "uploaded_files": [{
        "name": uploaded_file["name"],
        "path": uploaded_file["path"],
        "role": uploaded_file["role"],
        "size": uploaded_file["size"]
    }],
    "user_description": "测试任务",
    "data_profile": None
}

try:
    start_res = post_json(f"{BASE}/api/tasks/fast/start", {"task_config": task_config})
    print(f"  任务创建成功: {start_res['task_id']}, phase={start_res['phase']}")
    task_id = start_res["task_id"]

    # 4. Poll status
    print("\n[4/4] Poll Task Status")
    import time
    for i in range(3):
        time.sleep(1)
        status = get(f"{BASE}/api/tasks/fast/{task_id}/status")
        print(f"  第{i+1}次查询: phase={status['phase']}, optimize_round={status.get('optimize_round', 0)}")
        if status["phase"] in ("failed", "completed", "presenting"):
            break

    print("\n" + "=" * 50)
    print("联调测试完成！")
    print("=" * 50)
    print(f"\n任务ID: {task_id}")
    print("注意：由于未配置 LLM，PlanCodingAgent 会在 planning 阶段报错。")
    print("请在 backend/.env 中配置 LLM 后重新测试完整流程。")

except urllib.error.HTTPError as e:
    err = json.loads(e.read().decode())
    print(f"  任务启动失败: {err}")
    print("\n提示：这是预期的行为——未配置 LLM 时 Agent 无法调用模型。")

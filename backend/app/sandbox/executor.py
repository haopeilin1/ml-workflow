"""
沙箱执行器
- AST 安全检查（禁止危险导入和调用）
- 子进程隔离执行
- 超时控制
- stdout/stderr 捕获
- 指标 JSON 解析
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from app.config import settings
except ImportError:
    import sys
    class _Settings:
        SANDBOX_TIMEOUT = 300
        PYTHON_EXECUTABLE = sys.executable
    settings = _Settings()

try:
    from app.models.schemas import ExecutionMetrics
except ImportError:
    class ExecutionMetrics:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class SecurityError(Exception):
    """安全检查异常"""
    pass


class SecurityChecker(ast.NodeVisitor):
    """
    AST 安全检查器
    
    禁止：
    - 危险模块导入（os, subprocess, socket, urllib, requests 等）
    - 危险函数调用（eval, exec, compile, __import__）
    - 危险属性访问（os.system, os.popen, os.exec*, os.fork, os.kill 等）
    """
    
    # 严格模式下禁止导入的顶层模块
    FORBIDDEN_IMPORTS = {
        'os', 'subprocess', 'socket', 'urllib', 'requests', 'httpx',
        'ftplib', 'smtplib', 'telnetlib', 'http', 'webbrowser',
        'shutil', 'sys', 'importlib', 'pathlib'
    }
    
    # 产物生成模式下允许导入的模块（但仍禁止危险操作）
    ARTIFACT_ALLOWED_IMPORTS = {'os', 'shutil', 'sys', 'pathlib'}
    
    def __init__(self, artifact_mode: bool = False):
        self.artifact_mode = artifact_mode
        super().__init__()
    
    # 禁止调用的函数名（全局）
    FORBIDDEN_CALLS = {'eval', 'exec', 'compile', '__import__'}
    
    # 禁止的属性调用：模块 -> 禁止的方法集合
    FORBIDDEN_ATTR_CALLS = {
        'os': {'system', 'popen', 'exec', 'execve', 'fork', 'kill', 
               'remove', 'rmdir', 'rename', 'unlink', 'chmod', 'chown'},
        'subprocess': {'run', 'call', 'check_call', 'check_output', 'Popen'},
        'socket': {'socket', 'create_connection', 'connect'},
    }
    
    def check(self, code: str) -> list:
        """检查代码，返回错误列表（空列表表示通过）"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"语法错误: {e}"]
        
        self.errors = []
        self.visit(tree)
        return self.errors
    
    def _is_forbidden_import(self, top_module: str) -> bool:
        """判断模块是否在禁止导入列表中（考虑产物模式例外）"""
        if top_module not in self.FORBIDDEN_IMPORTS:
            return False
        if self.artifact_mode and top_module in self.ARTIFACT_ALLOWED_IMPORTS:
            return False
        return True
    
    def visit_Import(self, node):
        for alias in node.names:
            top_module = alias.name.split('.')[0]
            if self._is_forbidden_import(top_module):
                self.errors.append(f"禁止导入模块: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            top_module = node.module.split('.')[0]
            if self._is_forbidden_import(top_module):
                self.errors.append(f"禁止从模块导入: {node.module}")
        # 检查 from os import system 这种形式
        for alias in node.names:
            if node.module in self.FORBIDDEN_ATTR_CALLS:
                if alias.name in self.FORBIDDEN_ATTR_CALLS[node.module]:
                    self.errors.append(f"禁止导入: {node.module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # 检查全局函数调用，如 eval("...")
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                self.errors.append(f"禁止调用函数: {node.func.id}")
        
        # 检查属性调用，如 os.system("...")
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                attr_name = node.func.attr
                if module_name in self.FORBIDDEN_ATTR_CALLS:
                    if attr_name in self.FORBIDDEN_ATTR_CALLS[module_name]:
                        self.errors.append(f"禁止调用: {module_name}.{attr_name}")
        
        self.generic_visit(node)


@dataclass
class SandboxResult:
    """沙箱执行结果"""
    success: bool
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    metrics: Optional[ExecutionMetrics]
    error_message: Optional[str] = None


class SandboxExecutor:
    """
    沙箱执行器
    
    执行流程：
    1. AST 安全检查
    2. 准备临时工作目录 + 数据文件
    3. 代码路径适配（/data/ -> data/）
    4. 子进程隔离执行（带超时）
    5. 解析 stdout 中的 JSON 指标
    """
    
    def __init__(self, timeout: int = None):
        self.timeout = timeout or settings.SANDBOX_TIMEOUT
        self.checker = SecurityChecker()
        self.artifact_checker = SecurityChecker(artifact_mode=True)
    
    def execute(
        self,
        code: str,
        data_dir: Path,
        task_type: str = "binary_classification",
        artifact_mode: bool = False,
        artifact_output_dir: Optional[Path] = None
    ) -> SandboxResult:
        """
        在沙箱中执行 Python 代码
        
        Args:
            code: Python 代码字符串
            data_dir: 数据文件目录（包含 train.csv, validation.csv, test.csv）
            task_type: 任务类型，用于指标解析提示
            artifact_mode: 是否为产物生成模式（允许文件写入）
            artifact_output_dir: 产物输出目录（artifact_mode=True 时必填）
            
        Returns:
            SandboxResult
        """
        start_time = time.time()
        
        # 1. 安全检查
        checker = self.artifact_checker if artifact_mode else self.checker
        security_errors = checker.check(code)
        if security_errors:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                returncode=-1,
                execution_time=0.0,
                metrics=None,
                error_message=f"安全检查未通过: {'; '.join(security_errors)}"
            )
        
        # 2. 准备临时工作目录
        work_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
        data_path = work_dir / "data"
        data_path.mkdir()
        
        # 产物模式下创建 output 目录
        output_path = work_dir / "output"
        if artifact_mode:
            output_path.mkdir()
        
        try:
            # 复制数据文件到工作目录
            if data_dir.exists():
                for src_file in data_dir.iterdir():
                    if src_file.is_file():
                        dst = data_path / src_file.name
                        with open(src_file, 'rb') as fsrc, open(dst, 'wb') as fdst:
                            fdst.write(fsrc.read())
            
            # 3. 代码路径适配
            adapted_code = self._adapt_paths(code)
            
            # 4. 写入代码文件
            code_file = work_dir / "script.py"
            code_file.write_text(adapted_code, encoding='utf-8')
            
            # 5. 子进程执行
            python_exe = settings.PYTHON_EXECUTABLE or sys.executable
            
            # 构建干净的环境变量
            env = {
                **os.environ,
                'PYTHONPATH': str(work_dir),
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONNOUSERSITE': '1',
            }
            
            proc = subprocess.run(
                [python_exe, str(code_file)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )
            
            execution_time = time.time() - start_time
            
            # 6. 解析指标
            metrics = self._parse_metrics(proc.stdout)
            
            # 7. 产物模式下，将产物复制到持久化目录
            if artifact_mode and artifact_output_dir:
                self._collect_artifacts(work_dir, artifact_output_dir)
            
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                execution_time=execution_time,
                metrics=metrics,
                error_message=proc.stderr if proc.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                returncode=-2,
                execution_time=self.timeout,
                metrics=None,
                error_message=f"执行超时（超过 {self.timeout} 秒）"
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                returncode=-3,
                execution_time=time.time() - start_time,
                metrics=None,
                error_message=f"执行异常: {str(e)}"
            )
        finally:
            # 产物模式下保留产物，否则清理临时目录
            if not artifact_mode:
                self._cleanup(work_dir)
    
    def _adapt_paths(self, code: str) -> str:
        r"""
        将代码中的路径适配为沙箱内的相对路径
        
        处理项：
        1. /data/ → data/（Docker 风格绝对路径转相对路径）
        2. .xlsx / .xls → .csv（数据切分器统一转为 CSV）
        3. 将 Windows 反斜杠路径分隔符 data\xxx 转为 data/xxx
        """
        adapted = code
        
        # 1. 替换引号包裹的 /data/ 路径为相对路径
        adapted = adapted.replace("'/data/", "'data/").replace('"/data/', '"data/')
        adapted = adapted.replace("'/data'", "'data'").replace('"/data"', '"data"')
        
        # 2. 将 Excel 后缀替换为 CSV（沙箱中只有 CSV 文件）
        # 处理各种引号包裹的情况
        for old_ext in ['.xlsx', '.xls', '.XLSX', '.XLS']:
            new_ext = '.csv'
            adapted = adapted.replace(old_ext, new_ext)
        
        # 3. 将反斜杠路径分隔符转为正斜杠（Windows 兼容）
        # 注意：只处理 data\ 开头的情况，避免转义序列问题
        adapted = adapted.replace("'data\\", "'data/").replace('"data\\', '"data/')
        
        return adapted
    
    def _parse_metrics(self, stdout: str) -> Optional[ExecutionMetrics]:
        """
        从 stdout 中解析 JSON 指标
        
        策略：
        1. 从最后一行开始向上查找完整 JSON 对象
        2. 若失败，用正则在整个 stdout 中搜索 JSON 对象
        """
        if not stdout or not stdout.strip():
            return None
        
        lines = stdout.strip().split('\n')
        
        # 策略1：从后向前找 JSON 行
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        return self._build_metrics(data)
                except json.JSONDecodeError:
                    continue
        
        # 策略2：正则匹配整个 stdout 中的 JSON
        # 匹配多行 JSON 对象（非贪婪）
        matches = re.findall(r'\{[\s\S]*?\}', stdout)
        for match in reversed(matches):
            try:
                data = json.loads(match)
                if isinstance(data, dict):
                    return self._build_metrics(data)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _build_metrics(self, data: dict) -> ExecutionMetrics:
        """从字典构建 ExecutionMetrics（支持字段名别名映射）"""
        # 别名映射：将常见 LLM 输出字段名映射到标准字段名
        aliases = {
            "validation_accuracy": "val_accuracy",
            "valid_accuracy": "val_accuracy",
            "validation_auc": "val_auc",
            "valid_auc": "val_auc",
            "validation_score": "val_score",
            "valid_score": "val_score",
            "validation_rmse": "val_rmse",
            "valid_rmse": "val_rmse",
            "train_accuracy": "train_score",
            "training_accuracy": "train_score",
            "training_auc": "train_auc",
        }
        normalized = {}
        for k, v in data.items():
            key = aliases.get(k, k)
            normalized[key] = v
        
        return ExecutionMetrics(
            metric_name=normalized.get("metric_name", ""),
            val_auc=normalized.get("val_auc"),
            val_accuracy=normalized.get("val_accuracy"),
            val_rmse=normalized.get("val_rmse"),
            val_score=normalized.get("val_score"),
            train_auc=normalized.get("train_auc"),
            train_score=normalized.get("train_score"),
            overfit_ratio=normalized.get("overfit_ratio"),
            overfit_severe=normalized.get("overfit_severe", False)
        )
    
    def _cleanup(self, work_dir: Path):
        """清理临时工作目录"""
        import shutil
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir)
        except Exception:
            pass
    
    def _collect_artifacts(self, work_dir: Path, artifact_output_dir: Path):
        """将产物从沙箱复制到持久化目录"""
        import shutil
        try:
            artifact_output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = work_dir / "output"
            if output_dir.exists():
                for src_file in output_dir.iterdir():
                    if src_file.is_file():
                        dst = artifact_output_dir / src_file.name
                        shutil.copy2(str(src_file), str(dst))
        except Exception as e:
            print(f"[Sandbox] 产物收集失败: {e}")


# 全局沙箱执行器实例
sandbox_executor = SandboxExecutor()

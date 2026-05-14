#!/usr/bin/env python3
"""
批量 Benchmark 测试脚本
支持分批运行，自动创建符号链接目录
"""

import argparse
import json
import logging
import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 任务目录映射（图片中的9个任务 + 其他）
TASK_DIR_MAP = {
    "2026年股票违约": "2026年股票抵押融资违约预测",
    "信用卡欺诈": "信用卡欺诈-二分类-类别极度不平衡",
    "共享单车时序": "共享单车租赁量预测-时序回归",
    "加州房价": "加利福尼亚房价预测-回归",
    "北京PM2.5时序": "北京PM2.5浓度预测-时序回归",
    "电商退货": "电商顾客退货预测",
    "睡眠障碍": "睡眠障碍预测",
    "红酒品质": "红酒品质预测-有序多分类",
    "银行欺诈": "银行账户欺诈",
    # 第二批
    "成人收入预测": "成人收入预测-二分类含缺失值",
    "电子商务客户流失": "电子商务客户流失预测",
    "肝硬化患者状态": "肝硬化患者状态预测",
    "黑色素瘤种类": "黑色素瘤种类",
    "机翼噪声": "机翼噪声预测",
    "垃圾邮件判别": "垃圾邮件判别-二分类-高维稀疏",
    "糖尿病预测": "糖尿病预测",
    "吸烟状况": "吸烟状况",
    "医疗保险费用": "医疗保险费用预测",
    "鸢尾花种类识别": "鸢尾花种类识别-极小样本多分类",
    "支付欺诈": "支付欺诈",
}

BATCH_1_TASKS = [
    "2026年股票违约", "信用卡欺诈", "共享单车时序", "加州房价",
    "北京PM2.5时序", "电商退货", "睡眠障碍", "红酒品质", "银行欺诈"
]

BATCH_2_TASKS = [
    "成人收入预测", "电子商务客户流失", "肝硬化患者状态", "黑色素瘤种类",
    "机翼噪声", "垃圾邮件判别", "糖尿病预测", "吸烟状况", "医疗保险费用",
    "鸢尾花种类识别", "支付欺诈"
]
# 注意：实际20个任务，第一批9个 + 第二批11个 = 20个

TEST_DATA_ROOT = Path("/home/hpl/ml-workflow/test_data")


def create_benchmark_dir(task_names: list, batch_name: str) -> Path:
    """创建临时 benchmark 目录，包含指定任务的符号链接"""
    batch_dir = Path(f"/tmp/benchmark_{batch_name}_{datetime.now().strftime('%m%d_%H%M')}")
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True)
    
    for name in task_names:
        dir_name = TASK_DIR_MAP.get(name)
        if not dir_name:
            logger.warning(f"未找到任务 '{name}' 的目录映射")
            continue
        src = TEST_DATA_ROOT / dir_name
        if not src.exists():
            logger.warning(f"任务目录不存在: {src}")
            continue
        dst = batch_dir / dir_name
        dst.symlink_to(src, target_is_directory=True)
        logger.info(f"  链接: {name} -> {dir_name}")
    
    return batch_dir


def run_benchmark(batch_dir: Path, num_runs: int = 3, max_wait: int = 1800, output_dir: Path = None):
    """运行 benchmark"""
    backend_dir = Path(__file__).parent
    
    # 使用项目 venv 的 Python，确保能导入 backend 模块
    from app.config import settings
    python_exe = settings.PYTHON_EXECUTABLE
    
    cmd = [
        python_exe, "-m", "scripts.run_benchmark",
        "--benchmark-dir", str(batch_dir),
        "--num-runs", str(num_runs),
        "--max-wait", str(max_wait),
    ]
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"benchmark_report_{batch_dir.name}.json"
        cmd.extend(["--output", str(report_path)])
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    logger.info(f"工作目录: {backend_dir}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(backend_dir)
    
    proc = subprocess.run(
        cmd,
        cwd=str(backend_dir),
        capture_output=False,
        text=True,
        env=env,
    )
    
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="批量 Benchmark 测试")
    parser.add_argument("--batch", choices=["1", "2", "all"], default="1", help="运行批次: 1=第一批9个, 2=第二批, all=全部")
    parser.add_argument("--num-runs", type=int, default=3, help="每个任务运行次数")
    parser.add_argument("--max-wait", type=int, default=1800, help="每个任务最大等待时间（秒）")
    parser.add_argument("--output-dir", default="/home/hpl/ml-workflow/backend/benchmark_results", help="结果输出目录")
    
    args = parser.parse_args()
    
    if args.batch == "1":
        tasks = BATCH_1_TASKS
        batch_name = "batch1"
    elif args.batch == "2":
        tasks = BATCH_2_TASKS
        batch_name = "batch2"
    else:
        tasks = BATCH_1_TASKS + BATCH_2_TASKS
        batch_name = "all"
    
    logger.info("=" * 60)
    logger.info(f"批量 Benchmark 测试 - {batch_name}")
    logger.info(f"任务数: {len(tasks)}, 每任务运行: {args.num_runs} 次")
    logger.info(f"并发: 3 任务并行")
    logger.info("=" * 60)
    
    # 创建临时目录
    logger.info("创建 benchmark 目录...")
    batch_dir = create_benchmark_dir(tasks, batch_name)
    logger.info(f"Benchmark 目录: {batch_dir}")
    
    # 运行 benchmark
    output_dir = Path(args.output_dir)
    success = run_benchmark(batch_dir, args.num_runs, args.max_wait, output_dir)
    
    # 清理
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    
    logger.info("=" * 60)
    logger.info(f"测试{'完成' if success else '失败'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

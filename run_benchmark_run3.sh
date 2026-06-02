#!/bin/bash
# 分批次端到端测评脚本
# 用法: ./run_benchmark_batch.sh batch1|batch2

BATCH=$1

if [ "$BATCH" == "batch1" ]; then
    EVAL_ID="run3_batch1"
    TASKS=(
        "2026年股票抵押融资违约预测"
        "信用卡欺诈-二分类-类别极度不平衡"
        "共享单车租赁量预测-时序回归"
        "加利福尼亚房价预测-回归"
        "北京PM2.5浓度预测-时序回归"
        "电商顾客退货预测"
        "睡眠障碍预测"
        "红酒品质预测-有序多分类"
        "银行账户欺诈"
    )
elif [ "$BATCH" == "batch2" ]; then
    EVAL_ID="run3_batch2"
    TASKS=(
        "医疗保险费用预测"
        "吸烟状况"
        "垃圾邮件判别-二分类-高维稀疏"
        "成人收入预测-二分类含缺失值"
        "支付欺诈"
        "机翼噪声预测"
        "电子商务客户流失预测"
        "糖尿病预测"
        "肝硬化患者状态预测"
        "鸢尾花种类识别-极小样本多分类"
        "黑色素瘤种类"
    )
else
    echo "Usage: $0 batch1|batch2"
    exit 1
fi

cd backend
OUTPUT_BASE="outputs/${EVAL_ID}"
LOG_FILE="${OUTPUT_BASE}/batch.log"

mkdir -p "$OUTPUT_BASE"

echo "========================================" | tee -a "$LOG_FILE"
echo "开始运行: $BATCH" | tee -a "$LOG_FILE"
echo "Eval ID: $EVAL_ID" | tee -a "$LOG_FILE"
echo "任务数: ${#TASKS[@]}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for task in "${TASKS[@]}"; do
    TASK_RESULT_DIR="${OUTPUT_BASE}/${task}/run_1"
    if [ -f "${TASK_RESULT_DIR}/task_result.json" ]; then
        echo "[SKIP] $task already done" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[RUN] $task" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    venv/Scripts/python.exe -m scripts.run_single_task \
        --benchmark-dir "../test_data/${task}" \
        --num-runs 1 \
        --max-wait 1800 \
        --eval-id "$EVAL_ID" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "[DONE] $task" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "$BATCH 全部完成!" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 生成汇总报告
venv/Scripts/python.exe -m scripts.summarize_batch "$EVAL_ID" "$OUTPUT_BASE" 2>&1 | tee -a "$LOG_FILE"

#!/bin/bash
# AutoDL 一键部署脚本
# 使用方法: bash autodl_one_click.sh

set -e

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   AutoDL 象棋AI 一键部署脚本"
echo "========================================"
echo ""
echo "工作目录: $SCRIPT_DIR"
echo ""

# 检测配置
CPU_CORES=$(nproc)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "未知")
echo "系统配置:"
echo "   CPU: ${CPU_CORES}核"
echo "   GPU: ${GPU_NAME}"
echo ""

# 根据CPU核心数决定进程数
if [ "$CPU_CORES" -le 4 ]; then
    NUM_COLLECT=2
elif [ "$CPU_CORES" -le 8 ]; then
    NUM_COLLECT=3
else
    NUM_COLLECT=5
fi

echo "将启动 ${NUM_COLLECT} 个数据收集进程"
echo ""

# 1. 安装依赖
echo "[1/4] 安装依赖..."
pip install numpy pygame -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
echo "完成"
echo ""

# 2. 创建目录
echo "[2/4] 创建目录..."
mkdir -p "$SCRIPT_DIR/models"
mkdir -p "$SCRIPT_DIR/logs"
echo "完成"
echo ""

# 3. 优化配置
echo "[3/4] 优化配置..."
if [ -f "config.py" ]; then
    # 备份
    cp config.py config.py.bak.$(date +%Y%m%d_%H%M%S)

    # 修改数据路径为当前目录
    sed -i "s|'train_data_buffer_path': 'train_data_buffer.pkl'|'train_data_buffer_path': '$SCRIPT_DIR/train_data_buffer.pkl'|g" config.py

    # 减少等待时间
    sed -i "s|'train_update_interval': 600|'train_update_interval': 60|g" config.py

    echo "配置已优化:"
    grep "game_batch_num\|batch_size\|play_out\|train_update_interval\|train_data_buffer_path" config.py | head -5
fi
echo ""

# 4. 停止旧进程
echo "[4/4] 启动训练..."
pkill -f "python collect.py" 2>/dev/null || true
pkill -f "python train.py" 2>/dev/null || true
sleep 3

# 启动数据收集进程
echo "启动 ${NUM_COLLECT} 个数据收集进程..."
for i in $(seq 1 $NUM_COLLECT); do
    LOG_FILE="$SCRIPT_DIR/logs/collect_${i}.log"
    nohup python -u collect.py > "$LOG_FILE" 2>&1 &
    echo "   进程 $i 已启动 -> $LOG_FILE"
done

# 等待数据收集
echo "等待15秒让数据收集进程启动..."
sleep 15

# 检查collect进程
COLLECT_RUNNING=$(ps aux | grep "python collect.py" | grep -v grep | wc -l)
echo "运行中的collect进程: $COLLECT_RUNNING"

if [ "$COLLECT_RUNNING" -eq 0 ]; then
    echo "警告: collect进程未启动，请检查日志"
    echo "查看日志: tail -f $SCRIPT_DIR/logs/collect_1.log"
fi

# 启动训练进程
echo "启动训练进程..."
TRAIN_LOG="$SCRIPT_DIR/logs/train.log"
nohup python -u train.py > "$TRAIN_LOG" 2>&1 &
sleep 3

# 检查train进程
if ps aux | grep "python train.py" | grep -v grep > /dev/null; then
    echo "训练进程已启动"
else
    echo "警告: 训练进程可能未启动，请检查日志"
fi

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "训练配置:"
grep "game_batch_num" config.py | head -1
echo ""
echo "监控命令:"
echo "   实时日志: tail -f $TRAIN_LOG"
echo "   GPU状态: watch -n 1 nvidia-smi"
echo "   进程状态: ps aux | grep python"
echo ""
echo "预计完成时间: 15-18小时"
echo "========================================"

# 显示最新日志
sleep 2
echo ""
echo "最新日志（最后10行）:"
tail -10 "$TRAIN_LOG" 2>/dev/null || echo "日志生成中..."
echo ""

#!/bin/bash
# ============================================
# 查看所有 Actor 进程状态
# 用法: bash check_actors.sh
# ============================================

echo "========================================="
echo "  Actor 进程状态查询"
echo "========================================="
echo ""

COUNT=$(ps -elf | grep "python entry.py --actor_id=" | grep -v grep | wc -l)

if [ "$COUNT" -eq 0 ]; then
    echo "没有 Actor 进程在运行。"
    echo ""
    echo "========================================="
    exit 0
fi

echo "当前运行 $COUNT 个 Actor 进程:"
echo ""

ps -elf | grep "python entry.py --actor_id=" | grep -v grep | while read line; do
    PID=$(echo "$line" | awk '{print $4}')
    CPU=$(echo "$line" | awk '{print $3}') 
    ACTOR_ID=$(echo "$line" | grep -oP '(?<=--actor_id=)[0-9]+')
    
    # 获取 GPU 显存占用
    MEM_KB=$(grep VmRSS /proc/$PID/status 2>/dev/null | awk '{print $2}')
    MEM_MB=$((${MEM_KB:-0} / 1024))

    echo "  Actor $ACTOR_ID | PID: $PID | CPU: ${CPU}% | 内存: ${MEM_MB}MB"
done

echo ""
echo "========================================="
echo "  Actor 日志速览 (最新 3 行)"
echo "========================================="

for logfile in /aiarena/logs/actor/actor_*.log; do
    if [ -f "$logfile" ]; then
        ACTOR_NAME=$(basename "$logfile")
        echo ""
        echo "--- $ACTOR_NAME ---"
        tail -1 "$logfile" 2>/dev/null
    fi
done

echo ""
echo "========================================="
echo "  MemPool 最新 LANE 样本数量"
echo "========================================="
# 尝试从 learner 日志获取 recv_speed
LEARNER_LOG="/aiarena/logs/learner/train.log"
if [ -f "$LEARNER_LOG" ]; then
    tail -3 "$LEARNER_LOG" 2>/dev/null
fi

echo ""
echo "========================================="
echo "  提示"
echo "  - 查看实时日志: tail -f /aiarena/logs/actor/actor_*.log"
echo "  - 停止所有进程: bash stop_actors.sh"
echo "========================================="
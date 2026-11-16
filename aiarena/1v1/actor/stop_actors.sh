#!/bin/bash
# ============================================
# 停止所有 Actor 进程
# 用法: bash stop_actors.sh
# ============================================

echo "正在查找所有 Actor 进程..."

PIDS=$(ps -elf | grep "python entry.py --actor_id=" | grep -v grep | awk '{print $4}')

if [ -z "$PIDS" ]; then
    echo "没有发现正在运行的 Actor 进程。"
    exit 0
fi

echo "找到以下 Actor 进程："
ps -elf | grep "python entry.py --actor_id=" | grep -v grep

echo ""
echo "正在终止..."

for pid in $PIDS; do
    kill "$pid" 2>/dev/null
    echo "  killed PID: $pid"
done

sleep 1

REMAINING=$(ps -elf | grep "python entry.py --actor_id=" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "仍有进程未退出，强制终止..."
    ps -elf | grep "python entry.py --actor_id=" | grep -v grep | awk '{print $4}' | xargs kill -9 2>/dev/null
fi

echo ""
echo "========================================="
echo "  所有 Actor 已停止"
echo "========================================="
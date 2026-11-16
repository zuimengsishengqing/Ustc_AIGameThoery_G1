#!/bin/bash
# ============================================
# 批量启动多个「平台维度」Actor 进程（entry_platform.py）
# 用法: bash start_actors_platform.sh [Actor数量]
# 示例: bash start_actors_platform.sh 4
# 与 start_actors.sh 并列，唯一区别是入口从 entry.py 换成 entry_platform.py
# ============================================

# 默认 Actor 数量，可通过第一个参数覆盖
ACTOR_NUM=${1:-4}

# ========== 配置区（按需修改）==========
MEM_POOL_ADDR="127.0.0.1:35200"
MODEL_POOL_ADDR=""
MAX_EPISODE=-1
LOG_DIR="/aiarena/logs/actor"

# 自动检测 GameCore Server 地址（Windows 宿主机 IP + 端口）
if [ -z "$GAMECORE_SERVER_ADDR" ]; then
    HOST_IP=$(ip route show default 2>/dev/null | awk '{print $3}')
    if [ -n "$HOST_IP" ]; then
        export GAMECORE_SERVER_ADDR="${HOST_IP}:23432"
    else
        export GAMECORE_SERVER_ADDR="127.0.0.1:23432"
    fi
fi

# 强制只打鲁班(112)
export CAMP_DRIVER_CONFIG="WzExMl0="

mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  Actor 批量启动脚本 (平台维度)"
echo "  Actor 数量: $ACTOR_NUM"
echo "  MemPool: $MEM_POOL_ADDR"
echo "  GameCore: $GAMECORE_SERVER_ADDR"
echo "  日志目录: $LOG_DIR"
echo "========================================="

for ((i=0; i<ACTOR_NUM; i++)); do
    LOG_FILE="$LOG_DIR/actor_platform_${i}.log"
    echo "启动 Actor $i ... (日志: $LOG_FILE)"

    nohup python entry_platform.py \
        --actor_id=$i \
        --mem_pool_addr="$MEM_POOL_ADDR" \
        --model_pool_addr="$MODEL_POOL_ADDR" \
        --max_episode=$MAX_EPISODE \
        > "$LOG_FILE" 2>&1 &

    echo "  PID: $!"
    sleep 1
done

echo ""
echo "========================================="
echo "  已启动 $ACTOR_NUM 个平台维度 Actor 进程"
echo "  查看状态: ps -elf | grep 'entry_platform.py'"
echo "  查看日志: tail -f /aiarena/logs/actor/actor_platform_*.log"
echo "========================================="
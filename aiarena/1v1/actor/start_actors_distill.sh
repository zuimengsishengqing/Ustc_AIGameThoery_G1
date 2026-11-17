#!/bin/bash
# ============================================
# 批量启动多个「蒸馏模式」Actor 进程（entry_platform.py + 教师模型）
# 用法: bash start_actors_distill.sh [Actor数量] [教师ckpt源文件] [起始Actor编号]
# 示例: bash start_actors_distill.sh 4 /aiarena/anlys/model1111.pth
#       bash start_actors_distill.sh 4 "" 5   # 从 actor_id=5 开始（跳过 0~4）
#
# 前置条件：先用 prepare_teacher_ckpt.py 把教师 ckpt 整理成
# {TEACHER_CKPT_DIR}/model.pth 格式（LocalTorchPredictor 期望的目录结构）。
# 若第二个参数给出源 ckpt 文件路径，本脚本会自动调用 prepare_teacher_ckpt.py
# 完成这一步；否则默认认为 TEACHER_CKPT_DIR 下已经准备好 model.pth。
# ============================================

# 默认 Actor 数量，可通过第一个参数覆盖
ACTOR_NUM=${1:-4}
# 教师 ckpt 源文件（可选），第二个参数
TEACHER_SRC_CKPT="${2:-}"
# 起始 Actor 编号（可选），第三个参数，默认 0
ACTOR_ID_OFFSET="${3:-0}"

# ========== 配置区（按需修改）==========
MEM_POOL_ADDR="127.0.0.1:35200"
MODEL_POOL_ADDR=""
MAX_EPISODE=-1
LOG_DIR="/aiarena/logs/actor"

# 教师 ckpt 目标目录，需与 config_platform.Config.TEACHER_CKPT_PATH 保持一致
export AIARENA_TEACHER_CKPT_PATH="${AIARENA_TEACHER_CKPT_PATH:-/aiarena/checkpoints_local_teacher}"
# 开启蒸馏模式（custom.py / config_platform.py / algorithm_platform.py 均读取此开关）
export AIARENA_DISTILL=1

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

if [ -n "$TEACHER_SRC_CKPT" ]; then
    echo "准备教师 ckpt: $TEACHER_SRC_CKPT -> $AIARENA_TEACHER_CKPT_PATH/model.pth"
    python prepare_teacher_ckpt.py "$TEACHER_SRC_CKPT" --dest "$AIARENA_TEACHER_CKPT_PATH"
    if [ $? -ne 0 ]; then
        echo "prepare_teacher_ckpt.py 失败，终止启动"
        exit 1
    fi
fi

echo "========================================="
echo "  Actor 批量启动脚本 (蒸馏模式)"
echo "  Actor 数量: $ACTOR_NUM"
echo "  起始 Actor 编号: $ACTOR_ID_OFFSET"
echo "  MemPool: $MEM_POOL_ADDR"
echo "  GameCore: $GAMECORE_SERVER_ADDR"
echo "  Teacher ckpt: $AIARENA_TEACHER_CKPT_PATH/model.pth"
echo "  日志目录: $LOG_DIR"
echo "========================================="

for ((i=0; i<ACTOR_NUM; i++)); do
    ACTOR_ID=$((i + ACTOR_ID_OFFSET))
    LOG_FILE="$LOG_DIR/actor_distill_${ACTOR_ID}.log"
    echo "启动 Actor $ACTOR_ID ... (日志: $LOG_FILE)"

    nohup python entry_platform.py \
        --actor_id=$ACTOR_ID \
        --mem_pool_addr="$MEM_POOL_ADDR" \
        --model_pool_addr="$MODEL_POOL_ADDR" \
        --max_episode=$MAX_EPISODE \
        > "$LOG_FILE" 2>&1 &

    echo "  PID: $!"
    sleep 1
done

echo ""
echo "========================================="
echo "  已启动 $ACTOR_NUM 个蒸馏模式 Actor 进程"
echo "  查看状态: ps -elf | grep 'entry_platform.py'"
echo "  查看日志: tail -f /aiarena/logs/actor/actor_distill_*.log"
echo ""
echo "  learner 侧请用 train_platform.py（同样读取 AIARENA_DISTILL 开关）:"
echo "    AIARENA_DISTILL=1 python train_platform.py"
echo "========================================="
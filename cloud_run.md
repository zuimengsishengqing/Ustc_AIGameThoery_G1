## 启动 Gamecore 任务（win-似乎不行！转yi）
启动 Gamecore 服务（Wine 模式）
export SIMULATOR_USE_WINE=1
sh /rl_framework/remote-gc-server/run_gamecore_server.sh
# 启动 Gamecore 服务（Wine 模式）
---------------
export SIMULATOR_USE_WINE=1
sed -i 's/\r$//' /rl_framework/remote-gc-server/*.sh

# 获取当前容器的真实 IP 并保存到共享存储，方便 Actor 读
# （一般默认第一个网卡就是内部 IP，如果是多网卡，你可以用 hostname -I 看一眼再处理）
IP=$(hostname -I | awk '{print $1}')
echo "$IP:23432" > /aiarena/gpfs/ningsz/gamecore_addr.txt
echo "Gamecore Server running at $IP:23432"

sh /rl_framework/remote-gc-server/run_gamecore_server.sh
----------------
启动后，在日志中找到平台分配给它的 IP 地址。假设为 10.0.1.5，那么 Gamecore 地址就是 10.0.1.5:23432。

获取gamecore的IP：
root@task-100154-ningsz-hokgamecore-master-0:/# 
hostname -I
这次gamecore
10.244.50.25

寻找日志：
ls -ltr /aiarena/task_logs/ | tail -n 5

tail -n 40 -f /aiarena/task_logs/task-100154-ningsz-hokgamecore-master-0.log

## 2. 启动 Learner 任务
在“启动命令”中，将其内部 IP 写入配置文件。假设 Learner 的 IP 是 10.244.50.25：
最终测试成功版本：
---------------

# 1. 转换换行符
sed -i 's/\r$//' /aiarena/scripts/learner/*.sh

# 2. 强制安装合并代码
pip uninstall rl_framework -y > /dev/null 2>&1 || true
for pkg in common learner mem_pool model_pool monitor predictor; do
    cd /aiarena/gpfs/ningsz/hok/rl_framework/$pkg && pip install . --no-deps
done

# 3. 【新增】安装缺失的依赖包
pip install tensorboard

# 4. 建立软链接
ln -sfn /aiarena/gpfs/ningsz/hok/aiarena /aiarena/code

# 5. 日志重定向
mkdir -p /aiarena/gpfs/ningsz/hok/logs/learner
sed -i 's|LOG_DIR="/aiarena/logs/learner/"|LOG_DIR="/aiarena/gpfs/ningsz/hok/logs/learner/"|g' /aiarena/scripts/learner/start_learner.sh
sed -i 's|/aiarena/logs/learner/|/aiarena/gpfs/ningsz/hok/logs/learner/|g' /aiarena/scripts/learner/start_learner.sh
sed -i 's|/aiarena/logs/learner/train.log|/aiarena/gpfs/ningsz/hok/logs/learner/train.log|g' /aiarena/code/learner/train.py
sed -i 's|/aiarena/logs/learner/|/aiarena/gpfs/ningsz/hok/logs/learner/|g' /aiarena/code/learner/config/common.conf

# 6. 写入 IP 并启动
IP=$(hostname -I | awk '{print $1}')
echo "$IP root 36000 1" > /aiarena/scripts/learner/learner.iplist
# 【关键修复】写一份到共享存储，让 Actor 能读到！
echo "$IP root 36000 1" > /aiarena/gpfs/ningsz/learner_addr.txt

sh /aiarena/scripts/learner/start_learner.sh

# 7. 监听日志
touch /aiarena/gpfs/ningsz/hok/logs/learner/train.log
echo "========== 开始监听日志 =========="
tail -f /aiarena/gpfs/ningsz/hok/logs/learner/train.log

---------------

这次leaner：
10.244.52.40

## 3. 启动 Actor 任务（可多开）
这是最容易出错的一步。启动命令必须同时指定 Learner IP 和 Gamecore Server 地址：

bash
'''
------------
# 1. 转换换行符
sed -i 's/\r$//' /aiarena/scripts/actor/*.sh
sed -i 's/\r$//' /aiarena/gpfs/ningsz/hok/aiarena/actor/*.sh

# 2. 拷贝代码到本地并安装
rm -rf /tmp/rl_framework_build && mkdir -p /tmp/rl_framework_build
cp -r /aiarena/gpfs/ningsz/hok/rl_framework/* /tmp/rl_framework_build/
pip uninstall rl_framework -y > /dev/null 2>&1 || true
for pkg in common learner mem_pool model_pool monitor predictor; do
    cd /tmp/rl_framework_build/$pkg && pip install . --no-deps -q
done
# 【关键修复】补齐缺失的基础依赖包
pip install tensorboard psutil pyyaml h5py absl-py protobuf numpy -q

pip install tensorboard -q

# 3. 建立软链接
ln -sfn /aiarena/gpfs/ningsz/hok/aiarena /aiarena/code

# 4. 获取 Learner IP
while [ ! -f /aiarena/gpfs/ningsz/learner_addr.txt ]; do sleep 2; done
LEARNER_IP=$(awk '{print $1}' /aiarena/gpfs/ningsz/learner_addr.txt)
export GAMECORE_SERVER_ADDR=$(cat /aiarena/gpfs/ningsz/gamecore_addr.txt)

# 5. 强行注入环境变量
export MEM_POOL_ADDR="${LEARNER_IP}:35200"
export MODEL_POOL_ADDR="${LEARNER_IP}:10014"

echo "配置信息: Gamecore=$GAMECORE_SERVER_ADDR, MemPool=$MEM_POOL_ADDR, ModelPool=$MODEL_POOL_ADDR"

# 6. 直接执行我们的魔改版脚本，只开 1 个 Actor
bash /aiarena/gpfs/ningsz/hok/aiarena/actor/start_actors.sh 1

# 7. 保持容器存活
sleep 24h
------------------------
'''

NO_MODEL_POOL 是最大的坑：官方文档 [3†L26-L28] 说明，只有当 Actor 和 Learner 在同一个容器时，才需要设置 NO_MODEL_POOL=1 来跳过启动 model pool。既然你的 Actor 和 Learner 是分离的独立任务，就绝对不能设置这个变量，否则 Actor 将无法获取模型，训练会停滞。你看到我提供的启动命令里有 unset NO_MODEL_POOL，其目的就是确保这个变量不会被意外设置。
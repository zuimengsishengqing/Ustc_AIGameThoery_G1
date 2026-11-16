dockerfile.base (Ubuntu 20.04 + Python 3.8 + 系统工具)
    │
    ├── dockerfile.base.torch ──→ gpu_base_v2.0.1     ← 第1层
    │       PyTorch 2.0.1 + CUDA 11.6
    │
    └── dockerfile.dev ──→ battle (target)
            └── rl_framework 全部6个包               ← 第2层
            └── hok_env
            └── aiarena/1v1/ + aiarena/3v3/
            └── aiarena/scripts/
            └── Grafana + InfluxDB + 监控

Docker Hub 上发布的标签:
  tencentailab/hok_env:gpu_v2.0.1        ← 第1层 + 第2层（完整版）
  tencentailab/hok_env:gpu_base_v2.0.1   ← 仅第1层

结论：tencentailab/hok_env:gpu_v2.0.1 用于 Learner 训练，百分之百完整，什么都不缺。
需要注意的两个点
1. CUDA 版本：11.6 vs 实验室的 11.8
兼容的。 CUDA 驱动是向前兼容的——物理机装了 CUDA 11.8 驱动，可以运行 CUDA 11.6 编译的 PyTorch。3090 的驱动版本肯定 >= 11.8，完全能跑。

2. 镜像体积：偏大（含 TF + Horovod + Grafana 等 1v1 用不上的东西）
但这个不影响功能，只是上传慢一点。

镜像 ①：Learner（GPU 训练）
来源：tencentailab/hok_env:gpu_v2.0.1（Docker Hub 预构建 ）

组件	包含？	证据
CUDA 11.6 + PyTorch 2.0.1	✅	dockerfile.base.torch:18 torch==2.0.1
rl_framework/learner	✅	dockerfile.dev:L36 pip install
rl_framework/mem_pool	✅	dockerfile.dev:L37
rl_framework/model_pool	✅	dockerfile.dev:L38
rl_framework/predictor	✅	dockerfile.dev:L39
rl_framework/common	✅	dockerfile.dev:L35
hok_env 包	✅	dockerfile.dev:L48 pip install -e /hok_env
aiarena/1v1/learner/	✅	dockerfile.dev:L72 target battle
aiarena/1v1/actor/	✅	同上（battle 复制全部 1v1）
aiarena/1v1/common/	✅	同上
aiarena/scripts/	✅	dockerfile.dev:L56
Grafana + InfluxDB（监控）	✅	dockerfile.dev:L16-L21
Learner 镜像：✅ 100% 完整，什么都不缺。

镜像 ②：Actor（CPU 采样）
来源：tencentailab/hok_env:cpu_v2.0.1（同一份 dockerfile.dev，只是换 CPU base）

和 Learner 镜像的代码内容完全一样，区别仅在于基座是 PyTorch CPU 版而非 GPU 版。

组件	包含？
全部 rl_framework 6 个包	✅
hok_env	✅
aiarena/1v1/actor/ 全部文件	✅
aiarena/scripts/start_actor.sh	✅
Actor 镜像：✅ 100% 完整。

实际上 Actor 和 Learner 可以用同一个镜像（gpu_v2.0.1），Actor 推理代码硬编码了 torch.device("cpu")，所以即使在 GPU 镜像里跑 Actor 也只用 CPU。

### Gamecore Wine构建：
Gamecore + Wine 的运行方式
run_gamecore_server.sh 里有两套路径：

bash

# 路径 A（推荐）：有 Linux 原生 server 包装器
if [ -f "gamecore-server-linux-amd64" ] && [ -z "${SIMULATOR_USE_WINE}" ]; then
    # server 本身是 Linux 原生程序
    if [ -z "${SIMULATOR_USE_WINE}" ]; then
        ./gamecore-server-linux-amd64 server ...       # 纯 Linux
    else
        ./gamecore-server-linux-amd64 server ... \     # server 是 Linux 的
            --simulator-remote-bin .../sgame_simulator_remote_zmq  # 模拟器走 Wine
    fi
else
    # 路径 B（兜底）：全走 Wine
    wine gamecore-server.exe server ...                # 全部用 Wine
fi
你用 SIMULATOR_USE_WINE=1 触发的是路径 A 的第二个分支——混合模式：

text



┌─────────────────────────────────────────┐
│            Docker 容器 (Linux)           │
│                                         │
│  gamecore-server-linux-amd64 (原生)     │  ← HTTP API，Linux 原生程序
│         │                               │
│         │ 通过 Wine 调用                 │
│         ▼                               │
│  sgame_simulator_remote_zmq.exe         │  ← Windows 二进制，Wine 运行
│  (对局模拟器，每帧推进游戏)              │
│                                         │
└─────────────────────────────────────────┘
Dockerfile.gamecore 的三阶段构建
dockerfile.gamecore 分三步走：

阶段	做什么
Stage 1 gamecore	解压 hok_env_gamecore.zip → 得到 gamecore/ 目录
Stage 2-2 wine_base	dpkg --add-architecture i386 → 安装 3rd/wine_debs/ 里的 Wine .deb
Stage 3 最终镜像	把 Wine + gamecore + license.dat + remote-gc-server 脚本全部合到一起
text



dockerfile.gamecore 最终镜像内容：
├── /rl_framework/gamecore/
│   ├── bin/
│   │   └── sgame_simulator_remote_zmq.exe   ← Windows 模拟器（Wine 跑）
│   ├── lib/                                  ← Wine DLL 依赖
│   ├── core_assets/
│   │   └── license.dat                       ← 你的许可证
│   └── gamecore-server-linux-amd64           ← Linux server 包装器
├── /rl_framework/remote-gc-server/
│   ├── run_gamecore_server.sh                ← 启动脚本
│   └── run_and_monitor_gamecore_server.sh    ← 带守护的启动
├── /aiarena/logs/                            ← 日志目录
└── Wine 运行时（/opt/wine-stable/）
三张镜像的最终分工





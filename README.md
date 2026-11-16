# 腾讯王者荣耀AI开放环境（Honor of Kings AI Open Environment）

![avatar](./docs/hok_1v1.png)

## 更新：3v3 模式现已上线

```
python3.8 -c "from hok.hok3v3.unit_test.test_env import run_test; run_test()"
```

关于 3v3 模式的更多信息，请参考 [hok3v3](https://doc.aiarena.tencent.com/paper/hok3v3/latest/hok3v3_env/honor-of-kings/)。

关于使用 `hok_env` 环境和内置的 `rl_framework` 进行集群训练的说明，请查阅 [cluster_zh.md](docs/cluster_zh.md) 文档。

## 简介

[![PyPI](https://github.com/tencent-ailab/hok_env/actions/workflows/python-publish.yml/badge.svg)](https://github.com/tencent-ailab/hok_env/actions/workflows/python-publish.yml)
[![Image](https://github.com/tencent-ailab/hok_env/actions/workflows/gpu.yml/badge.svg)](https://github.com/tencent-ailab/hok_env/actions/workflows/gpu.yml)

- [Hok_env](https://github.com/tencent-ailab/hok_env) 是 MOBA 游戏《[王者荣耀](https://pvp.qq.com/)》的开放AI环境。

- 本仓库主要包含 Hok_env SDK、一个强化学习训练框架，以及基于该训练框架实现的 PPO 算法。Hok_env SDK 用于与王者荣耀的游戏核心（Gamecore）进行交互。

- 本仓库还包含以下论文的实现代码：

    > **Honor of Kings Arena: an Environment for Generalization in Competitive Reinforcement Learning**（王者荣耀竞技场：一个面向竞争性强化学习泛化能力的环境）\
    > Hua Wei*, Jingxiao Chen*, Xiyang Ji*, Hongyang Qin, Minwen Deng, Siqin Li, Liang Wang, Weinan Zhang, Yong Yu, Lin Liu, Lanxiao Huang, Deheng Ye, Qiang Fu, Wei Yang. (*同等贡献) \
    > **NeurIPS Datasets and Benchmarks 2022** \
    > 项目主页：https://github.com/tencent-ailab/hok_env \
    > arXiv：https://arxiv.org/abs/2209.08483

    > **摘要**：*本文介绍了王者荣耀竞技场（Honor of Kings Arena），一个基于王者荣耀（全球目前最受欢迎的游戏之一）的强化学习（RL）环境。与以往大多数工作中研究的环境相比，我们的环境为竞争性强化学习带来了新的泛化挑战。这是一个多智能体问题，其中一个智能体与对手竞争；它需要具备泛化能力，因为它需要控制多种目标英雄并与多种对手进行竞争。我们详细描述了王者荣耀领域的观察空间、动作空间和奖励函数，并提供了一个开源的 Python 接口用于与游戏引擎通信。我们在王者荣耀竞技场中提供了二十个目标英雄，涵盖了多种任务，并展示了在可行计算资源下基于强化学习方法的初步基线结果。最后，我们展示了王者荣耀竞技场带来的泛化挑战及其可能的解决方案。所有软件（包括环境类）均已公开发布在：https://github.com/tencent-ailab/hok_env。文档地址：https://aiarena.tencent.com/hok/doc/。*

- 当前 hok_env 已支持的英雄：
  - 鲁班七号（lubanqihao）
  - 芈月（miyue）
  - 李白（libai）
  - 马可波罗（makeboluo / Marco Polo）
  - 狄仁杰（direnjie）
  - 关羽（guanyu）
  - 貂蝉（diaochan）
  - 露娜（luna）
  - 韩信（hanxin）
  - 花木兰（huamulan）
  - 不知火舞（buzhihuowu / Mai Shiranui）
  - 橘右京（jvyoujing / Ukyou Tachibana）
  - 后羿（houyi）
  - 钟馗（zhongkui）
  - 干将莫邪（ganjiangmoye）
  - 铠（kai）
  - 公孙离（gongsunli）
  - 裴擒虎（peiqinhu）
  - 上官婉儿（shangguanwaner）

## 运行环境要求

- Python >= 3.6，<= 3.9

- Windows 10/11 或 Linux Wine（用于部署 Windows 版游戏核心服务器）

- Docker（用于在 Linux 容器中部署 hok_env）

  - 对于 Windows 系统，需要 WSL 2（适用于 Linux 的 Windows 子系统 2.0 版本）

hok_env 的 Gamecore 运行在 Windows 平台上，而 **hok_env** 包需要部署在 Linux 平台上才能与 Gamecore 进行交互。

同时我们提供了 Docker 镜像用于在本地计算机上进行训练。在后续版本中，我们将发布兼容 Linux 的 gamecore server。

关于集群训练的配置，这里提供了一种在 Linux 上运行 Windows Gamecore 的解决方案：[在 Linux 上运行 Windows Gamecore](./docs/run_windows_gamecore_on_linux_zh.md)。

## Gamecore 安装

### 下载 hok gamecore

你需要在此页面申请许可证和 gamecore：https://aiarena.tencent.com/aiarena/zh/open-gamecore

请将 `license.dat` 放置在文件夹 `hok_env_gamecore/gamecore/core_assets` 下。

![avatar](./docs/sgame_folder.png)

### 测试 gamecore

打开 CMD（命令提示符）：
```shell
 cd gamecore\bin
 set PATH=%PATH%;..\lib\
 .\sgame_simulator_remote_zmq.exe .\sgame_simulator.common.conf
```

`sgame_simulator_remote_zmq.exe` 需要一个参数：`sgame_simulator.common.conf`（配置文件路径）

你会看到以下消息：
```
PlayerNum:2
AbsPath:../scene/1V1.abs
PlayerInfo [CampID:0][HeroID:199][Skill:80104][AutoAi:1][AiServer::0:100] [Symbol 0 0 0] [Request:-1]
PlayerInfo [CampID:1][HeroID:199][Skill:80104][AutoAi:1][AiServer::0:100] [Symbol 0 0 0] [Request:-1]
SGame Simulator Begin
init_ret:0
seed: 3417
Symbols:
inHeroId: 199
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
inHeroId: 199
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
start_ret:0
Hero[0] inHeroId:199; outPlayerId:148
Hero[1] inHeroId:199; outPlayerId:149
[Hero Info] [HeroID:199] [RuntimeID:8] client_id:0.0.0.0_1658_0_20221202193534_148
[Hero Info] [HeroID:199] [RuntimeID:9] client_id:0.0.0.0_1658_0_20221202193534_149
boost_ret finished: 8, gameover_ai_server: 0
close_ret:0
uninit_ret:0
SGame Simulator End [FrameNum:8612][TimeUsed:7580ms]
```
Gamecore 已成功启动！

以下是 `sgame_simulator.common.conf` 配置文件的内容：
```json
{
    "abs_file": "../scene/1V1.abs",
    "core_assets": "../core_assets",
    "game_id": "kaiwu-base-35401-35400-6842-1669963820108111766-217",
    "hero_conf": [
        {
            "hero_id": 199
        },
        {
            "hero_id": 199
        }
    ]
}
```

输出文件：
```
AIOSS_221202-1935_linux_1450111_1450123_1_1_20001_kaiwu-base-35401-35400-6842-1669963820108111766-217.abs
kaiwu-base-35401-35400-6842-1669963820108111766-217.json
kaiwu-base-35401-35400-6842-1669963820108111766-217.stat
kaiwu-base-35401-35400-6842-1669963820108111766-217_detail.stat
```

## 1v1

### 观察空间与动作空间

请参考 https://aiarena.tencent.com/hok/doc/quickstart/index.html

### 使用方法

关于 hok1v1 的基本使用方式，请参考 [hok1v1/unit_test](hok_env/hok/hok1v1/unit_test/test_env.py)。

关于 hok1v1 的训练代码，请参考 [aiarena/1v1](aiarena/1v1)。

### 在 WSL 中使用演示脚本测试 gamecore

你可以在 WSL（适用于 Linux 的 Windows 子系统）中使用一个简单的 Python 脚本来测试 gamecore。

#### 确保你的电脑支持 WSL2

关于 WSL2 的安装与升级，请参考链接：https://docs.microsoft.com/zh-cn/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package

你需要在 WSL 中安装 Python 3.6 及一些必要的依赖项。

#### 在 WSL2 中运行测试脚本

0. 在 WSL2 外部启动 gamecore server

    ```shell
    cd gamecore
    gamecore-server.exe server --server-address :23432
    ```

1. 在 Python 环境中安装 hok_env

    ```shell
    ## git clone 本仓库后
    cd hok_env/hok_env
    pip install -e .
    ```

2. 运行测试脚本

    ```shell
    cd /hok_env/hok/hok1v1/unit_test
    python test_env.py
    ```

如果你看到以下消息，说明你已成功与 Hok_env 建立连接并完成了一局游戏。恭喜！

```
# python test_env.py
127.0.0.1:23432 127.0.0.1
======= test_send_action
camp_config {'mode': '1v1', 'heroes': [[{'hero_id': 132}], [{'hero_id': 133}]]}
common_ai [False, True]
try to get first state...
first state:  dict_keys(['observation', 'legal_action', 'reward', 'done', 'model_output_name', 'game_id', 'player_id', 'frame_no', 'sub_action_mask', 'req_pb', 'sgame_id'])
first frame: 0
----------------------run step 0
----------------------run step 100
----------------------run step 200
----------------------run step 300
----------------------run step 400
----------------------run step 500
----------------------run step 600
----------------------run step 700
----------------------run step 800
----------------------run step 900
----------------------run step 1000
----------------------run step 1100
2023-08-23 13:13:57.782 | INFO     | hok.common.log:info:85 - game not end, send close game at first: cur_frame_no(3525)
[{
    "player_id": 8,
    "frame_no": 30,
    "observation": array([1.00000000e00, 1.00000000e00, 0.00000000e00, 0.00000000e00, ...]),
    "reward": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "legal_action": array([1.0, 1.0, 1.0, 0.0, 0.0, ...]),
    "sub_action_mask": {
        0: array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1: array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ...
        11: array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    },
    "req_pb": hok.hok1v1.lib.interface.AIFrameState,
}, ...]
```

### 修改 1v1 游戏配置

在运行游戏环境之前，你需要在运行路径下创建 `config.json` 文件。

示例：
```json
{
  "reward_money": "0.006",
  "reward_exp": "0.006" ,
  "reward_hp_point": "2.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.6",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "5.0",
  "reward_last_hit": "0.5",
  
  "log_level": "4"
}
```

此配置文件包含了各项子奖励系数以及 protobuf 处理部分的日志级别（log_level）。
该文件仅在创建 `HoK1v1` 实例时加载，
即使你调用 `HoK1v1.reset`，任何修改都不会被重新加载。

在大多数情况下，log_level 应设置为 `4`，以避免产生无用的日志信息。
只有当你使用本环境遇到错误时，才可能需要将 `log_level` 设为更低的值，以便我们获取更多关于错误的详细调试信息。

## 3v3

### 观察空间与动作空间

更多信息请参考 [hok3v3](https://doc.aiarena.tencent.com/paper/hok3v3/latest/hok3v3_env/honor-of-kings/)。

### 使用方法

假设你已在 `127.0.0.1:23432` 启动了 gamecore server，且运行 hok_env 的 IP 地址为 `127.0.0.1`。

以下是 hok3v3 环境的基本使用方式：

- 获取环境实例：

    ```python
    GC_SERVER_ADDR = os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432")
    AI_SERVER_ADDR = os.getenv("AI_SERVER_ADDR", "127.0.0.1")
    reward_config = RewardConfig.default_reward_config.copy()
    
    env = get_hok3v3(GC_SERVER_ADDR, AI_SERVER_ADDR, reward_config)
    ```

- 重置环境并开始新游戏

    ```python
    use_common_ai = [True, False]
    camp_config = {
        "mode": "3v3",
        "heroes": [
            [{"hero_id": 190}, {"hero_id": 173}, {"hero_id": 117}],
            [{"hero_id": 141}, {"hero_id": 111}, {"hero_id": 107}],
        ],
    }
    env.reset(use_common_ai, camp_config, eval_mode=True)
    ```

- 游戏循环与动作预测

    ```python
    gameover = False
    while not gameover:
        for i, is_comon_ai in enumerate(use_common_ai):
            if is_comon_ai:
                continue
    
            continue_process, features, frame_state = env.step_feature(i)
            gameover = frame_state.gameover
            # 只在每 3 帧预测一次
            if not continue_process:
                continue
    
            probs = random_predict(features, frame_state)
            ok, results = env.step_action(i, probs, features, frame_state)
            if not ok:
                raise Exception("step action failed")
    
    env.close_game(force=True)
    ```

你可以通过以下方式获取默认的奖励配置：
```python
reward_config = RewardConfig.default_reward_config.copy()
```

以下是[奖励配置示例](aiarena/3v3/actor/config/config.py)：
```python
reward_config = {
    "whether_use_zero_sum_reward": 1,
    "team_spirit": 0,
    "time_scaling_discount": 1,
    "time_scaling_time": 4500,
    "reward_policy": {
        "hero_0": {
            "hp_rate_sqrt_sqrt": 1,
            "money": 0.001,
            "exp": 0.001,
            "tower": 1,
            "killCnt": 1,
            "deadCnt": -1,
            "assistCnt": 1,
            "total_hurt_to_hero": 0.1,
            "atk_monster": 0.1,
            "win_crystal": 1,
            "atk_crystal": 1,
        },
    },
    "policy_heroes": {
        "hero_0": [169, 112, 174],
    },
}
```

你可以通过英雄名称获取 hero_id，参考 [HERO_DICT](hok_env/hok/common/camp.py)：
```python
from hok.common.camp import HERO_DICT
print(HERO_DICT)
```

关于上文介绍的完整代码，请参考 [hok3v3/test_env](hok_env/hok/hok3v3/unit_test/test_env.py)。

你可以通过以下脚本运行测试代码：

```shell
python3.8 -c "from hok.hok3v3.unit_test.test_env import run_test; run_test()"
```

如果环境运行正常，你会看到如下消息：
```
2023-12-28 12:54:28.106 | INFO     | hok.hok3v3.unit_test.test_env:get_hok3v3:14 - Init libprocessor: /usr/local/lib/python3.8/dist-packages/hok/hok3v3/config.dat
2023-12-28 12:54:28.106 | INFO     | hok.hok3v3.unit_test.test_env:get_hok3v3:15 - Init reward: {'whether_use_zero_sum_reward': 1, 'team_spirit': 0.2, 'time_scaling_discount': 1, 'time_scaling_time': 4500, 'reward_policy': {'policy_name_0': {'hp_rate_sqrt': 1, 'money': 0.001, 'exp': 0.001, 'tower': 1, 'killCnt': 1, 'deadCnt': -1, 'assistCnt': 1, 'total_hurt_to_hero': 0.1, 'ep_rate': 0.1, 'win_crystal': 1}}, 'hero_policy': {1: 'policy_name_0'}, 'policy_heroes': {'policy_name_0': [1, 2]}}
2023-12-28 12:54:28.107 | INFO     | hok.hok3v3.unit_test.test_env:get_hok3v3:16 - Init gamecore environment: 127.0.0.1:23432 127.0.0.1
2023-12-28 12:54:28.107 | INFO     | hok.hok3v3.reward:update_reward_config:116 - Update reward config: time_scaling_time:4500, time_scaling_discount:1, team_spirit:0.2, whether_use_zero_sum_reward:1
2023-12-28 12:54:28.107 | INFO     | hok.hok3v3.reward:update_reward_config:124 - Update hero reward config: 1 -> {'hp_rate_sqrt': 1, 'money': 0.001, 'exp': 0.001, 'tower': 1, 'killCnt': 1, 'deadCnt': -1, 'assistCnt': 1, 'total_hurt_to_hero': 0.1, 'ep_rate': 0.1, 'win_crystal': 1}
2023-12-28 12:54:28.107 | INFO     | hok.hok3v3.reward:update_reward_config:124 - Update hero reward config: 2 -> {'hp_rate_sqrt': 1, 'money': 0.001, 'exp': 0.001, 'tower': 1, 'killCnt': 1, 'deadCnt': -1, 'assistCnt': 1, 'total_hurt_to_hero': 0.1, 'ep_rate': 0.1, 'win_crystal': 1}
2023-12-28 12:54:28.136 | INFO     | hok.hok3v3.server:start:31 - Start server at tcp://0.0.0.0:35151
2023-12-28 12:54:28.139 | INFO     | hok.hok3v3.env:reset:85 - Reset info: agent:0 is_common_ai:True
2023-12-28 12:54:28.139 | INFO     | hok.hok3v3.env:reset:85 - Reset info: agent:1 is_common_ai:False
2023-12-28 12:54:30.212 | INFO     | hok.hok3v3.unit_test.test_env:run_test:78 - ----------------------run step 0
2023-12-28 12:54:30.673 | INFO     | hok.hok3v3.unit_test.test_env:run_test:78 - ----------------------run step 100
2023-12-28 12:54:30.945 | INFO     | hok.hok3v3.unit_test.test_env:run_test:78 - ----------------------run step 200
```

## 集群训练

关于使用 `hok_env` 环境和内置的 `rl_framework` 进行集群训练的说明，请查阅 [cluster_zh.md](docs/cluster_zh.md) 文档。

## 回放工具：ABS 解析工具（与 Gamecore 一并提供）

观看游戏对局是直观了解你的智能体整场表现的最佳方式。我们提供了一个回放工具来可视化对局过程。

这是一款官方回放软件，用于解析 Gamecore 生成的 ABS 文件，并在王者荣耀的游戏界面中输出视频。
Gamecore 生成的 ABS 文件可在 gamecore 文件夹下找到。
你可以将 ABS 文件放入 `Replays` 文件夹中，然后运行 **ABSTOOL.exe** 来可视化对局。
![avatar](./docs/replay-tool.gif)

## 预构建镜像

https://hub.docker.com/r/tencentailab/hok_env

另请参阅：[run_with_prebuilt_image_zh.md](./docs/run_with_prebuilt_image_zh.md)

## 从 PyPI 安装

```
pip install hok
```

## 引用

如果你使用了 hok_env 的 Gamecore 或本仓库中的代码，请按以下格式引用我们的论文：

```
@inproceedings{wei2022hok_env,
  title={Honor of Kings Arena: an Environment for Generalization in Competitive Reinforcement Learning},
  author={Wei, Hua and Chen, Jingxiao and Ji, Xiyang and Qin, Hongyang and Deng, Minwen and Li, Siqin and Wang, Liang and Zhang, Weinan and Yu, Yong and Liu, Lin and Huang, Lanxiao and Ye, Deheng and Fu, Qiang and Yang, Wei},
  booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2022}
}
```

# 项目架构：
aiarena/1v1/
├── common/
│   ├── algorithm_torch.py  ← 模型定义 + PPO loss
│   ├── algorithm_tf.py     ← TensorFlow 版本（结构相同）
│   └── config.py           ← 超参数配置
│
├── learner/
│   └── train.py            ← 🎯 Learner 入口（组装 + 启动）
│
├── actor/
│   ├── entry.py            ← Actor 进程入口
│   ├── actor.py            ← 单局游戏采样循环
│   ├── agent.py            ← 单个英雄的决策代理
│   ├── sample_manager.py   ← 样本收集 + 打包 + 发送 MemPool
│   ├── model.py            ← Actor 端推理模型（与 Learner 同结构）
│   └── custom.py           ← Agent 工厂函数
│
└── rl_framework/           ← 🔧 基础设施框架
    └── learner/
        ├── framework/pytorch/
        │   ├── apd_benchmark.py   ← 核心训练循环
        │   ├── model_manager.py   ← 模型保存/同步
        │   └── node_info_*.py     ← 分布式节点管理
        └── dataset/
            ├── network_dataset_zmq.py    ← ZMQ 数据管道
            └── sample_generation/
                └── offline_rlinfo_adapter.py  ← 数据格式适配
### 完整流程
┌────────── GameCore (C++) ──────────┐
│  游戏引擎: 物理、战斗、状态更新      │
│  输出: 结构化特征向量 (frame_state) │
└────────────┬───────────────────────┘
             │ frame_state (809维)
             ▼
┌────────── Actor (Python) ──────────┐
│  agent.py:                         │
│    forward(feature) → logits+value │
│    采样 actions → env.step()       │
│                                    │
│  sample_manager.py:                │
│    收集 16 帧 → GAE 算 advantage   │
│    打包 16336 维 → ZMQ 发送        │
└────────────┬───────────────────────┘
             │ ZMQ (打包样本)
             ▼
┌────────── MemPool (C++) ───────────┐
│  内存池: 缓存 + 混洗 + 批处理       │
│  输出: (batch, 16336) 大张量        │
└────────────┬───────────────────────┘
             │ ZMQ (批量样本)
             ▼
┌────────── Learner (Python) ────────┐
│  algorithm_torch.py:               │
│    format_data() → forward()       │
│    compute_loss() → backward()     │
│    optimizer.step()                │
│                                    │
│  model_manager.py:                 │
│    保存 checkpoint → ModelPool     │
└────────────┬───────────────────────┘
             │ 模型同步
             ▼
┌────────── ModelPool ───────────────┐
│  模型仓库: 版本管理 + 分发          │
│  Actor 定期拉取最新模型             │
└────────────────────────────────────┘

## MemPool和采用的交互协议
server：
一个 ZMQ REP socket 监听端口
multiprocessing.Queue 作为内部缓冲
Actor push → 解包 → 入队
Learner pull → 出队
就这些，没有复杂的混洗、优先级队列、批量优化
client:
代码里有 socket_type == "mcp++" 分支，指向一个高性能 C++ 版 MemPool（比 Python ZMQ 版快很多）

## rl_framework分析：
rl_framework/
├─ common/           ← 通用工具层
├─ mem_pool/         ← 样本缓冲池
├─ model_pool/       ← 模型分发池
├─ predictor/        ← 模型推理接口
├─ monitor/          ← 监控上报
├─ learner/          ← Learner 训练框架
├─ send_model/       ← 模型同步工具
└─ setup.py × 5      ← 各子包安装脚本
###  被使用的部分：
common/
├─ logging/          ★ 几乎每个文件都 import
│   ├─ LOG (loguru)  → 统一的日志记录器
│   ├─ log_time      → @log_time("name") 装饰器，自动打点耗时
│   ├─ g_log_time    → 全局耗时字典，按阶段统计性能
│   └─ setup_logger  → 配置日志输出文件/级别
│
├─ utils/
│   └─ common_func.py  ★ model.py 用
│       └─ Singleton  → 单例元类，确保 Model 全局唯
mem_pool/
├─ mem_pool_api/
│   ├─ mem_pool_apis.py    ★ sample_manager.py 使用
│   │   ├─ MemPoolAPIs()   → 构造函数，连接 MemPool 服务器
│   │   ├─ push_samples()  → Actor 推样本
│   │   ├─ pull_sample()   → Learner 拉样本
│   │   └─ clean_samples() → 清空
│   └─ mem_pool_protocol.py → 编解码协议
│
└─ zmq_mem_pool_server/
    └─ zmq_mem_pool.py     → MemPool 服务端（独立进程运行）
        ├─ ZMQMEMPOOL()    → ZMQ REP 监听 + Queue 缓冲
        └─ pull_samples()  → Learner 拉取队列
predictor/
└─ predictor/
    ├─ local_torch_predictor.py   ★ agent.py 使用（PyTorch 后端）
    │   └─ LocalTorchPredictor()  → 加载 .pth 模型，forward 推理
    ├─ local_predictor.py         ★ agent.py 使用（TensorFlow 后端）
    │   └─ LocalPredictor()       → 加载 TF 模型
    ├─ remote_predictor.py        ❌ 远程推理（生产环境用）
    ├─ base_predictor.py          → 抽象基类
    └─ infer_input_output.py      → 输入输出规范
monitor/
└─ rl_framework/monitor/
    ├─ __init__.py
    │   └─ InfluxdbMonitorHandler  ★ entry.py 使用
    │       → 把 episode_infos (win, kill, reward...) 写入 InfluxDB
    └─ loglib/
        └─ influxdb_handler.py     → 底层 InfluxDB 客户端
learner/rl_framework/learner/
│
├─ dataset/
│   ├─ sample_generation/
│   │   └─ offline_rlinfo_adapter.py  ★ train.py 使用
│   │       └─ deserialization() → bytes → numpy
│   │
│   ├─ network_dataset/
│   │   ├─ pytorch/
│   │   │   ├─ network_dataset_zmq.py     ★ train.py 使用
│   │   │   │   → ZMQ 从 MemPool 拉数据 → PyTorch Dataset
│   │   │   └─ network_dataset_random.py  ★ train.py 使用（调试用）
│   │   │       → 随机生成假数据 → 快速验证训练流程
├─ framework/
│   ├─ pytorch/
│   │   ├─ apd_benchmark.py       ★ train.py 使用
│   │   │   └─ Benchmark.run()    → 主训练循环：load data → forward → loss → backward → save
│   │   ├─ model_manager.py       ★ train.py 使用
│   │   │   → 保存/恢复 checkpoint，推送模型到 ModelPool
│   │   ├─ node_info_ddp.py       ★ train.py 使用
│   │   │   → 分布式训练的 rank / world_size 信息
│   │   └─ node_info_hvd.py       ★ train.py 使用
│   │       → Horovod 分布式信息
│   └─ common/
│       ├─ config_control.py      ★ train.py 使用
│       │   → 管理训练配置（读取 JSON，热更新等）
│       └─ log_manager.py         ★ train.py 使用
│           → 日志管理器
### rl framework训练部分 apd_benchmark.py:
核心训练步；
def do_train_step(self, step_context, _input_datas):
    # ====== ① 清空梯度 ======
    self.optimizer.zero_grad()

    # ====== ② 前向传播（这部分你已读过） ======
    data_list = self.net.format_data(_input_datas)    # 拆包 batch → list
    rst_list  = self.net_wrapper(data_list)            # 模型 forward（MLP→LSTM→6头）
    total_loss, info_list = self.net.compute_loss(data_list, rst_list)  # PPO-Clip loss
    
    # ====== ③ 梯度回传 ★★★  ======
    total_loss.backward()
    # PyTorch 自动沿着计算图反向传播，计算所有 self.parameters() 的 .grad
    
    # ====== ④ Horovod 梯度同步（仅 Horovod 模式） ======
    if self.distributed_backend == "horovod":
        self.optimizer.synchronize()
        # 对所有 worker 的梯度做 allreduce 取平均
    
    # ====== ⑤ 梯度裁剪 ======
    if self.config_manager.use_grad_clip:
        torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config_manager.grad_clip_range
        )
        # 防止梯度爆炸，按全局 norm 裁剪
    
    # ====== ⑥ 记录 loss 和检测 inf/nan ======
    if self.local_step % self.config_manager.display_every == 0:
        step_context.set_forward_info(total_loss, info_list)
        if self.config_manager.check_values:
            step_context.check_has_inf_nan(total_loss, self.parameters)
    
    # ====== ⑦ 参数更新 ★★★ ======
    self.optimizer.step()
    # Adam: θ = θ - lr * m_hat / (sqrt(v_hat) + ε)
    
    # ====== ⑧ 学习率调度 ======
    if self.lr_scheduler:
        self.lr_scheduler.step()

混合精度路径的区别在于：
② 前向包在 torch.cuda.amp.autocast() 中
③ total_loss 先被 scaler.scale() 放大再 backward()，防止 FP16 小梯度下溢
⑤ 裁剪前先 scaler.unscale_(optimizer) 还原梯度 scale
⑦ 使用 scaler.step(optimizer) 代替 optimizer.step()，自动跳过 inf/nan 的 step
额外调用 scaler.update() 动态调整缩放因子

#### 数据流总结
1. Gamecore 产生一帧 (observation + reward)
       │
2. Agent.process() 调用模型推理 → action
       │
3. Actor._run_episode() 收集 (state, action, reward, value, prob)
       │
4. SampleManager.save_sample() 按帧存储，回填上一帧的 next_value + reward
       │
5. 对局结束 → save_last_sample() 设置终局 reward
       │
6. SampleManager._calc_reward()  GAE 反向计算 advantage + reward_sum
       │
7. SampleManager._format_data()  16帧打包为 16336 维 numpy 向量
       │
8. SampleManager._send_game_data()  ZMQ PUSH → MemPool
       │
9. NetworkDatasetZMQ  ZMQ PULL ← MemPool，攒够 batch_size 条
       │
10. Benchmark.do_train_step()
      format_data → forward → compute_loss → backward → optimizer.step()
       │
11. ModelManager.save_checkpoint() → ModelPool
       │
12. Agent 下次 reset() 时从 ModelPool 拉取新模型
##### learner内部：
NetworkDataset(ZMQ/Random)  →  _input_datas (一个 batch)
        │
        ▼
  format_data()         拆成 data_list (多个 tensor)
        │
        ▼
  net_wrapper(data_list)   前向: MLP → LSTM → 6头输出
        │
        ▼
  compute_loss()         PPO-Clip + GAE value loss + 熵正则化
        │
        ▼  total_loss (标量)
  backward()             ★ 自动微分，算出所有参数梯度
        │
        ▼
  optimizer.synchronize()  (Horovod only) 跨 GPU 梯度平均
        │
        ▼
  clip_grad_norm_()      梯度裁剪
        │
        ▼
  optimizer.step()       ★ Adam 更新参数
        │
        ▼
  lr_scheduler.step()    学习率衰减

### 进程启动与通信：
总结：进程间的"穿针引线"
你问的问题	答案
谁启动 Actor？	monitor_actor.sh 守护进程，死循环确保不挂
谁启动 Learner？	start_learner.sh，通过 mpirun/torchrun 启动 N 个进程
Actor 和 Learner 怎么通信？	不直接通信！通过 MemPool (ZMQ) 异步解耦
MemPool 在哪启动？	嵌入在 Learner 进程内（NetworkDatasetZMQ 初始化时自动拉起 ZMQMEMPOOL）
模型怎么分发？	预编译 Go 二进制 modelpool，Learner Push，Actor Pull
多个 Actor 怎么分配 MemPool？	select_mempool() 按 actor_id % mempool_num 负载均衡
对局对手怎么选？	keep_latest=False 的 Agent 80% 概率取最新模型（主控），20% 取历史模型（对手）

## hokenv：
hok_env/                          ← Python 包根目录 (setup.py 安装)
├── hok/
│   ├── common/                   ← 跨模式公共模块
│   │   ├── camp.py               ← 🔑 英雄阵容迭代器
│   │   ├── gamecore_client.py    ← 🔑 与 gamecore HTTP 通信
│   │   ├── server_process.py     ← 子进程管理
│   │   └── log.py                ← 日志封装
│   │
│   ├── hok1v1/                   ← 1v1 模式
│   │   ├── env1v1.py             ← 🔑 环境主类 (reset/step)
│   │   ├── server.py             ← 🔑 AI Server (ZMQ 接收 gamecore 请求)
│   │   ├── agent.py              ← Agent 基类
│   │   ├── hero_config.py        ← 读取 default_hero_config.json
│   │   ├── config.json           ← 🔑 奖励权重配置
│   │   ├── config.dat            ← 二进制配置 (libprocessor 读取)
│   │   ├── default_hero_config.json ← 🔑 英雄默认装备/技能/铭文
│   │   ├── lib/                  ← C++ 编译的动态库 (.so)
│   │   │   └── interface.cpython-39-*.so
│   │   ├── server_delay.py       ← 带延迟的 AI Server (测试用)
│   │   └── version.py
│   │
│   └── hok3v3/                   ← 3v3 模式 (结构对称)
│       ├── env.py
│       ├── server.py
│       ├── action_space.py
│       ├── reward.py             ← 3v3 专用奖励函数
│       ├── default_hero_config.json
│       └── lib/                  ← 3v3 专用 lib3v3.so
│
├── env.yaml                      ← Conda 环境声明
├── setup.py / pyproject.toml     ← pip install 入口
└── MANIFEST.in                   ← 打包清单


### 对手/英雄选择全流程
1. 英雄选择 — camp.py#L191-L230
默认 driver = "1v1_roundrobin_camp_heroes"，5 个英雄 两两组合：

hero_ids = [112, 169, 133, 199, 132]
 鲁班    后羿   狄仁杰  公孙离  马可波罗
itertools.product(heroes, heroes) → 5×5 = 25 种对阵组合，每轮 shuffle 打乱顺序，无限循环。

2. 对手类型 — config.py#L132-L142

ENEMY_TYPE = "network"   # ← 默认就是 RL 模型互打！
三种可选值：

环境变量	值	对手类型
不设置	—	network（自对弈）
ENEMY_TYPE=0	random	随机动作
ENEMY_TYPE=1	common_ai	规则 AI
ENEMY_TYPE=2	network	RL 模型
3. 谁做主 agent — entry.py#L128

main_agent = random.randint(0, AGENT_NUM - 1)  # 随机 camp0 或 camp1
每局随机选一边做"主 agent"（keep_latest=True），负责发送训练样本到 mempool。

4. 双方用什么模型 — actor.py#L82-L89

if len(load_models) == 1 and not agent.keep_latest:
    agent.reset(self.enemy_type, model_path=load_models[0])
else:
    agent.reset(self.enemy_type)
关键：load_models=[]（空列表），所以两个 agent 都走 reset("network")，都从同一个 checkpoint 加载权重 → 镜像自对弈。

        ┌─────────────────────────────┐
        │   /aiarena/checkpoints/     │
        │        model.pth            │
        └──────┬──────────┬───────────┘
               │          │
          camp0.load()  camp1.load()
               │          │
          RL Agent 0   RL Agent 1
               │          │
               └────互相对战────┘
双方用的是一模一样的模型，这就是标准的 self-play / 镜像对战。唯一的区别是随机选一边做主 agent 负责采集样本。
####  default_hero_config.json — 🔑 召唤师技能 & 铭文配置
每个英雄配置三项：

json
{
    "hero_id": 112,          // 英雄 ID
    "skill_id": 80115,      // 🔑 召唤师技能 (见下表)
    "symbol": [1512, ...]   // 🔑 铭文 (30个槽位)
}
80104	惩击
80108	终结
80110	狂暴
80115	闪现
80121	治疗

## 进化与训练策略细节：
### 多英雄泛化实现：
observation(453维) + hero_one_hot(20维) = 473维
向量拼接，后面20维度输入操控英雄，hero id作为条件输入

### 零和博弈进化
self-play 下 reward 是零和的（一边赢一边输）
但是reward本身，可以刷分，督促双方进步：
Reward 分解（config.json）:
  reward_money   × 0.006     ← 双方都可以打钱
  reward_exp     × 0.006     ← 双方都可以吃经验
  reward_hp      × 2.0       ← 双方都可以保持血量
  reward_tower   × 5.0       ← 双方都可以推塔 ⭐ 权重最大
  reward_kill    × -0.6      ← 击杀对手也有收益
  reward_dead    × -1.0
  reward_last_hit × 0.5
### 定期Eval：
actor.py:L25-L28 和 config.py:L143：
eval_freq = 5  # 每 5 局 eval 一次

eval_with_common_ai = (
    self.m_config_id == 0 and self._episode_num % eval_freq == 0
)
Eval 模式下行为：
- 主 agent 用最新 RL 模型
- 对手 agent 用 common_ai（游戏内置规则 AI）
- 不发送训练样本（纯评估）

# GameCore
## 目录
hok_env_gamecore/
├── version                                          # 构建版本号
│
├── gamecore/                                        # ★ 核心战斗引擎
│   ├── gamecore-server.exe                          # Windows 服务端（训练时启动，监听 23432）
│   ├── gamecore-server-linux-amd64                  # Linux 服务端（云端用）
│   │
│   ├── bin/                                         # 模拟器可执行文件
│   │   ├── sgame_simulator_remote_zmq.exe           # 远程模拟器（训练时用）
│   │   ├── sgame_simulator_remote_zmqBase.dll
│   │   ├── sgame_simulator_repeated_zmq.exe         # 重复模拟器（回放/验证用）
│   │   └── sgame_simulator_repeated_zmqBase.dll
│   │
│   ├── lib/                                         # 依赖库
│   │   ├── libgamecore.dll                          # ★ 核心：C++ 战斗引擎本体
│   │   ├── cygzmq.dll                               # ZMQ 通信库
│   │   ├── cygwin1.dll                              # Cygwin POSIX 运行时
│   │   └── cygstdc++-6.dll / cyggcc_s-seh-1.dll    # C++ STL / GCC 运行时
│   │
│   ├── core_assets/                                 # 游戏资源
│   │   ├── license.dat                              # ★ 许可证 ✅ 已放对位置！
│   │   ├── config/                                  # 游戏数值配置
│   │   └── customresources/                         # 动作/音效/UI/特效/精彩时刻
│   │       ├── actions/          # 受击/Buff/技能动作的 XML 定义
│   │       ├── sound/            # 音效 .bytes 文件
│   │       ├── ui3d/             # UI 3D 资源
│   │       ├── text/             # 本地化文本
│   │       └── wonderfulmoments/ # 精彩时刻（击杀/团战判定）
│   │
│   └── scene/                                       # 地图与英雄 ABS 文件
│       ├── 1V1.abs              # 1v1 墨家机关道
│       ├── 3V3.abs              # 3v3
│       ├── 5V5.abs              # 5v5 王者峡谷
│       ├── 5V5DLD.abs           # 5v5 深渊大乱斗
│       └── {英雄名}.abs         # 24 个可选英雄 (后羿/鲁班/李白/貂蝉...)
│
└── replay_tool/                                     # ABS 回放工具
    ├── ABSTool.exe                                  # Unity 独立应用
    └── ABSTool_Data/                                # Unity 运行时
        ├── Managed/              # C# 程序集 (UnityEngine/Assembly-CSharp)
        ├── Plugins/              # 原生插件 (GCloud/libcurl...)
        ├── Mono/                 # Mono 运行时
        └── Resources/            # Unity 预设资源
## 内置AI：
在gamecore里面：
ailab/ai_config/
├── game_ai_model_config.txt      ← 总入口：列出所有子模块路径
├── 1v1/                          ← 1v1 的内置 AI 配置
│   ├── strategy/                 ← 大局观（宏观决策）
│   ├── tactics/                  ← 战术（微观操作）
│   ├── rule/                     ← 规则系统（硬编码逻辑）
│   ├── buy_equipment/            ← 出装策略
│   ├── learn_skill/              ← 技能加点
│   ├── banpick/                  ← BP策略
│   └── game_situation_assessment/← 局势评估
│
└── 5v5/                          ← 5v5 的内置 AI 配置（结构相同）
内置AI逻辑：
Strategy（大局观）       →  决定"去哪里"（移动目标、占位）
    │  模型：每位英雄一个 Caffe 模型 (如 139strategy_struct.txt + weights)
    │  特征：thumb_vec（地图缩略图向量 + 英雄/野怪位置向量）
    │
    ▼
Tactics（战术操作）       →  决定"放什么技能"（攻击、技能、回撤）
    │  模型：每位英雄一个 Caffe 模型 (如 116tactics_struct.txt + weights)
    │  特征：img_vec (4805维图像特征) + vec_data (1569维向量特征)
    │  输出：11个Label（3个技能 + 普攻带目标 + 回撤 + 躲避技能）
    │
    ▼
Rule 系统（规则兜底）     →  硬编码安全逻辑（残血回城、必死不走等）

### 快速启动：
先！检查上次模型pth文件备份！（如果没开bak逻辑）

先开gamecore server：
cd E:\LearnAI\AIvs_class\hok_env\hok_env\hok_env_gamecore\gamecore
gamecore-server.exe server --server-address :23432

gamecore-server.exe server --server-address	23432
终端 1 — WSL：Learner
【问题：单卡ddp不需要，所以>1适配跳过；monitor子模块下载；backend取消暂时，防止保存太多；】
bash

wsl -d Ubuntu2204
source ~/hokenv/bin/activate

cd /mnt/e/LearnAI/AIvs_class/hok_env/hok_env/aiarena/1v1/learner

python train.py
等到看到 init finished 再启动终端 2。

终端 2 — WSL：Actor（单个，建议用优化sh脚本一次开多个）
bash

wsl -d Ubuntu2204
source ~/hokenv/bin/activate
export GAMECORE_SERVER_ADDR="$(ip route show default | awk '{print $3}'):23432"
cd /mnt/e/LearnAI/AIvs_class/hok_env/hok_env/aiarena/1v1/actor

#只打鲁班：(进文件夹后再用该命令，cd之前似乎无效？)
export CAMP_DRIVER_CONFIG="WzExMl0="

python entry.py \
  --actor_id=0 \
  --mem_pool_addr="127.0.0.1:35200" \
  --model_pool_addr="" \
  --max_episode=-1


#终端2优化版：
wsl -d Ubuntu2204
source ~/hokenv/bin/activate
cd /mnt/e/LearnAI/AIvs_class/hok_env/hok_env/aiarena/1v1/actor
bash start_actors.sh 4   

终端 3— WSL：TensorBoard
bash

wsl -d Ubuntu2204
source ~/hokenv/bin/activate

tensorboard --logdir=/aiarena/logs/learner/tensorboard --bind_all
浏览器：http://localhost:6006

#### 同时看两个日志
wsl -d Ubuntu2204
tail -f /aiarena/logs/learner/train.log /aiarena/logs/actor/actor_0.log


#### 关闭与进程：
bash check_actors.sh    # 查看所有 Actor 进程 + 内存 + 最新日志
bash stop_actors.sh     # 一键全部停止


#### ABSTool：
cd E:\LearnAI\AIvs_class\hok_env\hok_env\hok_env_gamecore\replay_tool

直接指定 abs 文件打开
.\ABSTool.exe "E:\LearnAI\AIvs_class\hok_env\hok_env\hok_env_gamecore\gamecore\simulator_output\AIOSS_261104-2342_linux_1450111_1450123_1_1_20001_kaiwu-actor-1v1-0-1793806917723614000-6.abs"

启动顺序总结

1. [Win CMD]   gamecore-server.exe     ← 必须先开
2. [WSL T1]    python train.py         ← 等到 init finished
3. [WSL T2]    python entry.py ...     ← 开始对局
4. [WSL T3]    tensorboard ...         ← 可随时开启/关闭



#### 配置与链接
服务端打开：
gamecore-server.exe server --server-address :23432

# 用 Windows 真实 IP（不要用 127.0.0.1）
export GAMECORE_SERVER_ADDR="$(ip route show default | awk '{print $3}'):23432"

echo $GAMECORE_SERVER_ADDR
# 应输出: 172.30.33.34:23432

# 启动开悟平台版本：
启动方式（与本地一一对应，只换入口文件）
① gamecore server（Windows，不变）

cd E:\LearnAI\AIvs_class\hok_env\hok_env\hok_env_gamecore\gamecore
gamecore-server.exe server --server-address :23432

② Learner（WSL 终端 1） —— 唯一区别是 train.py → train_platform.py

wsl -d Ubuntu2204
source ~/hokenv/bin/activate
cd /mnt/e/LearnAI/AIvs_class/hok_env/hok_env/aiarena/1v1/learner
python train_platform.py
看到 init finished 后再开终端 2。

③ Actor（WSL 终端 2） —— 唯一区别是 entry.py → entry_platform.py

wsl -d Ubuntu2204
source ~/hokenv/bin/activate
export GAMECORE_SERVER_ADDR="$(ip route show default | awk '{print $3}'):23432"
cd /mnt/e/LearnAI/AIvs_class/hok_env/hok_env/aiarena/1v1/actor
export CAMP_DRIVER_CONFIG="WzExMl0="

python entry_platform.py \
  --actor_id=0 \
  --mem_pool_addr="127.0.0.1:35200" \
  --model_pool_addr="" \
  --max_episode=-1
③ 批量版：已新建 start_actors_platform.sh（与 start_actors.sh 并列，仅把入口换成 entry_platform.py，日志前缀 actor_platform_*.log 避免和本地日志混）：

wsl -d Ubuntu2204
source ~/hokenv/bin/activate
cd /mnt/e/LearnAI/AIvs_class/hok_env/hok_env/aiarena/1v1/actor
bash start_actors_platform.sh 4
关闭：
pkill -f entry_platform.py
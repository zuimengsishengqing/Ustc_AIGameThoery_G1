# 学习笔记 3v3 算法

## 一、算法架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        3v3 PPO 训练架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐   样本     ┌──────────┐   参数     ┌──────────────┐  │
│   │  Actor   │ ────────> │ MemPool  │ ────────> │   Learner    │  │
│   │ (推理采样)│ <──────── │ (ZMQ队列) │ <──────── │ (PPO训练)    │  │
│   └──────────┘   模型同步  └──────────┘   梯度同步  └──────────────┘  │
│        │                                                    │       │
│        │  TCP/ZMQ                                          │       │
│        ▼                                                    ▼       │
│   ┌──────────┐                                     ┌──────────────┐ │
│   │ GameCore │                                     │  ModelPool   │ │
│   │ (Windows)│                                     │ (模型版本管理)│ │
│   └──────────┘                                     └──────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 使用的算法：**PPO（Proximal Policy Optimization）**

- **PPO-Clip** 变体，裁剪范围 `ε = 0.2`
- **GAE（Generalized Advantage Estimation）**：λ=0.95, γ=0.995
- **多智能体独立训练**：3 个英雄各自拥有独立的策略输出头，但共享底层编码
- **无共享 Critic**：每个英雄有独立的价值网络输出
- **无 LSTM**：3v3 的 Learner 模型不使用 LSTM（Actor 模型保留了 LSTM 占位但实际隐藏单元仅 16）

---

## 二、文件结构

```
aiarena/3v3/
├── actor/                          # Actor 端（推理 + 采样）
│   ├── entry.py                    # 入口：启动 Actor 进程
│   ├── actor.py                    # Actor 主循环（游戏交互 + 采样发送）
│   ├── agent/agent.py              # Agent：模型推理 + 样本处理
│   ├── sample_manager.py           # 样本管理器：GAE + 格式化 + 发送
│   ├── rl_data_info.py             # 单帧样本数据结构
│   ├── model/pytorch/model.py      # Actor 端 PyTorch 推理模型
│   └── config/
│       ├── config.py               # Actor 配置（奖励/超参）
│       └── model_config.py         # 模型维度配置
│
└── learner/                        # Learner 端（训练）
    ├── train.py                    # 训练入口
    ├── config/
    │   ├── Config.py               # Learner 配置
    │   ├── DimConfig.py            # 特征维度配置
    │   └── common.conf             # 框架配置
    └── networkmodel/pytorch/
        └── NetworkModel.py         # PyTorch 训练模型（核心）
```

---

## 三、核心类与函数分析

### 3.1 `NetworkModel` (Learner 训练模型) — [NetworkModel.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/learner/networkmodel/pytorch/NetworkModel.py)

#### 关键参数

| 参数 | 值 | 含义 |
|------|:---:|------|
| `LSTM_TIME_STEPS` | **16** | LSTM 序列长度（仅用于样本分批，模型不用 LSTM） |
| `LSTM_UNIT_SIZE` | **16** | LSTM 隐藏单元（极小，基本没用） |
| `INIT_LEARNING_RATE_START` | **0.0006** | 初始学习率 |
| `BETA_START` | **0.008** | 熵正则化系数 |
| `CLIP_PARAM` | **0.2** | PPO 裁剪参数 |
| `MIN_POLICY` | **1e-5** | 最小策略概率（防除零） |
| `HERO_FEATURE_IMG_CHANNEL` | **6** | 图像特征通道数 |
| `HERO_NUM` | **3** | 英雄数量 |

#### 输入数据维度 (每个英雄)

| 组件 | 维度 | 说明 |
|------|------|------|
| **图像特征** (IMG) | `6 × 17 × 17` (1734维) | 空间特征图（经 CNN 编码） |
| **向量特征** (VEC) | 2852 维 | 包含所有实体特征 |
| **总输入** | **4586 维/英雄** | IMG(1734) + VEC(2852) |

#### 模型结构（每个英雄独立前向）

```
输入: [feature_img 1734维, feature_vec 2852维] = 4586维

Step 1: CNN 编码图像特征
  feature_img (6,17,17)
    → Conv2d(6→18, kernel=5×5, padding='same')
    → ReLU → MaxPool2d(3,2)    # (18,7,7)
    → Conv2d(18→12, kernel=3×3, padding='same')  # (12,7,7)
    → Flatten → 588维              ← ★ 这就是 CNN 的输出

Step 2: 向量特征拆分编码
  feature_vec (2852维) → 拆分为:
  ┌─ hero_frd:    [251]×3  = 753维
  ├─ hero_emy:    [251]×3  = 753维
  ├─ hero_main:   [44]×1   = 44维
  ├─ soldier_frd: [25]×10  = 250维
  ├─ soldier_emy: [25]×10  = 250维
  ├─ organ_frd:   [29]×3   = 87维
  ├─ organ_emy:   [29]×3   = 87维
  ├─ monster:     [28]×20  = 560维
  └─ global_info: [68]×1   = 68维
  ─────────────────────────────────
  Total VEC = 753+753+44+250+250+87+87+560+68 = 2852维 ✓

Step 3: 各实体 MLP 编码 + MaxPool
  ┌─ hero_frd_mlp: [251]→512→512→128→128→64  (×3英雄→pool→64维)
  ├─ hero_emy_mlp: [251]→512→512→128→128→64  (×3英雄→pool→64维)
  ├─ hero_main_mlp: [44]→64→64→64→64→32       → 32维
  ├─ soldier_frd:  [25]→64→64→64→32           (×10小兵→pool→32维)
  ├─ soldier_emy:  [25]→64→64→64→32           (×10小兵→pool→32维)
  ├─ monster:      [28]→64→64→64→32           (×20怪物→pool→32维)
  ├─ organ_frd:    [29]→64→64→64→32           (×3塔→pool→32维)
  └─ organ_emy:    [29]→64→64→64→32           (×3塔→pool→32维)

Step 4: 拼接 → concat_mlp
  concat_dim = 588(CNN)+32+32+32+32+32+32+64+64+68= 1156 维
  → MLP(1156→256) → split:[64, 192]
      ├─ 64  → public_first (共享编码，用于团队池化)
      └─ 192 → public_second (私有编码，个体特定)

Step 5: 跨英雄编码融合（特征共享机制）
  3个英雄的 public_first[64] 堆叠 → MaxPool → pool_hero_public[64]
  再与各自的 public_second[192] 拼接 → [256]

Step 6: 多任务输出头 (每个英雄独立)
  [256] → MLP(256→64→label_size)
  ├─ label0 (which_button): 13类
  ├─ label1 (move_dir):     25类
  ├─ label2 (target_unit):  42类
  ├─ label3 (skill_dir):    42类
  ├─ label4 (skill_pos):    39类
  └─ value (价值估计):      1维
```

#### 关键方法

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__()` | L21~L194 | 构建 CNN + MLP 编码器 + 输出头 |
| `forward(data_list)` | L196~L356 | 对 3 个英雄分别前向，返回 `[hero0_result, hero1_result, hero2_result]` |
| `compute_loss(data_list, rst_list)` | L358~L520 | 对 3 个英雄分别计算 PPO 损失 |
| `format_data(datas)` | L522~L550 | 将 MemPool 原始数据解析为结构化列表 |

#### forward() 数据流详解

```
for hero_index in [0,1,2]:
    1. hero_feature = data_list[hero_index][0]  # 4586维
    2. 拆分为 feature_img(1734) + feature_vec(2852)
    3. feature_img → CNN → flatten(588)
    4. feature_vec → 拆分 → 8个实体组 → MLP编码 → MaxPool聚合
    5. 拼接 CNN输出 + 所有实体编码 + global_info = 1156维
    6. concat_mlp(1156→256) → split [64, 192]
    7. 收集到 hero_public_first_list, hero_public_second_list

# 跨英雄融合
pool_hero_public = max_pool(hero_public_first_list)  # 3英雄→1个64维

for hero_index in [0,1,2]:
    8. fc_public = [pool_hero_public(64), hero_public_second(192)] = 256维
    9. 5个 label MLP + 1个 value MLP
    10. 收集到 all_hero_result_list

return all_hero_result_list  # [[hero0_6outputs], [hero1_6outputs], [hero2_6outputs]]
```

#### compute_loss() 损失计算

```python
# 对每个英雄独立计算
for hero_index in [0,1,2]:
    value_cost = 0.5 × MSE(reward, predicted_value)   # 价值损失

    for task in [5 个子任务]:                          # 策略损失
        ratio = exp(log π_new - log π_old)
        surr1 = ratio × advantage
        surr2 = clip(ratio, 0.8, 1.2) × advantage
        policy_cost += -min(surr1, surr2)

    entropy_cost = -Σ(π × log(π))                      # 熵损失（最大化熵）

    total_loss += value_cost + policy_cost + 0.008 × entropy_cost
```

---

### 3.2 `Model` (Actor 推理模型) — [model.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/actor/model/pytorch/model.py)

与 Learner 的 `NetworkModel` **结构完全相同**，区别在于：

| 差异点 | Learner (NetworkModel) | Actor (Model) |
|--------|------------------------|---------------|
| 输入格式 | 批量 16 帧序列 `(batch, 16*4586)` | 单帧 `(1, 4586)` |
| format_data | 拆分序列帧 | 直接 unsqueeze |
| 输出 | `all_hero_result_list` | 展平后的 `probs_h0/h1/h2` + lstm 状态 |
| LSTM | 不使用 | 保留占位 `lstm_cell/lstm_hidden` |
| 用途 | 训练 | 推理 |

#### 关键方法

| 方法 | 功能 |
|------|------|
| `forward(data_list)` | 与 NetworkModel 相同的编码逻辑，但输出展平 |
| `format_data(datas)` | 将单帧数据转换为模型输入格式 |

---

### 3.3 `Agent` (Actor 端智能体) — [agent.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/actor/agent/agent.py)

| 参数/属性 | 值/类型 | 说明 |
|-----------|---------|------|
| `model` | Model 实例 | PyTorch 推理模型 |
| `_predictor` | LocalTorchPredictor | 加载和运行模型的预测器 |
| `_model_pool_api` | ModelPoolAPIs | 从模型池获取模型 |
| `keep_latest` | bool | 是否始终使用最新模型（仅主 agent 为 True） |
| `is_latest_model` | bool | 当前是否持有最新模型 |
| `lstm_unit_size` | 16 | LSTM 状态维度 |
| `model_version` | str | 当前加载的模型版本号 |

#### 关键方法

| 方法 | 功能 |
|------|------|
| `reset(agent_type)` | 重置 LSTM 状态；从模型池加载随机/最新模型 |
| `predict_process(features, frame_state)` | 对 3 个英雄的特征做推理，返回 probs 和 lstm_info |
| `_predict_process_torch(features, ...)` | PyTorch 推理：组装输入 → 模型前向 → 解析输出 |
| `sample_process(features, results, lstm_info, frame_state)` | 封装一帧的样本数据（feature, action, prob, value 等） |
| `compute_reward(features, frame_state)` | 自定义奖励计算（team_spirit + zero_sum + time_scaling） |
| `update_model()` | 如果是 latest 模型，从模型池拉取更新 |
| `_get_random_model()` | 80% 概率取最新模型，20% 取随机历史模型 |

---

### 3.4 `Actor` (游戏交互主循环) — [actor.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/actor/actor.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| `send_sample_frame` | **963** (约 10 秒) | 每隔多少帧发送一次样本 |
| `AGENT_NUM` | **2** | 两个阵营各一个 Agent |

#### `_run_episode()` 主循环流程

```
1. agents.reverse()                # 每局交换两个 Agent 的阵营
2. agent.reset() × 2               # 重置模型 + LSTM 状态
3. env.reset(use_common_ai, camp_config)
4. while not gameover:
     for agent in [0,1]:
       if is_common_ai: continue    # 规则型 AI 跳过

       5. env.step_feature(i)       # 获取特征
       6. agent.predict_process()   # 模型推理
       7. env.step_action(i)        # 执行动作
       8. agent.sample_process()    # 封装样本
       9. sample_manager.save_sample() 或 save_last_sample()

       10. 每 963 帧 或 游戏结束时:
             sample_manager.send_samples()  # GAE + 格式化 + ZMQ 发送
             sample_manager.reset()
             agent.update_model()
```

---

### 3.5 `SampleManager` (样本管理器) — [sample_manager.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/actor/sample_manager.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| `gamma` | **0.995** | 折扣因子 |
| `lamda` | **0.95** | GAE 参数 |
| `lstm_time_steps` | **16** | 每批样本帧数 |
| `lstm_unit_size` | **16** | LSTM 单元大小 |
| `hero_num` | **3** | 英雄数 |

#### 关键方法

| 方法 | 功能 |
|------|------|
| `save_sample(...)` | 逐帧保存样本到 `rl_data_map`（按 frame_no 有序字典） |
| `save_last_sample(agent_id, reward, value_s)` | 保存最后一帧的 reward 和 next_value |
| `send_samples()` | 触发 `_calc_reward()` → `_format_data()` → `_send_game_data()` |
| `_calc_reward()` | **GAE 反向计算** advantage 和 reward_sum |
| `_format_data()` | 将帧序列组合为 16 帧的 batch + LSTM 状态 → MemPool 格式 |
| `_clip_reward()` | 奖励裁剪到 [-100, 100] |

#### GAE 计算逻辑 (`_calc_reward`)

```python
# 从最后一帧反向计算
for frame_no in reversed(frames):
    for hero in [0,1,2]:
        delta = -value + reward + gamma * next_value
        gae = gae * gamma * lamda + delta
        advantage = gae
        reward_sum = gae + value           # 用于 value loss 的 target
```

#### 批样本格式 (`_format_data`)

每 16 帧组装为一个训练样本，每个英雄的样本包含：

```
[feature(4586)] + [legal_action(13+25+42+42+39=161)] 
+ [reward_sum(1)] + [advantage(1)]
+ [action(5)] + [probs(161)] + [is_train(1)] + [sub_action(5)]
= 4921 维/帧/英雄

× 16 帧 × 3 英雄 + lstm_cell(16) + lstm_hidden(16) ≈ 236K 字节
```

---

### 3.6 `entry.py` (Actor 启动入口) — [entry.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/actor/entry.py)

| 模块 | 功能 |
|------|------|
| `get_model_class(backend)` | 根据后端选择 PyTorch/TensorFlow Model 类 |
| `select_mempool()` | 负载均衡：轮询分配 Actor → MemPool |
| `auto_bind_cpu()` | CPU 亲和性绑定 |
| `run()` | 组装 Agent, AIServer, Environment, SampleManager, Actor |
| `main()` | CLI 参数解析入口 |

---

### 3.7 `RLDataInfo` (单帧数据结构) — [rl_data_info.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/actor/rl_data_info.py)

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_no` | int | 帧号 |
| `feature` | bytes | 4586 维特征向量 |
| `reward` | float | 即时奖励 |
| `reward_sum` | float | GAE 计算的累计奖励 (value target) |
| `value` | float | 当前状态价值估计 |
| `next_value` | float | 下一状态价值估计 |
| `advantage` | float | GAE 优势估计 |
| `action` | list | 5 维动作 |
| `sub_action` | list | 5 维子动作 |
| `prob` | np.array | 动作概率分布 |
| `legal_action` | np.array | 合法动作掩码 |
| `is_train` | bool | 是否为训练帧 |
| `lstm_info` | np.array(32) | LSTM cell + hidden 拼接 |

---

## 四、特征维度详解

### 输入特征的 4586 维构成

```
每个英雄的特征 = 图像特征(1734) + 向量特征(2852)

图像特征 (6×17×17=1734):
  6 通道的 17×17 空间特征图（GameCore 提供的 minimap 栅格化信息）

向量特征 (2852):
  ├─ 友方英雄: 251×3 = 753   (每个英雄 251 维：位置、血量、技能CD、装备等)
  ├─ 敌方英雄: 251×3 = 753
  ├─ 主控英雄: 44×1  = 44    (额外的自身详细信息)
  ├─ 友方小兵: 25×10 = 250   (10个小兵，每个 25 维)
  ├─ 敌方小兵: 25×10 = 250
  ├─ 友方防御塔: 29×3 = 87   (3座塔，每个 29 维)
  ├─ 敌方防御塔: 29×3 = 87
  ├─ 野怪: 28×20    = 560   (20个野怪，每个 28 维)
  └─ 全局信息: 68×1  = 68    (游戏时间、经济差等)
  ─────────────────────────────────
  总计: 753+753+44+250+250+87+87+560+68 = 2852 ✓
  全特征: 1734 + 2852 = 4586 ✓
```

### 动作空间 (5 个子任务 × 3 英雄)

| 子任务 | 类别数 | 含义 |
|--------|:---:|------|
| `which_button` | **13** | 按下哪个按钮 |
| `move_dir` | **25** | 移动方向（离散化） |
| `target_unit` | **42** | 目标单位 |
| `skill_dir` | **42** | 技能方向 |
| `skill_pos` | **39** | 技能位置 |

> 与 1v1 对比：3v3 多了 `monster` 编码器 + `CNN` 空间特征编码，动作头维度也不同（更多方向离散化）。

---

## 五、关键超参数汇总

| 超参数 | 1v1 | 3v3 | 说明 |
|--------|:---:|:---:|------|
| `LSTM_TIME_STEPS` | 16 | 16 | 序列长度 |
| `LSTM_UNIT_SIZE` | **512** | **16** | ⚠️ 3v3 基本不用 LSTM |
| `LEARNING_RATE` | 0.0001 | 0.0006 | 学习率 |
| `BETA (entropy)` | 0.025 | 0.008 | 熵正则系数 |
| `CLIP_PARAM` | 0.2 | 0.2 | PPO 裁剪范围 |
| `GAMMA` | 0.995 | 0.995 | 折扣因子 |
| `LAMDA` | 0.95 | 0.95 | GAE λ |
| `BATCH_SIZE` | — | — | 由 LSTM_TIME_STEPS 决定 |
| 优化器 | RMSProp | RMSProp | (框架层统一) |
| `SEND_SAMPLE_FRAME` | — | 963 | 采样发送间隔 |

---

## 六、与 1v1 的核心差异

| 维度 | 1v1 | 3v3 |
|------|-----|-----|
| **CNN** | ❌ 无 | ✅ Conv(5×5) + Conv(3×3) |
| **特征维度/英雄** | 809 | 4586 |
| **图像输入** | 无 | 6×17×17 空间特征图 |
| **英雄数** | 2 | 6 (3v3) |
| **实体编码器** | hero/soldier/organ | hero/soldier/organ/**monster** |
| **LSTM** | ✅ hidden=512 | ⚠️ hidden=16（基本不用） |
| **跨英雄融合** | ❌ (只有 1 个主控英雄) | ✅ public_first 池化融合 |
| **target_embed** | ✅ 目标选择辅助 | ❌ 无 |
| **动作头类别** | [12,16,16,16,16,8] | [13,25,42,42,39] |
| **样本格式** | 每个样本 1 个英雄 | 每个样本 3 个英雄串联 |
| **reward** | 零和（i 的 reward = -对手 reward） | team_spirit + zero_sum 混合 |

---

## 七、4 个存档算法

位于 `archive/3v3/` 目录：

| 算法 | 区别 |
|------|------|
| **PPO** | 3v3 论文基准，与 `learner/NetworkModel.py` 结构相同 |
| **MAPPO** | Multi-Agent PPO：集中式 Critic（使用所有智能体信息） |
| **CPPO** | Coordinated PPO：引入智能体间协调机制 |

---

## 八、奖励配置

来自 [config.py](file:///e:/LearnAI/AIvs_class/hok_env/hok_env/aiarena/3v3/actor/config/config.py)：

| 奖励项 | 系数 | 含义 |
|--------|:---:|------|
| `hp_rate_sqrt_sqrt` | 1 | 血量百分比损失（开四次方） |
| `money` | 0.001 | 经济奖励 |
| `exp` | 0.001 | 经验奖励 |
| `tower` | 1 | 推塔奖励 |
| `killCnt` | 1 | 击杀奖励 |
| `deadCnt` | -1 | 死亡惩罚 |
| `assistCnt` | 1 | 助攻奖励 |
| `total_hurt_to_hero` | 0.1 | 对英雄伤害 |
| `atk_monster` | 0.1 | 攻击野怪 |
| `win_crystal` | 1 | 摧毁水晶胜利 |
| `atk_crystal` | 1 | 攻击水晶 |

**团队奖励混合** (Agent 中的 `compute_reward`):
```python
final_reward = (1 - team_spirit) × self_reward 
             + team_spirit × ally_avg_reward 
             - zero_sum × enemy_avg_reward
```
默认 `team_spirit=0`（纯个人奖励），`zero_sum=1`（零和博弈）。
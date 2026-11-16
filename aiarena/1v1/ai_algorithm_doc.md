# 学习笔记 1v1

## algorithm_torch.py 代码全面分析

### 📁 文件结构总览

```
algorithm_torch.py  (共 ~782 行)
├── class Algorithm(nn.Module)          # [L19~L585] 核心算法模型（PPO）
│   ├── __init__()                      # [L20~L188]  构建神经网络层
│   ├── forward()                       # [L190~L400] 前向推理
│   ├── compute_loss()                  # [L402~L585] 计算PPO损失
│   └── format_data()                   # [L587~L624] 数据格式化
├── def make_fc_layer()                 # [L626~L648] 创建FC层的工具函数
├── class MLP(nn.Module)                # [L652~L685] 多层感知机封装
├── def _compute_conv_out_shape()       # [L687~L714] 计算Conv输出shape
└── def make_conv_layer()               # [L717~L782] 创建Conv层的工具函数
```

---

## 🔑 核心结论：1v1 模型中没有 CNN！

> **`make_conv_layer` 在 `algorithm_torch.py` 中虽然被定义了，但它是个「死代码」**——在 1v1 的模型中**从未被调用**。
>
> | 文件 | `make_conv_layer` | 是否有 CNN |
> |------|:---:|:---:|
> | **1v1** (algorithm_torch.py) | 仅定义，未使用 | ❌ 无 CNN |
> | **3v3** (NetworkModel.py / model.py) | 定义 + 使用 | ✅ 有 CNN (5×5 + 3×3) |
> | **archive/3v3/ppo/mappo** | 定义 + 使用 | ✅ 有 CNN |

### 为什么你的直觉是对的
你说「训练用不到CNN做图像识别吧」——**完全正确**。1v1 模型的输入是 **Gamecore 直接提供的结构化向量数据**（~809 维），不是像素图像。这些数据是游戏引擎通过 API 直接吐出来的（英雄血量、位置、距离、技能CD等数值），不需要任何图像识别。

---

## 1. `class Algorithm(nn.Module)` —— PPO 核心算法模型

继承自 `torch.nn.Module`，是整个 1v1 强化学习的核心，包含了**策略网络** + **价值网络** + **LSTM时序记忆**。

### 1.1 `__init__()` —— 网络结构构建 [L20~L188]

网络结构分为以下几个编码模块：

#### 📐 各模块的特征维度

| 模块 | 输入维度 (单个) | 网络结构 | 输出维度 |
|------|:---:|------|:---:|
| **hero_main_mlp**（主控英雄） | 14 | 14→64→32→**16** | 16 |
| **hero_mlp**（英雄共享编码） | 235 | 235→512→256→**128** | 128 |
| **hero_frd_fc**（友方英雄） | 128 | 128→**128** | 128 |
| **hero_emy_fc**（敌方英雄） | 128 | 128→**128** | 128 |
| **soldier_mlp**（小兵共享编码） | 18 | 18→64→64→**32** | 32 |
| **soldier_frd_fc**（友方小兵） | 32 | 32→**32** | 32 |
| **soldier_emy_fc**（敌方小兵） | 32 | 32→**32** | 32 |
| **organ_mlp**（防御塔共享编码） | 18 | 18→64→64→**32** | 32 |
| **organ_frd_fc**（友方防御塔） | 32 | 32→**32** | 32 |
| **organ_emy_fc**（敌方防御塔） | 32 | 32→**32** | 32 |
| **global_info**（全局信息） | 25 | 直接传入 | 25 |
| **concat_mlp**（拼接后全连接） | 389 | 389→**512** | 512 |
| **LSTM**（时序记忆） | 512 | hidden=512, 1层 | 512 |
| **value_mlp**（价值网络） | 512 | 512→64→**1** | 1 |

#### 🧩 拼接维度计算 (pre-LSTM)

```
hero_main:            16
hero_frd (友方1人):   128
hero_emy (敌方1人):   128
soldier_frd (10个小兵→pool): 32
soldier_emy (10个小兵→pool): 32
organ_frd (2座塔→pool):      32
organ_emy (2座塔→pool):      32
global_info:                 25
────────────────────────────────
concat_dim = 16+128+128+32+32+32+32+25 = 389
```

#### 🎯 输出头（多任务策略）

| 子任务 | 类别数 | 含义 |
|--------|:---:|------|
| which_button | **12** | 按下哪个按钮（移动/攻击/技能1-3等） |
| move_dir | **16** | 移动方向的离散化角度 |
| target_unit | **16** | 目标单位类型/编号 |
| skill_dir | **16** | 技能释放方向 |
| skill_pos | **16** | 技能释放位置 |
| target_embed | **8** (与button组合) | 目标嵌入辅助任务 |

每个子任务都有独立的 MLP 输出头：`label_mlp = [512 → label_size]`

---

### 1.2 `forward()` —— 前向推理 [L190~L400]

```python
def forward(self, data_list, inference=False):
```

#### 输入：一个 feature_vec (shape: batch × 809)

809 维向量包含：
- **英雄特征**: `hero_frd(235)` + `hero_emy(235)` + `hero_main(14)` = 484
- **小兵特征**: `soldier_1_10(4×18=72)` + `soldier_11_20(4×18=72)` = 144
- **防御塔特征**: `organ_1_2(2×18=36)` + `organ_3_4(2×18=36)` = 72
- **全局信息**: 25
- **legel_action**: 84（行为掩码）
- **总计**: 484+144+72+25+84 ≈ 809

#### 数据流处理步骤：

```
Step 1: 特征拆分 (split)
  feature_vec → [hero_vecs, soldier_vecs, organ_vecs, global_info]
  ↓ 进一步按 友方/敌方 拆分

Step 2: 各编码器并行处理 + Max Pooling
  ┌─ hero_main:  MLP(14→64→32→16)                    → 16维
  ├─ hero_emy:   MLP(235→512→256→128) → FC(128→128)  → max_pool → 128维
  ├─ hero_frd:   MLP(235→512→256→128) → FC(128→128)  → max_pool → 128维
  ├─ soldier_emy: MLP(18→64→64→32) → FC(32→32)       → max_pool → 32维
  ├─ soldier_frd: MLP(18→64→64→32) → FC(32→32)       → max_pool → 32维
  ├─ organ_emy:  MLP(18→64→64→32) → FC(32→32)        → max_pool → 32维
  └─ organ_frd:  MLP(18→64→64→32) → FC(32→32)        → max_pool → 32维

  Pooling 说明：对 10 个小兵做 max pooling 聚合为 1 个向量；
             对 2 座塔做 max pooling 聚合为 1 个向量

Step 3: 拼接 → Concat MLP(389→512)

Step 4: Reshape → LSTM(512, 512)
  将 512 维向量 reshape 为 [batch, 16, 512]（LSTM 时间步=16）
  LSTM 输出 → split回 [batch*16, 512]

Step 5: 多任务输出头
  ┌─ label0_mlp (12类) → which_button
  ├─ label1_mlp (16类) → move_dir
  ├─ label2_mlp (16类) → target_unit
  ├─ label3_mlp (16类) → skill_dir
  ├─ label4_mlp (16类) → skill_pos
  ├─ target_embed → label5 (8类)
  └─ value_mlp (1维)  → state value
```

#### target_embed 特殊机制
- 在编码时，从每个敌方英雄、小兵的输出中「截取」32 维作为 target_embed
- 这些 target_embed 堆叠后，通过矩阵乘法与 LSTM 输出计算**相似度分数**
- 用于让模型学会「当前应该关注哪个目标」

---

### 1.3 `compute_loss()` —— 损失计算 [L402~L585]

实现 **PPO-Clip** 算法的损失：

#### 损失组成：

```python
total_loss = value_cost + policy_cost + var_beta × entropy_cost
```

| 损失项 | 含义 | 计算方式 |
|--------|------|----------|
| `value_cost` | 价值网络损失 | `0.5 × MSE(reward, predicted_value)` |
| `policy_cost` | 策略损失（PPO-Clip） | `-min(ratio×adv, clip(ratio)×adv)` |
| `entropy_cost` | 熵正则化 | `-Σ(π × log(π))`，加负号意味着最小化负熵=最大化熵 |

#### PPO-Clip 核心逻辑：
```python
ratio = exp(log π_new - log π_old)           # 新旧策略的比值
surr1 = ratio × advantage                     # 无裁剪的代理损失
surr2 = clip(ratio, 1-ε, 1+ε) × advantage    # 裁剪后的代理损失
policy_loss = -min(surr1, surr2)              # 取较小值，防止策略更新过大
```

其中 `ε = CLIP_PARAM = 0.2`，即策略更新比值被限制在 [0.8, 1.2] 之内。

---

### 1.4 `format_data()` —— 数据预处理 [L587~L624]

将原始输入数据切分为：
- `feature`（725维 特征向量）
- `legal_action`（84维 合法动作掩码）
- `lstm_cell_state / lstm_hidden_state`（各512维 LSTM 初始状态）

---

## 2. 工具函数/类

### 2.1 `make_fc_layer()` [L626~L648]
创建全连接层，使用**正交初始化**（`nn.init.orthogonal`），偏置初始化为 0。

### 2.2 `class MLP(nn.Module)` [L652~L685]
多层感知机封装，支持：
- 可变层数和维度
- 可选的激活函数（默认 ReLU）
- 最后一层可选择是否加激活函数（`non_linearity_last`）

### 2.3 `_compute_conv_out_shape()` [L687~L714]
计算卷积后的特征图尺寸（工具函数）。

### 2.4 `make_conv_layer()` [L717~L782]
创建卷积层。**在 1v1 模型中未被调用**，仅在 3v3 的 NetworkModel 中使用。

---

## 3. 1v1 vs 3v3 模型关键差异

| 维度 | 1v1 | 3v3 |
|------|-----|-----|
| **输入特征** | 809维纯向量 | 4586维/英雄（向量 + 图像） |
| **CNN** | ❌ 无 | ✅ 有 (Conv 5×5 + Conv 3×3) |
| **图像输入** | 无 | 6通道 × 17×17 空间特征图 |
| **英雄数量** | 2 (1v1) | 6 (3v3) |
| **怪物编码** | ❌ 无 | ✅ 有 monster_mlp |
| **LSTM隐藏单元** | 512 | 16 |
| **输出头** | 6个子任务 | 5个子任务 × 3英雄 |
| **target_embed** | ✅ | ❌ |

### 3v3 的 CNN 到底处理什么？
3v3 的 CNN 处理的**不是像素图像**，而是 **Gamecore 提供的空间特征图**（类似 minimap 的栅格化数值信息），例如：
- 每个格子上的单位类型/密度
- 技能范围覆盖信息
- 地形信息

它是一个 6 通道 × 17×17 的**数值矩阵**，不是 RGB 图像。所以 3v3 的 CNN 做的是**空间特征编码**而非图像识别。

---

## 4. 总结

1. **1v1 模型是纯 MLP+LSTM 架构**，没有 CNN 成分
2. `make_conv_layer` 在 1v1 文件中只是「公用工具函数」，实际未被调用
3. **CNN 只出现在 3v3 模型**中，用于处理空间特征图（非像素图像）
4. 1v1 用到了 `hero_share` 模式（友方/敌方英雄共享同一套 MLP 权重）
5. 多实体聚合使用了 Max Pooling（对小兵、防御塔）
6. target_embed 机制是 1v1 的特色，帮助模型学习目标选择


## 有序字典与层的命名：
1. OrderedDict 就是有序字典
Python 的 OrderedDict（collections.OrderedDict）保证插入顺序 = 遍历顺序。

python



from collections import OrderedDict

od = OrderedDict()
od["b"] = 1
od["a"] = 2
od["c"] = 3

for k, v in od.items():
    print(k, v)
### 输出: b 1, a 2, c 3    ← 按插入顺序，不是字母序
补充：Python 3.7+ 之后，普通 dict 也保证插入顺序了，但代码写于较早时期（或出于显式语义），仍使用 OrderedDict。

2. 为什么包一层 OrderedDict 再传给 nn.Sequential？
这其实是为了给子模块起名字。来看两种写法的区别：

python



### 写法A：直接传，模块名字是数字索引
nn.Sequential(
    make_fc_layer(256, 128),
    nn.ReLU(),
    make_fc_layer(128, 64),
)
### → 内部模块名为 "0", "1", "2"

### 写法B：传 OrderedDict，模块名字是自定义字符串
nn.Sequential(
    OrderedDict([
        ("hero_frd_fc",  make_fc_layer(256, 128)),
        ("hero_frd_relu", nn.ReLU()),
        ("hero_frd_out", make_fc_layer(128, 64)),
    ])
)
### → 内部模块名为 "hero_frd_fc", "hero_frd_relu", "hero_frd_out"
好处：

方便调试：model.state_dict() 里的 key 是 hero_frd_fc.weight 而不是 0.weight，一眼就知道是哪个模块的参数
方便分类/模块管理：对应你这句代码 —— 所有英雄编码相关的层都以 hero_frd_ 为前缀，防御塔以 organ_ 为前缀，结构一目了然
方便迁移学习/加载权重：如果只想加载部分模块的预训练权重，有名字就能做精确匹配
python



### 有名字时，可以这样精确加载
pretrained = torch.load("checkpoint.pt")
model.hero_frd_fc.load_state_dict(pretrained["hero_frd_fc"])  # 精确匹配

## 轴的切分：
dim按照轴切分，例如：
feature_vec.shape = (batch_size × 16帧, 809特征)
                        ↑ axis=0        ↑ axis=1
沿着dim=1切：feature_vec.split(
    [hero总维度,  soldier总维度,  organ总维度,  global维度],   # 每段的宽度
    dim=1,                                                      # 沿特征轴切
)

 等价于在 axis=1 上切分：
 (N, 809) → [(N, 391), (N, 250), (N, 100), (N, 68)]
              hero域     soldier域   organ域    global域

注意语法：
xxlist.split([a,b,c],dim = 1) #在轴1上面切分，a、b、c三个部分

## fc_layer
make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1]),
全连接

## 英雄可攻击单位：
tar_embed_list 收集的是所有可攻击单位（敌方+友方英雄）的 32 维"目标嵌入"。后面 LSTM 状态会去和这些嵌入做点积，算出选哪个英雄作为攻击目标——这就是多任务头的 target_unit 输出。每个英雄在编码自己特征的同时，也产出一个"被别人选中时用的"表示向量。
先MLP提取英雄共通属性：
hero_mlp：不区分敌我，提取通用英雄属性（位置、血量、等级…本质都是数字）
后面套FNN
hero_frd_fc / hero_emy_fc：才区分阵营，让网络学习"队友和对手的行为意义不同"

这是经典设计——底层特征提取共享，顶层根据语义分化。

后面又128 切分成96和32，根据后面不同的loss信号使得网络干不同的事情，这两部分；

              |前 96 维	|后 32 维
下游用途	|进 max_pool → 拼接 → LSTM |	进 tar_embed_list → 目标选择
受哪些 loss 驱动|	全部 5 个动作头 + value	|仅 target_unit (label2)
学到什么|	"这个英雄状态如何？"（完整特征）|	"我作为目标时如何被识别？"（身份嵌入）

## 英雄之外其他单位
包括 小兵 ，防御塔
架构：
mlp -- FNN ，append（对于len的for循环）
最后concat - reshape -maxpool - reshape ；
并注意只有敌方的加入tar，我方小兵和防御塔不可攻击，直接不加入tar

另外特殊的，1v1里面的maxpool本质是摆设，恒等变换没什么意义；

np.prod是乘，不是梯度！算展平维度用的；
但是maxpool这一部分从插入两个1，再到取max唯一值最后拍平本质没有任何意义！
本质maxpool希望解决的问题是对于多目标时，提取最max表征，捕捉最关键。（比如特别靠近（位置）or残血（生命）？）

tar_embed_0 是一个空选项，没有攻击目标时，选择这个选项；为什么一定选择是0.1呢？0.1 是一个经验选值——足够小不会喧宾夺主，但足够让 weight 矩阵"感知"到这个输入的独特性。它就是让空目标在嵌入空间里有一个可学习的固定锚点，这个锚点不需要自己移动（恒为 0.1）本身0.1常数不会梯度反向传播更新，因为它不是权重不可学习；

## 公共拼接与lstm
拼接后先过一层public的总的mlp
然后进入lstm
结果再次拼接， state0，1分别获取hidden和cell

变量与方法学习：
lstm_time_steps 序列长度——一次处理 16 帧游戏画面； 每个batch元素是一段16帧的序列；
unsqueeze(0) = 在位置 0 插入一个维度；（满足LSTM的需求）
lstm的返回：
lstm_outputs	(batch, 16, 512)	每个时间步的输出隐藏状态
state[0] = h_n	(1, batch, 512)	第 16 帧后的最终隐藏状态
state[1] = c_n	(1, batch, 512)	第 16 帧后的最终细胞状态
====
时间步:  t=0      t=1      t=2  ...  t=15
         ↓        ↓        ↓           ↓
h_0 → [LSTM] → [LSTM] → [LSTM] ... → [LSTM] → h_n (state[0])
c_0 →  Cell  →  Cell  →  Cell  ... →  Cell  → c_n (state[1])
         ↓        ↓        ↓           ↓
      out[0]   out[1]   out[2]      out[15]   ← lstm_outputs (全部16步)
====
最后回到每帧一行（reshape）

## lstm后3个头：（都append到result list）
1.reshape_lstm_outputs_result —— label命名放入label_mlp_out 得到6个动作头
6个MLP动作头
12，16，16，16，16，8
whtchbtn，movedir等
2.target_embed_mlp X乘 （lstm_tar_embed_mlp    ----- reshapre） 得到reshape_label_result
3.对reshape_lstm_outputs_result进行value_mlp得到价值估计

最后根据是否为inference决定是否flatten展平（因为训练时候需要分开计算PPO loss）

## Loss 损失函数
### 传入data的解析：
data_keys = (
    "observation, reward, advantage,"
    "label0,label1,label2,label3,label4,label5,"     # 6 个动作标签
    "prob0,prob1,prob2,prob3,prob4,prob5,"            # 6 个旧策略概率
    "weight0,weight1,weight2,weight3,weight4,weight5," # 6 个权重
    "is_train, lstm_cell, lstm_hidden_state"
)
拆解list：
data_list (24段)
│
├─ [0] seri_vec:    12944维 → reshape(-1, 809)    → (batch×16, 809)  特征向量
├─ [1] usq_reward:  16维   → reshape(-1, 1)       → (batch×16, 1)    奖励
├─ [2] usq_advant:  16维   → reshape(-1, 1)       → (batch×16, 1)    优势
│
├─ [3]  label0:     16维   → reshape(-1, 1)       → (batch×16, 1)    动作标签×6
├─ [4]  label1                                                                │
├─ [5]  label2                   data_list[3 : 3+6]                           6
├─ [6]  label3                   = idx 3~8                                   个
├─ [7]  label4                                                               头
├─ [8]  label5                                                                │
│
├─ [9]  prob0:      192维  → reshape(-1, 12)     → (batch×16, 12)  旧概率×6
├─ [10] prob1:      256维  → reshape(-1, 16)                               │
├─ [11] prob2:      256维            data_list[3+6 : 3+12]                 6
├─ [12] prob3:      256维            = idx 9~14                           个
├─ [13] prob4:      256维                                                头
├─ [14] prob5:      128维  → reshape(-1, 8)                                │
│
├─ [15] weight0:    16维   → reshape(-1, 1)       → (batch×16, 1)    权重×6
├─ [16] weight1                                                              │
├─ [17] weight2                  data_list[3+12 : 3+18]                     6
├─ [18] weight3                  = idx 15~20                               个
├─ [19] weight4                                                            头
├─ [20] weight5                                                              │
│
├─ [21] is_train:   16维   → reshape(-1, 1)       → (batch×16, 1)  训练标记(-3)
├─ [22] lstm_cell:  512维                          → (batch, 512)   LSTM(-2)
└─ [23] lstm_hidden:512维                          → (batch, 512)   LSTM(-1)
### 3 + 1×len → 3 + 2×len → 3 + 3×len 公式
开头固定 3 段：
  [0] feature
  [1] reward
  [2] advantage        ← 基础偏移 = 3

第 1 块（label）： 3 + 0*6 = 3    开始  →  3:9   = idx 3~8   6个动作标签
第 2 块（prob）：  3 + 1*6 = 9    开始  →  9:15  = idx 9~14  6个旧概率
第 3 块（weight）：3 + 2*6 = 15   开始  →  15:21 = idx 15~20 6个权重

剩余：
  [-3] = idx 21 = is_train
  [-2] = idx 22 = lstm_cell
  [-1] = idx 23 = lstm_hidden
### 旧概率和权重：
prob:  采样时刻的 log π_old    →  PPO 用 ratio = exp(log π_new - log π_old) 算重要性采样比
weight: is_train 掩码          →  某些帧是规则AI控制的，击飞眩晕等，被迫mask没法操作出（不参与训练），weight=0 就跳过该帧

### 特征维度分两部分：
SERI_VEC_SPLIT_SHAPE = [(725,), (84,)]
                  ↑ 真实特征    ↑ 合法动作掩码
seri_vec (batch×16, 809)
    │
    split(dim=1, [725, 84])
    │
    ├── 前 725 维 → forward() 中的 feature_vec，进网络做推理
    │
    └── 后 84 维  → 合法动作标志，不进网络，直接用于 mask

### PPO clips计算部分：
I mask范围处理与概率：
task index循环： 6 个任务头各独立算一轮 PPO loss，最后求和
boundry = 10^20作用：超大的数，用来把 illegal action 的概率压到 0。
先把动作索引转化为one hot独热编码，后面用它从分布中取出实际行动动作概率；
然后（游戏状态）非法动作会减去 10^20，使 logit 变成 -inf，softmax 后概率为 0。这就是 Softmax 的数值稳定 trick：Log-Sum-Exp。
clamp 到 [-10^20, 1] 防止溢出。引入epsilon很小常量，
非法动作: 0 × exp + 1e-5 = 1e-5  ← 极小但不为0
合法动作: 1 × exp + 1e-5；
然后带入概率计算π_i = exp(logit_i) / Σ exp(logit_j) 得到list；
II ratio：
PPO 原公式: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
所以old单独有存储，新的pi是用onehot action 和 label_probalility乘法，保留实际执行动作概率；
后面就是calmp安全裁剪ratio到0~3，然后标准1+-clip param，最后（加权-掩码）平均loss
III 熵正则
H = -Σ p_i log p_i  →  熵正则化项，奖励函数中加入，防止过拟合
IIII汇总
self.entropy_cost = Σ entropy_element_items   # 所有任务头求和的 -H̄

total_loss = value_cost + policy_cost + var_beta × entropy_cost

## 工具函数部分：
make_fc_layer 全连接， orthogonal正交初始化和0初始化bias
MLP
convlayer（暂时不用）


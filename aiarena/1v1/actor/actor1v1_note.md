## actor.py
### 内置AI：
_get_common_ai true对应用内置的；false用训练的rl；可以做自博弈和评估；
reload重新加载；
_save_last_sample
注意reward返回，gamecore返回九维向量，最后一个元素是reward_sum总奖励因此用下标-1取；

#### 最后一帧单独计算reward：
save_last_sample 在做什么？
这是局末收尾：一局结束时（done=True），最后一帧没有"下一帧"的 value，GAE 的公式需要终端的 reward 代替：

正常帧: GAE_t = r_t + γ·V(s_{t+1}) - V(s_t)  ← 有下一帧 value
终局帧: GAE_last = r_last - V(s_last)          ← 没有下一帧，用 reward 替代
save_last_sample 就是存这个 r_last，让 SampleManager 能正确计算最后一帧的 GAE advantage。

### _run_episode 核心循环
run_episode() {
    env.reset()                          ← 开一局
    ↓
    while not done:                      ← 逐帧循环
        for each agent:                  ← 每个英雄
            agent.process(state_dict)    ← 模型推理 → 动作
        env.step(actions)                ← 执行动作
        sample_manager.save_sample()     ← 存样本
        ↓
    env.close_game()                     ← 局末收尾
    ↓
    统计 + 打印 + sample_manager.send_samples()  ← 发送 MemPool
}
random model 依照概率随机取 ， 模型或者随机取；
runepisode：
    def _run_episode(self, camp_config, eval=False, load_models=None):
类似的，贪心策略 ：
训练时候有概率随机动作，评估时候用确定性最优动作；（代码中action区分d_action）

### req_pbs 游戏内部装填快照
req_pbs = self.env.cur_req_pb   # 每帧 gamecore 返回的 protobuf 列表
req_pbs = [Player0的protobuf, Player1的protobuf]
              ↑ 主控方视角       ↑ 对手视角
它是 Gamecore 返回的游戏内部状态快照（protobuf 格式），每个玩家一份。包含该帧全部的战争迷雾内可见信息（英雄位置、血量、小兵、防御塔等）
req_pbs[i].hero_list
hero_list 是该玩家视角下所有英雄的终局状态列表。每个 hero_state 是一个 protobuf 对象，包含：
hero_state:
  ├── runtime_id     : int    ← 英雄唯一 ID（1024, 1025...）
  ├── moneyCnt       : float  ← 总经济
  ├── killCnt        : int    ← 击杀数
  ├── deadCnt        : int    ← 死亡数
  ├── assistCnt      : int    ← 助攻数
  ├── totalHurt      : float  ← 总伤害
  ├── totalHurtToHero: float  ← 对英雄伤害
  └── totalBeHurtByHero:float ← 承受英雄伤害
最后进行局末统计，信息上报监控与打印
区分与state_dict:

state_dict	req_pbs
数据量	809 维向量（压缩特征）	完整 protobuf（结构化）
用途	喂给模型做推理	局末统计/日志
帧内用？	✅ 每帧都用	❌ 仅打日志 frame_no
局末用？	❌ 不保留	✅ 读 hero_list 做 KPI

### 性能分析打点时间；
log_time_func 是计时打点，不是模拟时间
它是 rl_framework 的性能分析工具，记录每个阶段耗时：
log_time_func("agent_process")  ← 模型推理耗时
log_time_func("step")           ← Gamecore step 耗时
log_time_func("one_frame")      ← 整帧总耗时
log_time_func("one_episode")    ← 整局总耗时

### 大招使用率评判活跃：
 action[0] == 10 → 大招使用率统计
 h_act_num = "high-level action number" = 大招使用次数。h_act_rate = 每帧平均使用大招的比率。这是一个活跃度指标——大招是冷却最长的技能，频繁使用说明 AI 打得很"凶"。


### gamecore能提供特征向量？
 env1v1.py _step_feature():
parse_state, sgame_id = r_msgs[id]            # 收到 Gamecore 的 protobuf 状态
ret = self.lib_processor.FeatureProcess(parse_state, sgame_id)  # ← C++ 调用！
 ↑ 这是一个编译好的 .so/.dll 共享库

if ret[0] == 2:         # SEND_CCD_ONE_HERO
    state = self._state_tuple2np(ret[1:])[0]  # → 得到包含 observation, reward 等的 dict

### 推理是逐帧进行的；
agent里面progress根据每一step存的state dict去给到完整的前向传播；
LSTM 状态是跨帧维护的，每次 forward 输入上一帧的 (h,c)，输出当前帧的 (h',c') 供下一帧使用。这和训练时 16 帧一起喂给 LSTM 的效果是一致的。


## Agent.py
### 代码
random agent / agent;
set hero info
    def reset ： lstm的hidden和cell归零
    _update_model_list 
    模型读取 / 随机模型读取
    _get_latest_model
    关键：
    process里面走了_predict_process_torch模型前向传播（simple progress具体调用，dictionary组装，并且更新合法动作）
    _update_legal_action 合法动作；
    数据集保存_sample_process_for_saver
    _sample_masked_action 动作mask合法

### 合法mask与头的设计
前五个正常，对于非法10^20 稳定softmax输出概率0；
第六个头 target_embed依赖which_button ,
Gamecore 提供的 legal_action[5] 不是 8 维，而是 96 维 = 12 个按钮 × 8 个目标选项，是一个二维矩阵：
target_legal_action_o = legal_actions[5].reshape(12, 8)
#          目标选项(8维) →
#  按钮    [1,1,1,1,1,0,0,0]  ← 按钮0 (None1): 哪些目标合法
#   ↓      [0,1,1,1,1,0,0,0]  ← 按钮1 (None2)
#          [0,0,0,0,0,0,0,0]  ← 按钮2 (Move): 无目标
#          [1,1,1,0,0,0,0,0]  ← 按钮3 (Attack): 可攻击目标合法
#          [1,0,0,0,0,0,0,0]  ← 按钮4 (Skill1): 可选目标
#          ...
#          [0,1,1,0,0,0,0,0]  ← 按钮10 (Skill4/大招)


## entry启动：
select_mempool轮询内存池  对mempool个数取模分配；

main()                         ← 解析命令行参数
  ↓
run()                          ← 异常保护壳
  ↓
_run()                         ← 组装 7 个核心组件
  ├─ lib_processor             ← C++ 接口库初始化
  ├─ Agent × 2                 ← RL 决策代理
  ├─ GamecoreClient            ← 游戏客户端连接
  ├─ env (HoK1v1)              ← 游戏环境
  ├─ SampleManager             ← 样本收集 + MemPool
  ├─ Monitor (InfluxDB)        ← 监控上报
  └─ Actor                     ← 采样循环控制器
  ↓
actor.run()                    ← 启动采样

### ZMQ
ZeroMQ（ZMQ/ØMQ）是一个高性能异步消息库，在这个项目里做进程间通信的中转管道：
ZMQ 的特点：无中心代理、低延迟、自带缓冲、断线重连——非常适合 RL 这种生产者和消费者速率不匹配的场景（Actor 采样、Learner 训练速度不对等）。

## 日志系统：
┌──────────────────────────────────────────────────┐
│ 日志系统                                          │
│                                                  │
│ LOG (rl_framework)                                │
│   ├─ 本地文件: /aiarena/logs/actor/actor_0.log   │
│   └─ 用途: 调试、排查、性能分析                     │
│                                                  │
│ monitor_logger (InfluxDB)                         │
│   ├─ 远程时序数据库: InfluxDB:8086                │
│   └─ 用途: 训练曲线、KPI 看板（kill, win, reward） │
│                                                  │
│ 上报数据: episode_infos ← actor.py 局末统计       │
│   { reward, win, kill, death, hurt_per_frame,    │
│     money_per_frame, h_act_rate, ... }           │
└──────────────────────────────────────────────────┘
## 命令行解析：
Flag abs1.flags库
## 为什么是offline的？
OfflineRlInfoAdapter 中的 "Offline" = "Learner 从缓冲/文件中读数据，而不是从在线连接中读"。它对应的可能是 OnlineRlInfoAdapter（从直播连接中直接解析 protobuf 流）。这个项目只实现了 Offline 版本。
两个维度完全不相关：
学术上的 On-policy	学术上的 Off-policy
工程 Online	PPO (Actor→Learner 直连)	DQN with 在线 replay
工程 Offline	本项目: PPO + MemPool ←	DQN with ReplayBuffer

### sample_manager.py
SampleManager 在 Actor 的 Python 代码里，不是 MemPool 内部代码。它是 Actor ↔ MemPool 之间的桥梁。
技巧：reward 在当前帧才知道，所以回填到上一帧：
save_last_sample() — 终局收尾
_calc_reward() — GAE 计算
_format_data() — 16 帧打包
_send_game_data() — 推入 MemPool
actor.run_episode():
    ↓
    sample_manager.reset(agents, game_id)     ← 清理上一局的 rl_data_map
    ↓
    while not done:
        agent.process() → sample {reward, value, prob, action, ...}
        sample_manager.save_sample(**sample)   ← 每帧存入 rl_data_map
        env.step()
    ↓
    sample_manager.save_last_sample(reward)    ← 补终局 reward
    ↓
    sample_manager.send_samples()              ← 局末触发
        ├─ _calc_reward()     ← GAE 反向计算 advantage
        ├─ _format_data()     ← 16帧打包 + 序列化
        └─ _send_game_data()  ← ZMQ PUSH → MemPool



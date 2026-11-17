import numpy as np
from rl_framework.common.logging import logger as LOG

from agent import Agent as AgentBase

# 游戏原生 target 头维度（本地 8 动作头），平台第 9 维为占位、恒不可选
_GAME_NATIVE_TARGET_DIM = 8

# 教师（本地大模型）动作头与 legal_action 维度，用于蒸馏模式下教师推理
_TEACHER_LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
_TEACHER_LEGAL_ACTION_SIZE_LIST = [12, 16, 16, 16, 16, 8 * 12]

HERO_ID_INDEX_DICT = {
    112: 0,
    121: 1,
    123: 2,
    131: 3,
    132: 4,
    133: 5,
    140: 6,
    141: 7,
    146: 8,
    150: 9,
    154: 10,
    157: 11,
    163: 12,
    169: 13,
    175: 14,
    182: 15,
    193: 16,
    199: 17,
    502: 18,
    513: 19,
}


class Agent(AgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._platform_feature_process = None
        self._platform_feature_process_camp = None
        self._teacher_predictor = None
        self._teacher_lstm_cell = None
        self._teacher_lstm_hidden = None
        if self._use_distill():
            self._init_teacher_model()

    def _use_distill(self):
        # 蒸馏模式：教师策略采样，学生拟合教师软标签。仅平台模型支持蒸馏。
        return getattr(self.config, "DISTILL", False)

    def _init_teacher_model(self):
        # 教师为本地 725 维/8 动作头模型，需按单帧推理（lstm_time_steps=1）加载。
        from rl_framework.predictor.predictor.local_torch_predictor import (
            LocalTorchPredictor,
        )
        from common.algorithm_torch import Algorithm as TeacherAlgorithm

        teacher_model = TeacherAlgorithm()
        # 单帧推理：教师 forward 需按 1 帧 reshape LSTM 输入
        teacher_model.lstm_time_steps = 1
        self._teacher_predictor = LocalTorchPredictor(teacher_model)
        self._teacher_predictor.load_model(self.config.TEACHER_CKPT_PATH)
        self._teacher_lstm_cell = np.zeros([self.lstm_unit_size])
        self._teacher_lstm_hidden = np.zeros([self.lstm_unit_size])
        LOG.info(
            "distill teacher model loaded from %s" % self.config.TEACHER_CKPT_PATH
        )

    def _use_platform_feature(self):
        # 平台配置: 350 维输入、9 动作头；本地配置: 725 维输入、8 动作头
        return self.config.SERI_VEC_SPLIT_SHAPE[0][0] == 350

    def _get_platform_feature_process(self, camp):
        if (
            self._platform_feature_process is None
            or self._platform_feature_process_camp != camp
        ):
            from feature_platform import FeatureProcess

            self._platform_feature_process = FeatureProcess(camp)
            self._platform_feature_process_camp = camp
        return self._platform_feature_process

    def recompute_platform_observation(self, state_dict):
        # 平台分支：从 req_pb 重算 350 维 observation，与本地 725 维逻辑隔离
        req_pb = state_dict["req_pb"]
        feature = self._get_platform_feature_process(self.hero_camp).process_req_pb(
            req_pb
        )
        state_dict["observation"] = np.array(feature, dtype=np.float32)
        return state_dict

    def recompute_platform_legal_action(self, state_dict):
        # 平台分支：本地 172 维 legal_action -> 平台 184 维
        # 前 5 个动作头维度一致；target 头由 [12, 8] 扩为 [12, 9]，第 9 维恒不可选
        legal_action = np.asarray(state_dict["legal_action"], dtype=np.float32)
        top_size = self.label_size_list[0]
        platform_target_dim = self.label_size_list[-1]
        prefix_dim = int(np.sum(self.label_size_list[:-1]))
        target_len_local = top_size * _GAME_NATIVE_TARGET_DIM

        prefix = legal_action[:prefix_dim]
        target = legal_action[prefix_dim : prefix_dim + target_len_local].reshape(
            top_size, _GAME_NATIVE_TARGET_DIM
        )
        pad = np.zeros(
            [top_size, platform_target_dim - _GAME_NATIVE_TARGET_DIM],
            dtype=np.float32,
        )
        target = np.concatenate([target, pad], axis=1).reshape(-1)
        state_dict["legal_action"] = np.concatenate([prefix, target], axis=0)
        return state_dict

    def append_hero_identity(self, state_dict):
        # hero identity feature (ont-hot)
        runtime_id = state_dict["player_id"]
        hero_id = None
        for hero in state_dict["req_pb"].hero_list:
            if hero.runtime_id == runtime_id:
                hero_id = hero.config_id

        if hero_id is None:
            raise Exception("can not find config_id for runtime_id")

        hero_id_vec = np.zeros(
            [
                len(HERO_ID_INDEX_DICT),
            ],
            dtype=np.float32,
        )
        if HERO_ID_INDEX_DICT.get(hero_id) is not None:
            hero_id_vec[HERO_ID_INDEX_DICT[hero_id]] = 1
        else:
            LOG.debug("Unknown hero_id for network: %s" % hero_id)
        state_dict["observation"] = np.concatenate(
            (state_dict["observation"], hero_id_vec), axis=0
        )
        return state_dict

    def feature_post_process(self, state_dict):
        if self._use_platform_feature():
            # 浅拷贝：state_dict 实际是 env.cur_state[i] 的同一对象，
            # 直接写回 184 维 legal_action 会导致 env._check_action 按本地
            # 8 维 target 头 reshape 报错，因此只在副本上做扩维，不改动 env 状态
            state_dict = dict(state_dict)
            if self._use_distill():
                # 教师仍消费 env 原始 observation / legal_action，
                # 需在覆盖成平台 350/184 维之前保存引用。
                # 注意：本地教师模型输入是 725 维（raw 705 + 20 维 hero_id one-hot），
                # 而此处 state_dict["observation"] 是 raw 705 维，必须补上
                # append_hero_identity 才能与教师网络第一层 (512, 235)/(512, 425) 等
                # 权重形状对齐。不能直接在原 state_dict 上调用 append_hero_identity，
                # 否则会污染后续 recompute_platform_observation 所需的原始 observation。
                teacher_state = {
                    "observation": state_dict["observation"],
                    "player_id": state_dict["player_id"],
                    "req_pb": state_dict["req_pb"],
                }
                teacher_state = self.append_hero_identity(teacher_state)
                state_dict["teacher_observation"] = teacher_state["observation"]
                state_dict["teacher_legal_action"] = state_dict["legal_action"]
            state_dict = self.recompute_platform_observation(state_dict)
            state_dict = self.recompute_platform_legal_action(state_dict)
        else:
            state_dict = self.append_hero_identity(state_dict)
        return state_dict

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        if use_max:
            return np.argmax(probs)
        # 数值修正：随机初始化模型在多数动作头会输出近似均匀的分布，
        # float32 概率求和可能略大于 1（如 1.00000005），导致
        # np.random.multinomial 抛 "sum(pvals[:-1]) > 1.0"。
        # 这里转 float64 重新归一化并兜底，保证总和 <= 1。
        p = np.asarray(probs, dtype=np.float64)
        total = p.sum()
        if total > 0:
            p = p / total
        else:
            p = np.ones_like(p) / p.size
        if p.sum() > 1.0:
            idx = int(np.argmax(p))
            p[idx] -= p.sum() - 1.0
        return np.argmax(np.random.multinomial(1, p, size=1))

    # ---------- 蒸馏相关 ----------

    def reset(self, agent_type=None, model_path=None):
        super().reset(agent_type=agent_type, model_path=model_path)
        if self._use_distill() and self._teacher_predictor is not None:
            self._teacher_lstm_cell = np.zeros([self.lstm_unit_size])
            self._teacher_lstm_hidden = np.zeros([self.lstm_unit_size])

    def process(self, state_dict, battle=False):
        state_dict = self.feature_post_process(state_dict)
        if self._use_distill():
            return self._process_distill(state_dict, battle)

        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )
        pred_ret = self._predict_process_torch(feature_vec, legal_action)
        _, _, action, d_action = pred_ret
        if battle:
            return d_action
        return action, d_action, self._sample_process(state_dict, pred_ret)

    def _process_distill(self, state_dict, battle=False):
        # 1) 教师推理：动作驱动环境，prob 作为软标签写入样本
        teacher_prob, teacher_value, teacher_action, teacher_d_action, teacher_cell, teacher_hidden = self._predict_torch_with(
            self._teacher_predictor,
            state_dict["teacher_observation"],
            state_dict["teacher_legal_action"],
            self._teacher_lstm_cell,
            self._teacher_lstm_hidden,
            _TEACHER_LABEL_SIZE_LIST,
            _TEACHER_LEGAL_ACTION_SIZE_LIST,
        )
        self._teacher_lstm_cell = teacher_cell
        self._teacher_lstm_hidden = teacher_hidden

        # 2) 学生推理：仅追踪学生自身 LSTM 状态，动作被教师动作替代
        _, _, _, _, student_cell, student_hidden = self._predict_torch_with(
            self._predictor,
            state_dict["observation"],
            state_dict["legal_action"],
            self.lstm_cell,
            self.lstm_hidden,
            self.label_size_list,
            self.legal_action_size,
        )
        self.lstm_cell = student_cell
        self.lstm_hidden = student_hidden

        if battle:
            return teacher_d_action

        # 3) 教师 target 头 8 维 -> 平台样本 9 维（第 9 维恒不可选，概率 0）
        prob = self._pad_teacher_target_prob(teacher_prob)
        pred_ret = (prob, teacher_value, teacher_action, teacher_d_action)
        return teacher_action, teacher_d_action, self._sample_process(
            state_dict, pred_ret
        )

    def _predict_torch_with(
        self,
        predictor,
        feature,
        legal_action,
        lstm_cell,
        lstm_hidden,
        label_size_list,
        legal_action_size,
    ):
        input_list = [
            np.array(feature),
            np.array(legal_action),
            lstm_cell,
            lstm_hidden,
        ]
        output_list = predictor.inference(input_list)
        np_output = [output.numpy() for output in output_list]
        logits, value, new_cell, new_hidden = np_output[:4]
        prob, action, d_action = self._sample_masked_action_for(
            logits, legal_action, label_size_list, legal_action_size
        )
        return prob, value, action, d_action, new_cell, new_hidden

    def _sample_masked_action_for(
        self, logits, legal_action, label_size_list, legal_action_size
    ):
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [
            sum(label_size_list[: index + 1])
            for index in range(len(label_size_list))
        ]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits[0], label_split_size[:-1])
        for index in range(len(label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            action_list.append(self._legal_sample(probs, use_max=False))
            d_action_list.append(self._legal_sample(probs, use_max=True))

        index = len(label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [legal_action_size[0], legal_action_size[-1] // legal_action_size[0]],
        )
        one_hot_actions = np.eye(label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        action_list.append(self._legal_sample(probs, use_max=False))

        one_hot_actions = np.eye(label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_action_list.append(self._legal_sample(probs, use_max=True))

        return [prob_list], action_list, d_action_list

    def _pad_teacher_target_prob(self, prob):
        # 教师 target 头仅 8 维，平台样本 target 头为 9 维，第 9 维补 0（恒不可选）
        prob_list = list(prob[0])
        split_sizes = _TEACHER_LABEL_SIZE_LIST[:-1]
        heads = []
        offset = 0
        for size in split_sizes:
            heads.append(prob_list[offset : offset + size])
            offset += size
        target = prob_list[offset : offset + _TEACHER_LABEL_SIZE_LIST[-1]] + [0.0]
        heads.append(target)
        return [[p for head in heads for p in head]]
import numpy as np
from rl_framework.common.logging import logger as LOG

from agent import Agent as AgentBase

# 游戏原生 target 头维度（本地 8 动作头），平台第 9 维为占位、恒不可选
_GAME_NATIVE_TARGET_DIM = 8

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
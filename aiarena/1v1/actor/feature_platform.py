#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
平台（腾讯开悟 1v1）350 维特征处理，独立于本地 725 维逻辑。

结构完全对齐 code-hok1v1-IDE-63.0.9/agent_ppo/feature/feature_process：
    FeatureProcess = HeroProcess(67*2) + SoldierProcess(144, 全零占位) + OrganProcess(18*2*2)
    拼接顺序: main_hero(67) + enemy_hero(67) + soldier(144) + organ(72) = 350

与平台代码的唯一差别：平台环境的 observation["frame_state"] 已是字典，
本地 env1v1 拿到的是 interface.so 的 pybind protobuf 对象（req_pb），
因此这里额外提供 req_pb_to_frame_state() 做结构转换：
    - Hero 字段 camelCase -> snake_case (killCnt->kill_cnt, moneyCnt->money_cnt ...)
    - Hero.atk_range -> attack_range
    - Organ.type(ActorType 枚举 int) -> sub_type (21=塔, 24=水晶)
    - Organ.atk -> phy_atk
    - camp int(1/2) -> "PLAYERCAMP_1"/"PLAYERCAMP_2" 字符串
    - Hero.skill/common_skill -> skill_state.slot_states, slot_type int -> "SLOT_SKILL_x" 字符串
      (注意 SkillSlotType 枚举值不连续: SLOT_SKILL_4=5, SLOT_SKILL_5=6, 必须按枚举名映射)

归一化配置直接内嵌平台 hero_feature_config.ini / organ_feature_config.ini 原文，
保证特征顺序、归一化参数与平台逐位一致。
"""

import configparser
import math
from collections import OrderedDict

# =====================================================================
# 归一化配置（与平台 agent_ppo/feature/feature_process/*.ini 完全一致）
# =====================================================================

HERO_FEATURE_CONFIG_INI = """
[feature_config]
is_hero_alive = one_hot:1:eq
hero_level = one_hot:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:eq
hp = min_max:0:10000
hp_rate = min_max:0:1
max_hp = min_max:2860:12000
hp_recover = min_max:40:220
ep = min_max:0:3000
ep_rate = min_max:0:1
max_ep = min_max:0:3000
ep_recover = min_max:-10:70
phy_atk = min_max:100:900
mgc_atk = min_max:0:1000
phy_def = min_max:0:1200
mgc_def = min_max:0:700
kill_cnt = min_max:0:10
dead_cnt = min_max:0:8
money_cnt = min_max:0:12000
location_x = min_max:-60000:60000
location_z = min_max:-60000:60000
dist_from_all_heros = min_max:0:116000
mov_spd = min_max:1250:7000
attack_range = min_max:2800:8000
atk_spd = min_max:0:10000
phy_armor_hurt = min_max:0:240
mgc_armor_hurt = min_max:0:170
crit_rate = min_max:0:5000
crit_effe = min_max:9500:17000
phy_vamp = min_max:0:4000
mgc_vamp = min_max:0:3400
cd_reduce = min_max:0:3600
ctrl_reduce = min_max:0:3500
exp = min_max:0:2202
money = min_max:0:2000
revive_time = min_max:0:46
kill_income = min_max:20:400
summon_skill_cd = min_max:0:120000
summon_skill_type = one_hot:1:2:3:4:5:6:7:8:9:10:eq
common_attack_is_useable = one_hot:1:eq
skill_1_useable = one_hot:1:eq
hero_skill_1_cd = min_max:0:8000
skill_2_useable = one_hot:1:eq
hero_skill_2_cd = min_max:0:10000
skill_3_useable = one_hot:1:eq
hero_skill_3_cd = min_max:0:46000

[feature_functions]
is_hero_alive = is_alive
hero_level = get_hero_level
hp = get_from_actor_state
hp_rate = get_hp_rate
max_hp = get_from_actor_state
hp_recover = get_from_actor_state_values
ep = get_from_actor_state_values
ep_rate = get_ep_rate
max_ep = get_from_actor_state_values
ep_recover = get_from_actor_state_values
phy_atk = get_from_actor_state_values
mgc_atk = get_from_actor_state_values
phy_def = get_from_actor_state_values
mgc_def = get_from_actor_state_values
kill_cnt = get_from_hero
dead_cnt = get_from_hero
money_cnt = get_from_hero
location_x = get_location_x
location_z = get_location_z
dist_from_all_heros = get_dist_from_enemy_heros
mov_spd = get_from_actor_state_values
attack_range = get_from_actor_state
atk_spd = get_from_actor_state_values
phy_armor_hurt = get_from_actor_state_values
mgc_armor_hurt = get_from_actor_state_values
crit_rate = get_from_actor_state_values
crit_effe = get_from_actor_state_values
phy_vamp = get_from_actor_state_values
mgc_vamp = get_from_actor_state_values
cd_reduce = get_from_actor_state_values
ctrl_reduce = get_from_actor_state_values
exp = get_from_hero
money = get_from_hero
revive_time = get_from_hero
kill_income = get_from_actor_state
summon_skill_cd = get_summon_skill_cd
summon_skill_type = get_summon_skill_type
common_attack_is_useable = get_common_attack_is_useable
skill_1_useable = get_skill_1_useable
hero_skill_1_cd = get_hero_skill_1_cd
skill_2_useable = get_skill_2_useable
hero_skill_2_cd = get_hero_skill_2_cd
skill_3_useable = get_skill_3_useable
hero_skill_3_cd = get_hero_skill_3_cd
"""

ORGAN_FEATURE_CONFIG_INI = """
[feature_config]
is_alive = one_hot:1:eq
belong_to_main_camp = one_hot:1:eq
type = one_hot:21:22:23:24:eq
location_x = min_max:-41000:41000
location_z = min_max:-41000:41000
relative_location_x = min_max:0:1
relative_location_z = min_max:0:1
distance_from_each_hero = min_max:0:116000
hp = min_max:0:12000
hp_rate = min_max:0:1
max_hp = min_max:0:12000
atk = min_max:0:630
kill_income = min_max:0:150
attack_range = min_max:0:13000

[feature_functions]
is_alive = is_alive
belong_to_main_camp = belong_to_main_camp
type = get_organ_type
location_x = get_normal_organ_location_x
location_z = get_normal_organ_location_z
relative_location_x = relative_location_x
relative_location_z = relative_location_z
distance_from_each_hero = generate_distance_with_each_hero_feature
hp = get_hp
hp_rate = get_hp_rate
max_hp = get_max_hp
atk = get_atk
kill_income = get_kill_income
attack_range = get_atk_range
"""

# =====================================================================
# req_pb -> 平台 frame_state 字典 的映射常量
# =====================================================================

# Camp 枚举: PLAYERCAMP_1=1, PLAYERCAMP_2=2
CAMP_INT_TO_STR = {1: "PLAYERCAMP_1", 2: "PLAYERCAMP_2"}

# SkillSlotType 枚举值 -> 平台字符串（枚举值不连续: EX3=4, SLOT_SKILL_4=5, SLOT_SKILL_5=6）
SKILL_SLOT_INT_TO_NAME = {
    0: "SLOT_SKILL_0",
    1: "SLOT_SKILL_1",
    2: "SLOT_SKILL_2",
    3: "SLOT_SKILL_3",
    4: "SLOT_SKILL_EX3",
    5: "SLOT_SKILL_4",
    6: "SLOT_SKILL_5",
}

# 召唤师技能 configId 集合（与平台 get_summon_skill_type 的映射表同源），
# 用于从 Hero.skill 列表中识别哪个槽是召唤师技能 -> SLOT_SKILL_5
SUMMON_SKILL_CONFIG_IDS = {
    80102, 80109, 80104, 80116, 80108, 80110, 80105, 80103, 80107, 80121, 80115,
}

# ActorType 枚举（Organ.type）: ACTOR_TOWER=21, ACTOR_TOWER_HIGH=22,
# ACTOR_TOWER_SPRING=23, ACTOR_CRYSTAL=24；平台侧只消费 21/24
ORGAN_SUB_TYPE_TOWER = 21
ORGAN_SUB_TYPE_CRYSTAL = 24


def _vint3_to_dict(pos):
    return {"x": pos.x, "y": pos.y, "z": pos.z}


def _skill_to_slot_dict(skill, slot_name):
    return {
        "slot_type": slot_name,
        "configId": skill.config_id,
        "level": skill.level,
        "usable": bool(skill.usable),
        "cooldown": skill.cooldown,
        "cooldown_max": skill.cooldown_max,
    }


def _build_slot_states(hero):
    """Hero.common_skill + Hero.skill(List[Skill]) -> 平台 slot_states。

    普通攻击(common_skill)固定为 SLOT_SKILL_0；skill 列表按 configId 识别
    召唤师技能(->SLOT_SKILL_5)，其余按出现顺序依次填 SLOT_SKILL_1~4。
    """
    slot_states = [_skill_to_slot_dict(hero.common_skill, "SLOT_SKILL_0")]
    hero_slot_names = ["SLOT_SKILL_1", "SLOT_SKILL_2", "SLOT_SKILL_3", "SLOT_SKILL_4"]
    hero_slot_idx = 0
    for skill in hero.skill:
        if skill.config_id in SUMMON_SKILL_CONFIG_IDS:
            slot_states.append(_skill_to_slot_dict(skill, "SLOT_SKILL_5"))
        elif hero_slot_idx < len(hero_slot_names):
            slot_states.append(_skill_to_slot_dict(skill, hero_slot_names[hero_slot_idx]))
            hero_slot_idx += 1
    return slot_states


def _hero_pb_to_dict(hero):
    return {
        "player_id": hero.runtime_id,
        "config_id": hero.config_id,
        "runtime_id": hero.runtime_id,
        "camp": CAMP_INT_TO_STR[int(hero.camp)],
        "behav_mode": int(hero.behav_mode),
        "location": _vint3_to_dict(hero.location),
        "hp": hero.hp,
        "max_hp": hero.max_hp,
        "attack_range": hero.atk_range,
        "kill_income": hero.kill_income,
        "phy_atk": hero.phy_atk,
        "phy_def": hero.phy_def,
        "mgc_atk": hero.mgc_atk,
        "mgc_def": hero.mgc_def,
        "mov_spd": hero.mov_spd,
        "atk_spd": hero.atk_spd,
        "ep": hero.ep,
        "max_ep": hero.max_ep,
        "hp_recover": hero.hp_recover,
        "ep_recover": hero.ep_recover,
        "phy_armor_hurt": hero.phy_armor_hurt,
        "mgc_armor_hurt": hero.mgc_armor_hurt,
        "crit_rate": hero.crit_rate,
        "crit_effe": hero.crit_effe,
        "phy_vamp": hero.phy_vamp,
        "mgc_vamp": hero.mgc_vamp,
        "cd_reduce": hero.cd_reduce,
        "ctrl_reduce": hero.ctrl_reduce,
        "skill_state": {"slot_states": _build_slot_states(hero)},
        "level": hero.level,
        "exp": hero.exp,
        "money": hero.money,
        "revive_time": hero.revive_time,
        "kill_cnt": hero.killCnt,
        "dead_cnt": hero.deadCnt,
        "money_cnt": hero.moneyCnt,
    }


def _organ_pb_to_dict(organ):
    return {
        "config_id": organ.config_id,
        "runtime_id": organ.runtime_id,
        "sub_type": int(organ.type),
        "camp": CAMP_INT_TO_STR[int(organ.camp)],
        "location": _vint3_to_dict(organ.location),
        "hp": organ.hp,
        "max_hp": organ.max_hp,
        "phy_atk": organ.atk,
        "attack_range": organ.attack_range,
        "kill_income": organ.kill_income,
    }


def req_pb_to_frame_state(req_pb):
    """本地 interface.GetAIFrameState 返回的 pybind protobuf -> 平台 frame_state 字典。"""
    return {
        "frame_no": req_pb.frame_no,
        "hero_states": [_hero_pb_to_dict(hero) for hero in req_pb.hero_list],
        "npc_states": [_organ_pb_to_dict(organ) for organ in req_pb.organ_list],
    }


# =====================================================================
# 以下四个类逐行对齐平台 feature_process 实现（TODO 已全部补全）
# =====================================================================


class FeatureNormalizer:
    def one_hot(self, value, value_list, compare_method):
        if compare_method != "eq":
            raise ValueError("Unsupported compare method: " + compare_method)
        return [1 if value == v else 0 for v in value_list]

    def min_max(self, value, min_value, max_value):
        if value <= min_value:
            return 0.0
        if value >= max_value:
            return 1.0
        return float(value - min_value) / float(max_value - min_value)

    def parse_config(self, config_list):
        config_dict = {}
        for config_str in config_list:
            parts = config_str.split(":")
            feature_name = parts[0]
            method = parts[1]
            if method == "one_hot":
                values = list(map(int, parts[2:-1]))
                compare_method = parts[-1]
                config_dict[feature_name] = (self.one_hot, values, compare_method)
            elif method == "min_max":
                min_value = int(parts[2])
                max_value = int(parts[3])
                config_dict[feature_name] = (self.min_max, min_value, max_value)
            else:
                raise ValueError("Unsupported method: " + method)
        return config_dict


class _IniConfigLoader:
    """从内嵌 ini 字符串加载归一化配置与特征函数映射（替代平台读文件）。"""

    def load(self, ini_text):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read_string(ini_text)

        feature_config = []
        for feature, cfg in config["feature_config"].items():
            feature_config.append("{}:{}".format(feature, cfg))

        feature_func_map = {}
        for feature, func_name in config["feature_functions"].items():
            if hasattr(self, func_name):
                feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError("Unsupported function: {}".format(func_name))
        return feature_config, feature_func_map


class HeroProcess(_IniConfigLoader):
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        hero_feature_config, self.feature_func_map = self.load(HERO_FEATURE_CONFIG_INI)
        self.map_feature_to_norm = self.normalizer.parse_config(hero_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 67
        self.unit_buff_num = 1

    def process_vec_hero(self, frame_state):
        self.generate_hero_info_list(frame_state)
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(
            self.main_camp_hero_dict, "main_camp"
        )
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(
            self.enemy_camp_hero_dict, "enemy_camp"
        )
        return main_camp_hero_vector_feature, enemy_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero
        self.main_camp_hero_dict = OrderedDict(sorted(self.main_camp_hero_dict.items()))
        self.enemy_camp_hero_dict = OrderedDict(sorted(self.enemy_camp_hero_dict.items()))

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    # ---------------- 特征提取函数 ----------------

    def get_from_hero(self, hero, vector_feature, feature_name):
        value = hero[feature_name]
        vector_feature.append(value)

    def get_from_actor_state(self, hero, vector_feature, feature_name):
        value = hero[feature_name]
        vector_feature.append(value)

    def get_from_actor_state_values(self, hero, vector_feature, feature_name):
        value = hero[feature_name]
        vector_feature.append(value)

    def get_hp_rate(self, hero, vector_feature, feature_name):
        value = 0
        if hero["max_hp"] > 0:
            value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def get_ep_rate(self, hero, vector_feature, feature_name):
        value = 0
        if hero["max_ep"] > 0:
            value = hero["ep"] / hero["max_ep"]
        vector_feature.append(value)

    def get_summon_skill_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_5":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_common_attack_is_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_0":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_1_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_1":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_2_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_2":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_3_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_3":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_4_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_4":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_hero_skill_1_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_1":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_2_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_2":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_3_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_3":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_4_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_4":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_summon_skill_type(self, hero, vector_feature, feature_name):
        summon_skill_type_map = {
            "80102": 1,
            "80109": 2,
            "80104": 3,
            "80116": 3,
            "80108": 4,
            "80110": 5,
            "80105": 6,
            "80103": 7,
            "80107": 8,
            "80121": 9,
            "80115": 10,
        }
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_5":
                value = summon_skill_type_map.get(str(skill_state["configId"]), 0)
        vector_feature.append(value)

    def get_dist_from_enemy_heros(self, main_hero, vector_feature, feature_name):
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            dist = self.cal_dist(main_hero["location"], hero["location"])
            dist = dist * 100
            vector_feature.append(dist)

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt(
            (pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2
            + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2
        )
        return dist

    def get_hero_level(self, hero, vector_feature, feature_name):
        value = hero["level"]
        vector_feature.append(value)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 1.0 if hero["hp"] > 0 else 0.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)


class OrganProcess(_IniConfigLoader):
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        organ_feature_config, self.feature_func_map = self.load(ORGAN_FEATURE_CONFIG_INI)
        self.map_feature_to_norm = self.normalizer.parse_config(organ_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 18
        self.unit_buff_num = 2

    def process_vec_organ(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        local_vector_feature = []
        main_camp_organ_vector_feature = self.generate_one_type_organ_feature(
            self.main_camp_organ_dict, "main_camp"
        )
        local_vector_feature.extend(main_camp_organ_vector_feature)
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(
            self.enemy_camp_organ_dict, "enemy_camp"
        )
        local_vector_feature.extend(enemy_camp_organ_vector_feature)
        return local_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()
        for organ in frame_state["npc_states"]:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_subtype == ORGAN_SUB_TYPE_TOWER:
                organ_key = "tower"
            elif organ_subtype == ORGAN_SUB_TYPE_CRYSTAL:
                organ_key = "crystal"
            else:
                continue
            if organ_camp == self.main_camp:
                self.main_camp_organ_dict[organ_key] = organ
            else:
                self.enemy_camp_organ_dict[organ_key] = organ

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](organ, value)
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_organs_considered += 1

        if "crystal" in one_type_organ_info:
            organ = one_type_organ_info["crystal"]
            process_organ(organ)

        if "tower" in one_type_organ_info:
            organ = one_type_organ_info["tower"]
            process_organ(organ)

        if num_organs_considered < self.unit_buff_num:
            self.no_organ_feature(vector_feature, num_organs_considered)
        return vector_feature

    def no_organ_feature(self, vector_feature, num_organs_considered):
        for _ in range((self.unit_buff_num - num_organs_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    # ---------------- 特征提取函数 ----------------

    def get_hp(self, organ, vector_feature):
        value = organ["hp"]
        vector_feature.append(value)

    def get_max_hp(self, organ, vector_feature):
        value = organ["max_hp"]
        vector_feature.append(value)

    def get_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def get_atk(self, organ, vector_feature):
        value = organ["phy_atk"]
        vector_feature.append(value)

    def get_atk_range(self, organ, vector_feature):
        value = organ["attack_range"]
        vector_feature.append(value)

    def get_kill_income(self, organ, vector_feature):
        value = organ["kill_income"]
        vector_feature.append(value)

    def judge_in_main_hero_sight(self, organ, vector_feature):
        value = 0.0
        if self.judge_in_view(self.main_hero_info["location"], organ["location"]):
            value = 1.0
        vector_feature.append(value)

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt(
            (pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2
            + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2
        )
        return dist

    def is_alive(self, organ, vector_feature):
        value = 1.0 if organ["hp"] > 0 else 0.0
        vector_feature.append(value)

    def belong_to_main_camp(self, organ, vector_feature):
        value = 1.0 if organ["camp"] == self.main_hero_info["camp"] else 0.0
        vector_feature.append(value)

    def get_organ_type(self, organ, vector_feature):
        organ_sub_type = organ["sub_type"]
        value = 0
        if organ_sub_type == ORGAN_SUB_TYPE_TOWER:
            value = ORGAN_SUB_TYPE_TOWER
        elif organ_sub_type == ORGAN_SUB_TYPE_CRYSTAL:
            value = ORGAN_SUB_TYPE_CRYSTAL
        vector_feature.append(value)

    def generate_distance_with_each_hero_feature(self, organ, vector_feature):
        for _, hero in self.main_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], organ["location"])
            dist = dist * 100
            vector_feature.append(dist)
        for _, hero in self.enemy_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], organ["location"])
            dist = dist * 100
            vector_feature.append(dist)

    def get_normal_organ_location_x(self, organ, vector_feature):
        value = organ["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, organ, vector_feature):
        value = organ["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def relative_location_x(self, organ, vector_feature):
        organ_location_x = organ["location"]["x"]
        location_x = self.main_hero_info["location"]["x"]
        x_diff = organ_location_x - location_x
        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def relative_location_z(self, organ, vector_feature):
        organ_location_z = organ["location"]["z"]
        location_z = self.main_hero_info["location"]["z"]
        z_diff = organ_location_z - location_z
        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)


class SoldierProcess:
    """任务一简配：小兵特征全零占位，长度 18*4*2 = 144（与平台一致）。"""

    one_unit_feature_num = 18
    unit_buff_num = 4

    def __init__(self, camp):
        self.main_camp = camp

    def process_vec_soldier(self, frame_state):
        total_len = self.one_unit_feature_num * self.unit_buff_num * 2
        return [0] * total_len


class FeatureProcess:
    """聚合输出严格 350 维: hero(67+67) + soldier(144) + organ(36+36)。"""

    FEATURE_DIM = 350

    def __init__(self, camp):
        # 兼容本地 pybind 枚举阵营、int 阵营(1/2)与平台字符串阵营
        # 注意: env.player_camp 里存的是 interface.so 的枚举对象(非 Python int)，
        #       必须用 int(camp) 转成数值再映射，否则 == "PLAYERCAMP_x" 永远为 False
        if not isinstance(camp, str):
            camp = CAMP_INT_TO_STR[int(camp)]
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)

    def reset(self, camp):
        self.__init__(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_soldier_feature(self, frame_state):
        return self.soldier_process.process_vec_soldier(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        soldier_feature = self.process_soldier_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        main_camp_hero_vector_feature, enemy_camp_hero_vector_feature = (
            self.process_hero_feature(frame_state)
        )

        feature = (
            main_camp_hero_vector_feature
            + enemy_camp_hero_vector_feature
            + soldier_feature
            + organ_feature
        )
        assert len(feature) == self.FEATURE_DIM, "feature dim {} != {}".format(
            len(feature), self.FEATURE_DIM
        )
        return feature

    def process_req_pb(self, req_pb):
        """本地入口：req_pb(pybind protobuf) -> 350 维特征 list。"""
        frame_state = req_pb_to_frame_state(req_pb)
        return self.process_feature({"frame_state": frame_state})
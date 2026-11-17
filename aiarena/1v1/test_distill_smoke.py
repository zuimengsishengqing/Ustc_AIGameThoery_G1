# -*- coding: utf-8 -*-
"""
离线蒸馏冒烟测试（不依赖 gamecore / mem_pool / 真实对局）。

覆盖三件事：
1. 教师模型（common.algorithm_torch.Algorithm，本地 725 维/8 动作头）
   用 LocalTorchPredictor 加载真实 ckpt 后单帧推理，检查输出维度。
2. 学生模型（common.algorithm_platform.Algorithm，平台 350 维/9 动作头）
   随机初始化后单帧推理，检查输出维度。
3. 用教师对 16 帧合成轨迹逐帧推理得到软标签，按 config_platform.data_shapes
   的精确布局拼成一个 batch，喂给学生的 format_data/forward/compute_loss
   （distill 分支），验证 KL loss 可计算、可反向传播，且若干步优化后 loss 下降。

用法（WSL）:
    wsl -d Ubuntu2204
    source ~/hokenv/bin/activate
    cd /mnt/e/LearnAI/AIvs_class/hok_env/hok_env/aiarena/1v1
    python test_distill_smoke.py [--teacher_ckpt /aiarena/checkpoints_local_teacher]
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common.config import Config as LocalConfig
from common.config_platform import Config as PlatformConfig
from rl_framework.predictor.predictor.local_torch_predictor import LocalTorchPredictor


TEACHER_LABEL_SIZE_LIST = LocalConfig.LABEL_SIZE_LIST  # [12,16,16,16,16,8]
TEACHER_LEGAL_ACTION_SIZE_LIST = LocalConfig.LEGAL_ACTION_SIZE_LIST  # [12,16,16,16,16,96]
STUDENT_LABEL_SIZE_LIST = PlatformConfig.LABEL_SIZE_LIST  # [12,16,16,16,16,9]
STUDENT_LEGAL_ACTION_SIZE_LIST = PlatformConfig.LEGAL_ACTION_SIZE_LIST  # [12,16,16,16,16,108]

TEACHER_FEATURE_DIM = LocalConfig.SERI_VEC_SPLIT_SHAPE[0][0]  # 725
TEACHER_LEGAL_ACTION_DIM = int(np.sum(TEACHER_LEGAL_ACTION_SIZE_LIST))  # 172
STUDENT_FEATURE_DIM = PlatformConfig.SERI_VEC_SPLIT_SHAPE[0][0]  # 350
STUDENT_LEGAL_ACTION_DIM_FULL = int(np.sum(STUDENT_LEGAL_ACTION_SIZE_LIST))  # 184
# 样本内存的是 agent._update_legal_action 压缩后的 85 维（target 头只保留选中 button 那一行）
STUDENT_LEGAL_ACTION_DIM_SAMPLE = PlatformConfig.SERI_VEC_SPLIT_SHAPE[1][0]  # 85
LSTM_UNIT_SIZE = PlatformConfig.LSTM_UNIT_SIZE  # 512
LSTM_TIME_STEPS = PlatformConfig.LSTM_TIME_STEPS  # 16


def _random_legal_mask(size, rng, min_legal=2):
    mask = (rng.random(size) > 0.5).astype(np.float32)
    if mask.sum() < min_legal:
        mask[:min_legal] = 1.0
    return mask


def _build_shared_legal_masks(rng):
    """本地/平台前 5 个动作头（button/move/skill/...）语义完全一致，
    legal_action mask 应当共享；只有 target 头维度不同（本地 8 类，平台 9 类，
    平台第 9 类是恒不可选的占位）。这里生成一份共享的 mask，供教师/学生
    各自的 legal_action 拼装使用，避免测试里出现"教师认为合法、学生认为非法"
    这种现实中不会发生的错配（会人为放大 KL loss）。
    """
    head_sizes = TEACHER_LABEL_SIZE_LIST[:-1]  # [12,16,16,16,16]，教师/学生一致
    head_masks = [_random_legal_mask(s, rng) for s in head_sizes]
    # target 头：本地 [12, 8]，平台 [12, 9]（第 9 列恒 0，占位不可选）
    target_mask_teacher = np.stack(
        [_random_legal_mask(TEACHER_LABEL_SIZE_LIST[-1], rng) for _ in range(TEACHER_LEGAL_ACTION_SIZE_LIST[0])]
    )  # [12, 8]
    target_mask_student = np.concatenate(
        [target_mask_teacher, np.zeros([TEACHER_LEGAL_ACTION_SIZE_LIST[0], 1], dtype=np.float32)],
        axis=1,
    )  # [12, 9]
    return head_masks, target_mask_teacher, target_mask_student


def _build_teacher_input(rng, shared=None):
    feature = rng.random(TEACHER_FEATURE_DIM).astype(np.float32)
    if shared is None:
        head_masks, target_mask_teacher, _ = _build_shared_legal_masks(rng)
    else:
        head_masks, target_mask_teacher, _ = shared
    legal_action = np.concatenate(
        head_masks + [target_mask_teacher.reshape(-1)]
    ).astype(np.float32)
    assert legal_action.shape[0] == TEACHER_LEGAL_ACTION_DIM, legal_action.shape
    return feature, legal_action


def _build_student_input(rng, shared=None):
    feature = rng.random(STUDENT_FEATURE_DIM).astype(np.float32)
    if shared is None:
        head_masks, _, target_mask_student = _build_shared_legal_masks(rng)
    else:
        head_masks, _, target_mask_student = shared
    legal_action = np.concatenate(
        head_masks + [target_mask_student.reshape(-1)]
    ).astype(np.float32)
    assert legal_action.shape[0] == STUDENT_LEGAL_ACTION_DIM_FULL, legal_action.shape
    return feature, legal_action


def test_teacher_inference(teacher_ckpt_dir, rng):
    from common.algorithm_torch import Algorithm as TeacherAlgorithm

    net = TeacherAlgorithm()
    net.lstm_time_steps = 1
    predictor = LocalTorchPredictor(net)
    predictor.load_model(teacher_ckpt_dir)

    feature, legal_action = _build_teacher_input(rng)
    lstm_cell = np.zeros([LSTM_UNIT_SIZE], dtype=np.float32)
    lstm_hidden = np.zeros([LSTM_UNIT_SIZE], dtype=np.float32)

    out = predictor.inference([feature, legal_action, lstm_cell, lstm_hidden])
    logits, value, new_cell, new_hidden = [o.numpy() for o in out]

    expected_logits_dim = int(np.sum(TEACHER_LABEL_SIZE_LIST))  # 84
    assert logits.shape == (1, expected_logits_dim), logits.shape
    assert value.shape == (1, 1), value.shape
    assert new_cell.shape[-1] == LSTM_UNIT_SIZE
    assert new_hidden.shape[-1] == LSTM_UNIT_SIZE
    print(
        "[OK] teacher inference: logits%s value%s cell%s hidden%s"
        % (logits.shape, value.shape, new_cell.shape, new_hidden.shape)
    )
    return net, predictor


def test_student_inference(rng):
    from common.algorithm_platform import Algorithm as StudentAlgorithm

    net = StudentAlgorithm()
    net.lstm_time_steps = 1
    predictor = LocalTorchPredictor(net)

    feature, legal_action = _build_student_input(rng)
    lstm_cell = np.zeros([LSTM_UNIT_SIZE], dtype=np.float32)
    lstm_hidden = np.zeros([LSTM_UNIT_SIZE], dtype=np.float32)

    out = predictor.inference([feature, legal_action, lstm_cell, lstm_hidden])
    logits, value, new_cell, new_hidden = [o.numpy() for o in out]

    expected_logits_dim = int(np.sum(STUDENT_LABEL_SIZE_LIST))  # 85
    assert logits.shape == (1, expected_logits_dim), logits.shape
    assert value.shape == (1, 1), value.shape
    print(
        "[OK] student inference: logits%s value%s cell%s hidden%s"
        % (logits.shape, value.shape, new_cell.shape, new_hidden.shape)
    )
    return net


def _teacher_frame_prob(predictor, feature, legal_action, lstm_cell, lstm_hidden):
    """跑一帧教师推理，返回 (per-head prob list, value, new_cell, new_hidden)。

    prob 的计算方式与 actor/custom.py::_sample_masked_action_for 保持一致：
    前 5 个头直接对 legal mask 做 masked softmax；target 头先按 button 动作
    选出对应行再 masked softmax。这里 button 动作直接取 argmax 简化处理
    （冒烟测试只关心维度/数值合理性，不要求与真实采样动作一致）。
    """
    out = predictor.inference([feature, legal_action, lstm_cell, lstm_hidden])
    logits, value, new_cell, new_hidden = [o.numpy() for o in out]
    logits = logits[0]

    split_sizes = [sum(TEACHER_LABEL_SIZE_LIST[: i + 1]) for i in range(len(TEACHER_LABEL_SIZE_LIST))]
    legal_splits = np.split(legal_action, split_sizes[:-1])
    logits_splits = np.split(logits, split_sizes[:-1])

    def masked_softmax(lg, mask):
        boundary = 1e20
        lg = lg - np.max(lg - (1 - mask) * boundary)
        lg = np.clip(lg, -boundary, 1)
        exp = mask * np.exp(lg) + 1e-5
        return exp / exp.sum()

    prob_list = []
    action_list = []
    for i in range(len(TEACHER_LABEL_SIZE_LIST) - 1):
        p = masked_softmax(logits_splits[i], legal_splits[i])
        prob_list.append(p)
        action_list.append(int(np.argmax(p)))

    # target 头：legal_action 是 [12, 8] 展平，先按 button 动作取对应行
    target_legal = legal_splits[-1].reshape(
        TEACHER_LEGAL_ACTION_SIZE_LIST[0],
        TEACHER_LEGAL_ACTION_SIZE_LIST[-1] // TEACHER_LEGAL_ACTION_SIZE_LIST[0],
    )
    target_row = target_legal[action_list[0]]
    p_target = masked_softmax(logits_splits[-1], target_row)
    prob_list.append(p_target)

    flat_prob = np.concatenate(prob_list).astype(np.float32)
    return (
        flat_prob,
        value.reshape(-1),
        new_cell.reshape(-1),
        new_hidden.reshape(-1),
        action_list,
    )


def _build_distill_batch(teacher_predictor, batch_size, rng):
    """按 config_platform.data_shapes 的精确布局，拼出 [batch_size, total_len] 的合成样本。

    每个 batch 元素是一条 16 帧轨迹：教师逐帧推理得到软标签(prob)，
    学生特征/legal_action 用随机合成数据（冒烟测试只验证 loss 计算链路，
    不要求学生特征与教师特征语义对应）。教师/学生共享同一份 legal_action
    mask（前 5 头语义一致），避免"教师认为合法、学生认为非法"这种现实中
    不会发生的错配人为放大 KL loss。
    """
    data_shapes = PlatformConfig.data_shapes
    cut_points = [s[0] for s in data_shapes]
    total_len = int(np.sum(cut_points))
    assert total_len == 9584, total_len

    batch = np.zeros([batch_size, total_len], dtype=np.float32)

    for b in range(batch_size):
        teacher_cell = np.zeros([LSTM_UNIT_SIZE], dtype=np.float32)
        teacher_hidden = np.zeros([LSTM_UNIT_SIZE], dtype=np.float32)

        # 每条轨迹开头记录一份 lstm 初始状态（对应 data_shapes[-2:]）
        init_cell = teacher_cell.copy()
        init_hidden = teacher_hidden.copy()

        frames = []
        for _ in range(LSTM_TIME_STEPS):
            shared_masks = _build_shared_legal_masks(rng)
            t_feature, t_legal = _build_teacher_input(rng, shared_masks)
            s_feature, s_legal_full = _build_student_input(rng, shared_masks)

            prob, value, teacher_cell, teacher_hidden, teacher_action_list = (
                _teacher_frame_prob(
                    teacher_predictor, t_feature, t_legal, teacher_cell, teacher_hidden
                )
            )

            # 学生 legal_action 在样本里是压缩后的 85 维：
            # 前 5 个头原样保留，target 头 [12,9] 只取 button 动作对应行(9维)。
            # button 动作用教师实际选中的动作（与真实 actor 蒸馏管线一致：
            # agent.py::_sample_process 里 _update_legal_action 用的是 pred_ret
            # 里的 action，即教师动作），而不是学生自己的独立随机 mask argmax。
            student_split_sizes = [
                sum(STUDENT_LABEL_SIZE_LIST[: i + 1])
                for i in range(len(STUDENT_LABEL_SIZE_LIST))
            ]
            s_legal_splits = np.split(s_legal_full, student_split_sizes[:-1])
            button_action = teacher_action_list[0]
            target_legal_matrix = s_legal_splits[-1].reshape(
                STUDENT_LABEL_SIZE_LIST[0],
                STUDENT_LEGAL_ACTION_SIZE_LIST[-1] // STUDENT_LABEL_SIZE_LIST[0],
            )
            s_legal_compressed = np.concatenate(
                list(s_legal_splits[:-1]) + [target_legal_matrix[button_action]]
            ).astype(np.float32)
            assert s_legal_compressed.shape[0] == STUDENT_LEGAL_ACTION_DIM_SAMPLE

            # observation 段 = feature(350) + legal_action(85) 拼接，per-frame 435 维
            seri_vec_frame = np.concatenate([s_feature, s_legal_compressed])
            assert seri_vec_frame.shape[0] == PlatformConfig.DATA_SPLIT_SHAPE[0]

            # label：每头一个动作 id（这里直接用 prob 的 argmax 冒充，仅为跑通链路；
            # distill 分支的 compute_loss 实际不使用 label 字段）
            prob_splits = np.split(
                prob,
                [sum(TEACHER_LABEL_SIZE_LIST[: i + 1]) for i in range(len(TEACHER_LABEL_SIZE_LIST))][:-1],
            )
            # 教师 target 头是 8 维，学生样本 target 头 label/prob 需要 9 维（第 9 维占位）
            label_per_head = [int(np.argmax(p)) for p in prob_splits[:-1]]
            teacher_target_prob = prob_splits[-1]
            student_target_prob = np.concatenate([teacher_target_prob, [0.0]]).astype(np.float32)
            label_per_head.append(int(np.argmax(teacher_target_prob)))

            frames.append(
                {
                    "seri_vec": seri_vec_frame,
                    "reward": value[0],
                    "advantage": 0.0,
                    "label": label_per_head,
                    "prob": [p.astype(np.float32) for p in prob_splits[:-1]] + [student_target_prob],
                    "weight": [1.0] * len(STUDENT_LABEL_SIZE_LIST),
                    "is_train": 1.0,
                }
            )

        # 按 data_keys 顺序拼装：observation, reward, advantage,
        # label0-5, prob0-5, weight0-5, is_train, lstm_cell, lstm_hidden_state
        row = np.zeros([total_len], dtype=np.float32)
        offset = 0

        def put(arr):
            nonlocal offset
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            row[offset : offset + arr.shape[0]] = arr
            offset += arr.shape[0]

        # observation: [16帧 × 435] = 6960
        put(np.concatenate([f["seri_vec"] for f in frames]))
        # reward / advantage: 各 [16]
        put([f["reward"] for f in frames])
        put([f["advantage"] for f in frames])
        # label0-5: 各 [16]
        for head_idx in range(len(STUDENT_LABEL_SIZE_LIST)):
            put([f["label"][head_idx] for f in frames])
        # prob0-5: 各 [head_size * 16]
        for head_idx in range(len(STUDENT_LABEL_SIZE_LIST)):
            put(np.concatenate([f["prob"][head_idx] for f in frames]))
        # weight0-5: 各 [16]
        for head_idx in range(len(STUDENT_LABEL_SIZE_LIST)):
            put([f["weight"][head_idx] for f in frames])
        # is_train: [16]
        put([f["is_train"] for f in frames])
        # lstm_cell / lstm_hidden_state: 各 [512]
        put(init_cell)
        put(init_hidden)

        assert offset == total_len, (offset, total_len)
        batch[b] = row

    return batch


def test_distill_loss(teacher_ckpt_dir, rng, batch_size=2, opt_steps=30):
    from common.algorithm_torch import Algorithm as TeacherAlgorithm
    from common.algorithm_platform import Algorithm as StudentAlgorithm

    teacher_net = TeacherAlgorithm()
    teacher_net.lstm_time_steps = 1
    teacher_predictor = LocalTorchPredictor(teacher_net)
    teacher_predictor.load_model(teacher_ckpt_dir)

    batch = _build_distill_batch(teacher_predictor, batch_size, rng)
    print("[OK] built synthetic distill batch: shape=%s" % (batch.shape,))

    # 强制打开 distill 开关（Config.DISTILL 是 import 时读取的环境变量，
    # 冒烟测试里直接改写实例属性，避免依赖外部环境）
    student_net = StudentAlgorithm()
    student_net.distill = True
    student_net.lstm_time_steps = LSTM_TIME_STEPS

    datas = torch.from_numpy(batch)
    optimizer = torch.optim.RMSprop(
        student_net.parameters(),
        lr=PlatformConfig.INIT_LEARNING_RATE_START,
        alpha=PlatformConfig.RMSPROP_DECAY,
        momentum=PlatformConfig.RMSPROP_MOMENTUM,
        eps=PlatformConfig.RMSPROP_EPSILON,
    )

    losses = []
    for step in range(opt_steps):
        optimizer.zero_grad()
        data_list = student_net.format_data(datas)
        rst_list = student_net(data_list)
        total_loss, info_list = student_net.compute_loss(data_list, rst_list)
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.item()))

    print(
        "[OK] distill loss over %d steps: first=%.6f last=%.6f min=%.6f"
        % (opt_steps, losses[0], losses[-1], min(losses))
    )
    assert np.isfinite(losses).all(), "loss contains nan/inf: %s" % losses
    assert losses[-1] < losses[0], (
        "distill loss did not decrease: first=%.6f last=%.6f" % (losses[0], losses[-1])
    )
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_ckpt",
        default=PlatformConfig.TEACHER_CKPT_PATH,
        help="directory containing model.pth for the teacher network",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--opt_steps", type=int, default=30)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("=== 1) teacher single-frame inference ===")
    test_teacher_inference(args.teacher_ckpt, rng)

    print("=== 2) student single-frame inference ===")
    test_student_inference(rng)

    print("=== 3) distill loss forward/backward over %d steps ===" % args.opt_steps)
    test_distill_loss(args.teacher_ckpt, rng, args.batch_size, args.opt_steps)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

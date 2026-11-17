# -*- coding: utf-8 -*-
"""
把本地训练产出的教师 ckpt（anlys/*.pth 等）整理成 LocalTorchPredictor.load_model
期望的目录格式：{TEACHER_CKPT_PATH}/model.pth，内容为
{"network_state_dict": ..., ("optimizer_state_dict": ...), "step": ...}。

用法:
    python prepare_teacher_ckpt.py <src_ckpt_path> [--dest DEST_DIR]

若不给 --dest，则默认写入 config_platform.Config.TEACHER_CKPT_PATH
（受环境变量 AIARENA_TEACHER_CKPT_PATH 覆盖）。
"""
import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _resolve_default_dest():
    from common.config_platform import Config

    return Config.TEACHER_CKPT_PATH


def prepare(src_path, dest_dir):
    if not os.path.isfile(src_path):
        raise FileNotFoundError("src ckpt not found: %s" % src_path)

    checkpoint = torch.load(src_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "network_state_dict" not in checkpoint:
        raise ValueError(
            "unexpected ckpt format in %s: expected dict with key "
            "'network_state_dict', got keys=%s"
            % (src_path, list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint))
        )

    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, "model.pth")
    tmp_file = dest_file + ".tmp"

    # 若源文件本身已是标准 {"network_state_dict": ...} 格式，直接原样拷贝即可，
    # 不需要重新 torch.save（避免 optimizer_state 等大字段被意外改写）。
    shutil.copyfile(src_path, tmp_file)
    os.replace(tmp_file, dest_file)

    net_keys = list(checkpoint["network_state_dict"].keys())
    print("prepared teacher ckpt: %s -> %s" % (src_path, dest_file))
    print("step: %s" % checkpoint.get("step"))
    print("network_state_dict keys (%d): %s ..." % (len(net_keys), net_keys[:5]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="source teacher checkpoint file (*.pth)")
    parser.add_argument(
        "--dest",
        default=None,
        help="destination directory (default: Config.TEACHER_CKPT_PATH)",
    )
    args = parser.parse_args()

    dest_dir = args.dest or _resolve_default_dest()
    prepare(args.src, dest_dir)


if __name__ == "__main__":
    main()

"""
一次性转换脚本: PyTorch fsmn-vad → MLX safetensors

用法:
    python convert.py --pt-path /path/to/model.pt --output-dir /path/to/fsmn-vad-mlx
"""

import argparse
import json
import os

import mlx.core as mx
import numpy as np


def sanitize(pt_weights: dict) -> dict:
    """
    将 PyTorch fsmn-vad 权重映射到 MLX 命名和布局。

    PyTorch key 格式:
        encoder.in_linear1.linear.weight
        encoder.fsmn.0.linear.linear.weight
        encoder.fsmn.0.fsmn_block.conv_left.weight  [128, 1, 20, 1]
        encoder.fsmn.0.affine.linear.weight

    MLX key 格式:
        in_linear1.weight
        fsmn.0.linear.weight
        fsmn.0.fsmn_block.conv_left.weight  [128, 20, 1]
        fsmn.0.affine.weight
    """
    mlx_weights = {}

    for k, v in pt_weights.items():
        new_k = k

        # 1. 去掉 "encoder." 前缀
        if new_k.startswith("encoder."):
            new_k = new_k[len("encoder.") :]

        # 2. 去掉多余的 ".linear" (in_linear1.linear.weight → in_linear1.weight)
        #    但保留 fsmn.X.linear.weight (这是层内的 linear 投影)
        #    规则: "xxx.linear.weight" 或 "xxx.linear.bias" 中 xxx 不以数字结尾时去掉
        #    更简洁: 把 ".linear.linear." 替换为 ".linear."
        #            把 "in_linear1.linear." 替换为 "in_linear1."
        #            把 "in_linear2.linear." 替换为 "in_linear2."
        #            把 "out_linear1.linear." 替换为 "out_linear1."
        #            把 "out_linear2.linear." 替换为 "out_linear2."
        #            把 ".affine.linear." 替换为 ".affine."
        new_k = new_k.replace("in_linear1.linear.", "in_linear1.")
        new_k = new_k.replace("in_linear2.linear.", "in_linear2.")
        new_k = new_k.replace("out_linear1.linear.", "out_linear1.")
        new_k = new_k.replace("out_linear2.linear.", "out_linear2.")
        # fsmn.X.linear.linear.weight → fsmn.X.linear.weight
        new_k = new_k.replace(".linear.linear.", ".linear.")
        # fsmn.X.affine.linear.weight → fsmn.X.affine.weight
        new_k = new_k.replace(".affine.linear.", ".affine.")

        # 3. Conv2d 权重转换: [128, 1, 20, 1] → [128, 20, 1]
        v_np = v.numpy() if hasattr(v, "numpy") else v
        if "conv_left.weight" in new_k and len(v_np.shape) == 4:
            # PyTorch Conv2d depthwise: [out, in/groups, kH, kW] = [128, 1, 20, 1]
            # MLX Conv1d: [out, kernel_size, in/groups] = [128, 20, 1]
            v_np = v_np.squeeze(-1)  # [128, 1, 20]
            v_np = v_np.transpose(0, 2, 1)  # [128, 20, 1]

        mlx_weights[new_k] = mx.array(v_np)

    return mlx_weights


def convert(pt_path: str, output_dir: str):
    """执行转换."""
    import torch

    print(f"[1] 加载 PyTorch 权重: {pt_path}")
    pt_weights = torch.load(pt_path, map_location="cpu")
    print(f"    共 {len(pt_weights)} 个参数")

    print(f"[2] Sanitize 权重映射...")
    mlx_weights = sanitize(pt_weights)

    print("    映射结果:")
    for k, v in mlx_weights.items():
        print(f"      {k:45s} {list(v.shape)}")

    print(f"\n[3] 保存到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存权重

    mx.save_safetensors(os.path.join(output_dir, "model.safetensors"), mlx_weights)

    # 保存 config
    config = {
        "model_type": "fsmn",
        "architecture": "fsmn_vad",
        "encoder": {
            "input_dim": 400,
            "input_affine_dim": 140,
            "fsmn_layers": 4,
            "linear_dim": 250,
            "proj_dim": 128,
            "lorder": 20,
            "rorder": 0,
            "lstride": 1,
            "rstride": 0,
            "output_affine_dim": 140,
            "output_dim": 248,
        },
        "sample_rate": 16000,
        "n_mels": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "lfr_m": 5,
        "lfr_n": 1,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("    ✅ 转换完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt-path", type=str, required=True, help="PyTorch model.pt 路径"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="MLX 模型输出目录"
    )
    args = parser.parse_args()
    convert(args.pt_path, args.output_dir)

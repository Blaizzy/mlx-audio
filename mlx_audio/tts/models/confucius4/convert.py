# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""Convert original Confucius4-TTS (+ deps) torch weights into an MLX model dir.
torch is used ONLY here (conversion), never at inference.

    python -m mlx_audio.tts.models.confucius4.convert --out ./confucius4-model
"""

import argparse
import glob
from pathlib import Path

import numpy as np


def _fold_weight_norm(sd):
    folded, vg = {}, {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            vg.setdefault(k[:-9], {})["v"] = v
        elif k.endswith(".weight_g"):
            vg.setdefault(k[:-9], {})["g"] = v
        else:
            folded[k] = v
    for base, d in vg.items():
        g, v = d["g"].float(), d["v"].float()
        folded[base + ".weight"] = g * v / v.flatten(1).norm(dim=1).view(-1, 1, 1)
    return folded


def main():
    import mlx.core as mx
    import safetensors.torch
    import torch
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./confucius4-model")
    ap.add_argument(
        "--quantize",
        choices=["none", "int8", "int4"],
        default="none",
        help="int8/int4: quantize (group 64) the T2S body matmuls; semantic_head kept fp32",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, d):
        mx.save_safetensors(
            str(out / name),
            {
                k: mx.array(v.detach().cpu().numpy().astype(np.float32))
                for k, v in d.items()
            },
        )
        print("saved", name, len(d))

    # T2S GPT-2 weights. With --quantize int8, 8-bit (group 64) the body
    # matmuls (attn/mlp); keep semantic_head + norms + embeddings fp32 — this
    # preserves token-selection fidelity (8-bit head audibly degrades it).
    t2s_pt = hf_hub_download(
        "netease-youdao/Confucius4-TTS", filename="t2s_model.safetensors"
    )
    qbits = {"int8": 8, "int4": 4}.get(a.quantize, 8)
    if a.quantize != "none":
        Wt = mx.load(t2s_pt)
        body = {
            f"transformer.h.{i}.{n}.weight"
            for i in range(24)
            for n in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        }
        qd = {}
        for k, v in Wt.items():
            if k in body:  # GPT-2 Conv1D [in,out] -> [out,in] for quantized_matmul
                wq, sc, bs = mx.quantize(mx.transpose(v), group_size=64, bits=qbits)
                qd[k] = wq
                qd[k[:-7] + ".scales"] = sc
                qd[k[:-7] + ".biases"] = bs
            else:
                qd[k] = v
        mx.save_safetensors(str(out / "t2s_model.safetensors"), qd)
        print(f"saved t2s_model.safetensors ({a.quantize} body: {len(body)} matmuls)")
    else:
        save("t2s_model.safetensors", safetensors.torch.load_file(t2s_pt))

    # S2A (.pt, fold weight_norm)
    s2a_pt = hf_hub_download("netease-youdao/Confucius4-TTS", filename="s2a_model.pt")
    sd = torch.load(s2a_pt, map_location="cpu", weights_only=False)
    save(
        "s2a_mlx.safetensors",
        _fold_weight_norm(sd.state_dict() if hasattr(sd, "state_dict") else sd),
    )

    # BigVGAN (fold weight_norm)
    bv = torch.load(
        hf_hub_download(
            "nvidia/bigvgan_v2_22khz_80band_256x", filename="bigvgan_generator.pt"
        ),
        map_location="cpu",
        weights_only=False,
    )
    save("bigvgan_mlx.safetensors", _fold_weight_norm(bv.get("generator", bv)))

    # w2v-bert: feature_projection + layers 0..16 only
    from transformers import Wav2Vec2BertModel

    wsd = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval().state_dict()
    keep = {
        k: v
        for k, v in wsd.items()
        if k.startswith("feature_projection.")
        or (k.startswith("encoder.layers.") and int(k.split(".")[2]) <= 16)
    }
    if a.quantize != "none":
        # quantize per-layer ffn + attn linears (feature_projection kept fp32:
        # in=160 not divisible by group 64, and it is tiny anyway)
        wlin = {
            f"encoder.layers.{i}.{n}.weight"
            for i in range(17)
            for n in (
                "ffn1.intermediate_dense",
                "ffn1.output_dense",
                "ffn2.intermediate_dense",
                "ffn2.output_dense",
                "self_attn.linear_q",
                "self_attn.linear_k",
                "self_attn.linear_v",
                "self_attn.linear_out",
            )
        }
        qd = {}
        for k, v in keep.items():
            arr = mx.array(v.detach().cpu().numpy().astype(np.float32))
            if k in wlin:
                wq, sc, bs = mx.quantize(arr, group_size=64, bits=qbits)
                qd[k] = wq
                qd[k[:-7] + ".scales"] = sc
                qd[k[:-7] + ".biases"] = bs
            else:
                qd[k] = arr
        mx.save_safetensors(str(out / "w2vbert_mlx.safetensors"), qd)
        print(f"saved w2vbert_mlx.safetensors ({a.quantize} linears: {len(wlin)})")
    else:
        save("w2vbert_mlx.safetensors", keep)

    # w2v stats
    st = torch.load(
        (
            hf_hub_download("facebook/w2v-bert-2.0", filename="wav2vec2bert_stats.pt")
            if False
            else (
                glob.glob(str(Path(t2s_pt).parent / "wav2vec2bert_stats.pt"))[0]
                if glob.glob(str(Path(t2s_pt).parent / "wav2vec2bert_stats.pt"))
                else hf_hub_download(
                    "netease-youdao/Confucius4-TTS", filename="wav2vec2bert_stats.pt"
                )
            )
        ),
        map_location="cpu",
    )
    np.savez(
        str(out / "w2v_stats.npz"),
        mean=st["mean"].numpy().astype(np.float32),
        std=torch.sqrt(st["var"]).numpy().astype(np.float32),
    )
    print("saved w2v_stats.npz")

    # CAMPPlus: sanitize via mlx-audio's CAMPPlus then save (no torch at inference)
    from mlx.utils import tree_unflatten

    from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus

    cm = CAMPPlus(feat_dim=80, embedding_size=192)
    csd = torch.load(
        hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin"),
        map_location="cpu",
    )
    csd = csd.get("state_dict", csd)
    san = cm.sanitize(
        {k: mx.array(v.float().numpy()) for k, v in csd.items() if torch.is_tensor(v)}
    )
    mx.save_safetensors(str(out / "campplus.safetensors"), dict(san))
    print("saved campplus.safetensors", len(san))

    # precompute SeamlessM4T fbank mel matrix + povey window (torch-free at runtime)
    from transformers.audio_utils import mel_filter_bank, window_function

    mel = mel_filter_bank(
        num_frequency_bins=257,
        num_mel_filters=80,
        min_frequency=20,
        max_frequency=8000,
        sampling_rate=16000,
        norm=None,
        mel_scale="kaldi",
        triangularize_in_mel_space=True,
    )
    win = window_function(400, "povey", periodic=False)
    np.savez(
        str(out / "fbank_filters.npz"),
        mel=mel.astype(np.float32),
        window=win.astype(np.float32),
    )
    print("saved fbank_filters.npz")

    # tokenizer (loaded torch-free at runtime via tokenizers.Tokenizer.from_file)
    import shutil

    ck = out / "checkpoints"
    ck.mkdir(exist_ok=True)
    tok_json = hf_hub_download(
        "netease-youdao/Confucius4-TTS", filename="tokenizer.json"
    )
    shutil.copy(tok_json, ck / "tokenizer.json")
    print("saved checkpoints/tokenizer.json")

    import json

    (out / "config.json").write_text(
        json.dumps(
            {
                "model_type": "confucius4",
                "sample_rate": 22050,
                "quant_bits": qbits,
                "quant_group_size": 64,
            },
            indent=2,
        )
    )
    print("saved config.json")
    print("DONE ->", out)


if __name__ == "__main__":
    main()

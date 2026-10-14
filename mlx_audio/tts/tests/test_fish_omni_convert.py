import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.tts.models.fish_omni.convert import Converter, dequantize_fp8

EOS, SEM_START, CODEBOOK = 10, 20, 16
SEM_END = SEM_START + CODEBOOK - 1
VOCAB = 64
H = 128  # FP8 blocks are 128 wide
LAYERS = 4  # 3 linear + 1 full attention
EXPERTS = 4


def fp8_block(rng, rows, cols):
    weight = rng.normal(size=(rows, cols)).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, size=(rows // 128, cols // 128)).astype(np.float32)
    return mx.to_fp8(mx.array(weight)), mx.array(scale)


def write_checkpoint(path, rng):
    def bf16(*shape, scale=0.05):
        return mx.array(rng.normal(size=shape).astype(np.float32) * scale).astype(
            mx.bfloat16
        )

    w = {}
    lm = "model.language_model."
    w[lm + "embed_tokens.weight"] = bf16(VOCAB, H)
    w[lm + "norm.weight"] = bf16(H)
    w["lm_head.weight"] = bf16(VOCAB, H)
    for l in range(LAYERS):
        p = f"{lm}layers.{l}."
        w[p + "input_layernorm.weight"] = bf16(H)
        w[p + "post_attention_layernorm.weight"] = bf16(H)
        if (l + 1) % 4:
            a = p + "linear_attn."
            w[a + "A_log"] = bf16(4)
            w[a + "dt_bias"] = bf16(4)
            w[a + "conv1d.weight"] = bf16(256, 1, 4)
            w[a + "in_proj_qkv.weight"] = bf16(256, H)
            w[a + "in_proj_z.weight"] = bf16(128, H)
            w[a + "in_proj_a.weight"] = bf16(4, H)
            w[a + "in_proj_b.weight"] = bf16(4, H)
            w[a + "norm.weight"] = bf16(32)
            w[a + "out_proj.weight"] = bf16(H, 128)
        else:
            a = p + "self_attn."
            w[a + "q_proj.weight"] = bf16(256, H)
            w[a + "k_proj.weight"] = bf16(64, H)
            w[a + "v_proj.weight"] = bf16(64, H)
            w[a + "o_proj.weight"] = bf16(H, 128)
            w[a + "q_norm.weight"] = bf16(64)
            w[a + "k_norm.weight"] = bf16(64)
        m = p + "mlp."
        w[m + "gate.weight"] = bf16(EXPERTS, H)
        w[m + "shared_expert_gate.weight"] = bf16(1, H)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w[f"{m}shared_expert.{proj}.weight"] = bf16(128, H)
            for e in range(EXPERTS):
                weight, scale = fp8_block(rng, 128, 128)
                w[f"{m}experts.{e}.{proj}.weight"] = weight
                w[f"{m}experts.{e}.{proj}.weight_scale_inv"] = scale

    D = H  # the tower consumes backbone hidden states
    w["audio_decoder.codebook_embeddings.weight"] = bf16(CODEBOOK * 4, D)
    for prefix, count, emb in (
        ("audio_decoder_mtp_shared.layers.", 2, "semantic_embedding"),
        ("audio_decoder_mtp.layers.", 3, "codebook_embedding"),
    ):
        for i in range(count):
            p = f"{prefix}{i}."
            w[p + emb + ".weight"] = bf16(CODEBOOK, D)
            for norm in ("enorm", "hnorm", "final_layernorm"):
                w[f"{p}{norm}.weight"] = bf16(D)
            w[p + "eh_proj.weight"] = bf16(D, 2 * D)
            t = p + "transformer_layer."
            w[t + "attention_norm.weight"] = bf16(D)
            w[t + "ffn_norm.weight"] = bf16(D)
            w[t + "attention.wqkv.weight"] = bf16(2 * 64 + 2 * 32, D)
            w[t + "attention.wo.weight"] = bf16(D, 64)
            w[t + "attention.q_norm.weight"] = bf16(16)
            w[t + "attention.k_norm.weight"] = bf16(16)
            w[t + "feed_forward.w1.weight"] = bf16(128, D)
            w[t + "feed_forward.w3.weight"] = bf16(128, D)
            w[t + "feed_forward.w2.weight"] = bf16(D, 128)
    for i in range(3):
        w[f"audio_decoder_mtp.heads.{i}.weight"] = bf16(CODEBOOK, D)

    mx.save_safetensors(str(path / "model-00001-of-00001.safetensors"), w)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in w}})
    )
    config = {
        "model_type": "fish_omni",
        "eos_token_id": EOS,
        "pad_token_id": 5,
        "semantic_start_token_id": SEM_START,
        "semantic_end_token_id": SEM_END,
        "quantization_config": {"quant_method": "bsgemm-moe"},
        "text_config": {
            "hidden_size": H,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "num_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "mlp_layer_types": ["moe"] * LAYERS,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.25,
            },
        },
        "audio_decoder_config": {
            "vocab_size": CODEBOOK,
            "num_codebooks": 4,
            "dim": H,
            "n_head": 4,
            "n_local_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "shared_enrichment_layers": 2,
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    return w


def write_tokenizer(path):
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {f"tok{i}": i for i in range(VOCAB)}
    vocab["<|im_end|>"] = EOS
    for i in range(CODEBOOK):
        vocab[f"<|semantic:{i}|>"] = SEM_START + i
    for name in list(vocab):
        if vocab[name] in (EOS, *range(SEM_START, SEM_END + 1)) and name.startswith(
            "tok"
        ):
            del vocab[name]
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="tok0")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))


def convert(source, output, expert_bits, embed_bits=None, dense_bits=None):
    output.mkdir()
    Converter(
        source,
        output,
        expert_bits=expert_bits,
        embed_bits=embed_bits,
        dense_bits=dense_bits,
        group_size=64,
    ).convert()


def test_convert_bf16_tensors(tmp_path):
    rng = np.random.default_rng(0)
    source = tmp_path / "src"
    source.mkdir()
    w = write_checkpoint(source, rng)
    convert(source, tmp_path / "out", None)
    out = {}
    for f in (tmp_path / "out").glob("model-*.safetensors"):
        out.update(mx.load(str(f)))

    expert = "model.language_model.layers.2.mlp.experts.3.up_proj."
    want = dequantize_fp8(w[expert + "weight"], w[expert + "weight_scale_inv"])
    got = out["backbone.layers.2.mlp.switch_mlp.up_proj.weight"][3]
    assert mx.allclose(got.astype(mx.float32), want, rtol=1e-2, atol=1e-2).item()

    lm_head = out["lm_head.weight"]
    assert lm_head.shape == (1 + CODEBOOK, H)
    assert mx.array_equal(lm_head[0], w["lm_head.weight"][EOS]).item()
    assert mx.array_equal(
        lm_head[1:], w["lm_head.weight"][SEM_START : SEM_END + 1]
    ).item()

    norm = "layers.0.input_layernorm.weight"
    shifted = w["model.language_model." + norm].astype(mx.float32) + 1
    assert mx.allclose(
        out["backbone." + norm].astype(mx.float32), shifted, atol=1e-2
    ).item()
    # The Gated DeltaNet output norm is not zero-centered.
    assert mx.array_equal(
        out["backbone.layers.0.linear_attn.norm.weight"],
        w["model.language_model.layers.0.linear_attn.norm.weight"],
    ).item()
    assert out["backbone.layers.0.linear_attn.conv1d.weight"].shape == (256, 4, 1)
    assert "tower.shared_layers.1.semantic_embedding.weight" in out
    assert "tower.codebook_heads.2.weight" in out


def test_quantized_model_loads_and_decodes(tmp_path):
    from mlx_audio.tts.utils import load_model

    rng = np.random.default_rng(1)
    source = tmp_path / "src"
    source.mkdir()
    write_checkpoint(source, rng)
    output = tmp_path / "q4"
    convert(source, output, 4, embed_bits=8, dense_bits=8)
    write_tokenizer(output)

    config = json.loads((output / "config.json").read_text())
    quant = config["quantization"]
    assert quant["backbone.embed_tokens"]["bits"] == 8
    assert quant["backbone.layers.3.self_attn.q_proj"]["bits"] == 8
    assert "lm_head" not in quant and "backbone.layers.0.mlp.gate" not in quant
    assert (
        config["quantization"]["backbone.layers.0.mlp.switch_mlp.gate_proj"]["bits"]
        == 4
    )
    assert "quantization_config" not in config

    model = load_model(str(output))
    assert model.codec is None
    prompt = np.zeros((6, 5), dtype=np.int32)
    prompt[:, 0] = [1, 2, 3, 4, SEM_START + 3, 7]
    prompt[4, 1:] = [3, 1, 2, 0]
    mx.random.seed(0)
    frames = list(
        model.generate_frames(
            prompt, max_new_tokens=4, temperature=0.7, top_k=30, top_p=0.7
        )
    )
    for frame in frames:
        assert frame.shape == (1, 1, 5)
        main = frame[0, 0, 0].item()
        assert SEM_START <= main <= SEM_END
        assert frame[0, 0, 1].item() == main - SEM_START

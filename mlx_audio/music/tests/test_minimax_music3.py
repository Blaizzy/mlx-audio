"""Contracts for the native MiniMax Music 3 MLX integration."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten


def test_minimax_music3_is_registered_as_music() -> None:
    from mlx_audio.music.utils import MODEL_REMAPPING
    from mlx_audio.registry import classify_model

    assert MODEL_REMAPPING["minimax_music3"] == "minimax_music3"
    assert classify_model("minimax_music3", "MiniMax-Music3") == "music"


def test_tiny_generation_returns_finite_stereo_audio() -> None:
    from mlx_audio.music.models.minimax_music3.minimax_music3 import Model, ModelConfig
    from mlx_audio.tts.models.base import GenerationResult

    model = Model(ModelConfig.tiny())
    result = next(
        model.generate(
            text="Genre: acoustic pop. BPM: 96. Warm female vocal.",
            lyrics="[verse]\nMorning light\n[chorus]\nSing with me",
            duration=0.08,
            steps=2,
            seed=7,
        )
    )

    assert isinstance(result, GenerationResult)
    assert result.sample_rate == 44_100
    assert result.audio.ndim == 2
    assert result.audio.shape[1] == 2
    assert np.isfinite(np.asarray(result.audio)).all()
    assert result.prompt["tokens"] == result.token_count
    assert result.prompt["tokens-per-sec"] > 0
    assert result.audio_samples["samples"] == result.samples
    assert result.audio_samples["samples-per-sec"] > 0


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [
        (1, [0]),
        (200, [0]),
        (201, [0, 100]),
        (300, [0, 100]),
        (301, [0, 100, 200]),
    ],
)
def test_chunk_starts_match_the_official_200_by_100_windows(
    num_frames: int, expected: list[int]
) -> None:
    from mlx_audio.music.models.minimax_music3.minimax_music3 import _chunk_starts

    assert _chunk_starts(num_frames) == expected


def test_waveform_crops_match_the_official_chunk_stitching() -> None:
    from mlx_audio.music.models.minimax_music3.config import LATENT_HOP_LENGTH
    from mlx_audio.music.models.minimax_music3.minimax_music3 import _crop_waveform

    samples = 500 * LATENT_HOP_LENGTH
    waveform = mx.arange(samples, dtype=mx.float32).reshape(1, 1, -1)
    first = _crop_waveform(waveform, 0, 3)
    middle = _crop_waveform(waveform, 1, 3)
    last = _crop_waveform(waveform, 2, 3)

    assert first.shape[-1] == (500 - 258) * LATENT_HOP_LENGTH
    assert middle.shape[-1] == (500 - 86 - 258) * LATENT_HOP_LENGTH
    assert last.shape[-1] == (500 - 86) * LATENT_HOP_LENGTH
    assert float(first[0, 0, 0].item()) == 0
    assert float(middle[0, 0, 0].item()) == 86 * LATENT_HOP_LENGTH
    assert float(last[0, 0, 0].item()) == 86 * LATENT_HOP_LENGTH


def test_flow_schedule_runs_from_noise_to_data() -> None:
    from mlx_audio.music.models.minimax_music3.euler import make_sigma_schedule

    np.testing.assert_allclose(
        make_sigma_schedule(4),
        np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32),
    )


def test_quantization_policy_targets_only_large_generation_linears() -> None:
    from mlx_audio.music.models.minimax_music3.minimax_music3 import Model, ModelConfig

    model = Model(ModelConfig.tiny())
    large_linear = nn.Linear(64, 64, bias=False)
    embedding = nn.Embedding(128, 64)

    assert model.model_quant_predicate(
        "language_model.model.layers.0.self_attn.q_proj", large_linear
    )
    assert model.model_quant_predicate(
        "rvq_depth_decoder.layers.0.attn.to_q", large_linear
    )
    assert model.model_quant_predicate(
        "transformer.transformer_blocks.0.ff_in", large_linear
    )
    assert not model.model_quant_predicate("language_model.lm_head", large_linear)
    assert not model.model_quant_predicate(
        "language_model.model.embed_tokens", embedding
    )
    assert not model.model_quant_predicate("vocoder.conv_in", large_linear)


@pytest.mark.parametrize(
    ("mode", "group_size", "bits"),
    [
        ("affine", 64, 4),
        ("mxfp4", 32, 4),
        ("mxfp8", 32, 8),
        ("nvfp4", 16, 4),
    ],
)
def test_all_mlx_quantization_modes_rebuild_the_same_quantized_topology(
    mode: str, group_size: int, bits: int
) -> None:
    from mlx_audio.music.models.minimax_music3.minimax_music3 import Model, ModelConfig
    from mlx_audio.utils import apply_quantization

    model = Model(ModelConfig.tiny())
    config = {"quantization": {"group_size": group_size, "bits": bits, "mode": mode}}
    selected = {
        f"{path}.scales": mx.ones((1,))
        for path, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and module.weight.shape[-1] % group_size == 0
        and model.model_quant_predicate(path, module)
    }

    apply_quantization(
        model,
        config,
        selected,
        model.model_quant_predicate,
    )

    assert isinstance(
        model.language_model.model.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )
    assert model.language_model.model.layers[0].self_attn.q_proj.mode == mode
    assert isinstance(model.language_model.lm_head, nn.Linear)
    assert isinstance(model.vocoder.conv_in, nn.Conv1d)


def _component_configs(config) -> dict[str, dict]:
    return {
        "language_model": {
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "tie_word_embeddings": config.tie_word_embeddings,
            "rope_parameters": {"rope_theta": config.rope_theta},
        },
        "rvq_depth_decoder": {
            "hidden_size": config.hidden_size,
            "num_layers": config.depth_num_layers,
            "num_attention_heads": config.depth_num_heads,
            "intermediate_size": config.depth_intermediate_size,
            "audio_vocab_size": config.audio_vocab_size,
            "num_codebooks": config.num_codebooks,
        },
        "condition_encoder": {
            "condition_hidden_dim": config.hidden_size,
            "num_condition_layers": config.num_condition_layers,
            "out_dim": config.condition_out_dim,
            "input_sampling_rate": config.input_sampling_rate,
            "input_hop_length": config.input_hop_length,
            "output_sampling_rate": config.output_sampling_rate,
            "output_hop_length": config.output_hop_length,
        },
        "transformer": {
            "in_channels": config.dit_in_channels,
            "condition_dim": config.condition_out_dim,
            "num_layers": config.dit_num_layers,
            "num_attention_heads": config.dit_num_heads,
            "attention_head_dim": config.dit_head_dim,
            "ff_inner_dim": config.dit_ff_inner_dim,
            "rotary_dim": config.dit_rotary_dim,
            "fourier_embedding_dim": config.dit_fourier_dim,
        },
        "vocoder": {
            "latent_channels": config.dit_in_channels,
            "decoder_input_dim": config.vocoder_input_dim,
            "decoder_hidden_dim": config.vocoder_hidden_dim,
            "upsampling_ratios": list(config.vocoder_upsampling_ratios),
            "sampling_rate": config.sample_rate,
        },
    }


def _to_official_tensor(key: str, value: mx.array) -> mx.array:
    if value.ndim != 3 or not key.endswith(".weight"):
        return value
    if "conv_t" in key:
        return value.transpose(2, 0, 1)
    return value.transpose(0, 2, 1)


def _write_official_tiny_tree(source: Path) -> None:
    from mlx_audio.music.models.minimax_music3.minimax_music3 import Model, ModelConfig

    config = ModelConfig.tiny()
    model = Model(config)
    source.mkdir(parents=True)
    (source / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxMusic3ModularPipeline",
                "_diffusers_version": "0.40.0.dev0",
            }
        )
    )

    modules = {
        "language_model": model.language_model,
        "rvq_depth_decoder": model.rvq_depth_decoder,
        "condition_encoder": model.condition_encoder,
        "transformer": model.transformer,
        "vocoder": model.vocoder,
    }
    for name, module in modules.items():
        folder = source / name
        folder.mkdir()
        (folder / "config.json").write_text(
            json.dumps(_component_configs(config)[name])
        )
        official = {}
        for key, value in tree_flatten(module.parameters()):
            official_key = key
            if name == "transformer" and ".to_out.0." in official_key:
                official_key = official_key.replace(".to_out.0.", ".to_out.")
            official[official_key] = _to_official_tensor(key, value)

        if name == "vocoder":
            weight = official.pop("conv_in.weight")
            norm = mx.sqrt(mx.sum(weight.astype(mx.float32) ** 2, axis=(1, 2)))
            official["conv_in.weight_v"] = weight
            official["conv_in.weight_g"] = norm.reshape(-1, 1, 1)

        mx.save_safetensors(
            str(folder / "diffusion_pytorch_model.safetensors"), official
        )


@pytest.mark.parametrize(
    ("mode", "group_size", "bits"),
    [
        ("affine", 64, 4),
        ("mxfp4", 32, 4),
        ("mxfp8", 32, 8),
        ("nvfp4", 16, 4),
    ],
)
def test_official_tree_converts_and_loads_in_every_mlx_quantization_mode(
    tmp_path: Path, mode: str, group_size: int, bits: int
) -> None:
    from mlx_audio.convert import convert
    from mlx_audio.music.utils import load_model

    source = tmp_path / "MiniMax-Music3"
    _write_official_tiny_tree(source)

    destination = tmp_path / mode
    convert(
        str(source),
        str(destination),
        quantize=True,
        q_mode=mode,
    )

    saved_config = json.loads((destination / "config.json").read_text())
    assert saved_config["model_type"] == "minimax_music3"
    assert saved_config["quantization"]["mode"] == mode
    assert saved_config["quantization"]["group_size"] == group_size
    assert saved_config["quantization"]["bits"] == bits
    assert not (destination / "modular_model_index.json").exists()

    loaded = load_model(destination)
    projection = loaded.language_model.model.layers[0].self_attn.q_proj
    assert isinstance(projection, nn.QuantizedLinear)
    assert projection.mode == mode
    result = next(
        loaded.generate(
            text="Warm acoustic pop",
            lyrics="[verse]\nMorning light",
            duration=0.04,
            steps=1,
            seed=3,
        )
    )
    assert result.audio.shape[1] == 2
    assert np.isfinite(np.asarray(result.audio)).all()

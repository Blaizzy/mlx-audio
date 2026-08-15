"""Opt-in numerical parity tests against the official PyTorch implementations.

These tests deliberately instantiate the official Transformers and Diffusers
modules, then load their PyTorch parameters through the MiniMax Music 3 MLX
conversion path.  They are kept opt-in because mlx-audio does not otherwise
depend on PyTorch or a development checkout of Diffusers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

_DIFFUSERS_ENV = "MLX_AUDIO_MINIMAX_MUSIC3_DIFFUSERS"
_DIFFUSERS_COMMIT = "dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_DIFFUSERS_ENV),
    reason=f"set {_DIFFUSERS_ENV} to a pinned Diffusers checkout",
)


def _official_classes():
    checkout = Path(os.environ[_DIFFUSERS_ENV]).resolve()
    source = checkout / "src"
    if not source.is_dir():
        pytest.fail(f"{_DIFFUSERS_ENV} does not point to a Diffusers checkout")

    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == _DIFFUSERS_COMMIT

    sys.path.insert(0, str(source))
    try:
        from diffusers.models.autoencoders.minimax_music3_vocoder import (
            MiniMaxMusic3Vocoder,
        )
        from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import (
            MiniMaxMusic3ConditionEncoder,
        )
        from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import (
            MiniMaxMusic3RVQDepthDecoder,
        )
        from diffusers.models.transformers.transformer_minimax_music3 import (
            MiniMaxMusic3Transformer1DModel,
        )
    finally:
        sys.path.pop(0)

    return (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )


def _to_mlx(value) -> mx.array:
    return mx.array(value.detach().cpu().numpy())


def _load_torch_module(torch_module, mlx_module, *, vocoder: bool = False) -> None:
    from mlx_audio.tts.models.minimax_music3.conversion import (
        _fuse_weight_norm_pairs,
        _remap_tensor,
        _sanitize_key,
    )

    state = {key: _to_mlx(value) for key, value in torch_module.state_dict().items()}
    if vocoder:
        state = _fuse_weight_norm_pairs(state)

    converted = {}
    for source_key, value in state.items():
        key = _sanitize_key(source_key)
        if key is not None:
            converted[key] = _remap_tensor(key, value)
    mlx_module.load_weights(list(converted.items()), strict=True)
    mx.eval(mlx_module.parameters())


def _assert_close(actual: mx.array, expected, *, atol: float = 2e-4) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().cpu().numpy(),
        rtol=2e-4,
        atol=atol,
    )


def test_official_qwen3_logits_match_dense_mlx() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from mlx_audio.lm.models.qwen3 import Model as MLXQwen3
    from mlx_audio.lm.models.qwen3 import ModelArgs

    torch.manual_seed(11)
    official_config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=1_000_000.0,
        tie_word_embeddings=False,
        attention_bias=False,
    )
    official = transformers.Qwen3ForCausalLM(official_config).eval()
    mlx_model = MLXQwen3(
        ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            rms_norm_eps=1e-5,
            rope_theta=1_000_000.0,
            tie_word_embeddings=False,
            vocab_size=64,
        )
    )
    _load_torch_module(official, mlx_model)

    token_ids = np.array([[1, 7, 3, 12, 5], [4, 2, 9, 6, 8]], dtype=np.int32)
    with torch.inference_mode():
        expected = official(torch.from_numpy(token_ids).long()).logits
    actual = mlx_model(mx.array(token_ids))
    mx.eval(actual)

    _assert_close(actual, expected)


def test_official_flow_scheduler_matches_mlx() -> None:
    pytest.importorskip("torch")
    _official_classes()
    from diffusers import FlowMatchEulerDiscreteScheduler

    from mlx_audio.tts.models.minimax_music3.euler import make_sigma_schedule

    num_steps = 7
    official = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1,
        invert_sigmas=True,
    )
    official.set_timesteps(
        sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps),
        device="cpu",
    )

    np.testing.assert_allclose(
        official.sigmas.cpu().numpy(),
        make_sigma_schedule(num_steps),
        rtol=0,
        atol=1e-7,
    )


def test_official_music3_components_match_dense_mlx() -> None:
    torch = pytest.importorskip("torch")
    (
        TorchConditionEncoder,
        TorchDepthDecoder,
        TorchTransformer,
        TorchVocoder,
    ) = _official_classes()

    from mlx_audio.tts.models.minimax_music3.config import ModelConfig
    from mlx_audio.tts.models.minimax_music3.depth import RVQDepthDecoder
    from mlx_audio.tts.models.minimax_music3.dit import FlowMatchingTransformer
    from mlx_audio.tts.models.minimax_music3.fusion import ConditionEncoder
    from mlx_audio.tts.models.minimax_music3.vocoder import Vocoder

    config = ModelConfig.tiny()
    torch.manual_seed(17)
    rng = np.random.default_rng(17)

    torch_condition = TorchConditionEncoder(
        condition_hidden_dim=config.hidden_size,
        num_condition_layers=config.num_condition_layers,
        out_dim=config.condition_out_dim,
        input_sampling_rate=config.input_sampling_rate,
        input_hop_length=config.input_hop_length,
        output_sampling_rate=config.output_sampling_rate,
        output_hop_length=config.output_hop_length,
    ).eval()
    mlx_condition = ConditionEncoder(config)
    _load_torch_module(torch_condition, mlx_condition)
    condition_input = rng.standard_normal(
        (2, 5, config.num_condition_layers * config.hidden_size), dtype=np.float32
    )
    with torch.inference_mode():
        expected_condition = torch_condition(torch.from_numpy(condition_input))
    actual_condition = mlx_condition(mx.array(condition_input))
    mx.eval(actual_condition)
    _assert_close(actual_condition, expected_condition)

    torch_depth = TorchDepthDecoder(
        hidden_size=config.hidden_size,
        num_layers=config.depth_num_layers,
        num_attention_heads=config.depth_num_heads,
        intermediate_size=config.depth_intermediate_size,
        audio_vocab_size=config.audio_vocab_size,
        num_codebooks=config.num_codebooks,
        max_position_embeddings=config.depth_max_position_embeddings,
    ).eval()
    mlx_depth = RVQDepthDecoder(config)
    _load_torch_module(torch_depth, mlx_depth)
    depth_input = rng.standard_normal((2, 5, config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        expected_depth = torch_depth(torch.from_numpy(depth_input))
    actual_depth = mlx_depth(mx.array(depth_input))
    mx.eval(actual_depth)
    _assert_close(actual_depth, expected_depth)

    torch_transformer = TorchTransformer(
        in_channels=config.dit_in_channels,
        condition_dim=config.condition_out_dim,
        num_layers=config.dit_num_layers,
        num_attention_heads=config.dit_num_heads,
        attention_head_dim=config.dit_head_dim,
        ff_inner_dim=config.dit_ff_inner_dim,
        rotary_dim=config.dit_rotary_dim,
        fourier_embedding_dim=config.dit_fourier_dim,
    ).eval()
    mlx_transformer = FlowMatchingTransformer(config)
    _load_torch_module(torch_transformer, mlx_transformer)
    latent_input = rng.standard_normal((2, config.dit_in_channels, 7), dtype=np.float32)
    transformer_condition = rng.standard_normal(
        (2, 7, config.condition_out_dim), dtype=np.float32
    )
    timestep = np.array([0.2, 0.8], dtype=np.float32)
    with torch.inference_mode():
        expected_velocity = torch_transformer(
            torch.from_numpy(latent_input),
            torch.from_numpy(timestep),
            torch.from_numpy(transformer_condition),
        ).sample
    actual_velocity = mlx_transformer(
        mx.array(latent_input),
        mx.array(timestep),
        mx.array(transformer_condition),
    )
    mx.eval(actual_velocity)
    _assert_close(actual_velocity, expected_velocity)

    torch_vocoder = TorchVocoder(
        latent_channels=config.dit_in_channels,
        decoder_input_dim=config.vocoder_input_dim,
        decoder_hidden_dim=config.vocoder_hidden_dim,
        upsampling_ratios=config.vocoder_upsampling_ratios,
        sampling_rate=config.sample_rate,
    ).eval()
    mlx_vocoder = Vocoder(config)
    _load_torch_module(torch_vocoder, mlx_vocoder, vocoder=True)
    vocoder_input = rng.standard_normal(
        (1, config.dit_in_channels, 6), dtype=np.float32
    )
    with torch.inference_mode():
        expected_waveform = torch_vocoder(torch.from_numpy(vocoder_input))
    actual_waveform = mlx_vocoder(mx.array(vocoder_input))
    mx.eval(actual_waveform)
    _assert_close(actual_waveform, expected_waveform, atol=5e-4)

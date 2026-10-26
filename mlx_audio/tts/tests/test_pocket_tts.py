"""Compatibility and sampling regressions; no downloaded checkpoints required."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers

from mlx_audio.tts.models.pocket_tts import Model, ModelConfig
from mlx_audio.tts.models.pocket_tts.conditioners import (
    ContinuousConditioner,
    SpeakerCountConditioner,
    TokenizedText,
)
from mlx_audio.tts.models.pocket_tts.convert import (
    _prepare_quantized_modules,
    convert,
    convert_weights,
    load_source_weights,
)
from mlx_audio.tts.models.pocket_tts.flow_lm import FlowLMModel, lsd_decode
from mlx_audio.tts.models.pocket_tts.weights import _seanet_names


def flat_config():
    return {
        "flow_lm": {
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 24,
            "max_period": 10000,
            "n_bins": 8,
            "lut_dim": 8,
            "flow_dim": 8,
            "flow_depth": 1,
            "ldim": 4,
        },
        "mimi": {
            "channels": 1,
            "sample_rate": 16,
            "frame_rate": 2,
            "dimension": 8,
            "quantizer_dimension": 4,
            "quantizer_output_dimension": 8,
            "n_filters": 2,
            "n_residual_layers": 1,
            "ratios": [2],
            "kernel_size": 3,
            "last_kernel_size": 3,
            "residual_kernel_size": 3,
            "dilation_base": 2,
            "compress": 2,
            "transformer_d_model": 8,
            "transformer_num_heads": 2,
            "transformer_num_layers": 1,
            "transformer_layer_scale": 0.01,
            "transformer_context": 16,
            "transformer_max_period": 10000,
            "transformer_dim_feedforward": 16,
        },
    }


@pytest.fixture
def config(tmp_path):
    tok = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "Hello": 1,
                "world": 2,
                ".": 3,
                "!": 4,
                "?": 5,
                "test": 6,
                "Another": 7,
            },
            unk_token="[UNK]",
        )
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    cfg = flat_config()
    cfg["model_path"] = str(tmp_path)
    return cfg


def test_legacy_defaults_and_nested_roundtrip(config):
    nested = asdict(ModelConfig.from_dict(config))
    for field in ("temp", "lsd_decode_steps", "eos_threshold"):
        nested.pop(field)
    cfg = ModelConfig.from_dict(nested)
    assert (cfg.temp, cfg.lsd_decode_steps, cfg.eos_threshold) == (
        0.7,
        1,
        -4.0,
    )
    assert cfg.flow_lm.transformer.hidden_scale == 3
    assert cfg.mimi.transformer.output_dimensions == (8,)
    assert Model(cfg).flow_lm.transformer.layers[0].linear1.weight.shape == (24, 8)


def test_new_config_fields(config):
    config.update(
        temp=0.8,
        lsd_decode_steps=4,
        eos_threshold=4.0,
        noise_clamp=2.5,
        audio_prompt_min_duration=1,
        audio_prompt_max_duration=10,
        conditioners=[
            {"name": "num_speakers", "type": "lut", "lut": {"n_bins": 31, "dim": 4}}
        ],
        fuser={
            "sum": ["num_speakers"],
            "prepend": ["transcript_in_segment", "speaker_wavs"],
        },
    )
    model = Model(config)
    assert (
        model.temp,
        model.lsd_decode_steps,
        model.eos_threshold,
    ) == (0.8, 4, 4.0)
    assert model.flow_lm.num_speakers.embed.weight.shape == (32, 4)
    assert ModelConfig.from_dict(asdict(model.config)) == model.config


class CountingBackbone:
    def __init__(self):
        self.calls = []
        self.inputs = []

    def __call__(self, x, cache):
        self.calls.append(cache)
        self.inputs.append(x)
        return x + cache


class RecordingFlow:
    def __init__(self):
        self.conditions = []

    def __call__(self, c, s, t, x):
        self.conditions.append(c)
        return c[:, :2]


def make_flow():
    backbone, sampler = CountingBackbone(), RecordingFlow()
    model = FlowLMModel(None, sampler, backbone, dim=2, ldim=2)
    model.input_linear.weight = mx.eye(2)
    model.out_norm = None
    model.out_eos.weight = mx.ones((1, 2))
    model.out_eos.bias = mx.zeros((1,))
    return model, backbone, sampler


def test_sampling_uses_exactly_one_backbone_pass():
    model, backbone, sampler = make_flow()
    out, eos = model(
        mx.array([[[2.0, 3.0]]]),
        mx.zeros((1, 0, 2)),
        1,
        1,
        0.0,
        None,
        0.0,
    )
    assert backbone.calls == [1]
    assert len(sampler.conditions) == 1
    assert out.tolist() == [[3.0, 4.0]]
    assert eos.item()


def test_multistep_flow_reuses_backbone_condition_and_eos():
    model, backbone, sampler = make_flow()
    out, eos = model(
        mx.array([[[2.0, 3.0]]]),
        mx.zeros((1, 0, 2)),
        1,
        4,
        0.0,
        None,
        9.0,
    )
    assert backbone.calls == [1]
    assert len(sampler.conditions) == 4
    for cond in sampler.conditions:
        assert cond.tolist() == [[3.0, 4.0]]
    assert out.tolist() == [[3.0, 4.0]]
    assert not eos.item()


def test_speaker_condition_only_applies_to_audio_tokens():
    model, backbone, _ = make_flow()
    model.num_speakers = lambda tokens: mx.array([[10.0, 20.0]])
    out = model.backbone(
        mx.array([[[2.0, 3.0]]]), mx.array([[[4.0, 5.0]]]), mx.zeros((1, 1, 2)), 0
    )
    assert out.tolist() == [[[12.0, 23.0]]]
    assert backbone.inputs[0][:, :1].tolist() == [[[4.0, 5.0]]]


def test_speaker_condition_is_cached_after_load_and_refreshed_on_reload(config):
    config["conditioners"] = [
        {"name": "num_speakers", "type": "lut", "lut": {"n_bins": 31, "dim": 4}}
    ]
    model = Model(config)
    weights = dict(tree_flatten(model.parameters()))
    weights["flow_lm.num_speakers.embed.weight"] = mx.full((32, 4), 2.0)
    weights["flow_lm.num_speakers.output_proj.weight"] = mx.ones((8, 4))
    project = SpeakerCountConditioner.__call__
    with patch.object(
        SpeakerCountConditioner, "__call__", autospec=True, side_effect=project
    ) as calls:
        model.load_weights(list(weights.items()), strict=True)
        assert calls.call_count == 1
        assert mx.array_equal(
            model.flow_lm._conditioning_embedding, mx.full((1, 8), 8.0)
        ).item()

        # Prefill and successive audio steps only add the materialized vector.
        state = model.init_state()
        model._run_flow_lm(state, text_tokens=mx.array([[1, 2]]))
        for _ in range(3):
            latent, _ = model._run_flow_lm(
                state, backbone_input_latents=mx.zeros((1, 1, 4))
            )
            mx.eval(latent)
        assert calls.call_count == 1
        assert set(dict(tree_flatten(model.parameters()))) == set(weights)

        model.load_weights(
            [("flow_lm.num_speakers.output_proj.weight", mx.full((8, 4), 3.0))],
            strict=False,
        )
        assert calls.call_count == 2
        assert mx.array_equal(
            model.flow_lm._conditioning_embedding, mx.full((1, 8), 24.0)
        ).item()


def test_lsd_matches_reference_update():
    x = mx.array([[1.0, 2.0]])
    calls = []

    def velocity(s, t, current):
        calls.append((s.item(), t.item()))
        return current + s + t

    expected = x
    for i in range(4):
        expected = expected + (expected + i / 4 + (i + 1) / 4) / 4
    actual = lsd_decode(velocity, x, 4)
    assert mx.array_equal(actual, expected).item()
    assert calls == [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


def test_stored_speaker_latents_are_projected(config, tmp_path):
    model = Model(config)
    model.speaker_proj_weight = mx.eye(8)
    voice = tmp_path / "voice.safetensors"
    latents = mx.arange(16).reshape(1, 8, 2).astype(mx.float32)
    mx.save_safetensors(str(voice), {"speaker_wavs": latents})
    with patch.object(Model, "_run_flow_lm") as run:
        model.get_state_for_audio_prompt(voice)
    assert mx.array_equal(
        run.call_args.kwargs["audio_conditioning"], latents.transpose(0, 2, 1)
    ).item()


def test_overrides_restore_when_prompt_loading_fails(config):
    model = Model(config)
    with pytest.raises(ValueError, match="positive integer"):
        next(model.generate("Hello", lsd_decode_steps=0, temperature=0.2))
    assert (model.temp, model.lsd_decode_steps) == (0.7, 1)


def test_native_weights_are_unchanged(config):
    model = Model(config)
    weights = dict(tree_flatten(model.parameters()))
    assert model.sanitize(weights) is weights
    model.load_weights(list(weights.items()), strict=True)


def upstream_weights(model, weights):
    names = _seanet_names("mimi.decoder", model.config.mimi.seanet, False)
    inverse = sorted(((v, k) for k, v in names.items()), key=lambda pair: -len(pair[0]))
    raw = {}
    for key, value in weights.items():
        if key.startswith(("mimi.encoder", "mimi.downsample")):
            continue
        target = key
        if key.startswith("mimi.decoder."):
            for dest, source in inverse:
                if key.startswith(dest + "."):
                    target = source + key[len(dest) :]
                    break
            target = target.replace(".conv.conv.", ".conv.").replace(
                ".convtr.convtr.", ".convtr."
            )
            if value.ndim == 3:
                value = (
                    value.transpose(2, 0, 1)
                    if ".convtr." in target
                    else value.transpose(0, 2, 1)
                )
        elif value.ndim == 3:
            value = value.transpose(0, 2, 1)
        if target.startswith("mimi.upsample."):
            target = target.replace(".convtr.convtr.convtr.", ".convtr.convtr.")
        target = target.replace(
            "flow_lm.conditioner.",
            "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        )
        for name in ("num_speakers", "duration_delta"):
            target = target.replace(
                f"flow_lm.{name}.",
                f"flow_lm.condition_provider.conditioners.{name}.",
            )
        raw[target] = value
    return raw


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_upstream_decoder_mapping_and_optional_encoders(config, dtype):
    model = Model(config)
    weights = {k: v.astype(dtype) for k, v in tree_flatten(model.parameters())}
    raw = upstream_weights(model, weights)
    sanitized = convert_weights(model, raw)
    assert model.mimi.encoder is None
    for key, value in sanitized.items():
        assert value.dtype == weights[key].dtype, key
        assert mx.array_equal(value, weights[key]).item(), key
    model.load_weights(list(sanitized.items()), strict=True)
    with pytest.raises(ValueError, match="omits the speaker encoder"):
        model._encode_audio(mx.zeros((1, 1, 16)))
    assert model.sanitize(sanitized) is sanitized


@pytest.mark.parametrize("num_layers", [6, 9])
@pytest.mark.parametrize("model_type", [None, "gradium_phonon_tts"])
def test_official_checkpoint_loads_and_converts(
    config, tmp_path, num_layers, model_type
):
    from mlx_audio.tts.utils import load

    config["flow_lm"].update(num_layers=num_layers, time_rms_norm="rms", gelu="erf")
    config["mimi"].update(encoder_enabled=False, gelu="erf")
    config["conditioners"] = [
        {"name": "num_speakers", "type": "lut", "lut": {"n_bins": 31, "dim": 4}},
        {
            "name": "duration_delta",
            "type": "continuous",
            "continuous": {"dim": 4, "scale_factor": 1000.0, "max_period": 10000.0},
        },
    ]
    config["fuser"] = {"sum": ["num_speakers", "duration_delta"]}
    model = Model(config)
    weights = {
        key: value.astype(mx.bfloat16) if key.startswith("flow_lm.") else value
        for key, value in tree_flatten(model.parameters())
    }
    raw = upstream_weights(model, weights)
    raw["flow_lm.condition_provider.conditioners.duration_delta.learnt_padding"] = (
        mx.zeros((1, 1, 8))
    )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), raw)
    mx.save_safetensors(
        str(tmp_path / "default-voice.safetensors"),
        {"speaker_wavs": mx.ones((1, 8, 2))},
    )
    config.pop("model_path")
    if model_type is not None:
        config["model_type"] = model_type
    (tmp_path / "config.json").write_text(json.dumps(config))

    # Resolve the architecture from the config, independently of the folder name.
    loaded = load(tmp_path, strict=True)
    loaded_weights = dict(tree_flatten(loaded.parameters()))
    assert loaded_weights.keys() == weights.keys()
    for key, value in weights.items():
        assert loaded_weights[key].dtype == value.dtype, key
        assert mx.array_equal(loaded_weights[key], value).item(), key
    assert loaded.mimi.encoder is None
    assert loaded._resolve_audio_prompt(None, None, False) == (
        tmp_path / "default-voice.safetensors"
    )
    assert loaded._resolve_audio_prompt("default", None, False) == (
        tmp_path / "default-voice.safetensors"
    )
    expected_condition = loaded.flow_lm.num_speakers(mx.array([1]))
    expected_condition += loaded.flow_lm.duration_delta(mx.array([0.0]))
    assert mx.array_equal(
        loaded.flow_lm._conditioning_embedding, expected_condition
    ).item()

    # Servers preload on one thread and synthesize on an audio worker. Private
    # cached arrays must not retain unevaluated work on the loading stream.
    def generate_on_worker():
        mx.random.seed(0)
        audio = next(loaded.generate("Hello world.")).audio
        mx.eval(audio)
        return audio

    with ThreadPoolExecutor(max_workers=1) as executor:
        audio = executor.submit(generate_on_worker).result(timeout=30)
    assert audio.size > 0 and mx.all(mx.isfinite(audio)).item()

    destination = tmp_path.with_name(tmp_path.name + "-converted")
    convert(str(tmp_path), mlx_path=str(destination))
    converted = load(destination, strict=True)
    mx.random.seed(0)
    converted_audio = next(converted.generate("Hello world.")).audio
    assert mx.array_equal(audio, converted_audio).item()


@pytest.mark.parametrize("mode", ["rms", "rms_var"])
def test_timestep_normalization_matches_formula(config, mode):
    config["flow_lm"].update(flow_dim=4, time_rms_norm=mode)
    norm = Model(config).flow_lm.flow_net.time_embed[0].mlp[3]
    norm.alpha = mx.full((4,), 2.0)
    values = np.array([[1.0, 2.0, 4.0, 8.0]], dtype=np.float32)
    denominator = (
        np.mean(values**2, axis=-1, keepdims=True)
        if mode == "rms"
        else np.var(values, axis=-1, keepdims=True, ddof=1)
    )
    expected = values * 2 / np.sqrt(denominator + 1e-5)
    np.testing.assert_allclose(np.array(norm(mx.array(values))), expected, rtol=1e-6)


def test_continuous_conditioning_matches_sinusoidal_formula():
    conditioner = ContinuousConditioner(dim=4, output_dim=4, scale_factor=2.0)
    values = mx.array([0.0, 1.0])
    expected = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [math.cos(2.0), math.cos(0.0002), math.sin(2.0), math.sin(0.0002)],
        ]
    )
    np.testing.assert_allclose(np.array(conditioner(values)), expected, atol=1e-6)


@pytest.mark.parametrize("gelu", ["erf", "tanh"])
def test_codec_gelu_matches_config(config, gelu):
    from mlx_audio.codec.models.mimi.modules.transformer import MlpNoGating

    config["mimi"]["gelu"] = gelu
    codec_config = Model(config).mimi.decoder_transformer.transformer.cfg
    layer = MlpNoGating(replace(codec_config, d_model=4, dim_feedforward=4))
    layer.linear1.weight = layer.linear2.weight = mx.eye(4)
    values = np.array([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float32)
    if gelu == "erf":
        expected = (
            0.5
            * values
            * (1 + np.array([[math.erf(x / math.sqrt(2)) for x in values[0]]]))
        )
    else:
        expected = (
            0.5
            * values
            * (1 + np.tanh(math.sqrt(2 / math.pi) * (values + 0.044715 * values**3)))
        )
    np.testing.assert_allclose(np.array(layer(mx.array(values))), expected, atol=1e-6)


def test_loader_does_not_merge_voice_files(config, tmp_path):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"weight": mx.ones((1,))})
    mx.save_safetensors(
        str(tmp_path / "alba.safetensors"), {"audio_prompt": mx.ones((1,))}
    )
    assert list(load_source_weights(tmp_path)) == ["weight"]


def test_runtime_loader_requires_converted_weights(config, tmp_path):
    from mlx_audio.tts.utils import load

    (tmp_path / "config.json").write_text(
        json.dumps({**config, "model_type": "pocket_tts"})
    )
    (tmp_path / "model.q8.gguf").write_bytes(b"conversion-only input")
    with pytest.raises(FileNotFoundError, match="No weight files"):
        load(str(tmp_path), strict=True)


def test_bundled_latents_work_without_encoder(config, tmp_path):
    config["mimi"]["encoder_enabled"] = False
    model = Model(config)
    assert model.mimi.encoder is None
    assert model.speaker_proj_weight.shape == (8, 8)
    assert model.sample_rate == 16
    model.speaker_proj_weight = mx.arange(64).reshape(8, 8).astype(mx.float32)
    latents = mx.arange(16).reshape(1, 8, 2).astype(mx.float32)
    voice = tmp_path / "voice.safetensors"
    mx.save_safetensors(str(voice), {"speaker_wavs": latents})
    with patch.object(Model, "_run_flow_lm") as run:
        model.get_state_for_audio_prompt(voice)
    expected = latents.transpose(0, 2, 1) @ model.speaker_proj_weight.T
    assert mx.array_equal(run.call_args.kwargs["audio_conditioning"], expected).item()
    with pytest.raises(ValueError, match="omits the speaker encoder"):
        model._encode_audio(mx.ones((1, 1, 16)))


def test_legacy_reference_audio_uses_main_encoder(config):
    model = Model(config)
    model.speaker_proj_weight = mx.eye(8)
    audio = mx.ones((1, 1, 16))
    latents = model.mimi.encode_to_latent(audio).transpose(0, 2, 1)
    # Metal may round even an identity projection, so include it in the oracle.
    expected = latents @ model.speaker_proj_weight.T
    actual = model._encode_audio(audio)
    assert mx.array_equal(actual, expected).item()


def test_prompt_duration_limits(config):
    config.update(audio_prompt_min_duration=0.5, audio_prompt_max_duration=1.0)
    model = Model(config)
    with pytest.raises(ValueError, match="at least"):
        model.get_state_for_audio_prompt(mx.ones((4,)))
    with patch.object(
        Model, "_encode_audio", return_value=mx.ones((1, 2, 8))
    ) as encode:
        model.get_state_for_audio_prompt(mx.ones((32,)))
    assert encode.call_args.args[0].shape[-1] == 16


def test_upstream_seanet_indices():
    cfg = ModelConfig.from_dict(flat_config()).mimi.seanet
    assert _seanet_names("mimi.encoder", cfg, True) == {
        "mimi.encoder.model.0": "mimi.encoder.init_conv1d",
        "mimi.encoder.model.1.block.1": "mimi.encoder.layers.0.residuals.0.block.0",
        "mimi.encoder.model.1.block.3": "mimi.encoder.layers.0.residuals.0.block.1",
        "mimi.encoder.model.3": "mimi.encoder.layers.0.downsample",
        "mimi.encoder.model.5": "mimi.encoder.final_conv1d",
    }
    assert _seanet_names("mimi.decoder", cfg, False) == {
        "mimi.decoder.model.0": "mimi.decoder.init_conv1d",
        "mimi.decoder.model.2": "mimi.decoder.layers.0.upsample",
        "mimi.decoder.model.3.block.1": "mimi.decoder.layers.0.residuals.0.block.0",
        "mimi.decoder.model.3.block.3": "mimi.decoder.layers.0.residuals.0.block.1",
        "mimi.decoder.model.5": "mimi.decoder.final_conv1d",
    }


def test_conversion_infers_source_packing_without_quantization_metadata(config):
    # Packing inference belongs to conversion; normal loading uses exported metadata.
    config["flow_lm"].update(d_model=64, lut_dim=64, flow_dim=64, dim_feedforward=128)
    model = Model(config)
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        class_predicate=lambda p, m: p.startswith("flow_lm.transformer")
        and hasattr(m, "to_quantized"),
    )
    weights = dict(tree_flatten(model.parameters()))
    other = Model(config)
    _prepare_quantized_modules(other, weights)
    other.load_weights(list(other.sanitize(weights).items()), strict=True)
    layer = other.flow_lm.transformer.layers[0].linear1
    assert isinstance(layer, nn.QuantizedLinear)
    assert (layer.bits, layer.group_size) == (8, 32)
    x = mx.ones((1, 64))
    assert mx.array_equal(
        layer(x), model.flow_lm.transformer.layers[0].linear1(x)
    ).item()


def test_missing_decoder_weights_still_fail_strict_loading(config):
    model = Model(config)
    weights = dict(tree_flatten(model.parameters()))
    weights.pop("mimi.decoder.init_conv1d.conv.conv.weight")
    with pytest.raises(ValueError, match="Missing"):
        model.load_weights(list(model.sanitize(weights).items()), strict=True)


@pytest.mark.parametrize(
    "key",
    [
        "flow_lm.condition_provider.conditioners.transcript_in_segment.learnt_padding",
        "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding",
        "flow_lm.conditioner.learnt_padding",
    ],
)
def test_unused_padding_weights_are_discarded(config, key):
    model = Model(config)
    weights = dict(tree_flatten(model.parameters()))
    expected_keys = set(weights)
    weights[key] = mx.full((1, 1, 8), 0.25)
    weights = convert_weights(model, weights)
    assert set(weights) == expected_keys
    model.load_weights(list(weights.items()), strict=True)


def test_prefill_skips_sampling_and_preserves_random_sequence(config):
    model = Model(config)
    mx.random.seed(42)
    mx.random.normal((1, model.flow_lm.ldim))
    mx.random.normal((1, model.flow_lm.ldim))
    expected_noise = mx.random.normal((1, model.flow_lm.ldim))
    mx.random.seed(42)
    with patch.object(FlowLMModel, "__call__") as sample:
        state = model.init_state()
        model._run_flow_lm(state, audio_conditioning=mx.ones((1, 3, 8)))
        model._run_flow_lm(state, text_tokens=mx.array([[1, 2]]))
    sample.assert_not_called()
    assert model._get_flow_cache_num_frames(state) == 5
    assert mx.array_equal(
        mx.random.normal((1, model.flow_lm.ldim)), expected_noise
    ).item()


def test_prefill_preserves_legacy_float32_promotion(config):
    model = Model(config)
    model.flow_lm.conditioner.embed.weight = (
        model.flow_lm.conditioner.embed.weight.astype(mx.bfloat16)
    )
    tokens = mx.array([[1, 2]])
    old_state, new_state = model.init_state(), model.init_state()
    model.flow_lm(
        sequence=mx.zeros((1, 0, model.flow_lm.ldim)),
        text_embeddings=model.flow_lm.conditioner(TokenizedText(tokens)),
        cache=old_state["flow_cache"],
        lsd_decode_steps=1,
        temp=0.7,
        noise_clamp=None,
        eos_threshold=-4,
    )
    model._run_flow_lm(new_state, text_tokens=tokens)
    for old, new in zip(old_state["flow_cache"], new_state["flow_cache"]):
        assert new.keys.dtype == mx.float32
        assert mx.array_equal(old.keys, new.keys).item()
        assert mx.array_equal(old.values, new.values).item()


def test_bundled_voice_default_and_unknown_voice(config, tmp_path):
    config["mimi"]["encoder_enabled"] = False
    voices = tmp_path / "voices"
    voices.mkdir()
    for name in ("Toby", "Freya"):
        mx.save_safetensors(
            str(voices / f"{name}.safetensors"), {"emb": mx.ones((1, 2, 8))}
        )
    model = Model(config)
    assert model._resolve_audio_prompt(None, None, False).name == "Freya.safetensors"
    assert model._resolve_audio_prompt("freya", None, False).name == "Freya.safetensors"
    with pytest.raises(ValueError, match="Available voices"):
        model._resolve_audio_prompt("af_heart", None, False)


@pytest.mark.parametrize(
    "conversion", [{}, {"quantize": True}, {"dtype": "float16"}, {"dequantize": True}]
)
def test_conversion_preserves_assets_and_packed_weights(config, tmp_path, conversion):
    from mlx_audio.tts.utils import load

    config["flow_lm"].update(d_model=64, lut_dim=64, flow_dim=64, dim_feedforward=128)
    config.update(temp=0.8, weights_name="model.safetensors")
    model = Model(config)
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        class_predicate=lambda p, m: p.startswith("flow_lm.transformer")
        and hasattr(m, "to_quantized"),
    )
    # Match GGUF imports: packed uint32 weights with float16 scales and other tensors.
    source_weights = {
        k: v.astype(mx.float16) if mx.issubdtype(v.dtype, mx.floating) else v
        for k, v in tree_flatten(model.parameters())
    }
    # Training-only padding must be excluded from converted checkpoints.
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            **source_weights,
            "flow_lm.conditioner.learnt_padding": mx.ones((1, 1, 64), dtype=mx.float16),
        },
    )
    config.pop("model_path")
    (tmp_path / "config.json").write_text(json.dumps(config))
    voices = tmp_path / "voices"
    voices.mkdir()
    mx.save_safetensors(str(voices / "Freya.safetensors"), {"emb": mx.ones((1, 2, 64))})
    dest = tmp_path.with_name(tmp_path.name + "-converted")
    convert(str(tmp_path), mlx_path=str(dest), **conversion)
    with (
        patch(
            "mlx_audio.tts.models.pocket_tts.convert.load_source_weights",
            side_effect=AssertionError("conversion-only source loader"),
        ),
        patch(
            "mlx_audio.tts.models.pocket_tts.convert._prepare_quantized_modules",
            side_effect=AssertionError("conversion-only packing inference"),
        ),
    ):
        loaded = load(str(dest), strict=True)
    if not conversion:
        loaded_weights = dict(tree_flatten(loaded.parameters()))
        assert loaded_weights.keys() == source_weights.keys()
        for key, weight in source_weights.items():
            assert loaded_weights[key].dtype == weight.dtype, key
            assert mx.array_equal(loaded_weights[key], weight).item(), key
    assert loaded.temp == 0.8
    assert "flow_lm.conditioner.learnt_padding" not in dict(
        tree_flatten(loaded.parameters())
    )
    assert (
        loaded._resolve_audio_prompt(None, None, False)
        == dest / "voices" / "Freya.safetensors"
    )
    layer = loaded.flow_lm.transformer.layers[0].linear1
    if conversion.get("dequantize"):
        assert isinstance(layer, nn.Linear)
        assert loaded.config.quantization is None
    else:
        assert isinstance(layer, nn.QuantizedLinear)
        assert layer.weight.dtype == mx.uint32
        assert (layer.bits, layer.group_size) == (8, 32)
    assert loaded.flow_lm.conditioner.prepare("Hello world").tokens.shape == (1, 2)

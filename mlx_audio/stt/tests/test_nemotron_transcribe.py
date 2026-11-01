"""Small numerical and integration tests for the consolidated SpeechLM port."""

import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx_audio.stt.models.nemotron_transcribe import Model, ModelConfig
from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import (
    resolve_prompt,
    speaker_segments,
)
from mlx_audio.stt.models.nemotron_transcribe.perception import normalize_features


def tiny_config(features=False):
    encoder = dict(
        feat_in=8,
        d_model=16,
        n_layers=2,
        n_heads=2,
        subsampling_factor=2,
        ff_expansion=2,
        qk_norm=True,
        rotary_fraction=0.5,
    )
    return ModelConfig(
        llm_config=dict(
            model_type="nemotron_h",
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            max_position_embeddings=1024,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_bias=False,
            mamba_num_heads=2,
            mamba_head_dim=8,
            mamba_proj_bias=False,
            ssm_state_size=32 if features else 8,
            conv_kernel=4,
            n_groups=1,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            layers_block_type=["mamba", "attention", "moe"],
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=24,
            moe_shared_expert_intermediate_size=32,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
        ),
        perception=dict(
            output_dim=16,
            preprocessor=dict(features=8),
            modality_adapter={"_target_": "nemo.IdentityConnector", "d_model": 16},
        ),
        mtp=(
            dict(enabled=True, use_repeated_layer=True, hybrid_override_pattern="*E")
            if features
            else {}
        ),
        ctc_timestamps=(
            dict(
                decoder_config=dict(
                    feat_in=16,
                    num_classes=64,
                    n_layers=2,
                    n_heads=2,
                    ff_expansion=0.5,
                    qk_norm=True,
                    rotary_fraction=0.5,
                    upsampling_factor=2,
                    residual_scale=0.1,
                )
            )
            if features
            else {}
        ),
        pe_encoder_config=dict(
            asr_encoder_type="transformer",
            asr_encoder_cfg=encoder,
            speaker_feature_config_version=1,
            speaker_feature_mode="thresholded",
            speaker_activity_threshold=0.5,
            online_inference_length=4,
            chunk_left_context=1,
            chunk_right_context=1,
            diar_spkcache_len=8,
            diar_fifo_len=2,
            diar_spkcache_update_period=2,
            diarization_model_cfg=dict(
                high_resolution=True,
                preprocessor=dict(features=8),
                encoder={**encoder, "qk_norm": False, "rotary_fraction": 1.0},
                sortformer_modules=dict(
                    num_spks=2,
                    fc_d_model=16,
                    tf_d_model=8,
                    chunk_left_context=0,
                    chunk_right_context=0,
                ),
            ),
        ),
    )


def tiny_model(features=False):
    mx.random.seed(5)
    model = Model(tiny_config(features))
    model.eval()
    return model


def test_config_registry_and_detection():
    from mlx_audio.convert import Domain, detect_model_domain, get_model_type
    from mlx_audio.registry import classify_model

    cfg = asdict(tiny_config())
    cfg["model_type"] = "nemo_speechlm"
    assert ModelConfig.from_dict(cfg).text_config.hybrid_override_pattern == [
        "M",
        "*",
        "E",
    ]
    assert classify_model("nemo_speechlm") == "stt"
    assert detect_model_domain(cfg, "checkpoint") == Domain.STT
    assert get_model_type(cfg, "checkpoint", Domain.STT) == "nemotron_transcribe"


@pytest.mark.parametrize("features", [False, True])
def test_source_weight_mapping_and_idempotence(features):
    model = tiny_model(features)
    original = dict(tree_flatten(model.parameters()))
    source = {}
    for key, value in original.items():
        key = key.replace("llm.backbone.embeddings.", "llm.model.embed_tokens.")
        key = key.replace("llm.backbone.norm_f.", "llm.model.norm.")
        key = key.replace("llm.backbone.layers.", "llm.model.layers.")
        key = key.replace(".ffn.linear1.", ".ffn.net.0.").replace(
            ".ffn.linear2.", ".ffn.net.3."
        )
        key = key.replace(".preprocessor.", ".preprocessor.featurizer.")
        if key.endswith((".A_log", ".D", ".dt_bias")):
            before, after = key.rsplit(".", 1)
            key = before + "._fp32_params." + after
        if ".switch_mlp." in key:
            key = key.replace(".switch_mlp.fc1.weight", ".experts.gate_and_up_projs")
            key = key.replace(".switch_mlp.fc2.weight", ".experts.down_projs")
            value = value.transpose(0, 2, 1)
        elif key.endswith((".conv1d.weight", ".subpixel_upsample.weight")):
            value = value.transpose(0, 2, 1)
        elif key == "ctc_timestamp.decoder.upsample.weight":
            value = value.transpose(2, 0, 1)
        elif key == "ctc_timestamp.decoder.decoder_layers.0.weight":
            value = value[..., None]
        source[key] = value
    source["ctc_timestamp.tokenizer.model_proto"] = mx.zeros(5, dtype=mx.uint8)
    source["ctc_timestamp.decoder.upsample_factor"] = mx.array(2)
    source["llm.lm_head._extra_state"] = mx.zeros(0)
    converted = model.sanitize(source)
    assert converted.keys() == original.keys()
    for weights in [converted, model.sanitize(converted)]:
        for key, value in weights.items():
            np.testing.assert_array_equal(np.array(value), np.array(original[key]))
    model.load_weights(list(converted.items()), strict=True)


def test_nemo_encoded_nonfinite_config():
    config = tiny_config()
    config.llm_config["time_step_limit"] = [0.0, {"__float__": "Infinity"}]
    assert config.text_config.time_step_limit == (0.0, float("inf"))
    assert config.llm_config["time_step_limit"][1] == {"__float__": "Infinity"}
    model = Model(config)
    assert mx.all(mx.isfinite(model(mx.array([[1, 2, 3]])))).item()
    # Saving a converted checkpoint must retain standards-compliant JSON.
    json.dumps(asdict(config), allow_nan=False)


def test_cached_backbone_and_chunked_audio_embeddings():
    with mx.stream(mx.cpu):
        model = tiny_model()
        tokens = mx.array([[1, 8, 4, 5, 9, 2, 7]])
        embeddings = model.llm.backbone.embeddings(tokens)
        whole = model(tokens)
        np.testing.assert_allclose(
            np.array(whole), np.array(model(input_embeddings=embeddings)), atol=1e-6
        )
        cache = model.make_cache()
        pieces = [
            model(tokens[:, :3], cache=cache, input_embeddings=embeddings[:, :3]),
            model(tokens[:, 3:5], cache=cache, input_embeddings=embeddings[:, 3:5]),
            model(tokens[:, 5:], cache=cache),
        ]
        np.testing.assert_allclose(
            np.array(mx.concatenate(pieces, axis=1)), np.array(whole), atol=2e-5
        )


def test_normalization_and_thresholded_speaker_fusion():
    torch = pytest.importorskip("torch")
    with mx.stream(mx.cpu):
        model = tiny_model()
        x = np.random.default_rng(4).normal(size=(1, 8, 25)).astype(np.float32)
        t = torch.from_numpy(x)
        expected = (t - t.mean(-1, keepdim=True)) / (t.std(-1, keepdim=True) + 1e-5)
        np.testing.assert_allclose(
            np.array(normalize_features(mx.array(x))), expected.numpy(), atol=5e-7
        )
        for n in [1, 3]:
            assert mx.all(normalize_features(mx.ones((1, 8, n))) == 0).item()
        states = np.random.default_rng(3).normal(size=(1, 3, 16)).astype(np.float32)
        probs = np.array([[[0.5, 0.6], [0.3, 0.5], [0.9, 0.1]]], np.float32)
        positions = torch.arange(2)[:, None]
        angles = positions * torch.exp(torch.arange(0, 16, 2) * -(np.log(10000) / 16))
        kernel = torch.stack([angles.sin(), angles.cos()], -1).reshape(2, 16)
        expected = torch.nn.functional.layer_norm(torch.from_numpy(states), (16,))
        expected += (
            torch.nn.functional.layer_norm(
                torch.from_numpy((probs > 0.5).astype(np.float32)), (2,)
            )
            @ kernel
        )
        actual = model.perception.encoder.fuse(mx.array(states), mx.array(probs))
        np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=1e-6)


@pytest.mark.parametrize("frames", [1, 7, 8, 9, 16, 17, 41])
def test_windowed_perception_lengths_and_finiteness(frames):
    model = tiny_model()
    features = mx.random.normal((1, 8, frames))
    output = model.perception.encoder(features)
    assert output.shape == (1, (frames + 1) // 2, 16)
    assert mx.all(mx.isfinite(output)).item()


def test_left_context_not_added_to_speaker_history():
    model = tiny_model().perception.encoder.diarization_model
    state = model.init_streaming_state()
    features = mx.random.normal((1, 8, 10))
    chunk, _ = model.encoder.pre_encode(features, mx.array([10]))
    actual = model.streaming_step(
        features, state, central_frames=6, left_context_frames=2
    )
    assert actual.shape == (6, 2)
    history = mx.concatenate([state.spkcache, state.fifo], axis=1)
    np.testing.assert_allclose(np.array(history), np.array(chunk[:, 1:4]), atol=1e-7)


def test_native_prompts_and_speaker_markup():
    assert resolve_prompt("asr", None, None, None, "<|audio|>").endswith(" <|audio|>")
    assert "French" in resolve_prompt("translate", None, "French", None, "<|audio|>")
    assert resolve_prompt("qa", None, None, None, "<|audio|>") == "<|audio|>"
    assert "Neywa" in resolve_prompt("asr", None, None, ["Neywa"], "<|audio|>")
    with pytest.raises(ValueError):
        resolve_prompt("translate", None, None, None, "<|audio|>")
    with pytest.raises(ValueError):
        resolve_prompt("asr", "<|audio|><|audio|>", None, None, "<|audio|>")
    assert speaker_segments("<spk:0>Hello <ev:laugh> <spk:1>Hi") == [
        {"speaker": 0, "text": "Hello <ev:laugh>"},
        {"speaker": 1, "text": "Hi"},
    ]


def tokenizer_fixture(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<unk>": 0,
        "<|audio|>": 64,
        "<|im_end|>": 11,
        "hello": 12,
        "<spk:0>": 13,
        "<ev:laugh>": 14,
    }
    vocab.update({f"t{i}": i for i in range(65) if i not in vocab.values()})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|audio|>", "<spk:0>", "<ev:laugh>"]}
    )
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("features", [False, True])
def test_strict_load_and_quantized_roundtrip(tmp_path, quantized, features):
    from mlx_audio.convert import build_quant_predicate
    from mlx_audio.stt.utils import load_model

    config = tiny_config(features)
    config.llm_config.update(
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
        mamba_num_heads=8,
    )
    config.perception["output_dim"] = 64
    model = Model(config)
    model.eval()
    cfg = asdict(model.config)
    tokenizer_fixture(tmp_path)
    if features:
        tokenizer_fixture(tmp_path / "alignment_tokenizer")
    if quantized:
        nn.quantize(
            model, group_size=64, bits=4, class_predicate=build_quant_predicate(model)
        )
        cfg["quantization"] = {"bits": 4, "group_size": 64}
        assert not isinstance(model.perception.proj, nn.QuantizedLinear)
        if features:
            assert isinstance(model.llm.mtp.layers[0].eh_proj, nn.QuantizedLinear)
            assert not isinstance(
                model.ctc_timestamp.decoder.decoder_layers[0], nn.QuantizedLinear
            )
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    restored = load_model(tmp_path, strict=True)
    if features:
        assert restored._alignment_tokenizer is not None
        assert restored.llm.mtp is not None and restored.ctc_timestamp is not None
    tokens = mx.array([[1, 2, 3]])
    np.testing.assert_allclose(
        np.array(restored(tokens)), np.array(model(tokens)), atol=1e-6
    )
    if features:
        from mlx_audio.lm.models.cache import KVCache

        embeddings = model.llm.backbone.embeddings(tokens)
        hidden = model.llm.backbone(tokens)
        np.testing.assert_allclose(
            np.array(restored.llm.mtp(embeddings, hidden, [KVCache()])),
            np.array(model.llm.mtp(embeddings, hidden, [KVCache()])),
            atol=1e-6,
        )
        states = mx.random.normal((1, 5, 16))
        np.testing.assert_allclose(
            np.array(restored.ctc_timestamp.decoder(states)),
            np.array(model.ctc_timestamp.decoder(states)),
            atol=1e-6,
        )


def test_audio_placeholder_outside_embedding_vocab_and_streaming_eos(
    tmp_path, monkeypatch
):
    import mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe as implementation

    model = tiny_model()
    model._tokenizer = tokenizer_fixture(tmp_path)
    tokens, embeddings = model._prepare_inputs(mx.zeros(2000), "<|audio|>", None, False)
    assert len(tokens) == len(embeddings)
    assert mx.max(tokens).item() < model.config.llm_config["vocab_size"]
    captured = {}

    def generate(prompt, model, **kwargs):
        captured.update(kwargs)
        for token in [13, 12, 14, 11, 12]:
            yield token, None

    monkeypatch.setattr(implementation, "generate_step", generate)
    result = model.generate(mx.zeros(2000), prompt="<|audio|>")
    assert result.generation_tokens == 3
    assert "<spk:0>" in result.text and "<ev:laugh>" in result.text
    assert result.text.count("hello") == 1
    stream = list(model.generate(mx.zeros(2000), prompt="<|audio|>", stream=True))
    assert stream[-1].is_final
    assert all(part.start_time is None and part.end_time is None for part in stream)
    assert "".join(r.text for r in stream).strip() == result.text
    assert captured["input_embeddings"].shape[0] == result.prompt_tokens


def test_file_audio_is_resampled_and_downmixed(tmp_path):
    import wave

    from mlx_audio.stt.utils import load_audio

    path = tmp_path / "stereo.wav"
    samples = np.arange(4000, dtype=np.float32) / 32000
    tone = (np.sin(2 * np.pi * 440 * samples) * 16000).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(32000)
        f.writeframes(np.stack([tone, tone // 2], axis=-1).tobytes())
    model = tiny_model()
    model._tokenizer = tokenizer_fixture(tmp_path)
    actual_tokens, actual_embeddings = model._prepare_inputs(
        path, "<|audio|>", None, False
    )
    expected_tokens, expected_embeddings = model._prepare_inputs(
        load_audio(str(path), sr=16000), "<|audio|>", None, False
    )
    np.testing.assert_array_equal(np.array(actual_tokens), np.array(expected_tokens))
    np.testing.assert_allclose(
        np.array(actual_embeddings), np.array(expected_embeddings), atol=1e-6
    )


@pytest.mark.parametrize("features", [False, True])
def test_conversion_keeps_mamba_and_router_parameters_in_float32(tmp_path, features):
    from mlx_audio.convert import convert
    from mlx_audio.stt.utils import load_model

    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    model = tiny_model(features)
    weights = dict(tree_flatten(model.parameters()))
    for key, value in weights.items():
        if not model.cast_predicate(key):
            weights[key] = mx.full(value.shape, 1.1234567)
    (source / "config.json").write_text(json.dumps(asdict(model.config)))
    tokenizer_fixture(source)
    if features:
        tokenizer_fixture(source / "alignment_tokenizer")
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    convert(str(source), str(target), dtype="bfloat16", model_domain="stt")
    restored = load_model(target, strict=True)
    if features:
        assert restored._alignment_tokenizer is not None
    converted = dict(tree_flatten(restored.parameters()))
    for key, value in weights.items():
        if not model.cast_predicate(key):
            assert converted[key].dtype == mx.float32
            np.testing.assert_array_equal(np.array(converted[key]), np.array(value))
        else:
            assert converted[key].dtype == mx.bfloat16


def test_bfloat16_experts_use_float32_routing():
    torch = pytest.importorskip("torch")
    with mx.stream(mx.cpu):
        model = tiny_model()
        key = "llm.backbone.layers.2.mixer.gate.weight"
        weight = mx.random.normal((4, 16)).astype(mx.bfloat16)
        converted = model.sanitize({key: weight})
        model.load_weights(list(converted.items()), strict=False)
        assert converted[key].dtype == mx.float32
        gate = model.llm.layers[2].mixer.gate
        x = mx.random.normal((1, 10, 16)).astype(mx.bfloat16)
        indices, scores = gate(x)
        expected = torch.sigmoid(
            torch.from_numpy(np.array(x.astype(mx.float32)))
            @ torch.from_numpy(np.array(weight.astype(mx.float32))).T
        )
        chosen, expected_indices = torch.topk(expected, 2, dim=-1)
        chosen = chosen / chosen.sum(-1, keepdim=True) * 2.5
        order = mx.argsort(indices, axis=-1)
        expected_order = expected_indices.argsort(-1)
        np.testing.assert_array_equal(
            np.array(mx.take_along_axis(indices, order, -1)),
            expected_indices.gather(-1, expected_order).numpy(),
        )
        np.testing.assert_allclose(
            np.array(mx.take_along_axis(scores, order, -1)),
            chosen.gather(-1, expected_order).numpy(),
            atol=2e-7,
        )


def test_bfloat16_mamba_scan_and_gate_match_float32_reference():
    torch = pytest.importorskip("torch")
    from mlx_audio.lm.models.cache import ArraysCache
    from mlx_audio.stt.models.nemotron_transcribe.language_model import GatedRMSNorm

    with mx.stream(mx.cpu):
        mixer = tiny_model().llm.layers[0].mixer
        mixer.D = mx.array([1.1234567, 0.9876543])
        x = mx.random.normal((1, 5, 16)).astype(mx.bfloat16)
        B = mx.random.normal((1, 5, 8)).astype(mx.bfloat16)
        C = mx.random.normal((1, 5, 8)).astype(mx.bfloat16)
        dt = mx.random.normal((1, 5, 2)).astype(mx.bfloat16)
        tensor = lambda v: torch.from_numpy(np.array(v.astype(mx.float32)))
        xt, bt, ct = tensor(x).reshape(1, 5, 2, 8), tensor(B), tensor(C)
        delta = torch.nn.functional.softplus(tensor(dt) + tensor(mixer.dt_bias))
        decay = torch.exp(-torch.exp(tensor(mixer.A_log)) * delta)
        state = torch.zeros(1, 2, 8, 8)
        expected = []
        for i in range(5):
            state = (
                state * decay[:, i, :, None, None]
                + xt[:, i, :, :, None]
                * delta[:, i, :, None, None]
                * bt[:, i, None, None, :]
            )
            expected.append(
                (state * ct[:, i, None, None, :]).sum(-1)
                + xt[:, i] * tensor(mixer.D)[None, :, None]
            )
        expected = torch.stack(expected, 1).reshape(1, 5, 16).to(torch.bfloat16).float()
        cache = ArraysCache(size=2)
        y = mixer._ssm(x, B, C, dt, cache, None)
        np.testing.assert_allclose(
            np.array(y.astype(mx.float32)), expected.numpy(), atol=1e-5
        )
        np.testing.assert_allclose(np.array(cache[1]), state.numpy(), atol=2e-5)
        gate = mx.random.normal(y.shape).astype(mx.bfloat16)
        norm = GatedRMSNorm(16, 8, 1e-5)
        z = (tensor(y) * torch.nn.functional.silu(tensor(gate))).reshape(1, 5, 2, 8)
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)
        expected = z.reshape(1, 5, 16).to(torch.bfloat16).float()
        np.testing.assert_array_equal(
            np.array(norm(y, gate).astype(mx.float32)), expected.numpy()
        )


@pytest.mark.parametrize("keep", [1, 2, 4, 5])
@pytest.mark.parametrize("bfloat16", [False, True])
def test_hybrid_verification_rollback(keep, bfloat16):
    # Compare a rejected verification suffix to a cache that never saw it.
    with mx.stream(mx.gpu if bfloat16 else mx.cpu):
        model = tiny_model(True)
        if bfloat16:
            model.load_weights(
                [
                    (
                        key,
                        (
                            value.astype(mx.bfloat16)
                            if model.cast_predicate(key)
                            else value
                        ),
                    )
                    for key, value in tree_flatten(model.parameters())
                ]
            )
        tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        actual, expected = model.make_cache(), model.make_cache()
        model(tokens[:, :3], cache=actual)
        model(tokens[:, :3], cache=expected)
        for cache in actual:
            if hasattr(cache, "recording"):
                cache.recording = True
        model(tokens[:, 3:], cache=actual)
        model.llm.finish_verification(actual, keep, 5 - keep)
        model(tokens[:, 3 : 3 + keep], cache=expected)
        tolerance = 0.025 if bfloat16 else 2e-5
        for a, b in zip(actual, expected):
            for x, y in zip(a.state, b.state):
                np.testing.assert_allclose(
                    np.array(x.astype(mx.float32)),
                    np.array(y.astype(mx.float32)),
                    atol=tolerance,
                )
            if hasattr(a, "recording"):
                assert a.conv_inputs is None and a.ssm_inputs is None
        np.testing.assert_allclose(
            np.array(model(mx.array([[9]]), cache=actual).astype(mx.float32)),
            np.array(model(mx.array([[9]]), cache=expected).astype(mx.float32)),
            atol=tolerance,
        )


@pytest.mark.parametrize("max_tokens", [0, 1, 2, 19])
@pytest.mark.parametrize("num_draft_tokens", [1, 4, 8])
def test_mtp_generation_matches_greedy(max_tokens, num_draft_tokens):
    from mlx_audio.lm.generate import generate_step
    from mlx_audio.lm.sample_utils import make_repetition_penalty
    from mlx_audio.stt.models.nemotron_transcribe.speculative import (
        speculative_generate_step,
    )

    model = tiny_model(True)
    model.config.eos_token_id = []
    tokens = mx.array([1, 2, 3, 4, 5, 6])
    embeddings = model.llm.backbone.embeddings(tokens)
    # Make the audio replacement observable to the drafter as well as target.
    embeddings[1:3] = mx.random.normal((2, 16))
    processors = [make_repetition_penalty(1.1, 8)]
    expected = [
        token
        for token, _ in generate_step(
            tokens,
            model,
            input_embeddings=embeddings,
            max_tokens=max_tokens,
            prefill_step_size=3,
            logits_processors=processors,
        )
    ]
    stats = {}
    actual = [
        token
        for token, _ in speculative_generate_step(
            tokens,
            model,
            input_embeddings=embeddings,
            max_tokens=max_tokens,
            num_draft_tokens=num_draft_tokens,
            prefill_step_size=3,
            logits_processors=processors,
            stats=stats,
        )
    ]
    assert actual == expected
    assert len(actual) == max_tokens
    assert stats["accepted_tokens"] <= stats["proposed_tokens"]


def test_mtp_all_accepted_and_eos():
    from mlx_audio.stt.models.nemotron_transcribe.speculative import (
        speculative_generate_step,
    )

    model = tiny_model(True)
    model.llm.lm_head.weight = mx.zeros_like(model.llm.lm_head.weight)
    tokens = mx.array([1, 2, 3])
    stats = {}
    kwargs = dict(
        input_embeddings=model.llm.backbone.embeddings(tokens),
        max_tokens=14,
        stats=stats,
    )
    actual = list(
        speculative_generate_step(tokens, model, num_draft_tokens=4, **kwargs)
    )
    assert [token for token, _ in actual] == [0] * 14
    assert stats["accepted_tokens"] == stats["proposed_tokens"] > 0
    assert stats["verification_steps"] < 13
    model.config.eos_token_id = [0]
    assert [
        token for token, _ in speculative_generate_step(tokens, model, **kwargs)
    ] == [0]


def test_mtp_prefill_uses_shifted_audio_embeddings(monkeypatch):
    from mlx_audio.stt.models.nemotron_transcribe.speculative import (
        speculative_generate_step,
    )

    model = tiny_model(True)
    tokens = mx.array([1, 2, 3, 4, 5])
    embeddings = mx.random.normal((5, 16))
    calls = []
    draft = model.llm.mtp

    def capture(embeds, hidden, cache):
        calls.append(embeds)
        return draft(embeds, hidden, cache)

    model.llm.mtp = capture
    result = list(
        speculative_generate_step(
            tokens,
            model,
            input_embeddings=embeddings,
            max_tokens=1,
            prefill_step_size=2,
        )
    )
    shifted = mx.concatenate(calls, axis=1)[0]
    np.testing.assert_allclose(
        np.array(shifted[:-1]), np.array(embeddings[1:]), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(shifted[-1]),
        np.array(model.llm.backbone.embeddings(mx.array(result[0][0]))),
        atol=1e-6,
    )


def test_feature_validation_and_legacy_missing_heads(tmp_path):
    from mlx_audio.stt.utils import load_model

    model = tiny_model()
    with pytest.raises(ValueError, match="MTP weights"):
        model.generate(mx.zeros(2000), speculative=True)
    with pytest.raises(ValueError, match="CTC weights"):
        model.generate(mx.zeros(2000), word_timestamps=True)
    full = tiny_model(True)
    for kwargs, message in [
        ({"temperature": 0.2}, "temperature=0"),
        ({"num_draft_tokens": 0}, "1 to 8"),
        ({"num_draft_tokens": 9}, "1 to 8"),
    ]:
        with pytest.raises(ValueError, match=message):
            full.generate(mx.zeros(2000), speculative=True, **kwargs)
    full._alignment_tokenizer = object()
    with pytest.raises(ValueError, match="transcription task"):
        full.generate(
            mx.zeros(2000), task="translate", language="French", word_timestamps=True
        )
    # Old converted files declared the heads in config but did not save them.
    (tmp_path / "config.json").write_text(json.dumps(asdict(full.config)))
    tokenizer_fixture(tmp_path)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            key: value
            for key, value in tree_flatten(full.parameters())
            if not key.startswith(("llm.mtp.", "ctc_timestamp."))
        },
    )
    restored = load_model(tmp_path, strict=True)
    assert restored.llm.mtp is None and restored.ctc_timestamp is None
    partial = dict(tree_flatten(full.parameters()))
    partial.pop("llm.mtp.layers.0.eh_proj.weight")
    with pytest.raises(ValueError, match="Incomplete llm.mtp"):
        full.sanitize(partial)


def test_ctc_head_matches_torch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    with mx.stream(mx.cpu):
        model = tiny_model(True)
        decoder = model.ctc_timestamp.decoder
        tensor = lambda x: torch.from_numpy(np.array(x))
        x = mx.random.normal((1, 7, 16))
        upsampled = F.conv_transpose1d(
            tensor(x).transpose(1, 2),
            tensor(decoder.upsample.weight).permute(2, 0, 1),
            tensor(decoder.upsample.bias),
            stride=2,
        ).transpose(1, 2)

        def norm(x, layer):
            return F.layer_norm(
                x, (x.shape[-1],), tensor(layer.weight), tensor(layer.bias), layer.eps
            )

        def rope(x):
            dim = 2  # head width 8, rotary_fraction .5 => 4 rotary channels
            frequencies = 10000 ** (-torch.arange(dim).float() / dim)
            theta = torch.arange(x.shape[2])[:, None] * frequencies[None]
            a, b = x[..., :dim], x[..., dim : 2 * dim]
            return torch.cat(
                [
                    a * theta.cos() - b * theta.sin(),
                    a * theta.sin() + b * theta.cos(),
                    x[..., 2 * dim :],
                ],
                -1,
            )

        h = norm(upsampled, decoder.transformer.embed_norm)
        for layer in decoder.transformer.layers:
            z = norm(h, layer.norm1)
            qkv = F.linear(z, tensor(layer.attn.w_qkv.weight)).reshape(1, 14, 3, 2, 8)
            q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
            q, k = rope(norm(q, layer.attn.q_norm)), rope(norm(k, layer.attn.k_norm))
            attended = (
                F.scaled_dot_product_attention(q, k, v)
                .transpose(1, 2)
                .reshape(1, 14, 16)
            )
            h = h + F.linear(
                attended,
                tensor(layer.attn.out_proj.weight),
                tensor(layer.attn.out_proj.bias),
            )
            f = layer.ffn
            z = F.gelu(
                F.linear(
                    norm(h, layer.norm2),
                    tensor(f.linear1.weight),
                    tensor(f.linear1.bias),
                ),
                approximate="none",
            )
            h = h + F.linear(z, tensor(f.linear2.weight), tensor(f.linear2.bias))
        h = upsampled + 0.1 * norm(h, decoder.transformer.final_norm)
        expected = F.linear(
            h,
            tensor(decoder.decoder_layers[0].weight),
            tensor(decoder.decoder_layers[0].bias),
        )
        np.testing.assert_allclose(np.array(decoder(x)), expected.numpy(), atol=2e-6)
        # Released checkpoints keep the CTC head in FP32 with BF16 ASR states.
        rounded = x.astype(mx.bfloat16)
        np.testing.assert_allclose(
            np.array(decoder(rounded)),
            np.array(decoder(rounded.astype(mx.float32))),
            atol=1e-6,
        )


def test_ctc_path_repeats_and_impossible_alignment():
    from mlx_audio.stt.models.nemotron_transcribe.timestamps import ctc_path

    labels = np.array([0, 1, 1, 0, 1, 0, 2, 2, 0])
    logits = np.full((len(labels), 3), -20.0, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 0
    frames, path = ctc_path(logits, [1, 1, 2], np.ones(len(labels)))
    np.testing.assert_array_equal(frames, np.arange(len(labels)))
    np.testing.assert_array_equal(path, [0, 1, 1, 2, 3, 4, 5, 5, 6])
    with pytest.raises(ValueError, match="available speech frames"):
        ctc_path(logits[:2], [1, 1], np.ones(2))
    with pytest.raises(ValueError, match="available speech frames"):
        ctc_path(logits, [1], np.zeros(len(labels)))


def test_ctc_long_banded_alignment():
    from mlx_audio.stt.models.nemotron_transcribe.timestamps import ctc_path

    rng = np.random.default_rng(50)
    target = rng.integers(1, 40, 400).tolist()
    labels = np.zeros(len(target) * 4, np.int64)
    labels[1::4] = target
    logits = np.full((len(labels), 40), -20.0, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 0
    _, path = ctc_path(logits, target, np.ones(len(labels)))
    np.testing.assert_array_equal((path[1::4] - 1) // 2, np.arange(len(target)))


def test_word_alignment_speaker_permutation_and_streaming(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe as implementation
    from mlx_audio.stt.models.nemotron_transcribe.timestamps import align_words

    tokenizer = tokenizer_fixture(tmp_path)
    labels = [64, 12, 12, 64, 1, 1, 64, 64]
    logits = mx.full((8, 65), -20.0)
    logits[mx.arange(8), mx.array(labels)] = 0
    # Transcript 0 belongs to diarization 1; transcript 1 belongs to diarization 0.
    probabilities = np.array([[0, 1]] * 4 + [[1, 0]] * 4, np.float32)
    head = SimpleNamespace(
        decoder=SimpleNamespace(
            decoder_layers=[SimpleNamespace(weight=mx.zeros((65, 1)))], _factor=2
        ),
        logits=lambda *_: iter([logits]),
    )
    result = align_words(
        "<spk:0>hello <spk:1>t1", tokenizer, head, None, probabilities, 0.32, {}
    )
    assert [(w["text"], w["speaker"]) for w in result] == [("hello", 0), ("t1", 1)]
    assert all(0 <= w["start"] <= w["end"] <= 0.32 for w in result)
    model = tiny_model(True)
    model._tokenizer = model._alignment_tokenizer = tokenizer
    monkeypatch.setattr(
        implementation, "generate_step", lambda *a, **kw: iter([(12, None), (11, None)])
    )
    captured = []

    def align(*args, **kwargs):
        captured.append(args)
        return result

    monkeypatch.setattr(implementation, "align_words", align)
    output = model.generate(mx.zeros(2000), word_timestamps=True)
    stream = list(model.generate(mx.zeros(2000), word_timestamps=True, stream=True))
    assert output.segments == stream[-1].segments == result
    assert output.text == "".join(part.text for part in stream).strip() == "hello"
    assert stream[-1].is_final
    assert all(part.segments is None for part in stream[:-1])
    assert captured[0][3].shape[-1] == 16  # ASR states before speaker fusion/projection


def test_alignment_character_units_and_thai(tmp_path):
    from mlx_audio.stt.models.nemotron_transcribe.timestamps import tokenize_transcript

    tokenizer = tokenizer_fixture(tmp_path)
    tokenizer.add_tokens(["你", "好", "あ"])
    streams = tokenize_transcript(tokenizer, "你好。", 100)
    assert [word for word, _ in streams[0]] == ["你", "好"]
    assert [word for word, _ in tokenize_transcript(tokenizer, "あ", 100)[0]] == ["あ"]
    with pytest.raises(ValueError, match="Thai"):
        tokenize_transcript(tokenizer, "ไทย", 100)


def test_mtp_partial_acceptance_and_eos_inside_verified_block():
    from types import SimpleNamespace

    from mlx_audio.lm.models.cache import KVCache
    from mlx_audio.stt.models.nemotron_transcribe.speculative import (
        speculative_generate_step,
    )

    def append(cache, length):
        values = mx.zeros((1, 1, length, 1))
        cache[0].update_and_fetch(values, values)

    class Backbone:
        def embeddings(self, tokens):
            return mx.eye(17)[tokens]

        def __call__(self, inputs=None, cache=None, input_embeddings=None):
            if inputs is None:
                inputs = mx.argmax(input_embeddings, axis=-1)
            append(cache, inputs.shape[1])
            return mx.eye(17)[(inputs + 1) % 17]

    def draft(embeddings, hidden, cache):
        tokens = mx.argmax(embeddings, axis=-1)
        append(cache, tokens.shape[1])
        # Two correct proposals followed by an incorrect proposal, repeatedly.
        return mx.eye(17)[(tokens + 1 + (tokens % 3 == 0)) % 17]

    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=[]),
        llm=SimpleNamespace(
            backbone=Backbone(),
            lm_head=lambda hidden: hidden,
            mtp=draft,
            make_cache=lambda: [KVCache()],
            finish_verification=lambda cache, keep, rejected: cache[0].trim(rejected),
        ),
    )
    prompt = mx.array([1, 2, 3])
    stats = {}
    kwargs = {
        "input_embeddings": model.llm.backbone.embeddings(prompt),
        "max_tokens": 30,
        "num_draft_tokens": 4,
        "stats": stats,
    }
    assert [
        token for token, _ in speculative_generate_step(prompt, model, **kwargs)
    ] == [(4 + i) % 17 for i in range(30)]
    assert 0 < stats["accepted_tokens"] < stats["proposed_tokens"]
    model.config.eos_token_id = [6]
    assert [
        token for token, _ in speculative_generate_step(prompt, model, **kwargs)
    ] == [4, 5, 6]


def test_embedded_alignment_tokenizer_survives_conversion(tmp_path):
    import io

    import sentencepiece as spm

    from mlx_audio.convert import convert
    from mlx_audio.stt.utils import load_model

    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    model = tiny_model(True)
    buffer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world", "a small alignment tokenizer"]),
        model_writer=buffer,
        model_type="char",
        vocab_size=64,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    weights = dict(tree_flatten(model.parameters()))
    weights["ctc_timestamp.tokenizer.model_proto"] = mx.array(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    )
    (source / "config.json").write_text(json.dumps(asdict(model.config)))
    tokenizer_fixture(source)
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    convert(str(source), str(target), model_domain="stt")
    assert (
        target / "alignment_tokenizer" / "tokenizer.model"
    ).read_bytes() == buffer.getvalue()
    restored = load_model(target, strict=True)
    expected = spm.SentencePieceProcessor(model_proto=buffer.getvalue())
    assert restored._alignment_tokenizer.encode("hello") == expected.encode("hello")


@pytest.mark.parametrize("target", [[1, 2], [1, 1], [2]])
def test_ctc_search_matches_exhaustive_optimum(target):
    from itertools import product

    from mlx_audio.stt.models.nemotron_transcribe.timestamps import ctc_path

    rng = np.random.default_rng(15)
    log_probs = rng.normal(size=(6, 3)).astype(np.float32)
    probabilities = rng.uniform(0.1, 1.0, 6).astype(np.float32)
    frames, path = ctc_path(log_probs, target, probabilities)
    expanded = np.zeros(2 * len(target) + 1, np.int64)
    expanded[1::2] = target
    chosen = expanded[path]
    score = lambda labels: sum(
        log_probs[t, token] + (np.log(probabilities[t]) if token else 0)
        for t, token in enumerate(labels)
    )
    candidates = []
    for labels in product(range(3), repeat=6):
        collapsed = [
            token
            for i, token in enumerate(labels)
            if token and (i == 0 or token != labels[i - 1])
        ]
        if collapsed == target:
            candidates.append(score(labels))
    np.testing.assert_array_equal(frames, np.arange(6))
    np.testing.assert_allclose(score(chosen), max(candidates), atol=1e-6)

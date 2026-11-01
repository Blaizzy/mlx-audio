import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.sts.models.lfm3_audio import LFM3AudioProcessor, Model, ModelConfig
from mlx_audio.sts.models.lfm3_audio.codec import CodecConfig, LiquidCodec, Quantizer
from mlx_audio.sts.models.lfm3_audio.depth import normalize_eos, sample, sample_frame
from mlx_audio.sts.models.lfm3_audio.features import log_mel, num_frames


def tiny_config():
    return ModelConfig.from_dict(
        {
            "lfm": {
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "conv_L_cache": 3,
                "sliding_window": 3,
                "layer_types": ["conv", "full_attention", "sliding_attention"],
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 10000,
                        "partial_rotary_factor": 0.5,
                    },
                    "sliding_attention": {"rope_theta": 1000},
                },
            },
            "encoder": {
                "dim_in": 128,
                "dim_conv": 4,
                "dim_out": 16,
                "sliding_window": 3,
                "lfm": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "conv_L_cache": 3,
                    "layer_types": ["conv", "full_attention"],
                    "block_auto_adjust_ff_dim": False,
                    "rope_parameters": {"rope_theta": 1000},
                },
            },
            "depthformer": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "layer_types": ["full_attention"],
                "use_sandwich_norm": False,
                "rope_parameters": {"full_attention": {"rope_theta": 1000}},
            },
            "codebooks": 3,
            "audio_vocab_size": 9,
            "maskgit_steps": 3,
            "audio_token_id": 90,
            "audio_out_token_id": 91,
            "audio_start_token_id": 92,
            "text_end_token_id": 93,
            "mixed_start_token_id": 94,
            "eos_token_id": 95,
            "interleaved_n_text": 2,
            "interleaved_n_audio": 2,
        }
    )


@pytest.fixture
def model():
    mx.random.seed(0)
    model = Model(tiny_config())
    model.eval()
    return model


@pytest.fixture
def processor():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {f"word{i}": i for i in range(128)}
    for index, word in {
        0: "<|startoftext|>",
        1: "[UNK]",
        2: "<|im_start|>",
        3: "<|pad|>",
        90: "<|audio|>",
        91: "<|audio_out|>",
        92: "<|audio_start|>",
        93: "<|text_end|>",
        94: "<|mixed_start|>",
        95: "<|im_end|>",
    }.items():
        del vocab[f"word{index}"]
        vocab[word] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        unk_token="[UNK]",
        pad_token="<|pad|>",
        additional_special_tokens=[t for t in vocab if t.startswith("<|")],
    )
    tokenizer.chat_template = (
        "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n"
        "{% if m.content is string %}{{ m.content }}{% else %}"
        "{% for c in m.content %}{% if c.type == 'audio' %}<|audio|>"
        "{% elif c.type == 'audio_out' %}<|audio_out|>{% else %}{{ c.text }}"
        "{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    return LFM3AudioProcessor(tokenizer, tiny_config())


def test_training_and_normalized_config_roundtrip():
    config = tiny_config()
    restored = ModelConfig.from_dict(asdict(config))
    assert asdict(restored) == asdict(config)
    assert config.maskgit_schedule == [1, 1, 1]
    assert config.audio_eos_code == 8
    assert config.audio_encoder_config.text_config.model_type == "lfm2"


@pytest.mark.parametrize(
    "values",
    [
        {"maskgit_schedule": [4, 4]},
        {"maskgit_steps": 0},
        {"audio_eos_code": 2017},
        {"interleaved_n_audio": 0},
    ],
)
def test_invalid_configs(values):
    with pytest.raises(ValueError):
        ModelConfig.from_dict(values)


def test_cached_backbone_matches_full_forward_across_sliding_boundary(model):
    # Metal's float32 GEMM and GEMV use different arithmetic precision. Compare
    # on CPU so the tolerance isolates cache behavior rather than kernel drift.
    with mx.stream(mx.cpu):
        backbone = model.language_model.model
        tokens = mx.array([[4, 6, 8, 9, 10, 11, 12, 13, 14]])
        expected = backbone(tokens)
        cache = backbone.make_cache()
        chunks = [backbone(tokens[:, :4], cache=cache)]
        for i in range(4, tokens.shape[1]):
            chunks.append(backbone(tokens[:, i : i + 1], cache=cache))
        actual = mx.concatenate(chunks, axis=1)
        mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=3e-5, rtol=3e-5
    )
    assert cache[2].size() == 3
    assert cache[0][0].shape[1] == 2


def test_partial_rope_preserves_zero_frequency_pairs(model):
    attn = model.layers[1].self_attn
    x = mx.arange(16).reshape(1, 1, 1, 16).astype(mx.float32)
    rotated = np.asarray(attn._rope(x, 3))
    np.testing.assert_array_equal(rotated[..., 4:8], np.asarray(x)[..., 4:8])
    np.testing.assert_array_equal(rotated[..., 12:16], np.asarray(x)[..., 12:16])
    assert not np.allclose(rotated[..., :4], np.asarray(x)[..., :4])


def test_depth_decoder_is_bidirectional_and_hides_masked_codes(model):
    hidden = mx.ones((1, 64))
    codes = mx.array([[1, 2, 3]])
    masked = mx.array([[False, False, True]])
    original = model.depth_decoder(hidden, codes, masked)
    changed = model.depth_decoder(hidden, mx.array([[1, 2, 7]]), masked)
    np.testing.assert_array_equal(np.asarray(original), np.asarray(changed))
    revealed = model.depth_decoder(
        hidden, mx.array([[1, 2, 7]]), mx.zeros((1, 3), mx.bool_)
    )
    assert not np.allclose(np.asarray(original[:, 0]), np.asarray(revealed[:, 0]))


def test_encoder_padding_matches_individual_clips(model):
    a = mx.random.normal((1, 128, 9))
    b = mx.random.normal((1, 128, 25))
    batch = mx.concatenate([mx.pad(a, [(0, 0), (0, 0), (0, 16)]), b])
    actual = model.audio_tower(batch, mx.array([9, 25]))
    expected = mx.concatenate([model.audio_tower(a), model.audio_tower(b)])
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-5
    )


def test_speech_placeholder_count_and_code_range(model):
    with pytest.raises(ValueError, match="placeholders"):
        model.get_input_embeddings(mx.array([[90]]))
    with pytest.raises(ValueError, match="invalid codebook"):
        model.get_input_embeddings(
            mx.array([[91]]), audio_out_codes=mx.array([[1, 2, 9]])
        )
    h = model.get_input_embeddings(
        mx.array([[91]]), audio_out_codes=mx.array([[1, 2, 3]])
    )
    assert h.shape == (1, 1, 64)


@pytest.mark.parametrize("length", [1, 399, 400, 401, 560, 561, 16000])
def test_features_match_independent_stft(length):
    from scipy.signal.windows import hann
    from transformers.audio_utils import mel_filter_bank

    audio = np.random.default_rng(7).normal(0, 0.1, length).astype(np.float32)
    count = num_frames(length)
    padded = np.pad(audio, (0, 400 + (count - 1) * 160 - length))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160]
    power = np.abs(np.fft.rfft(windows * hann(400, sym=False))) ** 2
    filters = mel_filter_bank(
        201, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney"
    )
    expected = (
        np.maximum(np.log10(np.maximum(filters.T @ power.T, 1e-12)), -12) + 4
    ) / 4
    np.testing.assert_allclose(log_mel(audio), expected, atol=2e-5, rtol=2e-5)


def test_processor_expands_multiple_clips_and_speech_history(processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": np.zeros(400)},
                {"type": "text", "text": "word4"},
                {"type": "audio", "audio": np.zeros(2000)},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "audio_out", "codes": np.zeros((2, 3), np.int32)},
            ],
        },
    ]
    inputs = processor.prepare(messages)
    ids = np.asarray(inputs["input_ids"])[0]
    assert (ids == 90).sum() == 3
    assert (ids == 91).sum() == 3
    np.testing.assert_array_equal(np.asarray(inputs["audio_out_codes"])[-1], [8, 8, 8])
    assert messages[1]["content"][0]["codes"].shape == (2, 3)
    with pytest.raises(ValueError, match="placeholders"):
        processor("word4", audio=[np.zeros(400)])


def test_sanitizer_accepts_export_and_mlx_layouts(model):
    weights = dict(tree_flatten(model.parameters()))
    raw = {}
    for key, value in weights.items():
        if key.startswith("language_model.model."):
            key = key.replace("language_model.model.", "lfm.model.", 1)
        elif key.startswith("audio_tower.downsample."):
            key = key.replace("audio_tower.downsample.", "encoder.downsample.", 1)
        elif key.startswith("audio_tower.lfm."):
            key = key.replace("audio_tower.lfm.", "encoder.lfm.lfm.", 1)
        if value.ndim == 4:
            value = value.transpose(0, 3, 1, 2)
        elif key.endswith("lin_out.weight"):
            value = value[..., None]
        elif key.endswith("conv.conv.weight"):
            value = value.transpose(0, 2, 1)
        raw[key] = value
    raw["depth_embeddings.0.to_logits.weight"] = mx.zeros((9, 32))
    sanitized = model.sanitize(raw)
    assert set(sanitized) == set(weights)
    for key in weights:
        np.testing.assert_array_equal(
            np.asarray(sanitized[key]), np.asarray(weights[key])
        )
    again = model.sanitize(sanitized)
    assert {k: v.shape for k, v in again.items()} == {
        k: v.shape for k, v in weights.items()
    }


def test_sampling_penalty_is_per_codebook_and_eos_is_any_code():
    logits = mx.array([[[1.0, 3.0, 2.0], [3.0, 1.0, 2.0]]])
    history = mx.array([[[1], [0]]])
    chosen = sample(logits, temperature=0, repetition_penalty=2, history=history)
    np.testing.assert_array_equal(np.asarray(chosen), [[2, 2]])
    np.testing.assert_array_equal(
        np.asarray(normalize_eos(mx.array([[1, 8, 2]]), 8)), [[8, 8, 8]]
    )


def test_maskgit_reveal_order():
    masks = []

    class Depth:
        num_codebooks = 3

        def __call__(self, hidden, codes, masked):
            masks.append(np.asarray(masked))
            # Once a codebook becomes visible it affects later predictions.
            value = codes.sum(-1) % 5
            return mx.where(
                mx.arange(5)[None, None, :] == value[:, None, None], 10.0, 0.0
            ) + mx.zeros((1, 3, 5))

    codes = sample_frame(
        Depth(), mx.ones((1, 2)), [1, 2], temperature=0, order=mx.array([[2, 0, 1]])
    )
    assert codes.shape == (1, 3)
    np.testing.assert_array_equal(masks[0], [[True, True, True]])
    np.testing.assert_array_equal(masks[1], [[True, True, False]])


def test_interleaved_state_machine_and_eos_embedding(model, monkeypatch):
    import mlx_audio.sts.models.lfm3_audio.model as module

    text = iter([4, 5, 93, 95])
    frames = iter([mx.array([[1, 2, 3]]), mx.array([[2, 3, 4]]), mx.array([[1, 8, 2]])])
    monkeypatch.setattr(module, "sample", lambda *a, **kw: mx.array([next(text)]))
    monkeypatch.setattr(module, "sample_frame", lambda *a, **kw: next(frames))
    slots = list(model.stream_slots(mx.array([[94]]), max_tokens=12))
    assert [t for t, _ in slots] == [4, 5, 91, 91, 93, 91, 95]
    np.testing.assert_array_equal(np.asarray(slots[-2][1]), [8, 8, 8])


def test_asr_does_not_load_codec_and_standard_result(model, processor, monkeypatch):
    object.__setattr__(model, "_processor", processor)
    monkeypatch.setattr(model, "codec", lambda: pytest.fail("ASR must not load codec"))
    options = {}

    def slots(**kwargs):
        options.update(kwargs)
        yield 4, None
        yield 95, None

    monkeypatch.setattr(model, "stream_slots", slots)
    result = model.generate(np.zeros(400), task="asr")
    assert result.audio is None
    assert result.text == "word4"
    assert result.finish_reason == "stop"
    assert options["generation_mode"] == "text"


def test_thinking_chat_separates_reasoning(model, processor, monkeypatch):
    object.__setattr__(model, "_processor", processor)
    monkeypatch.setattr(
        processor.tokenizer, "decode", lambda *a, **kw: "A thought.</think>Paris."
    )
    monkeypatch.setattr(
        model, "stream_slots", lambda **kw: iter([(4, None), (95, None)])
    )
    result = model.generate(text="word4", task="chat", thinking=True)
    assert result.reasoning == "A thought."
    assert result.text == "Paris."


def test_stream_uses_untied_language_head():
    config = tiny_config()
    config.text_config.tie_word_embeddings = False
    model = Model(config)
    ids = mx.array([[4, 5]])
    expected = model(ids)[:, -1].argmax().item()
    slots = list(
        model.stream_slots(ids, generation_mode="text", max_tokens=1, temperature=0)
    )
    assert slots == [(expected, None)]


@pytest.mark.parametrize("ids", [[], [[1], [2]], [[[1]]]])
def test_generation_rejects_empty_or_batched_prompts(model, ids):
    with pytest.raises(ValueError, match="one prompt"):
        list(model.stream_slots(mx.array(ids)))


def test_codec_rejects_eos_and_handles_empty():
    # Validation and FSQ are independent of the expensive decoder trunk.
    codec = LiquidCodec.__new__(LiquidCodec)
    nn.Module.__init__(codec)
    codec.config = CodecConfig()
    assert codec.decode(mx.zeros((12, 0), mx.int32)).shape == (0,)
    with pytest.raises(ValueError, match="strip EOS"):
        codec.decode(mx.full((12, 1), 2016, mx.int32))
    quantizer = Quantizer(codec.config)
    assert quantizer(mx.zeros((1, 12, 2), mx.int32)).shape == (1, 2, 256)


def test_generic_load_convert_and_reload(tmp_path, model, processor):
    from mlx_audio.convert import convert
    from mlx_audio.registry import classify_model
    from mlx_audio.sts import load
    from mlx_audio.utils import load_model

    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(model.config)))
    processor.tokenizer.save_pretrained(source)
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    codec = source / "audio_tokenizer"
    codec.mkdir()
    (codec / "config.json").write_text("{}")
    mx.save_safetensors(str(codec / "model.safetensors"), {"test": mx.ones((1,))})
    assert classify_model("lfm3_audio") == "sts"
    assert isinstance(load_model(str(source)), Model)
    loaded = load(str(source), strict=True)
    convert(str(source), str(target), quantize=True, q_bits=8, q_group_size=64)
    from mlx_audio.convert import build_quant_predicate

    nn.quantize(
        loaded, group_size=64, bits=8, class_predicate=build_quant_predicate(loaded)
    )
    expected = loaded(mx.array([[4, 5, 6]]))
    reloaded = load(str(target), strict=True)
    after = reloaded(mx.array([[4, 5, 6]]))
    np.testing.assert_allclose(
        np.asarray(after)[..., :80],
        np.asarray(expected)[..., :80],
        atol=1e-6,
        rtol=1e-6,
    )
    assert isinstance(
        reloaded.language_model.model.embed_tokens,
        nn.QuantizedEmbedding,
    )
    assert not hasattr(reloaded.depth_decoder.depth_linear, "scales")
    assert (target / "audio_tokenizer" / "model.safetensors").read_bytes() == (
        codec / "model.safetensors"
    ).read_bytes()
    assert "mlx_audio.sts" in (target / "README.md").read_text()

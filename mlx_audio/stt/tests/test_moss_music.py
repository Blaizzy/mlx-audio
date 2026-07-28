import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.stt.models.moss_music import (
    AudioEncoderConfig,
    Model,
    ModelConfig,
    MossMusicEncoder,
    MossMusicFeatureExtractor,
    MossMusicProcessor,
    TextConfig,
)


class StubTokenizer:
    eos_token_id = 151645

    def __init__(self):
        self.offset = 1000

    def encode(self, text, add_special_tokens=False):
        if text in {str(i) for i in range(10)}:
            return [15 + int(text)]
        return [self.offset + (ord(ch) % 200) for ch in text]

    def decode(self, tokens, skip_special_tokens=True):
        table = {
            11: "<think>hidden</think>\n\nanswer",
            12: "hello",
            13: "[00:00] hello\n[00:02] world",
        }
        return "".join(table.get(int(t), f"T{int(t)}") for t in tokens)

    def batch_decode(self, batch, **kwargs):
        return [self.decode(tokens, **kwargs) for tokens in batch]

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<|AUDIO|>": 151654,
            "<|audio_bos|>": 151669,
            "<|audio_eos|>": 151670,
        }
        return mapping.get(token, self.encode(token)[0])


def tiny_config() -> ModelConfig:
    return ModelConfig(
        audio_config=AudioEncoderConfig(
            num_mel_bins=128,
            encoder_layers=3,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            d_model=8,
            output_dim=8,
            downsample_hidden_size=4,
            max_source_positions=128,
            deepstack_encoder_layer_indexes=[0, 1],
            n_window=20,
            conv_chunksize=2,
        ),
        language_config=TextConfig(
            vocab_size=256,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=512,
        ),
        adapter_hidden_size=32,
        deepstack_num_inject_layers=2,
        audio_token_id=40,
        audio_start_id=41,
        audio_end_id=42,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=1,
        default_prompt="describe",
    )


def make_processor(cfg: ModelConfig) -> MossMusicProcessor:
    proc = object.__new__(MossMusicProcessor)
    proc.config = cfg
    proc.audio_token_id = cfg.audio_token_id
    proc.audio_start_id = cfg.audio_start_id
    proc.audio_end_id = cfg.audio_end_id
    proc.enable_time_marker = cfg.enable_time_marker
    proc.feature_extractor = MossMusicFeatureExtractor(cfg.audio_config.num_mel_bins)
    proc.tokenizer = StubTokenizer()
    proc._digit_token_ids = {str(i): 15 + i for i in range(10)}
    proc.audio_tokens_per_second = 12.5
    proc.time_marker_every_seconds = 2
    proc.time_marker_every_audio_tokens = 25
    return proc


def test_config_parses_upstream_names():
    cfg = ModelConfig.from_dict(
        {
            "model_type": "moss_music",
            "audio_config": {"d_model": 8, "encoder_layers": 1},
            "language_config": {"hidden_size": 16, "num_hidden_layers": 2},
        }
    )
    cfg.__post_init__()
    assert isinstance(cfg.audio_config, AudioEncoderConfig)
    assert isinstance(cfg.language_config, TextConfig)
    assert cfg.audio_config.d_model == 8
    assert cfg.audio_config.n_window == 200
    assert cfg.language_config.hidden_size == 16
    assert cfg.enable_time_marker is True
    assert cfg.strip_thinking is True


def test_feature_extractor_shape():
    fx = MossMusicFeatureExtractor(num_mel_bins=128)
    mel, raw_len = fx(mx.zeros(16000))
    assert mel.shape[0] == 128
    assert mel.shape[1] == raw_len
    assert raw_len > 0


def test_feature_extractor_matches_transformers_whisper():
    transformers = pytest.importorskip("transformers")
    from transformers.models.whisper.feature_extraction_whisper import (
        WhisperFeatureExtractor,
    )

    wav = np.sin(2 * np.pi * 440 * np.arange(4000, dtype=np.float32) / 16000)
    fx = MossMusicFeatureExtractor(num_mel_bins=128)
    got, _ = fx(wav)
    ref = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
    )._np_extract_fbank_features(wav[None], device="cpu")[0]
    assert np.max(np.abs(np.array(got) - ref)) < 2e-4


def test_processor_time_markers_preserve_audio_token_count():
    cfg = tiny_config()
    proc = make_processor(cfg)
    ids = proc._build_audio_placeholder_ids(30)
    assert ids.count(cfg.audio_token_id) == 30
    assert 17 in ids  # second marker "2"


def test_processor_time_marker_override():
    cfg = tiny_config()
    proc = make_processor(cfg)
    proc.feature_extractor = lambda audio: (mx.zeros((128, 400)), 400)
    default = proc(text="describe", audio=mx.zeros(16000))
    disabled = proc(
        text="describe",
        audio=mx.zeros(16000),
        enable_time_marker=False,
    )
    assert int(mx.sum(default.audio_input_mask.astype(mx.int32))) == 50
    assert int(mx.sum(disabled.audio_input_mask.astype(mx.int32))) == 50
    assert 17 in default.input_ids.tolist()  # second marker "2"
    assert 17 not in disabled.input_ids.tolist()


def test_processor_builds_prompt_and_mask():
    cfg = tiny_config()
    proc = make_processor(cfg)
    proc.feature_extractor = lambda audio: (mx.zeros((128, 80)), 80)
    out = proc(text="describe", audio=mx.zeros(16000))
    assert out.audio_data.shape == (1, 128, 80)
    assert out.audio_data_seqlens.tolist() == [80]
    assert int(
        mx.sum(out.audio_input_mask.astype(mx.int32))
    ) == proc.conv3_downsample_len(80)
    assert out.input_ids[0].item() != cfg.audio_token_id


def test_encoder_returns_deepstack_shapes():
    cfg = tiny_config()
    enc = MossMusicEncoder(cfg.audio_config)
    mel = mx.zeros((1, cfg.audio_config.num_mel_bins, 80))
    out, deepstack = enc(mel, mx.array([80], dtype=mx.int32))
    assert out.shape == (1, MossMusicEncoder.compute_downsampled_length(80), 8)
    assert len(deepstack) == 2
    assert deepstack[0].shape == out.shape


def test_model_builds_embeddings_with_deepstack():
    cfg = tiny_config()
    model = Model(cfg)
    model._processor = make_processor(cfg)
    processed = model._processor(text="describe", audio=mx.zeros(16000))
    ids, embeds, deepstack, prompt_tokens = model._build_prompt_embeddings(processed)
    assert ids.shape[0] == embeds.shape[1] == prompt_tokens
    assert embeds.shape[2] == cfg.language_config.hidden_size
    assert deepstack is not None
    assert len(deepstack) == cfg.deepstack_num_inject_layers
    assert deepstack[0].shape == embeds.shape


def test_sanitize_transposes_conv2d_and_skips_position_buffer():
    weights = {
        "audio_encoder.conv1.weight": mx.zeros((4, 1, 3, 3)),
        "audio_encoder.conv2.weight": mx.zeros((4, 4, 3, 3)),
        "audio_encoder.embed_positions.inv_timescales": mx.zeros((4,)),
        "audio_encoder.layers.0.q_proj.weight": mx.zeros((4, 4)),
        "lm_head.weight": mx.zeros((8, 8)),
    }
    san = Model.sanitize(weights)
    assert san["audio_encoder.conv1.weight"].shape == (4, 3, 3, 1)
    assert san["audio_encoder.conv2.weight"].shape == (4, 3, 3, 4)
    assert "audio_encoder.embed_positions.inv_timescales" not in san
    assert "audio_encoder.layers.0.self_attn.q_proj.weight" in san
    assert "audio_encoder.layers.0.q_proj.weight" not in san
    assert "lm_head.weight" in san


def test_sanitize_covers_tiny_model_keys():
    model = Model(tiny_config())
    params = dict(tree_flatten(model.parameters()))
    weights = {}
    for key, value in params.items():
        shape = value.shape
        source_shape = shape
        if key.startswith("audio_encoder.conv") and key.endswith(".weight"):
            source_shape = (shape[0], shape[3], shape[1], shape[2])
        weights[key] = mx.zeros(source_shape, dtype=value.dtype)
    weights["audio_encoder.embed_positions.inv_timescales"] = mx.zeros((4,))
    san = Model.sanitize(weights)
    assert set(san) == set(params)


def test_quant_predicate_keeps_only_audio_encoder_unquantized():
    model = Model(tiny_config())
    assert model.model_quant_predicate("audio_encoder.layers.0.fc1", None) is False
    assert model.model_quant_predicate("audio_adapter.gate_proj", None) is True
    assert (
        model.model_quant_predicate(
            "deepstack_audio_merger_list.0.down_proj",
            None,
        )
        is True
    )
    assert model.model_quant_predicate("language_model.layers.0.mlp.gate_proj", None)
    assert model.model_quant_predicate("lm_head", None)


def test_strip_thinking_default():
    assert Model._strip_thinking("<think>hidden</think>\n\nanswer") == "answer"


def test_parse_structured_segments_from_timestamp_markers():
    segments = Model._parse_structured_segments(
        "Lyrics:\n[00:00] hello\n[00:02.5 - 00:04] world",
        audio_duration=5.0,
    )
    assert segments == [
        {
            "text": "hello",
            "start": 0.0,
            "end": 2.5,
            "kind": "timestamped_text",
            "marker": "[00:00]",
        },
        {
            "text": "world",
            "start": 2.5,
            "end": 4.0,
            "kind": "timestamped_text",
            "marker": "[00:02.5 - 00:04]",
        },
    ]


def test_parse_structured_segments_from_line_cues():
    segments = Model._parse_structured_segments(
        "00:01: intro\n00:03 - 00:05: hook",
        audio_duration=6.0,
    )
    assert segments == [
        {
            "text": "intro",
            "start": 1.0,
            "end": 3.0,
            "kind": "timestamped_text",
            "marker": "00:01:",
        },
        {
            "text": "hook",
            "start": 3.0,
            "end": 5.0,
            "kind": "timestamped_text",
            "marker": "00:03 - 00:05:",
        },
    ]


def test_parse_structured_segments_falls_back_to_audio_duration():
    segments = Model._parse_structured_segments("plain answer", audio_duration=3.25)
    assert segments == [
        {
            "text": "plain answer",
            "start": 0.0,
            "end": 3.25,
            "kind": "text",
            "marker": None,
        }
    ]


def test_tiny_generate_runs():
    cfg = tiny_config()
    model = Model(cfg)
    model._processor = make_processor(cfg)
    out = model.generate(
        mx.zeros(16000),
        max_tokens=2,
        temperature=0.0,
        enable_time_marker=False,
    )
    assert out.generation_tokens == 2
    assert out.prompt_tokens > 0
    assert isinstance(out.text, str)


def test_generate_returns_structured_segments_for_timestamped_text():
    cfg = tiny_config()
    model = Model(cfg)
    model._processor = make_processor(cfg)
    model._processor.feature_extractor = lambda audio: (mx.zeros((128, 80)), 80)

    def fake_generate_tokens(*args, **kwargs):
        yield 13
        yield cfg.eos_token_id

    model._generate_tokens = fake_generate_tokens
    out = model.generate(mx.zeros(48000), max_tokens=4, temperature=0.0)
    assert out.text == "[00:00] hello\n[00:02] world"
    assert out.segments == [
        {
            "text": "hello",
            "start": 0.0,
            "end": 2.0,
            "kind": "timestamped_text",
            "marker": "[00:00]",
        },
        {
            "text": "world",
            "start": 2.0,
            "end": 3.0,
            "kind": "timestamped_text",
            "marker": "[00:02]",
        },
    ]


@pytest.mark.requires_weights
def test_full_weight_smoke():
    model_path = os.environ.get("MOSS_MUSIC_MLX_PATH")
    if not model_path:
        pytest.skip("Set MOSS_MUSIC_MLX_PATH to run the full MOSS-Music smoke test")

    from mlx_audio.stt import load

    model = load(model_path, lazy=True)
    sample = Path(os.environ.get("MOSS_MUSIC_SAMPLE", "audio_000.wav"))
    if not sample.exists():
        pytest.skip(f"Reference sample not found: {sample}")
    prompt = os.environ.get(
        "MOSS_MUSIC_TEST_PROMPT",
        "Transcribe the lyrics of this song.",
    )
    out = model.generate(str(sample), prompt=prompt, max_tokens=128, temperature=0.0)
    text = out.text.lower()
    assert "quick brown fox" in text

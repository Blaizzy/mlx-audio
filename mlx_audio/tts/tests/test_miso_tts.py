from pathlib import Path
from types import SimpleNamespace

from mlx_audio.utils import get_model_class, get_model_name_parts


def test_miso_tts_default_config_matches_upstream_shape():
    from mlx_audio.tts.models.miso_tts import DEFAULT_CONFIG, MISO_TTS_WATERMARK

    assert DEFAULT_CONFIG["model_type"] == "miso_tts"
    assert DEFAULT_CONFIG["backbone_flavor"] == "llama-8B"
    assert DEFAULT_CONFIG["decoder_flavor"] == "llama-300M"
    assert DEFAULT_CONFIG["text_tokenizer"] == "meta-llama/Llama-3.2-1B"
    assert DEFAULT_CONFIG["text_tokenizer_apply_template"] is True
    assert DEFAULT_CONFIG["text_vocab_size"] == 128_256
    assert DEFAULT_CONFIG["audio_vocab_size"] == 2051
    assert DEFAULT_CONFIG["audio_num_codebooks"] == 32
    assert DEFAULT_CONFIG["num_hidden_layers"] == 32
    assert DEFAULT_CONFIG["num_attention_heads"] == 32
    assert DEFAULT_CONFIG["hidden_size"] == 4096
    assert DEFAULT_CONFIG["intermediate_size"] == 14_336
    assert DEFAULT_CONFIG["depth_decoder_config"]["num_hidden_layers"] == 8
    assert DEFAULT_CONFIG["depth_decoder_config"]["num_attention_heads"] == 24
    assert DEFAULT_CONFIG["depth_decoder_config"]["hidden_size"] == 1536
    assert DEFAULT_CONFIG["depth_decoder_config"]["intermediate_size"] == 6912
    assert DEFAULT_CONFIG["watermark_key"] == MISO_TTS_WATERMARK


def test_miso_tts_model_type_resolves_model_module():
    module, model_type = get_model_class(
        "miso_tts",
        get_model_name_parts("MisoTTS-bf16"),
        category="tts",
        model_remapping={},
    )

    assert model_type == "miso_tts"
    assert module.DEFAULT_CONFIG["model_type"] == "miso_tts"


def test_miso_tts_cached_hub_path_parts():
    path = Path("/tmp/hub/models--mlx-community--MisoTTS-bf16/snapshots/abcdef")

    assert "misotts" in get_model_name_parts(path)


def test_miso_tts_sanitize_maps_upstream_weight_names():
    from mlx_audio.tts.models.miso_tts import Model

    weights = {
        "audio_embeddings.weight": "audio_embeddings",
        "audio_head": "audio_head",
        "backbone.layers.0.attn.output_proj.weight": "attn",
        "backbone.layers.0.mlp.w1.weight": "w1",
        "backbone.layers.0.sa_norm.scale": "norm",
        "codebook0_head.weight": "codebook0_head",
        "decoder.layers.0.attn.k_proj.weight": "decoder_attn",
        "decoder.norm.scale": "decoder_norm",
        "projection.weight": "projection",
        "text_embeddings.weight": "text_embeddings",
    }

    sanitized = Model.sanitize(None, weights)

    assert sanitized["model.audio_embeddings.weight"] == "audio_embeddings"
    assert sanitized["model.audio_head"] == "audio_head"
    assert sanitized["model.backbone.layers.0.self_attn.o_proj.weight"] == "attn"
    assert sanitized["model.backbone.layers.0.mlp.gate_proj.weight"] == "w1"
    assert sanitized["model.backbone.layers.0.input_layernorm.weight"] == "norm"
    assert sanitized["model.codebook0_head.weight"] == "codebook0_head"
    assert sanitized["model.decoder.layers.0.self_attn.k_proj.weight"] == "decoder_attn"
    assert sanitized["model.decoder.norm.weight"] == "decoder_norm"
    assert sanitized["model.projection.weight"] == "projection"
    assert sanitized["model.text_embeddings.weight"] == "text_embeddings"


def test_miso_tts_generate_yields_decoded_audio_without_default_voice():
    import mlx.core as mx

    from mlx_audio.tts.models.miso_tts import Model

    class DummyTokenizer:
        def __init__(self):
            self.texts = []

        def encode(self, text, return_tensors):
            self.texts.append(text)
            return mx.array([[101, 102]], dtype=mx.int32)

    class DummyCoreModel:
        def __init__(self):
            self.calls = 0
            self.reset_count = 0
            self.token_shapes = []

        def reset_caches(self):
            self.reset_count += 1

        def generate_frame(self, tokens, tokens_mask, input_pos, sampler):
            self.calls += 1
            self.token_shapes.append((tokens.shape, tokens_mask.shape))
            if self.calls == 1:
                return mx.array([[7, 8]], dtype=mx.int32)
            return mx.array([[0, 0]], dtype=mx.int32)

    class DummyStreamingDecoder:
        def __init__(self):
            self.reset_count = 0
            self.decoded_shapes = []

        def reset(self):
            self.reset_count += 1

        def decode_frames(self, tokens):
            self.decoded_shapes.append(tokens.shape)
            return mx.ones((1, 1, tokens.shape[-1] * 4), dtype=mx.float32)

    model = Model.__new__(Model)
    model.model = DummyCoreModel()
    model._text_tokenizer = DummyTokenizer()
    model._streaming_decoder = DummyStreamingDecoder()
    model._frame_size = 3
    model._sample_rate = 24_000
    model._watermarker = None
    model._watermark_key = None
    model._runtime_initialized = True

    results = list(
        Model.generate(
            model,
            text="  hello",
            speaker=2,
            split_pattern=None,
            sampler=lambda logits: mx.array([0]),
            max_audio_length_ms=240,
            stream=False,
            verbose=False,
        )
    )

    assert model._text_tokenizer.texts == ["[2] hello"]
    assert model.model.reset_count == 1
    assert model._streaming_decoder.reset_count == 1
    assert model.model.token_shapes[0] == ((1, 2, 3), (1, 2, 3))
    assert model._streaming_decoder.decoded_shapes == [(1, 2, 1)]
    assert len(results) == 1
    assert results[0].samples == 4
    assert results[0].sample_rate == 24_000
    assert results[0].segment_idx == 0
    assert results[0].token_count == 1


def test_miso_tts_runtime_init_uses_configured_tokenizer_and_mimi(monkeypatch):
    from mlx_audio.tts.models.miso_tts import miso_tts as miso_module

    calls = {}

    class DummyMimi:
        cfg = SimpleNamespace(sample_rate=16_000)

        def eval(self):
            calls["mimi_eval"] = True

    def fake_tokenizer(repo):
        calls["tokenizer_repo"] = repo
        return "tokenizer"

    def fake_mimi_from_pretrained(repo):
        calls["mimi_repo"] = repo
        return DummyMimi()

    def fake_streaming_decoder(mimi):
        calls["decoder_mimi"] = mimi
        return "streaming_decoder"

    tiny_rope = {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }
    tiny_decoder = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "backbone_hidden_size": 16,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "max_position_embeddings": 16,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_codebooks": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": tiny_rope,
        "rope_theta": 500_000,
        "use_cache": True,
        "vocab_size": 32,
    }
    tiny_config = {
        "audio_num_codebooks": 2,
        "audio_vocab_size": 8,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "depth_decoder_config": tiny_decoder,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "max_position_embeddings": 16,
        "mlp_bias": False,
        "num_attention_heads": 2,
        "num_codebooks": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": tiny_rope,
        "rope_theta": 500_000,
        "text_tokenizer": "local-tokenizer",
        "text_tokenizer_apply_template": True,
        "text_vocab_size": 32,
        "tie_codebooks_embeddings": False,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 32,
    }

    monkeypatch.setattr(miso_module, "load_llama3_tokenizer", fake_tokenizer)
    monkeypatch.setattr(
        miso_module.sesame_module.Mimi,
        "from_pretrained",
        staticmethod(fake_mimi_from_pretrained),
    )
    monkeypatch.setattr(
        miso_module.sesame_module,
        "MimiStreamingDecoder",
        fake_streaming_decoder,
    )
    monkeypatch.setattr(
        miso_module.sesame_module,
        "load_watermarker",
        lambda: None,
        raising=False,
    )

    model = miso_module.Model(tiny_config)

    assert calls == {}
    assert model._text_tokenizer is None
    assert model._streaming_decoder is None
    assert model.sample_rate == 24_000

    model._ensure_runtime()

    assert calls["tokenizer_repo"] == "local-tokenizer"
    assert calls["mimi_repo"] == miso_module.sesame_module.MIMI_REPO
    assert calls["mimi_eval"] is True
    assert isinstance(calls["decoder_mimi"], DummyMimi)
    assert model._text_tokenizer == "tokenizer"
    assert model._streaming_decoder == "streaming_decoder"
    assert model.sample_rate == 16_000
    assert model._frame_size == 3

"""Weight-free tests for granite-speech-4.1-2b-plus support.

The plus checkpoint concatenates intermediate Conformer layer outputs onto the
final encoder output (``cat_hidden_layers``) and expects a system turn plus a
space between the audio placeholder and the instruction.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_audio.stt.models.granite_speech.config import (
    EncoderConfig,
    ModelConfig,
    ProjectorConfig,
    TextConfig,
)
from mlx_audio.stt.models.granite_speech.granite_speech import (
    PLUS_SYSTEM_PROMPT,
    ConformerAttention,
    CTCEncoder,
    EncoderProjector,
    Model,
)

HIDDEN_DIM = 8


def _tiny_encoder_config(**overrides):
    params = dict(
        input_dim=4,
        num_layers=2,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dim_head=4,
        output_dim=4,
        context_size=8,
        max_pos_emb=16,
    )
    params.update(overrides)
    return EncoderConfig(**params)


class TestEncoderCatHiddenLayers:
    def _run(self, cat_hidden_layers):
        encoder = CTCEncoder(_tiny_encoder_config(cat_hidden_layers=cat_hidden_layers))
        out = encoder(mx.zeros((1, 8, 4)))
        mx.eval(out)
        return out

    def test_none_keeps_hidden_dim(self):
        assert self._run(None).shape == (1, 8, HIDDEN_DIM)

    def test_empty_list_keeps_hidden_dim(self):
        assert self._run([]).shape == (1, 8, HIDDEN_DIM)

    def test_single_layer_doubles_dim(self):
        assert self._run([1]).shape == (1, 8, 2 * HIDDEN_DIM)

    def test_layer_zero_exports_input_linear_output(self):
        assert self._run([0, 1]).shape == (1, 8, 3 * HIDDEN_DIM)

    @staticmethod
    def _layer(encoder, idx, hidden):
        return encoder.layers[idx - 1](hidden, attention_dists=encoder._attention_dists)

    @staticmethod
    def _inject_mid_ctc(encoder, hidden):
        return hidden + encoder.out_mid(mx.softmax(encoder.out(hidden), axis=-1))

    def test_exported_mid_layer_includes_ctc_injection(self):
        # With two layers, layer 1 is the mid layer. HF adds the CTC term in
        # place, so the state it exports for that layer carries the injection.
        encoder = CTCEncoder(_tiny_encoder_config(cat_hidden_layers=[1]))
        x = mx.random.normal((1, 8, 4), key=mx.random.key(0))
        h1 = self._inject_mid_ctc(
            encoder, self._layer(encoder, 1, encoder.input_linear(x))
        )
        h2 = self._layer(encoder, 2, h1)

        expected = mx.concatenate([h1, h2], axis=-1)
        assert mx.allclose(encoder(x), expected, atol=1e-5).item()

    def test_exported_layer_before_mid_is_raw_layer_output(self):
        encoder = CTCEncoder(_tiny_encoder_config(num_layers=4, cat_hidden_layers=[1]))
        x = mx.random.normal((1, 8, 4), key=mx.random.key(0))
        h1 = self._layer(encoder, 1, encoder.input_linear(x))
        h2 = self._inject_mid_ctc(encoder, self._layer(encoder, 2, h1))
        h4 = self._layer(encoder, 4, self._layer(encoder, 3, h2))

        expected = mx.concatenate([h1, h4], axis=-1)
        assert mx.allclose(encoder(x), expected, atol=1e-5).item()

    def test_config_from_dict_keeps_cat_hidden_layers(self):
        cfg = EncoderConfig.from_dict({"cat_hidden_layers": [3], "num_layers": 16})
        assert cfg.cat_hidden_layers == [3]

    def test_projector_consumes_concatenated_features(self):
        config = ModelConfig(
            encoder_config=_tiny_encoder_config(cat_hidden_layers=[1]),
            projector_config=ProjectorConfig(
                hidden_size=HIDDEN_DIM,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                encoder_hidden_size=2 * HIDDEN_DIM,
            ),
            text_config=TextConfig(hidden_size=12),
        )
        encoder = CTCEncoder(config.encoder_config)
        projector = EncoderProjector(config)
        out = projector(encoder(mx.zeros((1, 8, 4))))
        mx.eval(out)
        # 8 frames -> 1 window of 15 -> window_size // downsample_rate queries
        num_queries = config.window_size // config.downsample_rate
        assert out.shape == (1, num_queries, 12)


def test_non_aligned_attention_preserves_bfloat16():
    config = _tiny_encoder_config(context_size=8)
    attention = ConformerAttention(config)
    attention.set_dtype(mx.bfloat16)

    seq = mx.arange(config.context_size)
    attention_dists = (
        mx.clip(
            seq[:, None] - seq[None, :],
            -config.context_size,
            config.context_size,
        )
        + config.max_pos_emb
    )
    output = attention(
        mx.zeros((1, config.context_size + 1, config.hidden_dim), dtype=mx.bfloat16),
        attention_dists,
    )
    mx.eval(output)

    assert output.dtype == mx.bfloat16


class StubTokenizer:
    """Records what _build_prompt renders; mimics the chat-template contract."""

    def __init__(self, chat_template="{{ messages }}"):
        self.chat_template = chat_template
        self.last_prompt = None

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        parts = [
            f"<|start_of_role|>{m['role']}<|end_of_role|>{m['content']}<|end_of_text|>\n"
            for m in chat
        ]
        if add_generation_prompt:
            parts.append("<|start_of_role|>assistant<|end_of_role|>")
        return "".join(parts)

    def encode(self, text):
        self.last_prompt = text
        return [0]


def _build_prompt(
    tokenizer, num_audio_tokens=2, prompt="do the thing", *, is_plus=True, **kwargs
):
    stub_model = SimpleNamespace(_tokenizer=tokenizer, is_plus=is_plus)
    Model._build_prompt(stub_model, num_audio_tokens, prompt, **kwargs)
    return tokenizer.last_prompt


class TestBuildPrompt:
    def test_plus_placeholder_has_reference_space(self):
        rendered = _build_prompt(StubTokenizer())
        assert "<|audio|><|audio|> do the thing<|end_of_text|>" in rendered

    def test_non_plus_placeholder_has_no_space(self):
        rendered = _build_prompt(StubTokenizer(), is_plus=False)
        assert "<|audio|><|audio|>do the thing<|end_of_text|>" in rendered

    def test_plus_placeholder_space_absorbs_leading_whitespace(self):
        rendered = _build_prompt(StubTokenizer(), prompt="  do the thing")
        assert "<|audio|><|audio|> do the thing<|end_of_text|>" in rendered

    def test_non_plus_prompt_is_verbatim(self):
        rendered = _build_prompt(
            StubTokenizer(), prompt="  do the thing", is_plus=False
        )
        assert "<|audio|><|audio|>  do the thing<|end_of_text|>" in rendered

    def test_system_turn_inserted_first(self):
        rendered = _build_prompt(StubTokenizer(), system_prompt=PLUS_SYSTEM_PROMPT)
        assert rendered.startswith(
            f"<|start_of_role|>system<|end_of_role|>{PLUS_SYSTEM_PROMPT}"
        )

    def test_no_system_turn_by_default(self):
        rendered = _build_prompt(StubTokenizer(), is_plus=False)
        assert rendered.startswith("<|start_of_role|>user<|end_of_role|>")

    def test_no_template_fallback(self):
        rendered = _build_prompt(StubTokenizer(chat_template=None))
        assert rendered == "USER: <|audio|><|audio|> do the thing\nASSISTANT:"


@pytest.mark.parametrize(
    ("is_plus", "expected_system_prompt"),
    [(True, PLUS_SYSTEM_PROMPT), (False, None)],
)
def test_generate_defaults_system_turn_for_plus_only(is_plus, expected_system_prompt):
    forwarded = {}

    def stream_generate(audio, **kwargs):
        forwarded.update(kwargs)
        return iter(())

    stub_model = SimpleNamespace(
        is_plus=is_plus,
        config=SimpleNamespace(model_type="granite_speech"),
        _stream_generate=stream_generate,
    )

    assert list(Model.generate(stub_model, mx.zeros((16000,)), stream=True)) == []
    assert forwarded["system_prompt"] == expected_system_prompt


class TestIsPlus:
    def _is_plus(self, config):
        return Model.is_plus.fget(SimpleNamespace(config=config))

    def test_by_model_type(self):
        assert self._is_plus(ModelConfig(model_type="granite_speech_plus"))

    def test_by_cat_hidden_layers_after_conversion(self):
        # mlx_audio.convert rewrites model_type to "granite_speech"; the
        # architectural fingerprint must still identify the plus variant.
        config = ModelConfig(
            model_type="granite_speech",
            encoder_config={"cat_hidden_layers": [3]},
        )
        assert self._is_plus(config)

    def test_non_plus(self):
        assert not self._is_plus(ModelConfig())


class TestSanitizeWeights:
    @pytest.mark.parametrize(
        ("name", "pytorch_shape", "mlx_shape"),
        [
            ("up_conv", (16, 8, 1), (16, 1, 8)),
            ("down_conv", (8, 16, 1), (8, 1, 16)),
            ("depth_conv", (16, 1, 5), (16, 5, 1)),
        ],
    )
    def test_convolution_conversion_is_idempotent(self, name, pytorch_shape, mlx_shape):
        key = f"encoder.layers.0.conv.{name}.weight"
        source = {key: mx.zeros(pytorch_shape)}

        converted = Model.sanitize(source)
        reloaded = Model.sanitize(converted)

        assert converted[key].shape == mlx_shape
        assert reloaded[key].shape == mlx_shape


@pytest.mark.parametrize(
    ("is_plus", "encoder_dtype", "expected_dtype"),
    [
        (True, mx.float32, mx.float32),
        (True, mx.bfloat16, mx.bfloat16),
        # 4.0/4.1 keep float32 activations regardless of the weight dtype.
        (False, mx.bfloat16, mx.float32),
    ],
)
def test_audio_feature_dtype(is_plus, encoder_dtype, expected_dtype):
    class RecordingEncoder:
        def __init__(self):
            self.input_linear = SimpleNamespace(
                weight=mx.zeros((1,), dtype=encoder_dtype)
            )
            self.input_dtype = None

        def __call__(self, features):
            self.input_dtype = features.dtype
            return features

    encoder = RecordingEncoder()
    stub_model = SimpleNamespace(
        encoder=encoder, projector=lambda features: features, is_plus=is_plus
    )

    output = Model.get_audio_features(stub_model, mx.zeros((1, 2, 4), dtype=mx.float32))

    assert encoder.input_dtype == expected_dtype
    assert output.dtype == expected_dtype

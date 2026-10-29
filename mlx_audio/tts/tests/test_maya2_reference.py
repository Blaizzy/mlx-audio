"""Opt-in differential checks against Maya Research's reference source."""

import importlib
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.lm.models.cache import make_prompt_cache
from mlx_audio.tts.models.maya2 import Model, ModelConfig
from mlx_audio.tts.models.maya2.maya2 import pack_prompt

SOURCE = os.environ.get("MAYA2_REFERENCE_SOURCE")
CHECKPOINT = os.environ.get("MAYA2_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    not SOURCE,
    reason="Set MAYA2_REFERENCE_SOURCE to the upstream maya2-native checkout",
)


@pytest.fixture(scope="module")
def reference():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, SOURCE)
    modules = {
        name: importlib.import_module("tts.model." + name)
        for name in ("depth_v2", "audio_io", "anchors")
    }
    torch.set_num_threads(4)
    yield torch, modules
    sys.path.remove(SOURCE)


def copy_weights(target, source):
    target.load_weights(
        [
            (k, mx.array(v.detach().float().numpy()))
            for k, v in source.state_dict().items()
        ]
    )


def assert_close(actual, expected, atol=2e-5, rtol=2e-5):
    expected = expected.detach().float().numpy()
    actual = np.asarray(actual.astype(mx.float32))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(actual - expected)))


def test_reference_components(reference, monkeypatch):
    from mlx_audio.tts.tests.test_maya2 import tiny_config

    torch, modules = reference
    depth_source = modules["depth_v2"]
    for key, value in {"D": 16, "H": 4, "FFN": 32, "LAYERS": 2, "RANK": 4}.items():
        monkeypatch.setattr(depth_source, key, value)
    cfg = tiny_config()
    torch.manual_seed(7)
    depth = depth_source.DepthV2(d_backbone=16, n_q=32, n_codes=8).eval()
    audio = modules["audio_io"].AudioInputEmbed(16, n_q=32, n_codes=8).eval()
    anchor = modules["anchors"].SpeakerAnchor(16, hidden=8, dropout_p=0).eval()
    model = Model(cfg)
    for target, source in (
        (model.depth, depth),
        (model.audio_in, audio),
        (model.speaker_anchor, anchor),
    ):
        copy_weights(target, source)
    h = torch.randn(2, 16)
    codes = torch.randint(0, 8, (2, 32))
    refs = torch.randn(2, 3, 192)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    mem = depth.spk_memory(refs, "ecapa")
    with torch.inference_mode():
        expected = depth(h, codes, spk_mem=mem, ref_mask=mask)
        actual = model.depth(
            mx.array(h.numpy()),
            mx.array(codes.numpy()),
            mx.array(mem.detach().numpy()),
            mx.array(mask.numpy()),
        )
        for x, y in zip(actual, expected):
            assert_close(x, y)
        assert_close(
            model.audio_in.frames(mx.array(codes.numpy())), audio.frames(codes, 32)
        )
        assert_close(model.audio_in.aeos(), audio.aeos(1)[0])
        assert_close(
            model.speaker_anchor(mx.array(refs.numpy()), mx.array(mask.numpy())),
            anchor(refs, ref_mask=mask),
        )
        # Exercise the learned null anchor after giving its output nonzero weights.
        anchor.fc2.weight.normal_(0, 0.1)
        copy_weights(model.speaker_anchor, anchor)
        assert_close(
            model.speaker_anchor(), anchor(refs[:1], null_mask=torch.tensor([True]))
        )
        for dtype, tolerance in ((torch.bfloat16, 0.03), (torch.float16, 0.004)):
            audio.to(dtype)
            model.audio_in.set_dtype(
                mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
            )
            assert_close(
                model.audio_in.frames(mx.array(codes.numpy())),
                audio.frames(codes, 32),
                atol=tolerance,
                rtol=tolerance,
            )


@pytest.mark.requires_weights
@pytest.mark.skipif(
    not CHECKPOINT, reason="Set MAYA2_CHECKPOINT to the original checkpoint directory"
)
def test_pretrained_reference(reference):
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    torch, modules = reference
    path = Path(CHECKPOINT)
    cfg = ModelConfig.from_dict({"model_path": str(path)})
    model = Model(cfg)
    model.eval()
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = {k: v.astype(mx.float32) for k, v in model.sanitize(weights).items()}
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    with torch.device("meta"):
        backbone = Qwen3Model(Qwen3Config.from_pretrained(path / "backbone"))
        depth = modules["depth_v2"].DepthV2().eval()
        audio = modules["audio_io"].AudioInputEmbed(2048).eval()
        anchor = modules["anchors"].SpeakerAnchor(2048).eval()
    backbone.rotary_emb = type(backbone.rotary_emb)(backbone.config, device="cpu")
    state = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            names = f.keys()
            state.update(
                {k.removeprefix("model."): f.get_tensor(k).float() for k in names}
            )
    for component, prefix in (
        (backbone, "backbone.lm.model."),
        (depth, "depth."),
        (audio, "audio_in."),
        (anchor, "speaker_anchor."),
    ):
        component.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
            assign=True,
        )
        component.eval()
    tokenizer = AutoTokenizer.from_pretrained(path / "backbone")
    packer_module = importlib.import_module("tts.data.packing_v2")
    tokens_module = importlib.import_module("tts.tokens")
    _, ids = tokens_module.extend_tokenizer(tokenizer)
    packer = packer_module.PackerV2(tokenizer, ids)
    model._tokenizer = tokenizer
    torch.manual_seed(42)
    refs = torch.nn.functional.normalize(torch.randn(1, 1, 192), dim=-1)
    ref_codes = torch.randint(0, 2048, (32, 3)).numpy()
    errors = {}
    with torch.inference_mode():
        for language, text in (
            ("en", "Hello from Maya."),
            ("hi", "नमस्ते, आप कैसे हैं?"),
        ):
            prompt = pack_prompt(tokenizer, text, language, "Ahana")
            packed = packer._span(
                {
                    "text_final": "{Ahana}: " + text,
                    "lang_final": language,
                    "codes": ref_codes,
                },
                first=True,
                cont=False,
            )
            assert prompt["tokens"] == packed[0][: (packed[1] == 0).sum()].tolist()
            embedding = backbone.embed_tokens(torch.tensor(prompt["tokens"])[None])
            embedding[:, prompt["sentinels"]] += anchor(
                refs, null_mask=torch.tensor([True])
            )[:, None]
            x = model.compose_inputs(prompt)
            errors[f"{language}_prompt"] = assert_close(x, embedding)
            expected = backbone(inputs_embeds=embedding, use_cache=True)
            cache = make_prompt_cache(model.backbone.lm)
            actual, logits = model(x, cache)
            errors[f"{language}_hidden"] = assert_close(
                actual, expected.last_hidden_state, atol=8e-4, rtol=2e-4
            )
            cb0_weight = state["backbone.cb0_head.weight"]
            expected_logits = torch.nn.functional.linear(
                expected.last_hidden_state, cb0_weight
            )
            errors[f"{language}_cb0"] = assert_close(
                logits, expected_logits, atol=5e-4, rtol=2e-4
            )
            np.testing.assert_array_equal(
                np.asarray(mx.argmax(logits, axis=-1)),
                expected_logits.argmax(-1).numpy(),
            )
        prompt = pack_prompt(
            tokenizer,
            "Hello.",
            "en",
            ref_text="नमस्ते।",
            ref_codes=ref_codes,
            ref_lang="hi",
        )
        x = model.compose_inputs(prompt, mx.array(refs.numpy()))
        embedding = backbone.embed_tokens(torch.tensor(prompt["tokens"])[None])
        kinds = np.array(prompt["kind"])
        embedding[:, kinds == 1] = audio.frames(torch.tensor(ref_codes.T.copy()), 32)
        embedding[:, kinds == 3] = audio.aeos(1)
        embedding[:, prompt["sentinels"]] += anchor(refs)[:, None]
        errors["clone_prompt"] = assert_close(x, embedding)
        expected = backbone(inputs_embeds=embedding, use_cache=True)
        cache = make_prompt_cache(model.backbone.lm)
        actual, _ = model(x, cache)
        errors["clone_hidden"] = assert_close(
            actual, expected.last_hidden_state, atol=8e-4, rtol=2e-4
        )
        h = expected.last_hidden_state[:, -1]
        codes = torch.randint(0, 2048, (1, 32))
        for name, mem in (
            ("null", depth.null_mem),
            ("ecapa", depth.spk_memory(refs, "ecapa")),
        ):
            expected_depth = depth(h, codes, spk_mem=mem)
            actual_depth = model.depth(
                mx.array(h.numpy()), mx.array(codes.numpy()), mx.array(mem.numpy())
            )
            errors[f"depth_{name}"] = max(
                assert_close(a, b, atol=4e-4, rtol=2e-4)
                for a, b in zip(actual_depth, expected_depth)
            )
            np.testing.assert_array_equal(
                np.stack([np.asarray(mx.argmax(a, axis=-1)) for a in actual_depth]),
                torch.stack(expected_depth).argmax(-1).numpy(),
            )
        feedback = audio.frames(codes, 32)[:, None]
        errors["feedback"] = assert_close(
            model.audio_in.frames(mx.array(codes.numpy()))[:, None], feedback
        )
        next_expected = backbone(
            inputs_embeds=feedback,
            past_key_values=expected.past_key_values,
            use_cache=True,
        )
        next_actual, _ = model(mx.array(feedback.numpy()), cache)
        errors["cached_hidden"] = assert_close(
            next_actual, next_expected.last_hidden_state, atol=8e-4, rtol=2e-4
        )
    print("Maya2 pretrained max absolute errors:", errors)


@pytest.mark.requires_weights
@pytest.mark.skipif(
    not CHECKPOINT, reason="Set MAYA2_CHECKPOINT for pretrained codec parity"
)
def test_pretrained_mimi(reference):
    from transformers import MimiModel

    from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder

    torch, _ = reference
    model = Mimi.from_pretrained("kyutai/moshiko-pytorch-bf16")
    original = MimiModel.from_pretrained("kyutai/mimi").eval()
    codes = np.random.default_rng(42).integers(0, 2048, (1, 32, 8), dtype=np.int64)
    with torch.inference_mode():
        expected = original.decode(torch.from_numpy(codes)).audio_values
    offline = model.decode(mx.array(codes))
    streamed = MimiStreamingDecoder(model).decode_frames(mx.array(codes))
    for actual in (offline, streamed):
        assert_close(actual, expected, atol=0.002, rtol=0.002)
        error = np.asarray(actual) - expected.numpy()
        assert np.sqrt(np.mean(error**2)) < 2e-4

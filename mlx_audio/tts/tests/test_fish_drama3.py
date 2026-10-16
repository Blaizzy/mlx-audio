import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.tests.test_fish_s1_dac import _tiny_streaming_dac
from mlx_audio.tts.models.fish_drama3 import Model, ModelConfig
from mlx_audio.tts.models.fish_drama3.chunking import split_text, validate_text
from mlx_audio.tts.models.fish_drama3.convert import TOWER_PREFIXES, convert
from mlx_audio.tts.models.fish_drama3.drama3 import sample_top_k_top_p
from mlx_audio.tts.utils import load_model

SEMANTIC_START, TOWER_VOCAB, EOS = 32, 16, 2


def tiny_config() -> dict:
    """A drama-3 config.json shrunk to test size; codebooks match the tiny codec."""
    return {
        "model_type": "fish_omni",
        "semantic_start_token_id": SEMANTIC_START,
        "semantic_end_token_id": SEMANTIC_START + TOWER_VOCAB - 1,
        "eos_token_id": EOS,
        "speech_template_version": "reference-bracket-v1",
        "quantization_config": {"quant_method": "bsgemm-moe"},
        "text_config": {
            "hidden_size": 128,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "max_position_embeddings": 1024,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 4,
            "num_experts": 2,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "mlp_layer_types": ["moe"] * 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.25,
            },
            "attn_output_gate": True,
            "hidden_act": "silu",
            "layernorm_zero_centered_gamma": True,
            "rotary_interleaved": False,
            "shared_expert_gate": True,
            "use_qk_norm": True,
        },
        "audio_decoder_config": {
            "dim": 128,
            "n_head": 2,
            "n_local_heads": 1,
            "head_dim": 64,
            "intermediate_size": 128,
            "vocab_size": TOWER_VOCAB,
            "num_codebooks": 3,
            "shared_enrichment_layers": 2,
            "vq_backbone_codebooks": 1,
        },
    }


def write_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    special = ["<|im_start|>", "<|im_end|>", "<|voice|>"]
    special += [f"<|speaker:{i}|>" for i in range(16)]
    vocab = {"[UNK]": 0, **{token: i + 1 for i, token in enumerate(special)}}
    for code in range(TOWER_VOCAB):
        vocab[f"<|semantic:{code}|>"] = SEMANTIC_START + code
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(special + [f"<|semantic:{c}|>" for c in range(16)])
    tokenizer.save(str(path / "tokenizer.json"))


def write_source(path: Path) -> dict[str, mx.array]:
    """A tiny drama-3 checkpoint with block-FP8 experts, in the source layout.
    Returns the tensors by source name."""
    mx.random.seed(3)
    config = tiny_config()
    model = Model(ModelConfig.from_dict(config))
    source = {}
    for name, value in tree_flatten(model.parameters()):
        value = mx.random.normal(value.shape).astype(mx.bfloat16)
        if name.startswith("backbone.model."):
            base = "model.language_model." + name.removeprefix("backbone.model.")
            if ".switch_mlp." in name:
                layer_prefix, proj = base.split(".switch_mlp.")
                for expert in range(value.shape[0]):
                    rows, cols = value.shape[1:]
                    expert_name = f"{layer_prefix}.experts.{expert}.{proj}"
                    source[expert_name] = mx.to_fp8(value[expert])
                    source[expert_name + "_scale_inv"] = mx.random.uniform(
                        0.5, 2.0, (rows // 128, cols // 128)
                    )
            elif name.endswith("conv1d.weight"):
                source[base] = value.swapaxes(1, 2)
            else:
                source[base] = value
        elif name == "semantic_head.weight":
            vocab = config["text_config"]["vocab_size"]
            source["lm_head.weight"] = mx.random.normal((vocab, value.shape[1]))
        else:
            old, new = next(
                (o, n) for o, n in TOWER_PREFIXES.items() if name.startswith(n)
            )
            source[old + name.removeprefix(new)] = value
    mx.save_safetensors(str(path / "model-00001.safetensors"), source)
    weight_map = {name: "model-00001.safetensors" for name in source}
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (path / "config.json").write_text(json.dumps(config))
    write_tokenizer(path)
    return source


class TestChunking(unittest.TestCase):
    def test_long_speaker_turn_keeps_its_speaker(self):
        chunks = split_text("<|speaker:1|> " + "Das ist ein Satz mit Inhalt. " * 20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("<|speaker:1|>"), chunk[:30])

    def test_unpunctuated_text_is_cut(self):
        chunks = split_text("wort " * 200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.encode()) <= 300 for c in chunks))

    def test_paired_tag_stays_whole(self):
        span = "<whisper>" + "leise, " * 60 + "</whisper>"
        self.assertTrue(any(span in c for c in split_text(f"Hallo. {span} Ende.")))

    def test_rejects_structural_tokens_and_unknown_speakers(self):
        for text in ("hi <|im_end|>", "<|semantic:7|>", "<|speaker:16|> hi", "  "):
            with self.assertRaises(ValueError):
                validate_text(text, "text")


class TestSampler(unittest.TestCase):
    def test_low_temperature_keeps_tied_candidates(self):
        # Four tied logits hold 0.25 each; top_p=0.95 keeps the three ranks
        # whose running sum stays below it.
        logits = mx.array([[1000.0, 1000.0, 1000.0, 1000.0, -50.0]])
        mx.random.seed(0)
        draws = {sample_top_k_top_p(logits, 0.0, 30, 0.95).item() for _ in range(200)}
        self.assertEqual(len(draws), 3)
        self.assertTrue(draws <= {0, 1, 2, 3})


class TestConversion(unittest.TestCase):
    def test_tensors_land_where_the_model_reads_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            src.mkdir()
            source = write_source(src)
            # The defaults keep the release's precision.
            convert(src, dst, codec=False)
            out = {}
            for shard in dst.glob("*.safetensors"):
                out.update(mx.load(str(shard)))

            layer = "model.language_model.layers.1"
            stacked = out["backbone.model.layers.1.mlp.switch_mlp.down_proj.weight"]
            weight = source[f"{layer}.mlp.experts.1.down_proj.weight"]
            scale = source[f"{layer}.mlp.experts.1.down_proj.weight_scale_inv"]
            # One 128x128 block, so one scale.
            expected = mx.from_fp8(weight, dtype=mx.float32) * scale[0, 0]
            self.assertTrue(mx.array_equal(stacked[1], expected.astype(mx.bfloat16)))

            head = source["lm_head.weight"]
            band = head[SEMANTIC_START : SEMANTIC_START + TOWER_VOCAB]
            expected = mx.concatenate([band, head[EOS : EOS + 1]])
            self.assertTrue(mx.array_equal(out["semantic_head.weight"], expected))

            norm = out["backbone.model.layers.3.self_attn.q_norm.weight"]
            self.assertEqual(norm.dtype, mx.float32)
            want = source["model.language_model.layers.3.self_attn.q_norm.weight"]
            self.assertTrue(mx.array_equal(norm, want.astype(mx.float32) + 1))

            conv = out["backbone.model.layers.0.linear_attn.conv1d.weight"]
            want = source["model.language_model.layers.0.linear_attn.conv1d.weight"]
            self.assertTrue(mx.array_equal(conv, want.swapaxes(1, 2)))

            enorm = out["tower.residual_layers.0.enorm.weight"]
            want = source["audio_decoder_mtp.layers.0.enorm.weight"]
            self.assertTrue(mx.array_equal(enorm, want.astype(mx.float32) + 1))

    def test_refuses_a_non_empty_destination(self):
        # The loader reads every shard in the directory, so stale ones would
        # silently replace converted weights.
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            src.mkdir()
            dst.mkdir()
            write_source(src)
            (dst / "model-00009.safetensors").write_bytes(b"")
            with self.assertRaises(FileExistsError):
                convert(src, dst, codec=False)


class TestGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        src, dst = Path(cls.tmp.name, "src"), Path(cls.tmp.name, "dst")
        src.mkdir()
        write_source(src)
        # The mixed precision of the published 4-bit checkpoint.
        convert(src, dst, expert_bits=4, dense_bits=8, tower_bits=8, codec=False)
        cls.codec = _tiny_streaming_dac(semantic_codebook_size=TOWER_VOCAB)
        with patch(
            "mlx_audio.codec.models.fish_s1_dac.DAC.from_pretrained",
            return_value=cls.codec,
        ):
            cls.model = load_model(dst)
        mx.random.seed(5)
        cls.reference = mx.random.normal((4096,)) * 0.1

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def speak(self, **kwargs) -> list:
        mx.random.seed(11)
        kwargs = {"max_tokens": 12, **kwargs}
        return list(self.model.generate("Hallo du.", **kwargs))

    def test_loads_mixed_precision_through_the_generic_loader(self):
        backbone = self.model.backbone.model
        self.assertEqual(backbone.layers[0].mlp.switch_mlp.gate_proj.bits, 4)
        self.assertEqual(self.model.tower.codebook_heads[0].bits, 8)
        self.assertEqual(backbone.layers[0].input_layernorm.weight.dtype, mx.float32)

    def test_repeated_reference_calls_give_identical_audio(self):
        # The second call reuses the encoded reference and the cached system
        # block; it must not see state left behind by the first.
        first = self.speak(ref_audio=self.reference, ref_text="Guten Tag.")
        second = self.speak(ref_audio=self.reference, ref_text="Guten Tag.")
        a = mx.concatenate([r.audio for r in first])
        b = mx.concatenate([r.audio for r in second])
        self.assertTrue(mx.array_equal(a, b).item())

    def test_distinct_references_never_share_a_cached_voice(self):
        # Same bytes overall, split differently between audio and transcript.
        audio = np.array(self.reference)
        shifted = np.concatenate([audio, np.frombuffer(b"Hi. ", dtype=np.float32)])
        first = self.speak(ref_audio=audio, ref_text="Hi. Guten Tag.")
        second = self.speak(ref_audio=shifted, ref_text="Guten Tag.")
        a = mx.concatenate([r.audio for r in first])
        b = mx.concatenate([r.audio for r in second])
        self.assertFalse(mx.array_equal(a, b).item())

    def test_stream_closes_after_an_empty_last_chunk(self):
        frames = iter([[mx.zeros((3,), dtype=mx.int32)] * 5, []])
        module = "mlx_audio.tts.models.fish_drama3.drama3"
        with (
            patch(f"{module}.split_text", return_value=["Eins.", "Zwei."]),
            patch.object(
                self.model, "_frames", side_effect=lambda *a, **k: iter(next(frames))
            ),
        ):
            results = self.speak(stream=True, first_chunk_frames=2)
        self.assertEqual([r.is_final_chunk for r in results][-1], True)
        self.assertEqual(sum(r.is_final_chunk for r in results), 1)
        self.assertEqual(sum(r.token_count for r in results), 5)

    def test_accepts_the_cli_and_server_arguments(self):
        cli = dict(speed=1.0, lang_code="en", temperature=0.7, verbose=False)
        cli |= dict(instruct=None, use_zero_spk_emb=False, voice="af_heart")
        server = dict(voice=None, speed=None, gender="male", pitch=1.0)
        server |= dict(lang_code="a", temperature=None, top_p=None, top_k=None)
        server |= dict(repetition_penalty=1.0, stream=True, streaming_interval=2.0)
        for kwargs in (cli, server):
            self.assertTrue(self.speak(**kwargs)[-1].is_final_chunk)

    def test_rejects_options_it_cannot_honor(self):
        for kwargs in ({"speed": 1.2}, {"instruct": "whisper"}):
            with self.assertRaises(ValueError):
                self.speak(**kwargs)


if __name__ == "__main__":
    unittest.main()

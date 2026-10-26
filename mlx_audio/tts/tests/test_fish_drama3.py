import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.tests.test_fish_s1_dac import _tiny_streaming_dac
from mlx_audio.tts.models.fish_drama3 import Model, ModelConfig
from mlx_audio.tts.models.fish_drama3.chunking import (
    flatten_lines,
    split_text,
    validate_text,
)
from mlx_audio.tts.models.fish_drama3.convert import TOWER_PREFIXES, convert
from mlx_audio.tts.models.fish_drama3.drama3 import sample_top_k_top_p
from mlx_audio.tts.utils import load_model

SEMANTIC_START, TOWER_VOCAB, EOS = 32, 16, 2
# Bytes the MLX cache may hold after a clear: finished GPU work still returns a
# few kilobytes, while one skipped clear leaves megabytes even with the tiny model.
CACHE_LEFTOVER = 256 * 1024


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
    def test_line_breaks_become_spaces_in_every_script(self):
        cases = (
            ("Erster Absatz.\n\nZweiter.", "Erster Absatz. Zweiter."),
            ("第一段。\r\n\n 第二段。\n", "第一段。 第二段。"),
            ("最初の段落。\n次の段落。", "最初の段落。 次の段落。"),
            ("좋은\n아침입니다.", "좋은 아침입니다."),
            ("한국어 \n English", "한국어 English"),
            ("中文\nEnglish", "中文 English"),
            ("Ｈｅｌｌｏ\nＷｏｒｌｄ", "Ｈｅｌｌｏ Ｗｏｒｌｄ"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(split_text(text), [expected])

    def test_unicode_line_breaks_and_surrounding_whitespace(self):
        for line_break in ("\n", "\r", "\r\n", "\v", "\f", "\x85", "\u2028", "\u2029"):
            with self.subTest(line_break=line_break):
                text = (
                    f" \t{line_break}First.\u3000{line_break}\u00a0Second.{line_break} "
                )
                self.assertEqual(split_text(text), ["First. Second."])

    def test_whitespace_without_line_breaks_is_preserved(self):
        for text in ("  First.\t\tSecond.\u3000", "First." + " " * 100_000 + "Second."):
            with self.subTest(length=len(text)):
                self.assertEqual(flatten_lines(text), text)

    def test_only_speaker_turns_keep_line_breaks(self):
        text = "<|speaker:0|> First.\n\n<|speaker:1|> Next\nline."
        expected = split_text("<|speaker:0|> First. <|speaker:1|> Next line.")
        chunks = split_text(text)
        self.assertEqual(chunks, expected)
        self.assertEqual(chunks[0].count("\n"), 1)

    def test_multiline_cues_stay_whole(self):
        text = "Before. <whisper>First\nsecond.</whisper> [laughs\nsoftly] After."
        chunks = split_text(text, limit=30)
        self.assertTrue(any("<whisper>First second.</whisper>" in c for c in chunks))
        self.assertTrue(any("[laughs softly]" in c for c in chunks))

    def test_many_paragraphs_preserve_words_and_chunk_limits(self):
        paragraphs = [f"Paragraph {i} has several words." for i in range(100)]
        chunks = split_text("\n\n".join(paragraphs))
        self.assertEqual(" ".join(chunks).split(), " ".join(paragraphs).split())
        self.assertTrue(all(len(chunk.encode()) <= 300 for chunk in chunks))
        self.assertTrue(all("\n" not in chunk for chunk in chunks))

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

    def test_no_top_k_draws_past_the_first_thirty(self):
        # Nearly flat but strictly falling, so the id of a candidate is its rank.
        logits = -0.001 * mx.arange(4097, dtype=mx.float32)[None]
        mx.random.seed(0)
        for top_k, beyond in ((None, True), (30, False)):
            draws = [
                sample_top_k_top_p(logits, 1.0, top_k, 0.9).item() for _ in range(200)
            ]
            self.assertEqual(max(draws) >= 30, beyond)

    def test_keep_crossing_keeps_the_token_that_crosses_top_p(self):
        logits = mx.log(mx.array([[0.85, 0.15]]))
        mx.random.seed(0)
        for keep, expected in ((False, {0}), (True, {0, 1})):
            draws = {
                sample_top_k_top_p(logits, 1.0, None, 0.9, keep_crossing=keep).item()
                for _ in range(200)
            }
            self.assertEqual(draws, expected)


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

    def test_per_layer_expert_bits_and_bf16_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            src.mkdir()
            write_source(src)
            convert(
                src,
                dst,
                expert_bits=3,
                dense_bits=8,
                tower_bits=8,
                head_bits=0,
                expert_group_size=32,
                expert_bits_map={1: 4},
                codec=False,
            )
            quant = json.loads((dst / "config.json").read_text())["quantization"]
            experts = "backbone.model.layers.{}.mlp.switch_mlp.down_proj"
            self.assertEqual(quant[experts.format(0)], {"group_size": 32, "bits": 3})
            self.assertEqual(quant[experts.format(1)], {"group_size": 32, "bits": 4})
            self.assertFalse(any("codebook_heads" in path for path in quant))
            self.assertEqual(
                quant["tower.residual_layers.0.transformer_layer.attention.wo"]["bits"],
                8,
            )

            with patch(
                "mlx_audio.codec.models.fish_s1_dac.DAC.from_pretrained",
                return_value=_tiny_streaming_dac(semantic_codebook_size=TOWER_VOCAB),
            ):
                model = load_model(dst)
            layers = model.backbone.model.layers
            self.assertEqual(layers[0].mlp.switch_mlp.down_proj.bits, 3)
            self.assertEqual(layers[1].mlp.switch_mlp.down_proj.bits, 4)
            self.assertEqual(layers[1].mlp.switch_mlp.down_proj.group_size, 32)
            self.assertNotIsInstance(model.tower.codebook_heads[0], nn.QuantizedLinear)

    def test_rejects_a_bits_map_for_missing_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            src.mkdir()
            write_source(src)
            with self.assertRaises(ValueError):
                convert(src, dst, expert_bits=3, expert_bits_map={9: 4}, codec=False)

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

    def test_a_chunk_decoded_in_pieces_sounds_like_one_codec_call(self):
        mx.random.seed(13)
        codes = mx.random.randint(0, TOWER_VOCAB, (12, 3)).astype(mx.int32)
        module = "mlx_audio.tts.models.fish_drama3.drama3"
        with (
            patch(f"{module}.DECODE_FRAMES", 5),
            patch.object(self.model, "_frames", return_value=iter(codes)),
        ):
            (result,) = self.speak()
        self.model.codec.reset_streaming_state()
        expected = self.model.codec.streaming_step(codes.T[None])[0, 0]
        self.assertEqual(result.audio.shape, expected.shape)
        self.assertTrue(mx.allclose(result.audio, expected, atol=1e-5).item())

    def decode_peak(self, codes: list[mx.array]) -> int:
        """Bytes one chunk's decode adds to MLX's active memory at its peak."""
        peaks = []
        # A busy machine returns freed buffers late or hands out cached ones up to
        # twice the size asked for, which only ever raises a peak: keep the least.
        for _ in range(3):
            self.model.codec.reset_streaming_state()
            # Arrays that earlier tests left in reference cycles would otherwise be
            # freed by a collection mid-decode, lowering the peak by their size.
            gc.collect()
            mx.clear_cache()
            before = mx.get_active_memory()
            mx.reset_peak_memory()
            self.model._result(codes, 0, 0.0, False, True)
            peaks.append(mx.get_peak_memory() - before)
        return min(peaks)

    def test_a_long_chunk_decodes_in_a_fraction_of_one_codec_call(self):
        def decode_peak(frames_per_call: int) -> int:
            with patch(f"{module}.DECODE_FRAMES", frames_per_call):
                return self.decode_peak(codes)

        mx.random.seed(17)
        codes = list(mx.random.randint(0, TOWER_VOCAB, (56, 3)))
        mx.eval(codes)
        module = "mlx_audio.tts.models.fish_drama3.drama3"
        # Measured: pieces of 8 frames peak at a fifth to a third of one call.
        self.assertLess(decode_peak(8), 0.5 * decode_peak(56))

    def test_decode_memory_does_not_grow_with_the_chunk(self):
        def decode_peak(frames: int) -> int:
            codes = list(mx.random.randint(0, TOWER_VOCAB, (frames, 3)))
            mx.eval(codes)
            return self.decode_peak(codes)

        mx.random.seed(19)
        module = "mlx_audio.tts.models.fish_drama3.drama3"
        with patch(f"{module}.DECODE_FRAMES", 8):
            short, long = decode_peak(56), decode_peak(1792)
        # Measured: 1.2 to 1.5 with each piece evaluated, 4 to 5 when all pieces
        # form one lazy graph.
        self.assertLess(long, 2 * short)

    def test_encoding_and_every_chunk_leave_the_mlx_cache_empty(self):
        # Buffers sized by one reference, prompt or chunk rarely fit the next, so
        # kept in MLX's cache they would pile up from call to call.
        self.model._encode_audio(np.array(self.reference[:3500]))
        self.assertLess(mx.get_cache_memory(), CACHE_LEFTOVER)
        module = "mlx_audio.tts.models.fish_drama3.drama3"
        with patch(f"{module}.split_text", return_value=["Eins.", "Zwei."]):
            mx.random.seed(11)
            for result in self.model.generate(
                "Eins. Zwei.",
                ref_audio=self.reference[:3000],
                ref_text="Guten Tag.",
                max_tokens=12,
            ):
                self.assertLess(mx.get_cache_memory(), CACHE_LEFTOVER)

    def test_a_stream_stopped_early_leaves_the_mlx_cache_empty(self):
        mx.random.seed(11)
        stream = self.model.generate(
            "Hallo du.",
            ref_audio=self.reference[:2500],
            ref_text="Guten Abend.",
            max_tokens=12,
            stream=True,
            first_chunk_frames=2,
        )
        next(stream)
        stream.close()
        self.assertLess(mx.get_cache_memory(), CACHE_LEFTOVER)

    def test_every_chunk_starts_with_an_empty_mlx_cache(self):
        # Two frames per chunk, streamed two at a time: no chunk has a closing
        # result, yet its prompt's buffers must not carry over to the next.
        model, frames, cached = self.model, self.model._frames, []

        def chunk_frames(*args):
            cached.append(mx.get_cache_memory())
            step = self.scripted_step([SEMANTIC_START + 1, SEMANTIC_START + 2])
            with patch.object(model, "_step", side_effect=step):
                yield from frames(*args)

        module = "mlx_audio.tts.models.fish_drama3.drama3"
        chunks = [" ".join(["Wort"] * 60) + "."] * 4
        with (
            patch(f"{module}.split_text", return_value=chunks),
            patch.object(model, "_frames", side_effect=chunk_frames),
        ):
            results = list(
                model.generate(
                    "Wort.",
                    stream=True,
                    first_chunk_frames=2,
                    streaming_interval=2 * model.codec.frame_length / model.sample_rate,
                )
            )
        self.assertEqual([r.token_count for r in results], [2, 2, 2, 2, 0])
        for size in cached[1:]:
            self.assertLess(size, CACHE_LEFTOVER)

    def scripted_step(self, mains: list[int]):
        """A Model._step that samples these main tokens, then EOS."""
        steps = iter(mains)

        def step(hidden, frame, history, *args):
            main = mx.array([next(steps, EOS)])
            c0 = mx.maximum(main - SEMANTIC_START, 0)
            new = mx.concatenate([main, c0, c0 + 1, c0 + 2])[None, None]
            return new.astype(mx.int32), hidden, history, main

        return step

    def run_frames(self, mains: list[int]) -> list[mx.array]:
        """The codes _frames yields when the steps sample these main tokens."""
        model = self.model
        frames, system_rows = model.build_frames([], [], "Hallo du.")
        voice = model._voice(None, None)
        with patch.object(model, "_step", side_effect=self.scripted_step(mains)):
            return list(model._frames(frames, system_rows, voice, 8, 0.7, 30, 0.7))

    def test_semantic_zero_right_before_eos_is_dropped(self):
        # <|semantic:0|> right before EOS decodes to a click, so it is dropped.
        codes = self.run_frames([SEMANTIC_START + 3, SEMANTIC_START])
        self.assertEqual([c[0].item() for c in codes], [3])
        # Anywhere else it is an ordinary frame.
        mains = [SEMANTIC_START, SEMANTIC_START, SEMANTIC_START + 5]
        codes = self.run_frames(mains)
        self.assertEqual([c[0].item() for c in codes], [0, 0, 5])
        # The frame cap is not EOS: a held frame is still yielded.
        codes = self.run_frames([SEMANTIC_START + 1] * 7 + [SEMANTIC_START])
        self.assertEqual(len(codes), 8)

    def test_accepts_the_cli_and_server_arguments(self):
        cli = dict(speed=1.0, lang_code="en", temperature=0.7, verbose=False)
        cli |= dict(instruct=None, use_zero_spk_emb=False, voice="af_heart")
        server = dict(voice=None, speed=None, gender="male", pitch=1.0)
        server |= dict(lang_code="a", temperature=None, top_p=None, top_k=None)
        server |= dict(repetition_penalty=1.0, stream=True, streaming_interval=2.0)
        for kwargs in (cli, server):
            self.assertTrue(self.speak(**kwargs)[-1].is_final_chunk)

    def test_main_token_draws_uncapped_and_codebook_heads_at_most_30(self):
        # The split is not audible on the tiny model, so the draws are recorded.
        calls = []

        def record(logits, temperature, top_k, top_p, keep_crossing=False):
            calls.append((top_k, top_p, keep_crossing))
            return sample_top_k_top_p(logits, temperature, top_k, top_p, keep_crossing)

        drama3 = "mlx_audio.tts.models.fish_drama3.drama3"
        with patch(f"{drama3}.sample_top_k_top_p", side_effect=record):
            self.speak()
            default = set(calls)
            calls.clear()
            self.speak(top_k=5)
            capped = set(calls)
        repetition_redraw = (30, 0.95, True)
        self.assertEqual(
            default, {(None, 0.9, True), repetition_redraw, (30, 0.9, False)}
        )
        self.assertEqual(capped, {(5, 0.9, True), repetition_redraw, (5, 0.9, False)})

    def test_server_requests_without_sampling_fields_get_these_defaults(self):
        try:
            from mlx_audio.server import SpeechRequest, _resolve_sampling_params
        except ImportError as error:
            self.skipTest(f"server extras missing: {error}")
        omitted = SpeechRequest(model="drama3", input="Hallo du.")
        self.assertEqual(
            _resolve_sampling_params(self.model, omitted),
            # Every candidate: the semantic band and EOS, so no cap.
            {
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": TOWER_VOCAB + 1,
                "repetition_penalty": 1.0,
            },
        )
        sent = SpeechRequest(model="drama3", input="Hallo du.", top_k=30)
        self.assertEqual(_resolve_sampling_params(self.model, sent)["top_k"], 30)

    def test_rejects_options_it_cannot_honor(self):
        for kwargs in ({"speed": 1.2}, {"instruct": "whisper"}):
            with self.assertRaises(ValueError):
                self.speak(**kwargs)


if __name__ == "__main__":
    unittest.main()

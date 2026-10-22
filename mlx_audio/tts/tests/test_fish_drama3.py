import contextlib
import gc
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional
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
DRAMA3 = "mlx_audio.tts.models.fish_drama3.drama3"
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


class FakeCodec:
    """The DAC streaming API over a frame counter, at the real codec's rate.

    A frame with c0 = 0 decodes to RMS 0.0002, as the model's silent tails do,
    any other frame to 0.05. The sign pattern depends on c1 and on the frame's
    position in the stream, so the audio shows which state decoded it. Every
    call width, reset and restore is recorded in `events`.
    """

    sample_rate, frame_length = 44100, 2048

    def __init__(self):
        self.position, self.events = 0, []

    def reset_streaming_state(self):
        self.position = 0
        self.events.append("reset")

    def streaming_state(self) -> int:
        return self.position

    def set_streaming_state(self, state: int):
        self.position = state
        self.events.append(("restore", state))

    def streaming_step(self, indices: mx.array) -> mx.array:
        codes = np.array(indices.reshape(-1, *indices.shape[-2:])[0])
        self.events.append(codes.shape[1])
        samples = np.arange(self.frame_length)
        frames = []
        for c0, c1 in codes[:2].T:
            level = 0.0002 if c0 == 0 else 0.05
            split = (37 * self.position + int(c1)) % self.frame_length
            frames.append(np.where(samples < split, level, -level))
            self.position += 1
        return mx.array(np.concatenate(frames).astype(np.float32))[None, None]


def spans(*runs: tuple[int, bool], salt: int = 0) -> list[mx.array]:
    """Frames from runs of (count, loud), their codes varied by salt."""
    frames = []
    for count, loud in runs:
        for _ in range(count):
            i = len(frames) + salt
            c0 = 1 + i % (TOWER_VOCAB - 1) if loud else 0
            frames.append(mx.array([c0, (3 * i) % TOWER_VOCAB, 0], dtype=mx.int32))
    return frames


def fake_decode(frames: list[mx.array]) -> mx.array:
    """What a fresh fake codec decodes from these frames."""
    return FakeCodec().streaming_step(mx.stack(frames, axis=1)[None])[0, 0]


def legacy_pieces(frames: int, first: Optional[int], interval: int) -> list[int]:
    """Frames per result of a chunk that is not the last, as streamed when each
    piece was decoded at its due point; first=None means no streaming."""
    pieces, emitted = [], 0
    for length in range(1, frames + 1):
        due = first if emitted == 0 else interval
        if first is not None and length - emitted >= due:
            pieces.append(length - emitted)
            emitted = length
    return pieces + ([frames - emitted] if frames > emitted else [])


def legacy_widths(pieces: list[int]) -> list[int]:
    """Codec call widths of decoding each piece at once in 64-frame calls."""
    return [min(64, piece - begin) for piece in pieces for begin in range(0, piece, 64)]


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

    def decode_peak(self, codes: list[mx.array], frames_per_call: int) -> int:
        """Peak memory of decoding codes in calls of frames_per_call frames,
        keeping the audio as a chunk does."""
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
            pieces = [
                self.model._decode(codes[begin : begin + frames_per_call])
                for begin in range(0, len(codes), frames_per_call)
            ]
            mx.eval(mx.concatenate(pieces))
            peaks.append(mx.get_peak_memory() - before)
            del pieces
        return min(peaks)

    def test_a_long_chunk_decodes_in_a_fraction_of_one_codec_call(self):
        mx.random.seed(17)
        codes = list(mx.random.randint(0, TOWER_VOCAB, (56, 3)))
        mx.eval(codes)
        # Measured: pieces of 8 frames peak at a fifth to a third of one call.
        self.assertLess(self.decode_peak(codes, 8), 0.5 * self.decode_peak(codes, 56))

    def test_decode_memory_does_not_grow_with_the_chunk(self):
        def chunk(frames: int) -> list[mx.array]:
            codes = list(mx.random.randint(0, TOWER_VOCAB, (frames, 3)))
            mx.eval(codes)
            return codes

        mx.random.seed(19)
        short = self.decode_peak(chunk(56), 8)
        long = self.decode_peak(chunk(1792), 8)
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

        # Plain functions, not mocks: a mock keeps its arguments in a cycle, so
        # a garbage collection may release them into the cache at any point.
        def chunk_frames(*args):
            cached.append(mx.get_cache_memory())
            step = self.scripted_step([SEMANTIC_START + 1, SEMANTIC_START + 2])
            with patch.object(model, "_step", step):
                yield from frames(*args)

        module = "mlx_audio.tts.models.fish_drama3.drama3"
        chunks = [" ".join(["Wort"] * 60) + "."] * 4
        with (
            patch(f"{module}.split_text", return_value=chunks),
            patch.object(model, "_frames", chunk_frames),
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
        with patch.object(model, "_step", self.scripted_step(mains)):
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

    def scripted(self, attempts: list, chunks=("Eins.",), retry_after=3.0, **kwargs):
        """Generates with the fake codec, _frames yielding the scripted attempts
        in order. Returns the results, the codec, how many frames each attempt
        yielded and the prompt each attempt got. retry_after=None keeps the
        model's own retry trigger."""
        codec, scripts, pulled, prompts = FakeCodec(), iter(attempts), [], []

        def frames(prompt, *args):
            prompts.append(prompt)
            pulled.append(0)
            for frame in next(scripts):
                pulled[-1] += 1
                yield frame

        # By default a retry trigger shorter than the end, so the scripts tell
        # the two apart.
        trigger = patch(f"{DRAMA3}.RETRY_AFTER_QUIET", retry_after)
        with (
            patch(f"{DRAMA3}.split_text", return_value=list(chunks)),
            trigger if retry_after else contextlib.nullcontext(),
            patch.object(self.model, "codec", codec),
            patch.object(self.model, "_frames", frames),
        ):
            results = list(self.model.generate("Hallo du.", **kwargs))
        return results, codec, pulled, prompts

    def prompt(self, text: str, *history: tuple[str, list]) -> mx.array:
        """The prompt rows for text after these turns, without a reference."""
        turns = [(turn, mx.stack(codes, axis=1)) for turn, codes in history]
        return self.model.build_frames([], turns, text)[0]

    def test_a_chunk_that_falls_silent_is_generated_again(self):
        silent, spoken = spans((10, True), (300, False)), spans((40, True), salt=5)
        (result,), codec, pulled, _ = self.scripted([silent, spoken])
        # 65 quiet frames end at frame 74, in the second 64-frame piece.
        self.assertEqual(pulled, [128, 40])
        self.assertEqual(codec.events, ["reset", 64, 64, ("restore", 0), 40])
        self.assertEqual(result.token_count, 40)
        self.assertTrue(mx.array_equal(result.audio, fake_decode(spoken)))

    def test_only_12_s_of_quiet_regenerate_a_chunk(self):
        # Complete takes paused for up to 10 s in the measured ones. 12 s are
        # 259 frames at 2048 samples and 44.1 kHz; a shorter pause is kept whole.
        spare = spans((40, True))
        kept = spans((10, True), (258, False), (30, True), salt=2)
        (result,), _, pulled, _ = self.scripted([kept, spare], retry_after=None)
        self.assertEqual(pulled, [298])
        self.assertTrue(mx.array_equal(result.audio, fake_decode(kept)))
        retried = spans((10, True), (259, False), (30, True), salt=2)
        (result,), _, pulled, _ = self.scripted([retried, spare], retry_after=None)
        self.assertEqual(pulled, [299, 40])
        self.assertTrue(mx.array_equal(result.audio, fake_decode(spare)))

    def test_a_chunk_silent_after_its_retries_ends_a_second_after_speech(self):
        prefixes = (5, 70, 30, 100)
        attempts = [spans((n, True), (400, False), salt=n) for n in prefixes]
        # The spare attempt lets an extra retry show in `pulled`.
        (result,), _, pulled, _ = self.scripted(attempts + [spans((40, True))])
        # Retries at the piece that shows 65 quiet frames, the end at the one
        # that shows 259 of them (frame 358).
        self.assertEqual(pulled, [128, 192, 128, 384])
        # The last attempt's 100 loud frames and the 21 quiet ones of 1 s.
        self.assertEqual(result.token_count, 121)
        self.assertTrue(mx.array_equal(result.audio, fake_decode(attempts[3][:121])))

    def test_an_all_quiet_text_closes_with_one_empty_final_result(self):
        # Without streaming, each chunk is retried three times and nothing goes
        # out.
        # A stream sends each chunk's first 21 quiet frames (1 s) at its due
        # points every 7 frames, so it cannot retry: 259 quiet frames end it.
        expected = {
            False: ([128, 128, 128, 320] * 2, [0]),
            True: ([259, 259], [7] * 6 + [0]),
        }
        for stream, (frames, counts) in expected.items():
            with (
                self.subTest(stream=stream),
                self.assertLogs(DRAMA3, "WARNING") as logs,
            ):
                results, _, pulled, prompts = self.scripted(
                    [spans((400, False))] * 8, ("Eins.", "Zwei."), stream=stream
                )
                self.assertEqual(pulled, frames)
                self.assertEqual([r.token_count for r in results], counts)
                finals = [r.is_final_chunk for r in results]
                self.assertEqual(finals, [False] * (len(counts) - 1) + [True])
                self.assertEqual(results[-1].audio.shape[0], 0)
                # Nothing of the first chunk is history, even what a stream sent.
                self.assertTrue(mx.array_equal(prompts[-1], self.prompt("Zwei.")))
                self.assertEqual(len(logs.records), 2)

    def test_a_short_quiet_chunk_that_ends_by_itself_is_kept_whole(self):
        quiet, spoken = spans((40, False)), spans((30, True))
        for stream in (False, True):
            with self.subTest(stream=stream):
                results, _, pulled, prompts = self.scripted(
                    [quiet, spoken], ("Eins.", "Zwei."), stream=stream
                )
                self.assertEqual(len(pulled), 2)
                first = [r.audio for r in results if r.segment_idx == 0]
                self.assertTrue(
                    mx.array_equal(mx.concatenate(first), fake_decode(quiet))
                )
                expected = self.prompt("Zwei.", ("Eins.", quiet))
                self.assertTrue(mx.array_equal(prompts[1], expected))

    def test_the_chunk_after_an_ended_one_continues_from_its_kept_audio(self):
        silent, spoken = spans((100, True), (400, False)), spans((20, True), salt=3)
        with patch(f"{DRAMA3}.SILENT_RETRIES", 0):
            results, codec, _, prompts = self.scripted(
                [silent, spoken],
                ("Eins.", "Zwei."),
                stream=True,
                first_chunk_frames=1000,
            )
        kept = silent[:121]
        ended = [r for r in results if r.segment_idx == 0]
        self.assertEqual(sum(r.token_count for r in ended), 121)
        audio = mx.concatenate([r.audio for r in ended])
        self.assertTrue(mx.array_equal(audio, fake_decode(kept)))
        # Six pieces up to the end, then the kept frames again in the calls
        # that decoded them, the last one cut short; then the next chunk.
        replay = [("restore", 0), 64, 57]
        self.assertEqual(codec.events, ["reset", *[64] * 6, *replay, 7, 7, 6])
        following = mx.concatenate([r.audio for r in results if r.segment_idx == 1])
        expected = fake_decode(kept + spoken)[121 * FakeCodec.frame_length :]
        self.assertTrue(mx.array_equal(following, expected))
        self.assertTrue(
            mx.array_equal(prompts[1], self.prompt("Zwei.", ("Eins.", kept)))
        )

    def test_audio_and_codec_calls_are_unchanged_without_a_long_silence(self):
        # Quiet runs of at most 21 frames (1 s), then of up to 60.
        short = spans((30, True), (20, False), (100, True), (21, False), (140, True))
        long = spans((10, True), (60, False), (150, True), (40, False), (30, True))
        # A quiet opening shorter than 1 s, due before and after it ends.
        opening = spans((15, False), (100, True), (20, False), (80, True))
        rate = FakeCodec.sample_rate / FakeCodec.frame_length
        cases = [(short, *due) for due in ((None, 7), (5, 7), (100, 90), (3, 70))]
        cases += [(opening, 5, 7), (opening, 18, 7)]
        for chunk, first, interval in cases:
            stream = first is not None
            kwargs = dict(stream=stream, first_chunk_frames=first)
            with self.subTest(first=first, interval=interval, frames=len(chunk)):
                results, codec, _, prompts = self.scripted(
                    [chunk, long],
                    ("Eins.", "Zwei."),
                    streaming_interval=interval / rate,
                    **kwargs,
                )
                pieces = legacy_pieces(len(chunk), first, interval)
                later = legacy_pieces(len(long), interval if stream else None, interval)
                widths = legacy_widths(pieces) + legacy_widths(later)
                self.assertEqual(codec.events, ["reset", *widths])
                audio = mx.concatenate([r.audio for r in results])
                self.assertTrue(mx.array_equal(audio, fake_decode(chunk + long)))
                counts = [r.token_count for r in results if r.segment_idx == 0]
                self.assertEqual(counts, pieces)
                expected = self.prompt("Zwei.", ("Eins.", chunk))
                self.assertTrue(mx.array_equal(prompts[1], expected))

    def test_due_points_on_a_64_frame_boundary_decode_each_frame_once(self):
        rate = FakeCodec.sample_rate / FakeCodec.frame_length
        for interval in (63, 64, 65):
            with self.subTest(interval=interval):
                # Both chunks end on a due point, so EOS leaves nothing to send.
                first = spans((3 * interval, True))
                last = spans((2 * interval, True), salt=1)
                results, codec, _, _ = self.scripted(
                    [first, last],
                    ("Eins.", "Zwei."),
                    stream=True,
                    first_chunk_frames=interval,
                    streaming_interval=interval / rate,
                )
                pieces = legacy_pieces(len(first), interval, interval)
                later = legacy_pieces(len(last), interval, interval)
                widths = legacy_widths(pieces) + legacy_widths(later)
                self.assertEqual(codec.events, ["reset", *widths])
                self.assertNotIn(0, codec.events)
                counts = [r.token_count for r in results]
                self.assertEqual(counts, [interval] * 5 + [0])
                finals = [r.is_final_chunk for r in results]
                self.assertEqual(finals, [False] * 5 + [True])
                audio = mx.concatenate([r.audio for r in results])
                self.assertTrue(mx.array_equal(audio, fake_decode(first + last)))
                if interval == 64:
                    # Every due point is also a full piece: one call each.
                    self.assertEqual(codec.events, ["reset"] + [64] * 5)

    def test_a_stream_retries_a_chunk_before_its_first_delivery(self):
        silent = [spans((10, True), (300, False), salt=n) for n in (1, 2)]
        spoken = spans((40, True), salt=3)
        results, codec, pulled, prompts = self.scripted(
            [*silent, spoken], stream=True, first_chunk_frames=1000
        )
        # Each silent attempt shows 65 quiet frames (to frame 74) in the piece
        # that ends at frame 128, before its first due point.
        self.assertEqual(pulled, [128, 128, 40])
        self.assertEqual(codec.events.count(("restore", 0)), 2)
        self.assertTrue(all(mx.array_equal(p, prompts[0]) for p in prompts))
        self.assertEqual(
            [(r.token_count, r.is_final_chunk) for r in results], [(40, True)]
        )
        self.assertTrue(mx.array_equal(results[0].audio, fake_decode(spoken)))

    def test_a_stream_retries_a_quiet_opening_only_while_it_has_sent_nothing(self):
        quiet, spoken = spans((300, False)), spans((40, True))
        # The 65th quiet frame is frame 64. A first due point at frame 64 comes
        # before it is decoded and sends up to 21 quiet frames, so the attempt
        # can only end, 259 quiet frames in, at the due point at frame 260.
        with self.assertLogs(DRAMA3, "WARNING"):
            results, _, pulled, _ = self.scripted(
                [quiet, quiet, spoken], stream=True, first_chunk_frames=64
            )
        self.assertEqual(pulled, [260])
        flags = [(r.token_count, r.is_final_chunk) for r in results]
        self.assertEqual(flags, [(21, False), (0, True)])
        # A first due point at frame 65 decodes frame 64 before sending, so the
        # attempt is retried with nothing sent.
        with self.assertNoLogs(DRAMA3, "WARNING"):
            results, codec, pulled, _ = self.scripted(
                [quiet, quiet, spoken], stream=True, first_chunk_frames=65
            )
        self.assertEqual(pulled, [65, 65, 40])
        self.assertEqual(codec.events.count(("restore", 0)), 2)
        self.assertEqual(
            [(r.token_count, r.is_final_chunk) for r in results], [(40, True)]
        )
        self.assertTrue(mx.array_equal(results[0].audio, fake_decode(spoken)))

    def test_a_stream_ends_a_silent_chunk_without_taking_back_audio(self):
        frames, spare = spans((30, True), (400, False)), spans((40, True))
        # The spare attempt lets a wrong retry show in `pulled`.
        results, _, pulled, _ = self.scripted(
            [frames, spare], stream=True, first_chunk_frames=5
        )
        # 259 quiet frames end at frame 288; the due point at 292 shows them.
        self.assertEqual(pulled, [292])
        self.assertEqual(sum(r.token_count for r in results), 30 + 21)
        audio = mx.concatenate([r.audio for r in results])
        self.assertTrue(mx.array_equal(audio, fake_decode(frames[:51])))
        self.assertEqual([r.is_final_chunk for r in results][-1], True)
        self.assertEqual(sum(r.is_final_chunk for r in results), 1)

    def test_a_held_pause_goes_out_once_when_speech_resumes(self):
        frames = spans((30, True), (40, False), (30, True))
        results, _, _, _ = self.scripted([frames], stream=True, first_chunk_frames=5)
        audio = mx.concatenate([r.audio for r in results])
        self.assertTrue(mx.array_equal(audio, fake_decode(frames)))
        # Due every 7 frames from frame 5. The pause sends 21 frames, holds the
        # rest, and they go out with the speech at frame 75.
        ends = list(itertools.accumulate(r.token_count for r in results))
        self.assertEqual(ends, [5, 12, 19, 26, 33, 40, 47, 51, 75, 82, 89, 96, 100])

    def test_audio_held_at_a_chunks_end_goes_out_in_its_closing_result(self):
        first, last = spans((20, True), (40, False)), spans((20, True), (40, False))
        results, _, _, _ = self.scripted(
            [first, last], ("Eins.", "Zwei."), stream=True, first_chunk_frames=5
        )
        audio = mx.concatenate([r.audio for r in results])
        self.assertTrue(mx.array_equal(audio, fake_decode(first + last)))
        self.assertTrue(all(r.token_count for r in results))
        self.assertEqual(sum(r.token_count for r in results), 120)
        self.assertEqual([r.is_final_chunk for r in results][-1], True)
        self.assertEqual(sum(r.is_final_chunk for r in results), 1)
        # A last chunk sent whole by its due points closes with an empty result.
        results, _, _, _ = self.scripted(
            [first, spans((21, True))], ("Eins.", "Zwei."), stream=True
        )
        self.assertEqual(
            (results[-1].token_count, results[-1].is_final_chunk), (0, True)
        )

    def test_a_retried_chunk_leaves_only_its_accepted_codes_as_history(self):
        silent, spoken = spans((10, True), (300, False)), spans((40, True), salt=5)
        _, _, pulled, prompts = self.scripted(
            [silent, spoken, spans((30, True))], ("Eins.", "Zwei.")
        )
        self.assertEqual(len(pulled), 3)
        self.assertTrue(mx.array_equal(prompts[0], prompts[1]))
        expected = self.prompt("Zwei.", ("Eins.", spoken))
        self.assertTrue(mx.array_equal(prompts[2], expected))

    def test_a_retry_draws_new_frames_and_the_call_stays_repeatable(self):
        model, frames, attempts = self.model, self.model._frames, []

        def recorded(*args):
            attempts.append([])
            for frame in frames(*args):
                attempts[-1].append(frame.tolist())
                yield frame

        # Every frame counts as quiet, and 8 of them call for a retry: the tiny
        # codec has 16 samples per frame, so 3 s would take 8269 frames.
        rate = model.sample_rate / model.codec.frame_length
        # The recipe's top-p keeps a single candidate in the tiny model.
        voice = dict(ref_audio=self.reference, ref_text="Guten Tag.", top_p=1.0)
        with (
            patch(f"{DRAMA3}.QUIET_RMS", 1.0),
            patch(f"{DRAMA3}.RETRY_AFTER_QUIET", 7.5 / rate),
            patch.object(model, "_frames", recorded),
        ):
            first = self.speak(**voice)
            retried = attempts[:]
            second = self.speak(**voice)
        self.assertLess(mx.get_cache_memory(), CACHE_LEFTOVER)
        # The first two attempts both ran 8 frames without EOS, so both were retried.
        self.assertGreaterEqual(len(retried), 3)
        self.assertGreaterEqual(min(len(codes) for codes in retried[:2]), 8)
        self.assertNotEqual(retried[0], retried[1])
        self.assertEqual(attempts[len(retried) :], retried)
        a = mx.concatenate([r.audio for r in first])
        b = mx.concatenate([r.audio for r in second])
        self.assertTrue(mx.array_equal(a, b).item())

    def test_one_warning_per_ended_chunk_and_none_for_a_retry(self):
        retried = [spans((10, True), (300, False)), spans((40, True))]
        with self.assertNoLogs(DRAMA3, "WARNING"):
            self.scripted(retried)
        with self.assertLogs(DRAMA3, "WARNING") as logs:
            self.scripted(retried + [spans((300, False))] * 4, ("Eins.", "Zwei."))
        (record,) = logs.records
        self.assertIn("chunk 1 ", record.getMessage())
        self.assertIn("3 retries", record.getMessage())

    def test_retries_and_ended_chunks_leave_the_mlx_cache_empty(self):
        model, frames, build = self.model, self.model._frames, self.model.build_frames
        loud, quiet = [SEMANTIC_START + 1] * 20, [SEMANTIC_START] * 300
        cached = []

        def chunk_frames(*args):
            with patch.object(model, "_step", self.scripted_step(next(scripts))):
                yield from frames(*args)

        def chunk_start(*args):
            cached.append(mx.get_cache_memory())
            return build(*args)

        def run(**kwargs):
            # Audio as small as the tiny codec's, at the real frame rate:
            # buffers the GPU releases after a clear stay a few kilobytes.
            codec = FakeCodec()
            codec.frame_length, codec.sample_rate = 16, 44100 * 16 / 2048
            with (
                patch(f"{DRAMA3}.split_text", return_value=["Eins.", "Zwei.", "Drei."]),
                patch(f"{DRAMA3}.RETRY_AFTER_QUIET", 3.0),
                patch(f"{DRAMA3}.SILENT_RETRIES", 1),
                patch.object(model, "codec", codec),
                patch.object(model, "_frames", chunk_frames),
                patch.object(model, "build_frames", chunk_start),
            ):
                yield from model.generate("Wort.", max_tokens=400, **kwargs)

        # A retry, a chunk ended after its retry, and a chunk that speaks.
        scripts = iter([loud[:10] + quiet, loud] + [loud[:10] + quiet] * 2 + [loud])
        for _ in run():
            self.assertLess(mx.get_cache_memory(), CACHE_LEFTOVER)
        self.assertEqual(len(cached), 3)
        for size in cached[1:]:
            self.assertLess(size, CACHE_LEFTOVER)
        # A stream closed at its first piece after a retry: the first due point
        # comes after the quiet run that calls for one.
        scripts = iter([quiet, loud * 6])
        stream = run(stream=True, first_chunk_frames=100)
        next(stream)
        stream.close()
        self.assertIsNone(next(scripts, None))
        self.assertLess(mx.get_cache_memory(), CACHE_LEFTOVER)

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

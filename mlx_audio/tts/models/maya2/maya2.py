# Portions derived from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya
# Research, licensed under the Apache License 2.0 (see LICENSE in this directory).
"""Maya-2 Native: streaming text-to-speech for 11 Indic languages and English, with zero-shot voice cloning.

A Qwen3-1.7B backbone predicts codebook 0 of a Mimi frame (80 ms) from a text prompt; a 31-level depth decoder fills
codebooks 1-31; Kyutai's Mimi turns the frames into 24 kHz audio. A speaker anchor (ECAPA-192 -> MLP) is added at every
span's <|audio|> sentinel and the depth decoder cross-attends to a speaker memory, so a short reference clip can be
cloned across languages.

The per-frame logic is the reference engine's for one stream (maya3/engine.py): controlled stop (argmax AEOS, tau
0.35, cumulative hazard 0.6, text-conditioned minimum length), codebook-0 sampling with temperature / top-k / top-p
and a presence penalty, a loop guard with an EOS boost, a duration cap, and 0.15 s gaps between text pieces.

Reference implementation: https://github.com/MayaResearch/maya2-native (Apache-2.0)
Weights: https://huggingface.co/maya-research/maya2-native
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen3 import ModelArgs as Qwen3Args
from mlx_audio.lm.models.qwen3 import Qwen3Model

from ..base import BaseModelArgs, GenerationResult
from . import guards as G
from .depth import DepthDecoder
from .prompt import (
    KIND_AEOS,
    KIND_AUDIO,
    LANGS,
    MODES,
    N_CODES,
    N_Q,
    Prompter,
    Voice,
    expected_frames,
    extend_tokenizer,
    load_frames_prior,
    load_named_voices,
    split_text,
)

ASSETS = Path(__file__).parent / "assets"
FRAME = 1920  # samples per 80 ms frame at 24 kHz
FP32_EPS = float(
    np.finfo(np.float32).eps
)  # torch nn.RMSNorm(eps=None) on bf16 computes in fp32 with fp32's eps
STOP_REASONS = {1: "aeos", 2: "aeos_argmax", 3: "aeos_tau", 4: "aeos_hazard"}
LANG_NAMES = {
    "assamese": "as",
    "bengali": "bn",
    "bangla": "bn",
    "english": "en",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
}
# script block -> language, for text without an explicit language (Devanagari -> hi, Bengali script -> bn)
SCRIPT_LANGS = [
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0B00, 0x0B7F, "or"),
    (0x0980, 0x09FF, "bn"),
    (0x0900, 0x097F, "hi"),
]


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "maya2"
    backbone: dict = field(default_factory=dict)  # Qwen3 decoder arguments
    n_q: int = N_Q
    n_codes: int = N_CODES
    depth_dim: int = 1024
    depth_heads: int = 16
    depth_ffn: int = 3072
    depth_layers: int = 4
    depth_groups: int = 11
    depth_rank: int = 128
    speaker_dim: int = 192
    anchor_hidden: int = 512
    sample_rate: int = 24000
    # serving numerics of the reference engine: fp16 depth decoder (closer to fp32 greedy than bf16)
    depth_dtype: str = "float16"
    default_voice: str = "Ahana"
    # Mimi (kyutai layout, as used across mlx-audio) and the SpeechBrain ECAPA speaker encoder in MLX
    mimi_repo: str = "kyutai/moshiko-pytorch-bf16"
    ecapa_repo: str = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
    ecapa_revision: Optional[str] = "e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77"
    # reference engine defaults (maya3/engine.py EngineConfig)
    temperature: float = 0.9  # codebook 0
    top_k: int = 64
    top_p: float = 1.0
    repetition_penalty: float = (
        0.8  # presence penalty over the last 12 codebook-0 codes
    )
    repetition_window: int = 12
    depth_temperature: float = 0.7
    depth_top_k: int = 64
    capacity: int = 2048
    hard_cap: int = 1500
    gap_s: float = 0.15
    chunk_s: float = 20.0
    seed: int = 20261024
    model_path: Optional[str] = None


class AudioInputEmbed(nn.Module):
    """Backbone input for one audio frame: sum of the 32 level embeddings / sqrt(32), RMS-normed; plus the AEOS input."""

    def __init__(self, dim, n_q=N_Q, n_codes=N_CODES):
        super().__init__()
        self.n_q, self.n_codes = n_q, n_codes
        self.table = nn.Embedding(n_q * n_codes, dim)
        self.aeos_in = mx.zeros((dim,))
        self.norm = nn.RMSNorm(dim, eps=FP32_EPS)

    def frames(self, codes):
        """codes [N, q] -> [N, dim] (summed in the weights' dtype, as the reference does)."""
        q = codes.shape[1]
        e = self.table(codes + mx.arange(q) * self.n_codes).sum(1) / math.sqrt(q)
        return self.norm(e)

    def aeos(self):
        return self.norm(self.aeos_in)


class SpeakerAnchor(nn.Module):
    """ECAPA-192 -> MLP -> backbone width, added at every span's <|audio|> sentinel; a learned null input otherwise."""

    def __init__(self, dim, spk_dim=192, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(spk_dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.null_vec = mx.zeros((spk_dim,))

    def __call__(self, ecapa=None):
        x = self.null_vec if ecapa is None else mx.array(ecapa, mx.float32)
        return self.fc2(nn.gelu(self.fc1(x.astype(self.fc1.weight.dtype))))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        args = Qwen3Args.from_dict({"model_type": "qwen3", **config.backbone})
        self.backbone = Qwen3Model(args)
        dim = args.hidden_size
        self.cb0_head = nn.Linear(dim, config.n_codes + 1, bias=False)
        self.audio_in = AudioInputEmbed(dim, config.n_q, config.n_codes)
        self.speaker_anchor = SpeakerAnchor(
            dim, config.speaker_dim, config.anchor_hidden
        )
        self.depth = DepthDecoder(
            dim,
            config.depth_dim,
            config.depth_heads,
            config.depth_ffn,
            config.depth_layers,
            config.depth_groups,
            config.depth_rank,
            config.n_q,
            config.n_codes,
            config.speaker_dim,
        )
        # runtime state (underscore attributes are not parameters)
        self._tokenizer = None
        self._prompter = None
        self._voices: Optional[Dict[str, Voice]] = None
        self._mimi = None
        self._decoder = None
        self._ecapa = None
        self._ready = False
        self._frame_fns = {}
        self._seed_rng = random.Random(config.seed)
        self._frames_prior = load_frames_prior(
            json.loads((ASSETS / "text_frames_prior.json").read_text())
        )
        self._duration = G.DurationPrior(
            json.loads((ASSETS / "duration_prior.json").read_text())
        )
        self._guards = G.GuardConfig()

    # ------------------------------------------------------------------ weights
    def sanitize(self, weights):
        """Accepts the official checkpoint (maya-research/maya2-native) and already-converted weights."""
        out = {}
        for k, v in weights.items():
            if k.startswith("backbone.lm.lm_head") or "rotary_emb.inv_freq" in k:
                continue  # tied to the input embeddings / recomputed
            if k.startswith("backbone.lm.model."):
                k = "backbone." + k[len("backbone.lm.model.") :]
            elif k == "backbone.cb0_head.weight":
                k = "cb0_head.weight"
            out[k] = v
        return out

    def model_quant_predicate(self, path, module):
        # quantize the backbone's projections only: the measured 8-bit setting (cb0 KL 4.5e-3 vs bf16). Embedding
        # tables are gathered, the depth decoder is precision-sensitive, and the heads are small.
        return path.startswith("backbone.layers.")

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def layers(self):
        return self.backbone.layers

    # ------------------------------------------------------------------ assets
    def _model_dir(self) -> Path:
        if self.config.model_path is None:
            raise ValueError(
                "ModelConfig.model_path is unset; cannot locate the tokenizer and voices."
            )
        return Path(self.config.model_path)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            path = self._model_dir() / "tokenizer"
            self._tokenizer, ids = extend_tokenizer(
                AutoTokenizer.from_pretrained(str(path))
            )
            self._prompter = Prompter(self._tokenizer, ids)
        return self._tokenizer

    @property
    def prompter(self) -> Prompter:
        self.tokenizer
        return self._prompter

    @property
    def voices(self) -> Dict[str, Voice]:
        if self._voices is None:
            path = self._model_dir() / "voices.json"
            self._voices = (
                load_named_voices(json.loads(path.read_text())) if path.exists() else {}
            )
            self._voices["plain"] = Voice("plain", "plain")
        return self._voices

    @property
    def mimi(self):
        if self._mimi is None:
            from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder

            self._mimi = Mimi.from_pretrained(self.config.mimi_repo)
            self._mimi.eval()
            self._decoder = MimiStreamingDecoder(self._mimi)
        return self._mimi

    def _prepare(self):
        """Once, after the weights are loaded: serving dtype for the depth decoder and its prefix table."""
        if not self._ready:
            self.depth.to_serving_dtype(getattr(mx, self.config.depth_dtype))
            self.mimi
            self._ready = True

    def _frame_fn(self, temp, top_k):
        key = (float(temp), int(top_k))
        if key not in self._frame_fns:
            self._frame_fns[key] = mx.compile(
                lambda h, cb0, cross, k: self.depth.frame(
                    h, cb0, cross, k, temp, top_k
                )[0]
            )
        return self._frame_fns[key]

    # ------------------------------------------------------------------ voices
    def make_voice(self, voice_id, ref_audio, ref_text=None, lang=None, mode="icl_spk"):
        """A cloned voice from a reference clip (3-10 s; 24 kHz array or a path) and, for in-context modes, its exact
        transcript and language. The clip is Mimi-encoded and ECAPA-embedded in MLX."""
        from mlx_audio.utils import load_audio

        if mode not in MODES or mode in ("plain", "named"):
            raise ValueError("mode must be one of icl_spk, icl, spk")
        use_icl, use_mem = MODES[mode]
        if use_icl and (not ref_text or not lang):
            raise ValueError(
                "ref_text and the reference language are required for cloning"
            )
        wav = (
            load_audio(ref_audio, sample_rate=self.sample_rate)
            if not isinstance(ref_audio, mx.array)
            else ref_audio
        )
        wav = wav.reshape(-1).astype(mx.float32)
        if wav.shape[0] > 30 * self.sample_rate:
            raise ValueError("reference audio longer than 30 s; 3-10 s works best")
        clips, ecapa = [], None
        if use_icl:
            self.mimi.reset_state()
            codes = np.array(self.mimi.encode(wav[None, None]))[0].astype(
                np.int64
            )  # [32, F]
            self._decoder.reset()
            clips = [{"text": ref_text, "lang": lang, "codes": codes}]
        if use_mem:
            ecapa = self._speaker_embedding(np.array(wav))
        return Voice(voice_id, mode, clips=clips, ecapa=ecapa)

    def register_voice(
        self, voice_id, ref_audio, ref_text=None, lang=None, mode="icl_spk"
    ):
        self.voices[voice_id.lower()] = self.make_voice(
            voice_id, ref_audio, ref_text, lang, mode
        )
        return voice_id

    def _speaker_embedding(self, wav24):
        from scipy.signal import resample_poly

        from mlx_audio.codec.models.ecapa_tdnn.speaker import (
            EcapaSpeakerEmbeddingExtractor,
        )

        if self._ecapa is None:
            self._ecapa = EcapaSpeakerEmbeddingExtractor.from_pretrained(
                self.config.ecapa_repo, revision=self.config.ecapa_revision
            )
        w16 = resample_poly(np.asarray(wav24, np.float32), 2, 3).astype(np.float32)
        return np.array(self._ecapa.encode(mx.array(w16)), np.float32)

    def _resolve_voice(self, voice) -> Voice:
        if isinstance(voice, Voice):
            return voice
        name = (voice or self.config.default_voice).strip().lower()
        if name not in self.voices:
            names = sorted(v.id for v in self.voices.values())
            raise ValueError(f"Unknown voice {voice!r}. Available: {', '.join(names)}")
        return self.voices[name]

    def _voice_state(self, v: Voice):
        """Per voice, once: depth cross-attention constants and the backbone KV of its reference prefix."""
        if "cross" not in v.cache:
            v.cache["cross"] = self.depth.speaker_constants(
                v.ecapa if v.use_mem else None
            )
            v.cache["kv"], v.cache["ts"] = None, 0
            if v.use_icl:
                prompt = self.prompter.build(v, "a", v.clips[0]["lang"])
                ts = prompt.prefix_len
                caches = [KVCache() for _ in self.backbone.layers]
                self._backbone(self._compose(prompt, v)[:ts], caches)
                v.cache["kv"] = [c.state for c in caches]
                v.cache["ts"] = ts
                mx.eval(v.cache["kv"])
            mx.eval(v.cache["cross"])
        return v.cache

    # ------------------------------------------------------------------ prompt -> embeddings
    def _compose(self, prompt, voice: Voice):
        """TTSModelV2.compose_inputs for one row: text embeddings, audio-frame and AEOS inputs, the speaker anchor
        added at every span sentinel."""
        emb = self.backbone.embed_tokens(mx.array(prompt.tokens))
        audio = np.nonzero(prompt.kind == KIND_AUDIO)[0]
        if audio.size:
            emb[mx.array(audio)] = self.audio_in.frames(
                mx.array(prompt.codes[audio])
            ).astype(emb.dtype)
        aeos = np.nonzero(prompt.kind == KIND_AEOS)[0]
        if aeos.size:
            emb[mx.array(aeos)] = self.audio_in.aeos().astype(emb.dtype)
        anchor = self.speaker_anchor(voice.ecapa if voice.use_mem else None).astype(
            emb.dtype
        )
        for c in prompt.sentinels:
            emb[c] = emb[c] + anchor
        return emb

    def _backbone(self, emb, caches):
        """[T, D] input embeddings at the caches' next positions -> final-norm hidden [T, D]."""
        return self.backbone(None, cache=caches, input_embeddings=emb[None])[0]

    # ------------------------------------------------------------------ one frame of codebook 0
    def _cb0(self, h, st, key):
        """Controlled stop + codebook-0 sampling (Engine._cb0 for one stream) -> (stop code, cb0, hazard)."""
        aeos_class = self.config.n_codes  # the cb0 head's extra class is end-of-speech
        logits = self.cb0_head(h).astype(mx.float32) / st["temp"]
        is_aeos = mx.arange(aeos_class + 1) == aeos_class
        logits = logits + st["bias"] * is_aeos
        if self.config.repetition_window > 0 and st["rep"] > 0 and st["recent"]:
            seen = np.zeros(aeos_class + 1, np.float32)
            seen[list(st["recent"])] = 1.0
            seen[aeos_class] = 0
            logits = logits - st["rep"] * mx.array(seen)
        kth = mx.topk(logits, st["top_k"], -1).min(-1, keepdims=True)
        logits = mx.where(logits < kth, -mx.inf, logits)
        if st["top_p"] < 1.0:
            order = mx.argsort(-logits, -1)
            sp = mx.take_along_axis(mx.softmax(logits, -1), order, -1)
            drop_sorted = (mx.cumsum(sp, -1) - sp) > st["top_p"]
            drop = mx.put_along_axis(mx.zeros_like(drop_sorted), order, drop_sorted, -1)
            logits = mx.where(drop, -mx.inf, logits)
        probs = mx.softmax(logits, -1)
        pa = probs[:, aeos_class]
        can = mx.array(st["frames"] >= st["minimum"])
        hz = mx.where(can, 1 - (1 - st["hazard"]) * (1 - pa), st["hazard"])
        code = mx.zeros_like(pa, dtype=mx.int32)
        code = mx.where(can & (hz >= 0.6), 4, code)
        code = mx.where(can & (pa >= 0.35), 3, code)
        code = mx.where(can & (mx.argmax(probs, -1) == aeos_class), 2, code)
        q = probs * ~is_aeos
        q = q / mx.maximum(q.sum(-1, keepdims=True), 1e-12)
        race = -mx.log(mx.random.uniform(shape=q.shape, key=key))
        cb0 = mx.minimum(mx.argmax(q / race, -1), aeos_class - 1)
        return code, cb0, hz

    # ------------------------------------------------------------------ one text piece
    def _piece(
        self, voice: Voice, text, lang, sampling, piece_idx, frames_done, max_frames
    ):
        """Yield int16 frames (np, 1920 samples) for one text piece; StopIteration.value = (frames, stop reason)."""
        cfg, g = self.config, self._guards
        state = self._voice_state(voice)
        prompt = self.prompter.build(voice, text, lang)
        n = len(prompt.tokens)
        ef, er = expected_frames(self._frames_prior, lang, len(text))
        loose = cap = min(cfg.hard_cap, int(2.0 * (ef + er) + 40))
        if g.enabled:
            cap = min(
                cap,
                max(
                    int(ef * 0.5) + 8,
                    self._duration.cap_frames(text, lang, g.dur_k, g.dur_margin_s),
                ),
            )
        if max_frames:
            cap = min(cap, max(1, int(max_frames) - frames_done))
        cap = min(cap, cfg.capacity - n - 2)
        if cap < 1:
            return 0, "capacity"
        minimum = max(4, int(0.5 * ef))
        if g.enabled and g.loop and g.aware_v2:
            suspend = G.suspend_windows_v2(text, lang, self._duration)
        elif g.enabled and g.loop and g.tag_aware and "[" in text:
            suspend = G.tag_windows(text, lang, self._duration, g.tag_hold_s)
        else:
            suspend = ()
        if sampling.get("seed") is not None:
            seed = (int(sampling["seed"]) * 1000003 + piece_idx * 7919) % (2**31 - 1)
        else:
            seed = self._seed_rng.randrange(1, 2**31 - 1)
        key = mx.random.key(seed)
        st = {
            "temp": float(sampling.get("temperature") or cfg.temperature),
            "top_k": int(sampling.get("top_k") or cfg.top_k),
            "top_p": float(sampling.get("top_p") or cfg.top_p),
            "rep": float(
                sampling["repetition_penalty"]
                if sampling.get("repetition_penalty") is not None
                else cfg.repetition_penalty
            ),
            "bias": 0.0,
            "recent": [],
            "frames": 0,
            "minimum": minimum,
            "hazard": mx.zeros((1,), mx.float32),
        }
        frame_fn = self._frame_fn(cfg.depth_temperature, cfg.depth_top_k)

        caches = [KVCache() for _ in self.backbone.layers]
        emb = self._compose(prompt, voice)
        if state["ts"]:
            for c, kv in zip(caches, state["kv"]):
                c.state = kv
            h = self._backbone(emb[state["ts"] :], caches)[-1:]
        else:
            h = self._backbone(emb, caches)[-1:]
        self._decoder.reset()
        hist = np.full(g.hist, -1, np.int64)
        frames, boost_at, reason = 0, None, None
        while True:
            key, k_cb0, k_depth = mx.random.split(key, 3)
            code, cb0, st["hazard"] = self._cb0(h, st, k_cb0)
            codes = frame_fn(h, cb0, state["cross"], k_depth)  # [1, 32]
            fb = self.audio_in.frames(codes).astype(h.dtype)
            pcm = self._decoder.decode_frames(
                codes.astype(mx.int32)[:, :, None]
            ).reshape(-1)
            pcm = mx.round(mx.clip(pcm, -1, 1) * 32767).astype(mx.int16)
            h = self._backbone(fb, caches)[
                -1:
            ]  # next frame's state, evaluated together (discarded on stop)
            mx.eval(code, cb0, pcm, h, st["hazard"])
            stop = int(code.item())
            st["frames"] += 1
            if stop:
                reason = STOP_REASONS[stop]
                break
            c0 = int(cb0.item())
            frames += 1
            yield np.array(pcm)
            hist[:-1] = hist[1:]
            hist[-1] = c0
            if cfg.repetition_window > 0:
                st["recent"] = (st["recent"] + [c0])[-cfg.repetition_window :]
            if frames >= cap:
                reason = (
                    "max_frames"
                    if max_frames and frames_done + frames >= max_frames
                    else "cap_guard" if g.enabled and cap < loose else "cap"
                )
                break
            if (
                g.enabled
                and g.loop
                and frames >= g.start
                and frames % g.every == 0
                and not any(a <= frames <= b for a, b in suspend)
            ):
                hit = G.loop_detect_batch(hist[None], np.array([frames]), g)[0]
                if hit is not None and g.mode == "boost":
                    if boost_at is None:
                        boost_at, st["bias"] = frames, g.boost
                    if frames - boost_at >= g.backstop:
                        reason = "loop_backstop"
                        break
                elif hit is not None:
                    reason = "loop_" + hit.split(":")[0]
                    break
                elif boost_at is not None:
                    boost_at, st["bias"] = None, 0.0
        return frames, reason

    # ------------------------------------------------------------------ public API
    def resolve_lang(self, lang_code, text):
        """The language to speak: the text's script decides, and ``lang_code`` (a code or a language name) picks
        among languages that share it (Devanagari: hi / mr; Bengali script: bn / as) or applies to Latin text.
        mlx-audio's CLI always passes lang_code="en", so it cannot override, e.g., Tamil script.
        """
        code = None
        if lang_code:
            code = str(lang_code).strip().lower()
            code = LANG_NAMES.get(code, code.split("-")[0])
            code = code if code in LANGS else None
        counts = {}
        for ch in text:
            for lo, hi, lang in SCRIPT_LANGS:
                if lo <= ord(ch) <= hi:
                    counts[lang] = counts.get(lang, 0) + 1
                    break
        if not counts:
            return code or "en"
        script = max(counts, key=counts.get)
        same_script = {"hi": "hi", "mr": "hi", "bn": "bn", "as": "bn"}
        if code and same_script.get(code, code) == same_script.get(script, script):
            return code
        return script

    def stream_pcm(
        self,
        text,
        lang=None,
        voice=None,
        ref_audio=None,
        ref_text=None,
        ref_lang=None,
        mode="icl_spk",
        max_frames=None,
        normalize=True,
        seed=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        info=None,
    ):
        """Yield int16 numpy PCM (24 kHz mono): one 80 ms frame at a time, and a 0.15 s gap between text pieces."""
        self._prepare()
        lang = self.resolve_lang(lang, text)
        if ref_audio is not None:
            if voice and str(voice).lower() not in (
                "plain",
                self.config.default_voice.lower(),
            ):
                raise ValueError("pass either voice=<name> or ref_audio=, not both")
            v = self.make_voice(
                "reference", ref_audio, ref_text, ref_lang or lang, mode
            )
        else:
            v = self._resolve_voice(voice)
        if normalize:
            from .text import normalise_text

            text = normalise_text(text, lang)
        text = text.strip()
        if not text:
            raise ValueError("text must be a non-empty string")
        sampling = {
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        pieces = (
            split_text(text, lang, self._frames_prior, self.config.chunk_s)
            if self.config.chunk_s > 0
            else [text]
        )
        total, stops = 0, []
        for i, piece in enumerate(pieces):
            gen = self._piece(v, piece, lang, sampling, i, total, max_frames)
            while True:
                try:
                    yield next(gen)
                except StopIteration as e:
                    n, reason = e.value
                    break
            total += n
            stops.append({"reason": reason, "frames": n})
            if max_frames and total >= max_frames:
                break
            if i + 1 < len(pieces):
                yield np.zeros(int(self.config.gap_s * self.sample_rate), np.int16)
        if info is not None:
            info.update(
                {
                    "lang": lang,
                    "voice": v.id,
                    "pieces": len(pieces),
                    "frames": total,
                    "stops": stops,
                }
            )

    def generate(
        self,
        text,
        voice=None,
        lang_code=None,
        ref_audio=None,
        ref_text=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_tokens=None,
        seed=None,
        normalize=True,
        stream=False,
        streaming_interval=0.4,
        verbose=False,
        **kwargs,
    ):
        """Synthesize ``text``; yields GenerationResult objects (one at the end, or chunks when ``stream``).

        Args:
            text: Text in any of the 12 languages (Tamil, Hindi, Telugu, ... English; code-mixed is fine).
            voice: A named voice (default ``Ahana``; see ``model.voices``) or a voice from ``register_voice``.
            lang_code: Language code or name (``ta``, ``hindi``, ...). Default: inferred from the text's script.
            ref_audio, ref_text: Clone a voice from a 3-10 s clip and its exact transcript (in-context + ECAPA).
                ``ref_lang`` (kwarg) names the clip's language when it differs from the text's.
            temperature, top_k, top_p, repetition_penalty: Codebook-0 sampling (reference defaults 0.9 / 64 / 1.0 /
                0.8, the last a presence penalty over the last 12 frames).
            max_tokens: Cap on audio frames (12.5 per second).
            seed: Reproducible output for the same text, voice and settings.
            normalize: Expand numbers, currency, dates, URLs... into words first (bundled normaliser).
            stream: Yield audio every ``streaming_interval`` seconds as it is generated.
        """
        start = time.perf_counter()
        info: dict = {}
        frames_per_chunk = max(
            1, int(round(streaming_interval * self.sample_rate / FRAME))
        )
        pcm_iter = self.stream_pcm(
            text,
            lang_code,
            voice,
            ref_audio,
            ref_text,
            kwargs.get("ref_lang"),
            kwargs.get("mode", "icl_spk"),
            max_tokens,
            normalize,
            seed,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            info,
        )
        chunks, pending, n_frames = [], [], 0
        for pcm in pcm_iter:
            n_frames += pcm.shape[0] == FRAME
            if stream:
                pending.append(pcm)
                if len(pending) >= frames_per_chunk:
                    yield self._result(
                        np.concatenate(pending),
                        start,
                        n_frames,
                        is_streaming_chunk=True,
                    )
                    pending, start, n_frames = [], time.perf_counter(), 0
            else:
                chunks.append(pcm)
        if verbose:
            print(
                f"[maya2] lang={info.get('lang')} voice={info.get('voice')} pieces={info.get('pieces')} "
                f"frames={info.get('frames')} stops={[s['reason'] for s in info.get('stops', [])]}"
            )
        if stream:
            audio = np.concatenate(pending) if pending else np.zeros(0, np.int16)
            yield self._result(
                audio, start, n_frames, is_streaming_chunk=True, is_final_chunk=True
            )
        else:
            if not chunks:
                raise ValueError("The model produced no audio.")
            yield self._result(np.concatenate(chunks), start, n_frames)
        mx.clear_cache()

    def _result(
        self, pcm16, start, frames, is_streaming_chunk=False, is_final_chunk=False
    ):
        audio = mx.array(pcm16.astype(np.float32) / 32767.0)
        samples = int(audio.shape[0])
        duration = samples / self.sample_rate
        elapsed = time.perf_counter() - start
        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=frames,
            audio_duration=f"{int(duration // 3600):02d}:{int(duration % 3600 // 60):02d}:{int(duration % 60):02d}."
            f"{int(duration % 1 * 1000):03d}",
            real_time_factor=duration / elapsed if elapsed > 0 else 0.0,
            prompt={
                "tokens": frames,
                "tokens-per-sec": round(frames / elapsed, 2) if elapsed > 0 else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

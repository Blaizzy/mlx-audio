import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from scipy.signal import resample_poly

from mlx_audio.audio_io import read as audio_read
from mlx_audio.codec.models.ecapa_tdnn.speaker import EcapaSpeakerEmbeddingExtractor
from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder
from mlx_audio.lm.models.cache import make_prompt_cache
from mlx_audio.lm.models.qwen3 import Model as Qwen3
from mlx_audio.lm.sample_utils import apply_top_p, make_sampler
from mlx_audio.tts.models.base import GenerationResult

from .config import LANGUAGES, ModelConfig
from .depth import DepthDecoder
from .normalization import normalise_text

VOICES = (
    "Ahana",
    "Vaidehi",
    "Malini",
    "Charvi",
    "Jahnavi",
    "Anirban",
    "Nakul",
    "Vasant",
    "Siddhant",
    "Satvik",
    "Abhay",
    "Nivedita",
    "Ishwari",
    "Ritvik",
    "Vrinda",
    "Sreekar",
    "Kayal",
    "Shaunak",
    "Akshat",
    "Tanushree",
)


class AudioRMSNorm(nn.RMSNorm):
    def __call__(self, x):
        # PyTorch's eps=None uses float32 accumulation epsilon for low precision.
        return mx.fast.rms_norm(x, self.weight, mx.finfo(mx.float32).eps)


class AudioInputEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.qwen_config().hidden_size
        self.n_codes = cfg.codebook_size
        self.table = nn.Embedding(cfg.num_codebooks * self.n_codes, d)
        self.aeos_in = mx.zeros((d,))
        self.norm = AudioRMSNorm(d)

    def frames(self, codes):
        q = codes.shape[-1]
        return self.norm(
            self.table(codes + mx.arange(q) * self.n_codes).sum(axis=-2) / math.sqrt(q)
        )

    def aeos(self):
        return self.norm(self.aeos_in)


class SpeakerAnchor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.speaker_dim, cfg.speaker_anchor_hidden)
        self.fc2 = nn.Linear(cfg.speaker_anchor_hidden, cfg.qwen_config().hidden_size)
        self.null_vec = mx.zeros((cfg.speaker_dim,))

    def __call__(self, refs=None, mask=None):
        if refs is None:
            x = self.null_vec[None]
        else:
            refs = refs.astype(mx.float32)
            if mask is None:
                x = refs.mean(axis=1)
            else:
                m = mask[..., None].astype(mx.float32)
                x = (refs * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1)
                x = mx.where(mask.any(axis=1, keepdims=True), x, self.null_vec)
        return self.fc2(nn.gelu(self.fc1(x.astype(self.null_vec.dtype))))


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lm = Qwen3(cfg.qwen_config())
        self.cb0_head = nn.Linear(
            cfg.qwen_config().hidden_size, cfg.codebook_size + 1, bias=False
        )

    def __call__(self, embeddings, cache=None):
        hidden = self.lm.model(None, input_embeddings=embeddings, cache=cache)
        return hidden, self.cb0_head(hidden)


def pack_prompt(
    tokenizer, text, lang, voice=None, ref_text=None, ref_codes=None, ref_lang=None
):
    if lang not in LANGUAGES or (ref_lang is not None and ref_lang not in LANGUAGES):
        raise ValueError(f"Supported language codes: {', '.join(LANGUAGES)}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must contain speech text")
    tokens, kinds, codes, sentinels = [], [], [], []
    q = ref_codes.shape[0] if ref_codes is not None else 32

    def span(content, language):
        ids = (
            []
            if language is None
            else [tokenizer.convert_tokens_to_ids(f"<|lang:{language}|>")]
        )
        ids += tokenizer.encode(content, add_special_tokens=False)
        ids.append(tokenizer.convert_tokens_to_ids("<|audio|>"))
        if any(i is None or i < 0 for i in ids):
            raise ValueError("Tokenizer is missing Maya2 control tokens")
        tokens.extend(ids)
        kinds.extend([0] * len(ids))
        codes.extend([[-1] * q for _ in ids])
        sentinels.append(len(tokens) - 1)

    if ref_codes is not None:
        if not ref_text or not ref_text.strip():
            raise ValueError("ref_text is required for in-context voice cloning")
        reference_language = ref_lang or lang
        span(ref_text, reference_language)
        n = ref_codes.shape[1]
        tokens.extend([0] * (n + 1))
        kinds.extend([1] * n + [3])
        codes.extend(np.asarray(ref_codes).T.tolist() + [[-1] * q])
        target_language = lang if reference_language != lang else None
    else:
        target_language = lang
    span(f"{{{voice}}}: {text.strip()}" if voice else text.strip(), target_language)
    return {"tokens": tokens, "kind": kinds, "codes": codes, "sentinels": sentinels}


class Model(nn.Module):
    preserve_ref_audio_path = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = "maya2"
        self.backbone = Backbone(config)
        self.depth = DepthDecoder(config)
        self.audio_in = AudioInputEmbedding(config)
        self.speaker_anchor = (
            SpeakerAnchor(config) if config.speaker_anchor_enabled else None
        )
        self._tokenizer = None
        self._codec = None
        self._speaker_encoder = None

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @property
    def voices(self):
        return list(VOICES)

    @property
    def model_quant_predicate(self):
        return lambda path, module: path.startswith("backbone.lm.")

    def sanitize(self, weights):
        result = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if (
                key == "backbone.lm.lm_head.weight"
                and self.config.qwen_config().tie_word_embeddings
            ):
                continue
            result[key] = value
        return result

    @classmethod
    def post_load_hook(cls, model, model_path):
        from transformers import AutoTokenizer

        model._tokenizer = AutoTokenizer.from_pretrained(Path(model_path) / "backbone")
        model.depth.set_dtype(mx.float16)
        return model

    def _load_codec(self):
        if self._codec is None:
            self._codec = Mimi.from_pretrained(
                self.config.codec_repo, self.config.codec_filename
            )
            self._codec.eval()
        return self._codec

    def _reference(self, audio, mode):
        if isinstance(audio, (str, Path)):
            audio, sample_rate = audio_read(str(audio), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sample_rate != self.sample_rate:
                divisor = math.gcd(sample_rate, self.sample_rate)
                audio = resample_poly(
                    audio, self.sample_rate // divisor, sample_rate // divisor
                )
        audio = mx.array(audio, dtype=mx.float32)
        if audio.ndim != 1 or audio.size < self.sample_rate // 10:
            raise ValueError(
                "Reference must contain at least 0.1 seconds of mono audio"
            )
        if not mx.all(mx.isfinite(audio)).item():
            raise ValueError("Reference contains non-finite samples")
        codes = None
        if mode in ("icl", "icl_spk"):
            codes = np.array(self._load_codec().encode(audio[None, None])[0])
        refs = None
        if mode in ("spk", "icl_spk"):
            if self._speaker_encoder is None:
                self._speaker_encoder = EcapaSpeakerEmbeddingExtractor.from_pretrained(
                    self.config.speaker_encoder_repo
                )
            audio16 = resample_poly(np.asarray(audio), 2, 3).astype(np.float32)
            refs = self._speaker_encoder.encode(audio16)[None, None]
        return codes, refs

    def compose_inputs(self, prompt, refs=None):
        x = self.backbone.lm.model.embed_tokens(mx.array(prompt["tokens"])[None])
        kinds = np.asarray(prompt["kind"])
        pos = np.flatnonzero(kinds == 1)
        if len(pos):
            codes = mx.array(np.asarray(prompt["codes"])[pos])
            x[:, mx.array(pos)] = self.audio_in.frames(codes)[None].astype(x.dtype)
        pos = np.flatnonzero(kinds == 3)
        if len(pos):
            x[:, mx.array(pos)] = self.audio_in.aeos().astype(x.dtype)
        if self.speaker_anchor is not None:
            x = x.at[:, mx.array(prompt["sentinels"])].add(
                self.speaker_anchor(refs)[:, None].astype(x.dtype)
            )
        return x

    def __call__(self, embeddings, cache=None):
        return self.backbone(embeddings, cache)

    def generate(
        self,
        text,
        voice=None,
        lang_code="en",
        ref_audio=None,
        ref_text=None,
        ref_lang=None,
        conditioning="icl_spk",
        temperature=0.9,
        top_k=64,
        top_p=1.0,
        depth_temperature=0.7,
        depth_top_k=64,
        presence_penalty=0.8,
        presence_window=12,
        max_tokens=1500,
        min_tokens=2,
        stream=False,
        streaming_interval=0.32,
        speed=1.0,
        normalize=True,
        **kwargs,
    ):
        if speed != 1.0:
            raise ValueError(
                "Maya2 supports speed=1.0; use punctuation to control pacing"
            )
        if conditioning not in ("icl_spk", "icl", "spk"):
            raise ValueError("conditioning must be icl_spk, icl, or spk")
        if max_tokens <= 0 or min_tokens < 0 or streaming_interval <= 0:
            raise ValueError("Token limits and streaming_interval must be positive")
        if temperature < 0 or depth_temperature < 0 or top_k < 0 or depth_top_k < 0:
            raise ValueError("Temperatures and top_k must be nonnegative")
        if not 0 < top_p <= 1 or presence_window < 0:
            raise ValueError("top_p must be in (0, 1] and presence_window nonnegative")
        if self._tokenizer is None:
            raise RuntimeError(
                "Load Maya2 with mlx_audio.tts.utils.load() before generating"
            )
        if normalize:
            text = normalise_text(text, lang_code)
        start = time.perf_counter()
        refs, ref_codes = None, None
        if ref_audio is not None:
            if voice is not None:
                raise ValueError("Use either a named voice or ref_audio")
            if conditioning != "spk" and not ref_text:
                raise ValueError("ref_text is required for in-context voice cloning")
            ref_codes, refs = self._reference(ref_audio, conditioning)
        elif voice is None:
            voice = "Ahana"
        if voice is not None:
            names = {v.lower(): v for v in VOICES}
            if voice.lower() not in names:
                raise ValueError(
                    f"Unknown voice {voice!r}; choose from {', '.join(VOICES)}"
                )
            voice = names[voice.lower()]
        prompt = pack_prompt(
            self._tokenizer, text, lang_code, voice, ref_text, ref_codes, ref_lang
        )
        embeddings = self.compose_inputs(prompt, refs)
        if (
            embeddings.shape[1] + max_tokens
            > self.config.qwen_config().max_position_embeddings
        ):
            raise ValueError("Prompt and max_tokens exceed the backbone context length")
        cache = make_prompt_cache(self.backbone.lm)
        memory = self.depth.speaker_memory(refs)
        decoder = MimiStreamingDecoder(self._load_codec())
        pending, history = [], []
        chunk_frames = max(1, round(streaming_interval * self.config.frame_rate))
        hazard, total_frames, segment = 0.0, 0, 0
        prompt_time = 0.0

        def result(final):
            nonlocal segment
            audio = mx.concatenate(pending, axis=-1).reshape(-1).astype(mx.float32)
            mx.eval(audio)
            elapsed = time.perf_counter() - start
            duration = audio.size / self.sample_rate
            item = GenerationResult(
                audio=audio,
                samples=audio.size,
                sample_rate=self.sample_rate,
                segment_idx=segment,
                token_count=total_frames,
                audio_duration=f"{duration:.2f}s",
                real_time_factor=elapsed
                / max(total_frames / self.config.frame_rate, 1e-6),
                prompt={
                    "tokens": embeddings.shape[1],
                    "time": prompt_time,
                    "tokens-per-sec": embeddings.shape[1] / max(prompt_time, 1e-6),
                },
                audio_samples={
                    "samples": audio.size,
                    "samples-per-sec": audio.size / max(elapsed, 1e-6),
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=stream,
                is_final_chunk=final,
            )
            segment += 1
            pending.clear()
            return item

        try:
            hidden, logits = self(embeddings, cache)
            mx.eval(hidden, logits)
            prompt_time = time.perf_counter() - start
            for _ in range(max_tokens):
                scaled = logits[:, -1].astype(mx.float32) / (temperature or 1.0)
                if history and presence_window and presence_penalty:
                    ids = mx.array(sorted(set(history[-presence_window:])))
                    scaled = scaled.at[:, ids].add(-presence_penalty)
                if top_k:
                    threshold = mx.sort(scaled, axis=-1)[
                        :, -min(top_k, scaled.shape[-1]) :
                    ][:, :1]
                    scaled = mx.where(scaled < threshold, -mx.inf, scaled)
                if top_p < 1:
                    scaled = apply_top_p(
                        scaled - mx.logsumexp(scaled, axis=-1, keepdims=True), top_p
                    )
                probs = mx.softmax(scaled, axis=-1)
                eos = self.config.codebook_size
                eos_prob = probs[0, eos].item()
                if total_frames >= min_tokens:
                    hazard = 1 - (1 - hazard) * (1 - eos_prob)
                    if (
                        mx.argmax(probs, axis=-1).item() == eos
                        or eos_prob >= 0.35
                        or hazard >= 0.6
                    ):
                        break
                # Keep codebook zero's sampling support valid when EOS dominates top-k.
                scaled = scaled.at[:, eos].add(-mx.inf)
                if not mx.any(mx.isfinite(scaled)).item():
                    scaled = logits[:, -1].astype(mx.float32).at[:, eos].add(-mx.inf)
                cb0 = make_sampler(temp=0.0 if temperature == 0 else 1.0)(scaled)
                codes = self.depth.generate(
                    hidden[:, -1], cb0, memory, depth_temperature, depth_top_k
                )
                mx.eval(codes)
                if stream and len(pending) >= chunk_frames:
                    yield result(False)
                pending.append(decoder.decode_frames(codes[:, :, None]))
                mx.eval(pending[-1])
                history.append(cb0.item())
                total_frames += 1
                feedback = self.audio_in.frames(codes)[:, None]
                hidden, logits = self(feedback, cache)
            if pending:
                yield result(True)
        finally:
            decoder.reset()

"""AR orchestration for the dots.tts MLX runtime — ``DotsTtsModel.generate``.

Imports mlx / numpy / stdlib (+ this package) and makes no calls. (+
transformers are pulled in transitively via mlx-lm; the math here is pure MLX.)
Audio I/O is WAV via stdlib ``wave`` + numpy; resampling is a numpy
Kaiser-windowed-sinc that mirrors
sinc_interp_kaiser (lowpass_filter_width=64, rolloff=0.95) to ~1e-6.

Ties every pure-MLX submodule (LLM trunk + eos head, AudioVAE encode/decode,
CAM++ x-vector, flow-matching DiT + coordinate_proj, patch encoder, the
hidden/latent/xvec projections) into the schedule-driven autoregressive decode that
upstream ``DotsTtsModel`` implements across ``_prepare_prompt_conditioning`` ->
``_prefill`` -> ``_decode`` -> vocode.

Distilled control flow (authoritative upstream refs in model.py / core.py):

  hidden_patch_size = 1, latent_patch_size = patch_size = 4, fm_hidden = 1024,
  llm_hidden = 1536, latent_dim = 128, hop_size = 1920.

  * prompt conditioning: load+resample prompt audio to 48 kHz mono, trim, pad to a
    multiple of patch_size*hop (=7680); x-vector from a 16 kHz fbank ->
    g_cond = xvec_proj(xvec * speaker_scale); prompt_latents =
    sample_from_latent(vae.encode(48k))[:, :-patch_size]; prompt_patches =
    normalize(prompt_latents).reshape(1, S, patch_size, 128).
  * prefill: embed schedule[:prefill_end], scatter patch_encoder(prompt_patches)
    into the prompt-span slots, one LLM forward; walk the prompt spans appending
    hidden (hidden_proj) + history (latent_proj) chunks to fm_sequence/fm_cfg_sequence.
  * decode: text runs -> LLM step + hidden chunk; audio spans -> EOS check (deferred
    stop), build attn_mask/pos_ids, FlowSolver.denoise one patch, append history,
    re-encode the new patch via patch_encoder over the FULL denormalized latent history
    (recompute-full; causal-equivalent to upstream's streaming decode_patch) -> take the
    last token -> LLM step, emit denorm patch.
  * vocode: concat emitted patches, AudioVAE.decode -> 48 kHz waveform.

The fm_sequence layout is ``[history... | latent_block(patch_size)]``: history is
causal, the latent block attends to all history + itself (``_build_fm_attn_mask``);
pos_ids run 0..fm_len-1 for history then fm_len..fm_len+patch_size-1 for the block
(``_build_fm_pos_ids``).
"""

from __future__ import annotations

import bisect
import gc
import math
import os
import wave
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

# Memory-ceiling safety guard: set BEFORE any heavy allocation.
_DOTS_MEMORY_LIMIT_GB = int(os.environ.get("MLX_AUDIO_DOTS_MEMORY_LIMIT_GB", "20"))
if _DOTS_MEMORY_LIMIT_GB > 0:
    mx.set_memory_limit(int(_DOTS_MEMORY_LIMIT_GB * (1 << 30)))

from .dit import FlowSolver  # noqa: E402
from .io_helper import IOHelper  # noqa: E402
from .layers import Linear  # noqa: E402
from .llm import DotsLLM  # noqa: E402
from .loader import (  # noqa: E402
    load_audiovae,
    load_coordinate_proj,
    load_dit,
    load_semantic_encoder,
    load_speaker,
)
from .speaker import kaldi_fbank  # noqa: E402
from .text import DotsTokenizer  # noqa: E402

_SPEAKER_FBANK_SAMPLE_RATE = 16000


def _resolve_num_steps(num_steps: int | None, mode: str) -> int:
    """Per-mode default for the solver step count.

    ``None`` resolves to 4 in meanflow mode (the published distilled NFE) and 10 in
    flow-matching mode (the existing default). An explicit value is always honored.
    """
    if num_steps is not None:
        return int(num_steps)
    return 4 if mode == "meanflow" else 10


# ---------------------------------------------------------------------------
# Audio I/O + resampling (pure numpy; no / the original implementation / librosa).
# ---------------------------------------------------------------------------
def _load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a PCM WAV file to a mono float32 ndarray in [-1, 1] + its sample rate."""
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sw} bytes")
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data.astype(np.float32), sr


def _kaiser_sinc_kernel(
    orig_freq: int,
    new_freq: int,
    *,
    lowpass_filter_width: int = 64,
    rolloff: float = 0.95,
    beta: float = 14.769656459379492,
) -> tuple[np.ndarray, int, int, int]:
    """Replicate the original implementation ``_get_sinc_resample_kernel`` (sinc_interp_kaiser).

    Returns ``(kernel[new_freq, K], width, orig_freq//gcd, new_freq//gcd)``.
    """
    g = math.gcd(int(orig_freq), int(new_freq))
    of = int(orig_freq) // g
    nf = int(new_freq) // g
    base_freq = min(of, nf) * rolloff
    width = math.ceil(lowpass_filter_width * of / base_freq)
    idx = np.arange(-width, width + of, dtype=np.float64)[None, None] / of
    t = np.arange(0, -nf, -1, dtype=np.float64)[:, None, None] / nf + idx
    t *= base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    inside = np.clip(1.0 - (t / lowpass_filter_width) ** 2, 0.0, None)
    window = np.i0(beta * np.sqrt(inside)) / np.i0(beta)
    t *= math.pi
    scale = base_freq / of
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(t == 0, 1.0, np.sin(t) / t)
    kernels = sinc * window * scale
    return kernels[:, 0, :].astype(np.float32), width, of, nf


def _resample(x: np.ndarray, orig_freq: int, new_freq: int, **kw) -> np.ndarray:
    """High-quality Kaiser-sinc resample (mirrors the original implementation.functional.resample)."""
    if orig_freq == new_freq:
        return x.astype(np.float32)
    kernel, width, of, nf = _kaiser_sinc_kernel(orig_freq, new_freq, **kw)
    length = x.shape[-1]
    xp = np.pad(x, (width, width + of))
    k = kernel.shape[-1]
    out_positions = (len(xp) - k) // of + 1
    starts = np.arange(out_positions) * of
    win = xp[starts[:, None] + np.arange(k)[None, :]]  # [out_positions, K]
    res = (win @ kernel.T).reshape(-1)  # interleave the nf phases
    target_length = math.ceil(nf * length / of)
    return res[:target_length].astype(np.float32)


def _trim_silence(x: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence below ``top_db`` dB of peak (librosa-style).

    Frame-energy gate over 2048-sample frames (512 hop), matching librosa.effects.trim
    defaults closely enough for prompt conditioning.
    """
    if x.size == 0:
        return x
    frame_length, hop = 2048, 512
    n = x.size
    n_frames = 1 + max(0, (n - frame_length) // hop)
    if n_frames <= 0:
        return x
    ref = np.max(np.abs(x)) + 1e-9
    threshold = ref * (10.0 ** (-top_db / 20.0))
    nonsilent = []
    for i in range(n_frames):
        s = i * hop
        frame = x[s : s + frame_length]
        rms = np.sqrt(np.mean(frame**2) + 1e-12)
        nonsilent.append(rms > threshold)
    nonsilent = np.asarray(nonsilent)
    if not nonsilent.any():
        return x
    first = int(np.argmax(nonsilent))
    last = int(len(nonsilent) - 1 - np.argmax(nonsilent[::-1]))
    start = first * hop
    end = min(n, (last + 1) * hop + frame_length)
    return x[start:end]


def _load_prompt_audio48k(
    prompt_audio: str | Path | mx.array | np.ndarray, sample_rate: int
) -> np.ndarray:
    """Load + trim a prompt clip to mono 48 kHz for prompt conditioning.

    Path inputs follow the runtime's native WAV path (load, trim, resample). Predecoded
    arrays are assumed to already be mono at ``sample_rate`` and are only trimmed.
    """
    if isinstance(prompt_audio, (str, Path)):
        raw, sr = _load_wav_mono(prompt_audio)
        raw = _trim_silence(raw, top_db=30.0)
        return _resample(raw, sr, sample_rate)

    raw = np.asarray(prompt_audio, dtype=np.float32)
    raw = np.squeeze(raw)
    if raw.ndim == 0:
        raise ValueError("prompt_audio array must contain at least one sample")
    if raw.ndim > 1:
        raw = raw.mean(axis=0)
    return _trim_silence(raw.astype(np.float32), top_db=30.0)


def _clean_onset(wav: np.ndarray, sr: int) -> np.ndarray:
    """Trim the fixed BigVGAN vocoder onset transient off the start of a dots.tts clip.

    dots.tts (via its BigVGAN vocoder) emits a deterministic ~50-150 ms low-level burst (a soft
    "hhh"/breath, byte-identical across clips) at sample 0, followed by a short silence gap before
    the real speech. It's a fixed ~-26 dBFS transient — on quiet dubs (peaks ~-20 dBFS) it's audible.

    Mirrors ``clean_tts_audio._speech_bounds`` (smoothed-RMS gate + sustained-run requirement) to find
    the *real* speech onset, then keeps ``[onset - lead_pad, end]``. Differences vs the MisoTTS cleaner:
    a tighter ``rel_db`` (12, not 16) and lighter smoothing (40 ms, not 150) so the threshold sits
    cleanly *between* the low transient and the louder speech body — the wide 16 dB / 150 ms gate would
    flag the transient itself as "speech" (it's within 16 dB of the body) and trim nothing. The lead-pad
    is kept small (30 ms) so a soft real onset consonant is never clipped; the transient + dead-air gap
    that precede the speech are both removed. A 10 ms fade-in suppresses the cut click.

    Pure numpy (no frameworks). Returns the trimmed-and-faded mono float32 array.
    """
    rel_db, hop_ms, smooth_ms, run_ms = 12.0, 10.0, 40.0, 60.0
    lead_pad_ms, fade_ms = 30.0, 10.0
    x = wav.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    hop = max(1, int(sr * hop_ms / 1000))
    n = max(1, (len(x) - hop) // hop)
    rms = np.array(
        [np.sqrt((x[i * hop : i * hop + hop] ** 2).mean() + 1e-12) for i in range(n)]
    )
    db = 20 * np.log10(rms + 1e-9)
    k = max(1, int(smooth_ms / hop_ms))
    sm = np.convolve(
        db, np.ones(k) / k, mode="same"
    )  # light smooth — keep transient/speech distinct
    above = sm > (sm.max() - rel_db)
    run = max(1, int(run_ms / hop_ms))  # require a sustained run (not the burst's edge)
    ok = np.convolve(above.astype(int), np.ones(run, dtype=int), mode="valid") == run
    starts = np.where(ok)[0]
    if len(starts) == 0:  # nothing looked like speech -> leave untouched
        return x
    s0 = int(starts[0] * hop)
    i0 = max(
        0, s0 - int(lead_pad_ms / 1000 * sr)
    )  # small lead-pad: never clip the first phoneme
    y = x[i0:].copy()
    nf = min(int(sr * fade_ms / 1000), len(y) // 2)  # anti-click fade-in at the cut
    if nf > 0:
        y[:nf] *= np.linspace(0.0, 1.0, nf, dtype=y.dtype)
    return y


# ---------------------------------------------------------------------------
# xvec_proj = Linear(512 -> 1024) -> affine LayerNorm(1024).
# ---------------------------------------------------------------------------
class _AffineLayerNorm:
    """``nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)`` (weight + bias)."""

    def __init__(self, weight: mx.array, bias: mx.array, eps: float = 1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        xf = x.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) ** 2, axis=-1, keepdims=True)
        out = (xf - mean) * mx.rsqrt(var + self.eps)
        out = out * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return out.astype(in_dtype)


class _XvecProj:
    """``Sequential(Linear(512, 1024), LayerNorm(1024))`` (the ``xvec_proj.*``)."""

    def __init__(self, linear: Linear, norm: _AffineLayerNorm):
        self.linear = linear
        self.norm = norm

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.linear(x))


class DotsTtsModel:
    """Full pure-MLX dots.tts TTS model: ``generate(text, prompt_audio, ...) -> wav``.

    Construct via ``DotsTtsModel.from_pretrained(path, dtype)``. Holds every wired
    submodule + the schedule helpers; runs the schedule-driven AR decode end to end.
    """

    def __init__(
        self,
        *,
        tokenizer: DotsTokenizer,
        llm: DotsLLM,
        vae,
        speaker,
        flow_solver: FlowSolver,
        patch_encoder,
        io_helper: IOHelper,
        hidden_proj: Linear,
        latent_proj: Linear,
        xvec_proj: _XvecProj,
        patch_size: int = 4,
        hop_size: int = 1920,
        latent_dim: int = 128,
        dtype: mx.Dtype = mx.float32,
        mode: str = "flow_matching",
    ):
        self.tokenizer = tokenizer
        self.llm = llm
        self.vae = vae
        self.speaker = speaker
        self.flow_solver = flow_solver
        self.patch_encoder = patch_encoder
        self.io = io_helper
        self.hidden_proj = hidden_proj
        self.latent_proj = latent_proj
        self.xvec_proj = xvec_proj
        self.patch_size = patch_size  # latent_patch_size
        self.hidden_patch_size = 1
        self.hop_size = hop_size
        self.fm_hidden = 1024
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.mode = mode
        self.sample_rate = 48000
        self._model_dir: Path | None = None
        self._compat_hash: str | None = None

    # region construction
    @classmethod
    def from_pretrained(
        cls, path: str | Path, dtype: mx.Dtype = mx.float32
    ) -> "DotsTtsModel":
        """Load + wire every submodule from a converted ``weights/dots_tts_mlx`` dir."""
        from .loader import ModelConfig

        path = Path(path)
        config = ModelConfig.from_checkpoint(path)

        core = str(path / "core.safetensors")
        vocoder = str(path / "vocoder.safetensors")
        speaker_st = str(path / "speaker.safetensors")
        llm_config_json = str(path / "llm_config.json")
        latent_stats = str(path / "latent_stats.npz")

        tokenizer = DotsTokenizer.from_pretrained(path / "tokenizer")
        llm = DotsLLM.from_core(
            core, llm_config_json, dtype=dtype, quantization=config.quantization
        )
        vae = load_audiovae(vocoder, dtype=dtype, with_encoder=True)
        speaker = load_speaker(speaker_st, dtype=dtype)
        dit = load_dit(core, dtype=dtype)
        coord = load_coordinate_proj(core, dtype=dtype)
        flow_solver = FlowSolver(dit, coord, latent_dim=config.latent_dim)
        patch_encoder = load_semantic_encoder(core, dtype=dtype)
        io_helper = IOHelper(latent_stats)

        raw = mx.load(core)
        hidden_proj = Linear(
            raw["hidden_proj.weight"].astype(dtype),
            raw["hidden_proj.bias"].astype(dtype),
        )
        latent_proj = Linear(
            raw["latent_proj.weight"].astype(dtype),
            raw["latent_proj.bias"].astype(dtype),
        )
        xvec_proj = _XvecProj(
            Linear(
                raw["xvec_proj.0.weight"].astype(dtype),
                raw["xvec_proj.0.bias"].astype(dtype),
            ),
            _AffineLayerNorm(
                raw["xvec_proj.1.weight"].astype(dtype),
                raw["xvec_proj.1.bias"].astype(dtype),
            ),
        )

        # hop_size = total vocoder upsample factor (= 1920 for this checkpoint); the
        # pad-multiple is patch_size * hop_size (= 7680). latent_dim from config.
        hop_size = math.prod(config.vocoder.upsample_rates)

        model = cls(
            tokenizer=tokenizer,
            llm=llm,
            vae=vae,
            speaker=speaker,
            flow_solver=flow_solver,
            patch_encoder=patch_encoder,
            io_helper=io_helper,
            hidden_proj=hidden_proj,
            latent_proj=latent_proj,
            xvec_proj=xvec_proj,
            patch_size=config.patch_size,
            hop_size=hop_size,
            latent_dim=config.latent_dim,
            dtype=dtype,
            mode=config.mode,
        )
        from .profile import model_compat_hash

        model._model_dir = path
        model._compat_hash = model_compat_hash(path)
        return model

    # endregion construction

    # region prompt conditioning
    def _xvector_from_audio48k(self, audio48k: np.ndarray) -> mx.array:
        """CAM++ x-vector from a 48 kHz mono waveform: resample to 16 kHz -> fbank."""
        audio16k = _resample(audio48k, 48000, _SPEAKER_FBANK_SAMPLE_RATE)
        # Crop to max_audio_seconds (10s) — start=0 deterministically (upstream).
        max_len = _SPEAKER_FBANK_SAMPLE_RATE * 10
        if audio16k.shape[-1] > max_len:
            audio16k = audio16k[:max_len]
        fbank = kaldi_fbank(audio16k, sample_rate=_SPEAKER_FBANK_SAMPLE_RATE)  # [T, 80]
        fbank_mx = mx.array(np.asarray(fbank), dtype=mx.float32)[None]  # [1, T, 80]
        xvec = self.speaker(fbank_mx)  # [1, 512]
        return xvec

    def _prepare_prompt_conditioning(
        self,
        prompt_audio48k: np.ndarray | None,
        *,
        speaker_scale: float,
    ) -> tuple[mx.array | None, mx.array | None, mx.array | None]:
        """Return ``(g_cond [1, 1024], prompt_patches [1, S, patch_size, 128] | None,
        prompt_denorm_latents [1, S*patch_size, 128] | None)``.

        For a reference clone (``prompt_audio48k`` given) this always builds the full
        **in-context** prompt: the speaker ``g_cond`` plus the prefill latents.
        ``prompt_patches`` is the NORMALIZED, reshaped prompt latent (FM history +
        prefill scatter); ``prompt_denorm_latents`` is the matching DENORMALIZED stream
        that seeds the patch-encoder recompute-full history (upstream feeds the
        denormalized ``prompt_latents_sampled`` into ``patch_encoder.prefill``). With no
        reference audio (plain text-to-speech) it returns ``(None, None, None)``.
        """
        if prompt_audio48k is None:
            return None, None, None

        # Pad to a multiple of patch_size * hop_size (= 7680).
        chunk = self.patch_size * self.hop_size
        n = prompt_audio48k.shape[-1]
        target = math.ceil(n / chunk) * chunk
        if target > n:
            prompt_audio48k = np.pad(prompt_audio48k, (0, target - n))

        xvec = self._xvector_from_audio48k(prompt_audio48k)  # [1, 512] fp32
        speaker_embedding = xvec * float(speaker_scale)
        g_cond = self.xvec_proj(speaker_embedding.astype(self.dtype))  # [1, 1024]
        g_cond = g_cond.astype(self.dtype)

        wav = mx.array(prompt_audio48k, dtype=mx.float32)[None, None]  # [1, 1, S]
        latent = self.vae.encode(wav)  # [1, 256, T]
        sampled = self.io.sample_from_latent(latent)  # [1, T, 128] (denormalized)
        sampled = sampled[:, : -self.patch_size]  # drop the trailing patch
        normalized = self.io.normalize(sampled)  # [1, S*patch_size, 128]
        s = normalized.shape[1] // self.patch_size
        keep = s * self.patch_size
        prompt_patches = normalized[:, :keep].reshape(
            1, s, self.patch_size, self.latent_dim
        )
        # Denormalized stream (same time span as prompt_patches) for the recompute-full
        # patch-encoder history; upstream patch_encoder operates on denormalized latents.
        prompt_denorm_latents = sampled[:, :keep]  # [1, S*patch_size, 128]
        return (
            g_cond,
            prompt_patches.astype(self.dtype),
            prompt_denorm_latents.astype(self.dtype),
        )

    # endregion prompt conditioning

    # region fm-sequence builders
    def _build_fm_attn_mask(self, fm_seq_len: int) -> mx.array:
        """``[1, L, L]`` bool mask; history causal, latent block attends to all + self.

        L = fm_seq_len + patch_size. Mirrors upstream ``_build_fm_attn_mask`` for the
        simple (non-bucketed) case where ``latent_start == fm_seq_len``.
        """
        p = self.patch_size
        h = self.hidden_patch_size
        latent_start = fm_seq_len  # total_len - patch_size == fm_seq_len here
        total = fm_seq_len + p
        mask = np.zeros((total, total), dtype=bool)
        block_start = fm_seq_len - h
        if block_start > 0:
            causal = ~np.triu(np.ones((block_start, block_start), dtype=bool), k=1)
            mask[:block_start, :block_start] = causal
        # The last hidden token (block_start..fm_seq_len) attends to all history + block.
        mask[block_start:fm_seq_len, :fm_seq_len] = True
        mask[block_start:fm_seq_len, latent_start:] = True
        # The latent block attends to all history + itself.
        mask[latent_start:, :fm_seq_len] = True
        mask[latent_start:, latent_start:] = True
        return mx.array(mask)[None]  # [1, L, L]

    def _build_fm_pos_ids(self, fm_seq_len: int) -> mx.array:
        """``[1, L]`` positions: 0..fm_len-1 (history), fm_len..fm_len+p-1 (block)."""
        p = self.patch_size
        total = fm_seq_len + p
        pos = np.zeros((total,), dtype=np.float32)
        pos[:fm_seq_len] = np.arange(fm_seq_len, dtype=np.float32)
        pos[fm_seq_len:] = np.arange(fm_seq_len, fm_seq_len + p, dtype=np.float32)
        return mx.array(pos)[None]  # [1, L]

    # endregion fm-sequence builders

    # region generate
    def generate(
        self,
        text: str,
        *,
        prompt_audio: str | Path | mx.array | np.ndarray | None = None,
        prompt_text: str | None = None,
        profile: "SpeakerProfile | None" = None,  # noqa: F821
        num_steps: int | None = None,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        language: str | None = None,
        seed: int = 42,
        max_generate_length: int = 500,
        eos_threshold: float = 0.8,
        trim_onset: bool = True,
        streaming_decode: bool = True,
    ) -> dict:
        """Synthesize ``text`` (optionally cloning ``prompt_audio``) -> 48 kHz wav.

        Returns ``{"audio": mx.array [1, T], "sample_rate": 48000,
        "num_patches": int}``.

        ``trim_onset`` (default on) removes the fixed ~50-150 ms BigVGAN vocoder onset
        transient (a soft "hhh"/breath at sample 0) via an energy gate + 10 ms fade. It
        is applied here — not bolted onto the CLI — so EVERY caller (Python API, the
        persistent service, the CLI) gets clean audio by default. Pass ``trim_onset=False``
        for the raw vocoder output.

        ``streaming_decode`` (default on) re-encodes each generated patch incrementally
        through a maintained conv_tail + per-layer KV caches (O(n) over the clip).
        ``streaming_decode=False`` selects the legacy recompute-full path (re-encodes the
        full history every patch, O(n^3)) — kept as a fallback and the on-device parity
        oracle. Both paths are numerically identical (the encoder is fully causal, no
        rotary/qk-norm).
        """
        mx.random.seed(int(seed))
        np.random.seed(int(seed))
        num_steps = _resolve_num_steps(num_steps, self.mode)

        # --- prompt conditioning: from a saved profile, or computed from prompt_audio. ---
        patch_emb_override = None
        if profile is not None:
            if prompt_audio is not None or prompt_text is not None:
                raise ValueError(
                    "generate(profile=…) is mutually exclusive with prompt_audio/prompt_text."
                )
            if self._compat_hash is not None:
                profile.check_compat(self._compat_hash)
            # integrity: the cached arrays must line up with prompt_patch_count.
            if (
                int(profile.prompt_patches.shape[1]) != profile.prompt_patch_count
                or int(profile.patch_emb.shape[1]) != profile.prompt_patch_count
                or int(profile.prompt_denorm_latents.shape[1])
                != profile.prompt_patch_count * self.patch_size
            ):
                raise ValueError(
                    "corrupt speaker profile: prompt_patch_count mismatch."
                )
            g_cond = profile.g_cond.astype(self.dtype)
            prompt_patches = profile.prompt_patches.astype(self.dtype)
            prompt_denorm_latents = profile.prompt_denorm_latents.astype(self.dtype)
            patch_emb_override = profile.patch_emb.astype(self.dtype)
            prompt_text = profile.prompt_text  # rebuild the identical schedule
            use_prompt_prefill = True
            # speaker_scale is already baked into profile.g_cond; the arg is ignored here.
            # RNG sync: the ref-audio path consumes exactly one mx.random.normal draw in
            # io.sample_from_latent (the VAE-latent sampling) before the decode loop. The
            # profile path skips that encode, so replicate the single RNG advance here —
            # MLX's global RNG advances per-call (shape-independent), so the decode-loop
            # noise then matches one-shot generate exactly (byte-parity). Do NOT remove:
            # the round-trip parity gate (test_profile_generate_matches_one_shot) depends on it.
            mx.random.normal((1,))
        else:
            prompt_audio48k = None
            if prompt_audio is not None:
                prompt_audio48k = _load_prompt_audio48k(prompt_audio, self.sample_rate)
            if prompt_audio48k is not None and not prompt_text:
                raise ValueError(
                    "Reference cloning requires the reference transcript "
                    "(prompt_text / --ref-text). The x-vector-only (no-transcript) clone "
                    "mode has been removed — it drifted on longer utterances. Pass the "
                    "transcript, or omit the reference audio for plain text-to-speech."
                )
            # cloning => always in-context prefill; no reference => plain TTS.
            use_prompt_prefill = prompt_audio48k is not None
            g_cond, prompt_patches, prompt_denorm_latents = (
                self._prepare_prompt_conditioning(
                    prompt_audio48k,
                    speaker_scale=speaker_scale,
                )
            )
        prompt_patch_count = (
            0 if prompt_patches is None else int(prompt_patches.shape[1])
        )

        # --- build the generation schedule. ---
        schedule_ids = self.tokenizer.build_generation_schedule(
            text,
            prompt_text=prompt_text,
            language=language,
            max_audio_tokens=max_generate_length,
        )
        schedule = mx.array(schedule_ids, dtype=mx.int32)[None]  # [1, N]
        span_id = self.tokenizer.audio_gen_span_id
        comp_id = self.tokenizer.audio_comp_span_id
        audio_ids = {span_id, comp_id}
        span_positions = [i for i, t in enumerate(schedule_ids) if t in audio_ids]
        if len(span_positions) < prompt_patch_count + 1:
            raise ValueError(
                f"schedule has {len(span_positions)} spans, need "
                f"{prompt_patch_count + 1} (prompt + >=1 decode)."
            )

        # --- FM sequence buffers (grown incrementally as lists of [1, n, 1024]). ---
        self._fm_chunks: list[mx.array] = []
        self._fm_cfg_chunks: list[mx.array] = []
        self._fm_seq_len = 0
        # All DENORMALIZED latent patches seen so far (each [1, patch_size, 128]); the
        # patch re-encode runs the patch_encoder over this full history every step so
        # the causal context (prior patches + ds_proj left-context) is preserved. Seed
        # with the prompt's denormalized latents (upstream feeds these into
        # patch_encoder.prefill) so the FIRST generated patch already has left-context.
        self._denorm_patch_history: list[mx.array] = []
        if prompt_denorm_latents is not None and prompt_patch_count > 0:
            for i in range(prompt_patch_count):
                start = i * self.patch_size
                self._denorm_patch_history.append(
                    prompt_denorm_latents[:, start : start + self.patch_size]
                )

        cache = self.llm.make_cache()

        # Streaming patch-decode state (default path). Prefill it from the SAME
        # denormalized prompt stream that seeds the recompute-full history above, so the
        # KV caches reflect recompute-full's prompt view. No RNG is drawn here (prefill is
        # deterministic) -> streaming vs recompute-full stay in RNG lock-step.
        self._streaming_decode = bool(streaming_decode)
        self._patch_encoder_state = None
        if self._streaming_decode:
            self._patch_encoder_state = self.patch_encoder.init_decode_state(
                dtype=self.dtype
            )
            if prompt_denorm_latents is not None and prompt_patch_count > 0:
                self.patch_encoder.prefill(
                    prompt_denorm_latents[:, : prompt_patch_count * self.patch_size],
                    self._patch_encoder_state,
                )

        llm_hiddens: mx.array | None = None

        # --- prefill. ---
        position = self._prefill(
            schedule_ids,
            schedule,
            span_positions=span_positions,
            prompt_patches=prompt_patches,
            prompt_denorm_latents=prompt_denorm_latents,
            cache=cache,
            patch_emb=patch_emb_override,
        )
        llm_hiddens = self._last_prefill_hidden

        # --- decode loop. ---
        g_cond_run = None if g_cond is None else g_cond.astype(self.dtype)
        null_g_cond = mx.zeros((1, self.fm_hidden), dtype=self.dtype)

        emitted: list[mx.array] = []
        should_drop_regenerated = use_prompt_prefill  # drop the first regenerated patch
        n_sched = len(schedule_ids)
        # span_cursor: index into span_positions for the NEXT audio span.
        span_cursor = bisect.bisect_left(span_positions, position)
        end_flag = False

        while position < n_sched and not end_flag:
            token_id = schedule_ids[position]
            if token_id in audio_ids:
                stop_after = self._should_stop_after_current_audio(
                    llm_hiddens, eos_threshold=eos_threshold
                )
                # build FM inputs + denoise one patch
                fm_seq = mx.concatenate(self._fm_chunks, axis=1)
                fm_cfg = mx.concatenate(self._fm_cfg_chunks, axis=1)
                fm_seq_len = self._fm_seq_len
                attn_mask = self._build_fm_attn_mask(fm_seq_len)
                pos_ids = self._build_fm_pos_ids(fm_seq_len)
                # pad both sequences with a zero latent block (placeholder slots).
                pad = mx.zeros((1, self.patch_size, self.fm_hidden), dtype=self.dtype)
                input_seq = mx.concatenate([fm_seq, pad], axis=1)
                cfg_seq = mx.concatenate([fm_cfg, pad], axis=1)
                noise = mx.random.normal((1, self.patch_size, self.latent_dim)).astype(
                    self.dtype
                )
                if self.mode == "meanflow":
                    patch = self.flow_solver.meanflow_sample(
                        input_sequence=input_seq,
                        attn_mask=attn_mask,
                        pos_ids=pos_ids,
                        g_cond=g_cond_run if g_cond_run is not None else null_g_cond,
                        num_steps=num_steps,
                        patch_size=self.patch_size,
                        noise=noise,
                    )  # normalized latent [1, patch_size, 128]
                else:
                    patch = self.flow_solver.denoise(
                        input_sequence=input_seq,
                        cfg_sequence=cfg_seq,
                        attn_mask=attn_mask,
                        pos_ids=pos_ids,
                        g_cond=g_cond_run if g_cond_run is not None else null_g_cond,
                        guidance_scale=guidance_scale,
                        num_steps=num_steps,
                        patch_size=self.patch_size,
                        noise=noise,
                    )  # normalized latent [1, patch_size, 128]
                mx.eval(patch)

                # consume the patch: append history + re-encode for the LLM.
                self._append_history_chunk(patch)
                denorm = self.io.denormalize(patch)  # [1, patch_size, 128]
                llm_hiddens, cache = self._reencode_and_step(cache, denorm)

                # if the next token is also an audio span, append a hidden chunk.
                if position + 1 < n_sched and schedule_ids[position + 1] in audio_ids:
                    self._append_hidden_chunk(llm_hiddens)

                position += 1
                span_cursor += 1

                if should_drop_regenerated:
                    should_drop_regenerated = False  # discard the prompt-tail patch
                else:
                    emitted.append(denorm)

                if stop_after:
                    end_flag = True
                continue

            # TEXT run: consume schedule[position:next_audio_position] in one LLM step.
            next_audio_position = (
                span_positions[span_cursor]
                if span_cursor < len(span_positions)
                else n_sched
            )
            text_chunk = schedule[:, position:next_audio_position]
            llm_hiddens, cache = self.llm.step(input_ids=text_chunk, cache=cache)
            self._append_hidden_chunk(llm_hiddens)
            position = next_audio_position

        if not emitted:
            raise RuntimeError(
                "Generation produced no payload latents (EOS fired immediately or "
                "the schedule provided no effective decode span)."
            )

        # --- vocode. ---
        latents = mx.concatenate(emitted, axis=1)  # [1, T_patches*patch_size, 128]
        wav = self.vae.decode(latents.transpose(0, 2, 1))  # [1, 1, T*1920]
        wav = wav[:, 0, :]  # [1, T*1920]
        mx.eval(wav)
        if trim_onset:
            cleaned = _clean_onset(
                np.asarray(wav[0].astype(mx.float32)), self.sample_rate
            )
            wav = mx.array(cleaned)[None]  # [1, T']
        return {
            "audio": wav,
            "sample_rate": self.sample_rate,
            "num_patches": len(emitted),
        }

    def generate_long(
        self,
        text: str,
        *,
        prompt_audio: str | Path | mx.array | np.ndarray | None = None,
        prompt_text: str | None = None,
        profile: "SpeakerProfile | None" = None,  # noqa: F821
        num_steps: int | None = None,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        language: str | None = None,
        seed: int = 42,
        max_generate_length: int = 500,
        eos_threshold: float = 0.8,
        trim_onset: bool = True,
        streaming_decode: bool = True,
        gap_ms: int = 80,
        max_chars: int | None = None,
        retry_degenerate: bool = True,
        max_retries: int = 2,
        validator: "Callable[[np.ndarray, str, int], bool] | None" = None,  # noqa: F821
    ) -> dict:
        """Render long/multilingual text by sentence-chunking.

        Splits ``text`` into TTS-sized chunks (sentence-first, word-safe length-cap
        fallback), generates each chunk INDEPENDENTLY via ``generate()`` (short context ->
        EOS fires correctly per sentence -> no truncation/drift; no O(n^2) growth), drains
        Metal between chunks, and concatenates with ``gap_ms`` silence between sentences.
        Numerically just N independent ``generate()`` calls -- no model/parity change.

        Conditioning/sampling args mirror ``generate``; ``profile`` is mutually exclusive
        with ``prompt_audio``/``prompt_text`` (enforced inside ``generate``). ``streaming_decode``
        (default on) is forwarded to each per-chunk ``generate`` call. Returns
        ``{"audio": [1, T], "sample_rate", "num_chunks", "num_patches"}``. (Speed is a CLI
        post-process, matching ``generate``.)

        ``num_steps=None`` (default) resolves per-mode in ``generate`` (4 meanflow / 10
        flow-matching).
        """
        from .chunking import (
            assemble_chunks,
            chunk_health,
            resolve_max_chars,
            split_for_generation,
        )

        cap = (
            int(max_chars) if max_chars else resolve_max_chars(text, language=language)
        )
        chunks = split_for_generation(text, max_chars=cap, language=language)
        if not chunks:
            raise ValueError("generate_long: text is empty after splitting.")

        max_retries = max(0, int(max_retries))
        pieces: list[np.ndarray] = []
        total_patches = 0
        total_retries = 0
        unrecovered = 0

        def _gen_once(chunk: str, chunk_seed: int):
            out = self.generate(
                chunk,
                prompt_audio=prompt_audio,
                prompt_text=prompt_text,
                profile=profile,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                language=language,
                seed=chunk_seed,
                max_generate_length=max_generate_length,
                eos_threshold=eos_threshold,
                trim_onset=trim_onset,
                streaming_decode=streaming_decode,
            )
            return np.asarray(out["audio"][0].astype(mx.float32)), int(
                out["num_patches"]
            )

        def _accept(audio_np: np.ndarray, chunk: str) -> bool:
            if not chunk_health(audio_np, chunk, self.sample_rate, language=language):
                return False
            if validator is not None:
                try:
                    return bool(validator(audio_np, chunk, self.sample_rate))
                except (
                    Exception
                ):  # noqa: BLE001 -- a flaky validator must not crash the render
                    return False
            return True

        for i, chunk in enumerate(chunks):
            best_audio, best_patches = _gen_once(
                chunk, seed
            )  # attempt 0 = current seed
            accepted = _accept(best_audio, chunk)
            attempt = 0
            while not accepted and retry_degenerate and attempt < max_retries:
                attempt += 1
                total_retries += 1
                mx.synchronize()
                mx.clear_cache()
                gc.collect()
                chunk_seed = int(seed) + 1_000_003 * (i + 1) + 101 * attempt
                cand_audio, cand_patches = _gen_once(chunk, chunk_seed)
                if _accept(cand_audio, chunk):
                    best_audio, best_patches, accepted = cand_audio, cand_patches, True
                elif cand_audio.size > best_audio.size:
                    best_audio, best_patches = cand_audio, cand_patches
            if not accepted:
                unrecovered += 1
            pieces.append(best_audio)
            total_patches += best_patches
            mx.synchronize()
            mx.clear_cache()
            gc.collect()

        full = assemble_chunks(pieces, self.sample_rate, gap_ms)
        return {
            "audio": mx.array(full)[None],
            "sample_rate": self.sample_rate,
            "num_chunks": len(chunks),
            "num_patches": total_patches,
            "retries": total_retries,
            "unrecovered": unrecovered,
        }

    # endregion generate

    # region decode helpers
    def _append_hidden_chunk(self, hidden_chunk: mx.array) -> None:
        """Project the last hidden token -> fm_sequence; zeros -> fm_cfg_sequence."""
        last_hidden = hidden_chunk[:, -self.hidden_patch_size :, :]  # [1, 1, 1536]
        projected = self.hidden_proj(last_hidden).astype(self.dtype)  # [1, 1, 1024]
        null_projected = self.hidden_proj(mx.zeros_like(last_hidden)).astype(self.dtype)
        self._fm_chunks.append(projected)
        self._fm_cfg_chunks.append(null_projected)
        self._fm_seq_len += projected.shape[1]

    def _append_history_chunk(self, latent_chunk: mx.array) -> None:
        """Project a latent patch [1, patch_size, 128] -> BOTH fm sequences."""
        history_latent = self.latent_proj(latent_chunk).astype(
            self.dtype
        )  # [1, p, 1024]
        self._fm_chunks.append(history_latent)
        self._fm_cfg_chunks.append(history_latent)
        self._fm_seq_len += history_latent.shape[1]

    def _reencode_and_step(self, cache, denorm_patch: mx.array):
        """Re-encode the new denormalized patch (with full causal context) -> one LLM step.

        RECOMPUTE-FULL: upstream ``_consume_audio_patch`` -> ``patch_encoder.decode_patch``
        re-encodes the new patch against a PERSISTENT per-layer KV cache (all prior
        patches) + a ``conv_tail`` (so the causal stride-2 ``ds_proj`` conv sees the
        previous patch's last frame, not a zero left-pad). Rather than port that
        incremental KV-cache/conv_tail streaming, we run ``patch_encoder`` over the FULL
        denormalized history every step: the encoder is fully causal (causal ds_proj
        conv with left-padding + causal transformer mask), so a recompute over
        ``[1, 4n, 128]`` is NUMERICALLY IDENTICAL to the streaming decode_patch — token n
        only ever attends to tokens 0..n and the conv's left-context is the natural
        prefix of the full sequence. The patch_encoder maps each 4-frame patch -> 1 LLM
        token (out_ds_rate=2 downsampled tokens grouped back by _project_embeddings), so
        n patches -> [1, n, 1536]; we take the LAST token as the new patch's embedding.
        O(n^2) over patches — fine for short clips; the faithful alternative to
        upstream's incremental decode_patch.
        """
        if getattr(self, "_streaming_decode", False):
            emb = self.patch_encoder.decode_patch(
                denorm_patch, self._patch_encoder_state
            ).astype(self.dtype)
            return self.llm.step(inputs_embeds=emb, cache=cache)
        # --- recompute-full fallback / parity oracle (kept verbatim, O(n^3)) ---
        self._denorm_patch_history.append(denorm_patch)
        history = mx.concatenate(self._denorm_patch_history, axis=1)  # [1, 4n, 128]
        emb_full = self.patch_encoder(history)  # [1, n, 1536]
        emb = emb_full[:, -1:, :].astype(self.dtype)  # newest patch -> [1, 1, 1536]
        return self.llm.step(inputs_embeds=emb, cache=cache)

    def _should_stop_after_current_audio(
        self, llm_hiddens: mx.array | None, *, eos_threshold: float
    ) -> bool:
        if llm_hiddens is None:
            return False
        logits = self.llm.eos_logits(llm_hiddens)  # [1, L, 2]
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        eos_p = float(probs[0, -1, 1])
        return eos_p > eos_threshold

    def enroll(
        self,
        prompt_audio: str | Path,
        prompt_text: str,
        *,
        speaker_scale: float = 1.5,
    ) -> "SpeakerProfile":  # noqa: F821
        """Compute + return a reusable SpeakerProfile for ``prompt_audio``.

        Caches the target-independent reference artifacts (g_cond, the AudioVAE-encoded
        prompt latents, and the patch-encoder embeddings). ``prompt_text`` is REQUIRED —
        it is stored so a later ``generate(profile=…)`` rebuilds the identical schedule.
        """
        from .profile import SpeakerProfile

        if not prompt_text:
            raise ValueError(
                "enroll() requires a non-empty prompt_text (the reference transcript)."
            )
        if self._compat_hash is None:
            raise ValueError(
                "enroll() requires a model built via from_pretrained (no compat hash)."
            )

        prompt_audio48k = _load_prompt_audio48k(prompt_audio, self.sample_rate)
        g_cond, prompt_patches, prompt_denorm_latents = (
            self._prepare_prompt_conditioning(
                prompt_audio48k, speaker_scale=speaker_scale
            )
        )
        s = int(prompt_patches.shape[1])
        flat = prompt_denorm_latents[:, : s * self.patch_size]  # denormalized
        patch_emb = self.patch_encoder(flat).astype(self.dtype)
        mx.eval(g_cond, prompt_patches, prompt_denorm_latents, patch_emb)

        return SpeakerProfile(
            g_cond=g_cond,
            prompt_patches=prompt_patches,
            prompt_denorm_latents=prompt_denorm_latents,
            patch_emb=patch_emb,
            prompt_text=prompt_text,
            prompt_patch_count=s,
            speaker_scale=float(speaker_scale),
            latent_dim=self.latent_dim,
            patch_size=self.patch_size,
            hop_size=self.hop_size,
            sample_rate=self.sample_rate,
            dtype=str(self.dtype).split(".")[-1],
            compat_hash=self._compat_hash,
        )

    def _prefill(
        self,
        schedule_ids: list[int],
        schedule: mx.array,
        *,
        span_positions: list[int],
        prompt_patches: mx.array | None,
        prompt_denorm_latents: mx.array | None = None,
        cache,
        patch_emb: mx.array | None = None,
    ) -> int:
        """Run the prefill forward + walk the prompt spans. Returns ``prefill_end``.

        Stores the last LLM hidden in ``self._last_prefill_hidden``.
        """
        prompt_patch_count = (
            0 if prompt_patches is None else int(prompt_patches.shape[1])
        )
        prefill_end = span_positions[prompt_patch_count]
        prompt_span_positions = span_positions[:prompt_patch_count]

        if prefill_end == 0:
            self._last_prefill_hidden = None
            return 0

        # embed schedule[:prefill_end]; scatter prompt patch embeddings into spans.
        head_ids = schedule[:, :prefill_end]  # [1, prefill_end]
        inputs_embeds = self.llm._model.model.embed_tokens(head_ids)
        inputs_embeds = inputs_embeds.astype(self.dtype)
        if prompt_span_positions:
            # patch_encoder over the full prompt latents -> [1, S, 1536].
            # The prompt_patches reshape is [1, S, patch_size, 128]; flatten the time
            # axis back to [1, S*patch_size, 128] for the encoder, which downsamples
            # by patch_size to [1, S, 1536].
            # patch_emb is supplied by the profile path (precomputed at enroll); else
            # recompute it here — numerically identical, so the ref-audio path is unchanged.
            if patch_emb is None:
                # The patch encoder operates on DENORMALIZED latents (matches upstream's
                # patch_encoder.prefill(prompt_latents) and our streaming-state prefill).
                # prompt_patches (normalized) is the FM-history input only -- NOT this.
                if prompt_denorm_latents is None:
                    raise ValueError(
                        "_prefill requires prompt_denorm_latents to recompute patch_emb "
                        "when patch_emb is not supplied (the profile path)."
                    )
                flat = prompt_denorm_latents[
                    :, : prompt_patch_count * self.patch_size
                ]  # [1, S*patch_size, 128] denormalized
                patch_emb = self.patch_encoder(flat)
            patch_emb = patch_emb.astype(inputs_embeds.dtype)  # [1, S, 1536]
            # scatter into the prompt span slots.
            idx = mx.array(prompt_span_positions, dtype=mx.int32)
            inputs_embeds[:, idx, :] = patch_emb[:, :prompt_patch_count, :]

        llm_hiddens, _ = self.llm.step(inputs_embeds=inputs_embeds, cache=cache)
        self._last_prefill_hidden = llm_hiddens[:, -1:, :]

        # walk the prompt spans appending hidden + history chunks.
        cursor = 0
        for prompt_index, span_position in enumerate(prompt_span_positions):
            if span_position > cursor:
                self._append_hidden_chunk(
                    llm_hiddens[:, span_position - 1 : span_position, :]
                )
            self._append_history_chunk(prompt_patches[:, prompt_index])  # [1, p, 128]
            next_is_span = span_position + 1 < len(schedule_ids) and schedule_ids[
                span_position + 1
            ] in (self.tokenizer.audio_gen_span_id, self.tokenizer.audio_comp_span_id)
            if next_is_span:
                self._append_hidden_chunk(
                    llm_hiddens[:, span_position : span_position + 1, :]
                )
            cursor = span_position + 1
        if prefill_end > cursor:
            self._append_hidden_chunk(llm_hiddens[:, prefill_end - 1 : prefill_end, :])
        return prefill_end

    # endregion decode helpers

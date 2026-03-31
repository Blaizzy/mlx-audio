"""LongCat-AudioDiT: Conditional Flow Matching TTS with DiT backbone.

Ports meituan-longcat/LongCat-AudioDiT to MLX. Components:
  1. UMT5 text encoder (frozen)
  2. DiT transformer (CrossDiT with AdaLN, RoPE, cross-attention)
  3. WAV-VAE audio codec (frozen, 24kHz, latent_dim=64)
"""

import math
import re
import time as time_module
from typing import Generator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.base import GenerationResult

from .config import ModelConfig
from .dit import AudioDiTTransformer
from .text_encoder import UMT5Encoder
from .vae import AudioDiTVae

# ---------------------------------------------------------------------------
# Duration estimation heuristic
# ---------------------------------------------------------------------------

EN_DUR_PER_CHAR = 0.082
ZH_DUR_PER_CHAR = 0.21


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["""\u2018\u2019]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _approx_duration(text: str, max_duration: float = 30.0) -> float:
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)


# ---------------------------------------------------------------------------
# APG helpers (Adaptive Projected Guidance)
# ---------------------------------------------------------------------------


class _MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: mx.array):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def _project(v0: mx.array, v1: mx.array):
    v0 = v0.astype(mx.float32)
    v1 = v1.astype(mx.float32)
    v1_norm = v1 / (mx.sqrt(mx.sum(v1 * v1, axis=(-1, -2), keepdims=True)) + 1e-8)
    v0_parallel = mx.sum(v0 * v1_norm, axis=(-1, -2), keepdims=True) * v1_norm
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel, v0_orthogonal


def _apg_forward(
    pred_cond,
    pred_uncond,
    guidance_scale,
    momentum_buffer=None,
    eta=0.0,
    norm_threshold=2.5,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        diff_norm = mx.sqrt(mx.sum(diff * diff, axis=(-1, -2), keepdims=True))
        scale_factor = mx.minimum(mx.ones_like(diff_norm), norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + guidance_scale * normalized_update


# ---------------------------------------------------------------------------
# Euler ODE solver
# ---------------------------------------------------------------------------


def _odeint_euler(fn, y0, t_steps):
    y = y0
    for i in range(len(t_steps) - 1):
        dt = t_steps[i + 1] - t_steps[i]
        y = y + fn(t_steps[i], y) * dt
    return y


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Text encoder (UMT5)
        self.text_encoder = UMT5Encoder(config.text_encoder_config)

        # DiT transformer
        self.transformer = AudioDiTTransformer(config)

        # WAV-VAE
        self.vae = AudioDiTVae(config.vae_config)

    def model_quant_predicate(self, p, m):
        """Skip quantization for VAE (conv-based codec)."""
        return not p.startswith("vae")

    @property
    def sample_rate(self) -> int:
        return self.config.sampling_rate

    def encode_text(self, input_ids: mx.array, attention_mask: mx.array):
        """Encode text with UMT5 and apply text_add_embed / text_norm_feat."""
        last_hidden, initial_embed = self.text_encoder(input_ids, attention_mask)
        d_model = self.config.text_encoder_config.d_model

        if self.config.text_norm_feat:
            last_hidden = _layer_norm(last_hidden, d_model)

        if self.config.text_add_embed:
            if self.config.text_norm_feat:
                initial_embed = _layer_norm(initial_embed, d_model)
            last_hidden = last_hidden + initial_embed

        return last_hidden.astype(mx.float32)

    def encode_prompt_audio(self, prompt_audio: mx.array):
        """Encode prompt audio to latent space. Returns (latent, num_frames)."""
        full_hop = self.config.latent_hop
        off = 3
        wav = prompt_audio
        if wav.ndim == 2:
            wav = wav[..., None]  # (B, L) -> (B, L, 1)
        if wav.ndim == 3 and wav.shape[-1] != 1:
            wav = wav.transpose(0, 2, 1)  # (B, 1, L) -> (B, L, 1)

        L = wav.shape[1]
        if L % full_hop != 0:
            pad_amount = full_hop - L % full_hop
            wav = mx.pad(wav, [(0, 0), (0, pad_amount), (0, 0)])
        wav = mx.pad(wav, [(0, 0), (0, full_hop * off), (0, 0)])

        latent = self.vae.encode(wav)  # (B, T, latent_dim)
        if off != 0:
            latent = latent[:, :-off, :]
        prompt_dur = latent.shape[1]
        return latent, prompt_dur

    @staticmethod
    def _format_duration(seconds: float) -> str:
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{seconds % 60:06.3f}"

    def _stream_decode(
        self,
        pred_latent: mx.array,
        sr: int,
        start_time: float,
        chunk_seconds: float = 2.0,
        overlap_seconds: float = 0.5,
    ) -> Generator[GenerationResult, None, None]:
        """Decode latents in overlapping chunks with cosine crossfade.

        Runs the VAE decoder on small overlapping windows of the latent
        sequence and yields audio as soon as each chunk is ready, giving
        much lower time-to-first-audio than full-sequence decoding.
        """
        ratio = self.config.vae_config.downsampling_ratio
        chunk_frames = max(1, int(chunk_seconds * sr / ratio))
        overlap_frames = max(0, int(overlap_seconds * sr / ratio))
        hop_frames = max(1, chunk_frames - overlap_frames)
        overlap_samples = overlap_frames * ratio
        # Context frames fed to the VAE decoder on each side so its
        # convolutions have proper context, then trimmed from the output.
        context_frames = overlap_frames

        total_frames = pred_latent.shape[1]
        prev_tail = None
        chunk_idx = 0
        cumulative_samples = 0

        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            is_last = end >= total_frames

            # Extend chunk with context on both sides for decoder conv context
            left_ctx = min(context_frames, start)
            right_ctx = min(context_frames, total_frames - end)
            ctx_start = start - left_ctx
            ctx_end = end + right_ctx

            # Decode extended chunk, then trim context audio
            audio_full = self.vae.decode(pred_latent[:, ctx_start:ctx_end, :])
            audio_full = audio_full.squeeze(-1).squeeze(0)  # (L,)
            mx.eval(audio_full)
            left_trim = left_ctx * ratio
            right_trim = right_ctx * ratio
            if right_trim > 0:
                audio_chunk = audio_full[left_trim:-right_trim]
            else:
                audio_chunk = audio_full[left_trim:]

            if prev_tail is not None and overlap_samples > 0:
                # Cosine crossfade with previous chunk's tail
                ol = min(overlap_samples, prev_tail.shape[0], audio_chunk.shape[0])
                t = mx.linspace(0, 1, ol)
                fade_in = 0.5 * (1 - mx.cos(mx.array(math.pi) * t))
                fade_out = 1.0 - fade_in
                blended = prev_tail[:ol] * fade_out + audio_chunk[:ol] * fade_in

                if is_last:
                    output = mx.concatenate([blended, audio_chunk[ol:]])
                else:
                    output = mx.concatenate([blended, audio_chunk[ol:-overlap_samples]])
                    prev_tail = audio_chunk[-overlap_samples:]
                    mx.eval(prev_tail)
            else:
                if is_last or overlap_frames == 0:
                    output = audio_chunk
                else:
                    output = audio_chunk[:-overlap_samples]
                    prev_tail = audio_chunk[-overlap_samples:]
                    mx.eval(prev_tail)

            mx.eval(output)
            num_samples = output.shape[0]
            cumulative_samples += num_samples
            processing_time = time_module.time() - start_time
            audio_dur = cumulative_samples / sr

            yield GenerationResult(
                audio=output,
                samples=num_samples,
                sample_rate=sr,
                segment_idx=chunk_idx,
                token_count=0,
                audio_duration=self._format_duration(audio_dur),
                real_time_factor=processing_time / max(audio_dur, 1e-6),
                prompt={"tokens": 0, "tokens-per-sec": 0},
                audio_samples={
                    "samples": num_samples,
                    "samples-per-sec": num_samples / max(processing_time, 1e-6),
                },
                processing_time_seconds=processing_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=True,
                is_final_chunk=is_last,
            )

            chunk_idx += 1
            start += hop_frames

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: str = "en",
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        steps: int = 16,
        cfg_strength: float = 4.0,
        guidance_method: str = "cfg",
        seed: int = 1024,
        stream: bool = False,
        streaming_interval: float = 2.0,
        chunk_seconds: float = 2.0,
        overlap_seconds: float = 0.5,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate audio from text.

        Args:
            text: Text to synthesize.
            voice: Unused (kept for API compat).
            speed: Unused (kept for API compat).
            ref_audio: Optional reference audio array for voice cloning (samples at 24kHz).
            ref_text: Optional text of the reference audio.
            steps: Number of ODE Euler steps (default 16).
            cfg_strength: Classifier-free guidance strength.
            guidance_method: "cfg" or "apg".
            seed: Random seed.
            stream: Stream audio chunks during VAE decoding.
            streaming_interval: Alias for chunk_seconds (CLI compat).
            chunk_seconds: Chunk length in seconds for streaming decode.
            overlap_seconds: Overlap between chunks for cosine crossfade.
        """
        import transformers

        start_time = time_module.time()
        mx.random.seed(seed)

        sr = self.config.sampling_rate
        full_hop = self.config.latent_hop
        max_duration = self.config.max_wav_duration
        repa_layer = self.config.repa_dit_layer

        # Tokenize
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.text_encoder_model
        )

        text = _normalize_text(text)
        no_prompt = ref_audio is None

        if not no_prompt and ref_text is not None:
            prompt_text = _normalize_text(ref_text)
            full_text = f"{prompt_text} {text}"
        else:
            full_text = text

        inputs = tokenizer([full_text], padding="longest", return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"]).astype(mx.float32)

        # Text encoding
        text_condition = self.encode_text(input_ids, attention_mask)
        text_condition_len = mx.sum(attention_mask, axis=1)

        batch = text_condition.shape[0]

        # Prompt audio encoding
        if not no_prompt:
            prompt_audio_mx = (
                mx.array(np.array(ref_audio))
                if not isinstance(ref_audio, mx.array)
                else ref_audio
            )
            prompt_latent, prompt_dur = self.encode_prompt_audio(
                prompt_audio_mx[None] if prompt_audio_mx.ndim < 3 else prompt_audio_mx
            )
        else:
            prompt_latent = mx.zeros((batch, 0, self.config.latent_dim))
            prompt_dur = 0

        # Duration estimation
        prompt_time = prompt_dur * full_hop / sr
        dur_sec = _approx_duration(text, max_duration=max_duration - prompt_time)
        if not no_prompt and ref_text is not None:
            approx_pd = _approx_duration(ref_text, max_duration=max_duration)
            ratio = float(np.clip(prompt_time / max(approx_pd, 1e-6), 1.0, 1.5))
            dur_sec = dur_sec * ratio

        duration = int(dur_sec * sr // full_hop)
        total_duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

        # Masks
        mask = mx.arange(total_duration)[None, :] < mx.array([total_duration])
        text_mask = (
            mx.arange(text_condition.shape[1])[None, :] < text_condition_len[:, None]
        )

        # Conditioning
        neg_text = mx.zeros_like(text_condition)

        if not no_prompt:
            gen_len = total_duration - prompt_dur
            latent_cond = mx.pad(prompt_latent, [(0, 0), (0, gen_len), (0, 0)])
            empty_latent_cond = mx.zeros_like(latent_cond)
        else:
            latent_cond = mx.zeros((batch, total_duration, self.config.latent_dim))
            empty_latent_cond = latent_cond

        # APG buffer
        apg_buffer = (
            _MomentumBuffer(momentum=-0.3) if guidance_method == "apg" else None
        )

        # Initial noise
        y = mx.random.normal((batch, total_duration, self.config.latent_dim))
        prompt_noise = y[:, :prompt_dur] if prompt_dur > 0 else None

        # Time steps
        t_steps = np.linspace(0, 1, steps).tolist()

        # Euler ODE integration
        for i in range(len(t_steps) - 1):
            t_val = t_steps[i]
            dt = t_steps[i + 1] - t_val
            t = mx.array([t_val])

            # Set prompt region
            if prompt_dur > 0:
                y_prompt = (
                    prompt_noise * (1 - t_val) + latent_cond[:, :prompt_dur] * t_val
                )
                y = mx.concatenate([y_prompt, y[:, prompt_dur:]], axis=1)

            output = self.transformer(
                x=y,
                text=text_condition,
                text_len=text_condition_len,
                time=t,
                mask=mask,
                cond_mask=text_mask,
                return_ith_layer=repa_layer,
                latent_cond=latent_cond,
            )
            pred = output["last_hidden_state"]

            if cfg_strength >= 1e-5:
                # Null forward for guidance
                y_null = mx.array(y)
                if prompt_dur > 0:
                    y_null = mx.concatenate(
                        [mx.zeros_like(y_null[:, :prompt_dur]), y_null[:, prompt_dur:]],
                        axis=1,
                    )

                null_output = self.transformer(
                    x=y_null,
                    text=neg_text,
                    text_len=text_condition_len,
                    time=t,
                    mask=mask,
                    cond_mask=text_mask,
                    return_ith_layer=repa_layer,
                    latent_cond=empty_latent_cond,
                )
                null_pred = null_output["last_hidden_state"]

                if guidance_method == "cfg":
                    pred = pred + (pred - null_pred) * cfg_strength
                else:
                    # APG
                    x_s = y[:, prompt_dur:]
                    pred_s = pred[:, prompt_dur:]
                    null_s = null_pred[:, prompt_dur:]
                    pred_sample = x_s + (1 - t_val) * pred_s
                    null_sample = x_s + (1 - t_val) * null_s
                    out = _apg_forward(
                        pred_sample,
                        null_sample,
                        cfg_strength,
                        apg_buffer,
                        eta=0.5,
                        norm_threshold=0.0,
                    )
                    out = (out - x_s) / (1 - t_val)
                    pred = mx.pad(out, [(0, 0), (prompt_dur, 0), (0, 0)])

            y = y + pred * dt
            mx.eval(y)

        # Decode
        pred_latent = y[:, prompt_dur:] if not no_prompt else y

        if stream:
            # streaming_interval from CLI takes priority over chunk_seconds
            cs = streaming_interval if streaming_interval != 2.0 else chunk_seconds
            yield from self._stream_decode(
                pred_latent, sr, start_time, cs, overlap_seconds
            )
        else:
            waveform = self.vae.decode(pred_latent)  # (B, L, 1)
            waveform = waveform.squeeze(-1)  # (B, L)

            mx.eval(waveform)
            processing_time = time_module.time() - start_time

            audio = waveform[0]
            num_samples = audio.shape[0]
            audio_duration = num_samples / sr

            yield GenerationResult(
                audio=audio,
                samples=num_samples,
                sample_rate=sr,
                segment_idx=0,
                token_count=0,
                audio_duration=self._format_duration(audio_duration),
                real_time_factor=processing_time / max(audio_duration, 1e-6),
                prompt={"tokens": 0, "tokens-per-sec": 0},
                audio_samples={
                    "samples": num_samples,
                    "samples-per-sec": num_samples / max(processing_time, 1e-6),
                },
                processing_time_seconds=processing_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

    def sanitize(self, weights: dict) -> dict:
        """Convert PyTorch AudioDiT weights to MLX format."""
        sanitized = {}

        # Collect weight_v/weight_g pairs for reconstruction
        weight_v_keys = [k for k in weights if k.endswith(".weight_v")]
        processed = set()

        for wv_key in weight_v_keys:
            base = wv_key[: -len(".weight_v")]
            wg_key = base + ".weight_g"
            if wg_key not in weights:
                continue

            weight_v = weights[wv_key]
            weight_g = weights[wg_key]

            # Determine if this is ConvTranspose1d
            # ConvTranspose1d is at decoder.layers.{block}.layers.1
            is_conv_transpose = bool(
                re.search(r"vae\.decoder\.layers\.\d+\.layers\.1\.weight", wv_key)
            )

            if weight_v.ndim == 3:
                # Reconstruct weight from weight_norm: w = g * v / ||v||
                if is_conv_transpose:
                    # PyTorch ConvTranspose1d: (in_ch, out_ch, ksize)
                    # norm over dims 1,2 (out_ch and ksize)
                    norm_axes = (1, 2)
                    norm = mx.sqrt(
                        mx.sum(weight_v * weight_v, axis=norm_axes, keepdims=True)
                        + 1e-12
                    )
                    weight = weight_g * weight_v / norm
                    # Transpose: (in_ch, out_ch, ksize) -> (out_ch, ksize, in_ch)
                    weight = weight.transpose(1, 2, 0)
                else:
                    # PyTorch Conv1d: (out_ch, in_ch, ksize)
                    # norm over dims 1,2 (in_ch and ksize)
                    norm_axes = (1, 2)
                    norm = mx.sqrt(
                        mx.sum(weight_v * weight_v, axis=norm_axes, keepdims=True)
                        + 1e-12
                    )
                    weight = weight_g * weight_v / norm
                    # Transpose: (out_ch, in_ch, ksize) -> (out_ch, ksize, in_ch)
                    weight = weight.transpose(0, 2, 1)

                sanitized[base + ".weight"] = weight
            else:
                sanitized[base + ".weight"] = weight_v

            processed.add(wv_key)
            processed.add(wg_key)

        # Process remaining weights
        for key, value in weights.items():
            if key in processed:
                continue

            new_key = key

            # --- Text encoder remapping ---
            if new_key.startswith("text_encoder.encoder.embed_tokens."):
                new_key = new_key.replace(
                    "text_encoder.encoder.embed_tokens.", "text_encoder.shared."
                )

            # T5 block layer remapping:
            # HF: block.{i}.layer.0.SelfAttention -> block.{i}.SelfAttention
            # HF: block.{i}.layer.0.layer_norm -> block.{i}.layer_norm_sa
            # HF: block.{i}.layer.1.DenseReluDense -> block.{i}.DenseReluDense
            # HF: block.{i}.layer.1.layer_norm -> block.{i}.layer_norm_ff
            new_key = re.sub(
                r"text_encoder\.encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.",
                r"text_encoder.block.\1.SelfAttention.",
                new_key,
            )
            new_key = re.sub(
                r"text_encoder\.encoder\.block\.(\d+)\.layer\.0\.layer_norm\.",
                r"text_encoder.block.\1.layer_norm_sa.",
                new_key,
            )
            new_key = re.sub(
                r"text_encoder\.encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.",
                r"text_encoder.block.\1.DenseReluDense.",
                new_key,
            )
            new_key = re.sub(
                r"text_encoder\.encoder\.block\.(\d+)\.layer\.1\.layer_norm\.",
                r"text_encoder.block.\1.layer_norm_ff.",
                new_key,
            )
            new_key = new_key.replace(
                "text_encoder.encoder.final_layer_norm.",
                "text_encoder.final_layer_norm.",
            )

            # --- Transformer Sequential index remapping ---
            # Embedder: proj.0 -> proj.0, proj.2 -> proj.1
            new_key = re.sub(r"\.proj\.2\.", ".proj.1.", new_key)
            # TimestepEmbedding: time_mlp.0 -> time_mlp.0, time_mlp.2 -> time_mlp.1
            new_key = re.sub(r"\.time_mlp\.2\.", ".time_mlp.1.", new_key)
            # AdaLNMLP: mlp.1 -> mlp.0
            new_key = re.sub(r"\.mlp\.1\.", ".mlp.0.", new_key)
            # Attention to_out: to_out.0 -> to_out
            new_key = re.sub(r"\.to_out\.0\.", ".to_out.", new_key)
            # FeedForward: ff.3 -> ff.1
            new_key = re.sub(r"\.ff\.3\.", ".ff.1.", new_key)

            # --- ConvNeXtV2 depthwise conv weight ---
            # dwconv.weight: (dim, 1, ksize) -> (dim, ksize, 1) for channels-last
            if "dwconv.weight" in new_key and value.ndim == 3:
                value = value.transpose(0, 2, 1)
                new_key = new_key.replace(".dwconv.weight", ".dwconv_weight")
            elif "dwconv.bias" in new_key:
                new_key = new_key.replace(".dwconv.bias", ".dwconv_bias")

            # --- Handle standalone Conv1d weights (non weight-normed) ---
            # These might be in the ConvNeXtV2 dwconv or other places

            sanitized[new_key] = value

        return sanitized


def _layer_norm(x: mx.array, dim: int, eps: float = 1e-6) -> mx.array:
    """Layer normalization."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps)

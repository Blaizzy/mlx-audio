"""Voxtral Realtime 4B - Main model orchestrator.

Inference pipeline:
1. Resample audio to 16kHz, pad (left silence + right silence)
2. Compute mel spectrogram
3. Run causal encoder -> 4x downsample -> adapter
4. Construct prompt: [BOS] + [STREAMING_PAD] * (n_left_pad + n_delay)
5. For each position: input = audio_embed + tok_embed(token_id)
6. Prefill decoder, then autoregressive generation until EOS
7. Decode tokens via Tekken tokenizer
"""

import math
import time
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import STTOutput
from .audio import compute_mel_filters, compute_mel_spectrogram
from .config import ModelConfig
from .decoder import Decoder, compute_time_embedding
from .encoder import AudioEncoder
from .tokenizer import TekkenTokenizer


# Derived streaming constants
SAMPLE_RATE = 16000
FRAME_RATE = 12.5
RAW_AUDIO_LENGTH_PER_TOK = int(SAMPLE_RATE // FRAME_RATE)  # 1280
HOP_LENGTH = 160
AUDIO_LENGTH_PER_TOK = RAW_AUDIO_LENGTH_PER_TOK // HOP_LENGTH  # 8


def _num_audio_tokens(audio_len):
    if audio_len % HOP_LENGTH != 0:
        audio_len = math.ceil(audio_len / HOP_LENGTH - 1)
    else:
        audio_len = audio_len // HOP_LENGTH
    return math.ceil(audio_len / AUDIO_LENGTH_PER_TOK)


def _num_delay_tokens(delay_ms):
    delay_len = int(delay_ms / 1000.0 * SAMPLE_RATE)
    return _num_audio_tokens(delay_len)


def _pad_audio_streaming(audio_array, n_left_pad_tokens, n_right_pad_tokens):
    """Pad audio for offline streaming mode.

    Left pad: n_left_pad_tokens * 1280 samples of silence
    Right pad: align to 1280 + n_right_pad_tokens * 1280 of silence
    """
    mult_of = RAW_AUDIO_LENGTH_PER_TOK
    n_samples = len(audio_array)
    align_pad = (mult_of - (n_samples % mult_of)) % mult_of
    right_pad = align_pad + n_right_pad_tokens * mult_of
    left_pad = n_left_pad_tokens * mult_of
    return np.pad(audio_array, (left_pad, right_pad))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = AudioEncoder(config.encoder_args)
        self.decoder = Decoder(config.decoder)

        # Will be set in post_load_hook
        self._tokenizer = None
        self._mel_filters = None

    def _ensure_mel_filters(self):
        if self._mel_filters is None:
            aec = self.config.audio_encoding_args
            filters_np = compute_mel_filters(
                num_mel_bins=aec.num_mel_bins,
                window_size=aec.window_size,
                sample_rate=aec.sampling_rate,
            )
            self._mel_filters = mx.array(filters_np, dtype=mx.float32)
        return self._mel_filters

    def _load_audio(self, audio_input: Union[str, Path, List[mx.array], mx.array, np.ndarray]) -> np.ndarray:
        """Load and resample audio from file path or array to 16kHz float32 numpy."""
        if isinstance(audio_input, (str, Path)):
            import soundfile as sf
            audio_np, sr = sf.read(str(audio_input), dtype="float32")
            audio_np = audio_np.flatten()
            if sr != SAMPLE_RATE:
                # Resample to 16kHz
                from scipy.signal import resample
                n_samples = int(len(audio_np) * SAMPLE_RATE / sr)
                audio_np = resample(audio_np, n_samples).astype(np.float32)
            return audio_np
        if isinstance(audio_input, list):
            audio_input = audio_input[0]
        return np.array(audio_input).flatten().astype(np.float32)

    def _encode_and_prefill(self, audio_np, verbose=False):
        """Shared encoder + prefill logic. Returns (adapter_out, prompt_len, cache, start_time)."""
        start_time = time.time()

        n_delay = _num_delay_tokens(self.config.transcription_delay_ms)
        n_left = self.config.n_left_pad_tokens
        n_right = (n_delay + 1) + 10

        padded = _pad_audio_streaming(audio_np, n_left, n_right)

        aec = self.config.audio_encoding_args
        mel_filters = self._ensure_mel_filters()
        audio_mx = mx.array(padded, dtype=mx.float32)
        mel = compute_mel_spectrogram(
            audio_mx,
            mel_filters,
            window_size=aec.window_size,
            hop_length=aec.hop_length,
            global_log_mel_max=aec.global_log_mel_max,
        )

        if mel.shape[1] % 2 != 0:
            mel = mel[:, 1:]

        if verbose:
            print(f"Audio: {len(audio_np)} samples ({len(audio_np)/SAMPLE_RATE:.1f}s)")
            print(f"Padded: {len(padded)} samples, Mel: {mel.shape[1]} frames")

        adapter_out = self.encoder(mel)
        mx.eval(adapter_out)

        n_audio = adapter_out.shape[0]
        if verbose:
            print(f"Adapter output: {n_audio} tokens")

        prompt_len = 1 + n_left + n_delay
        prompt_ids = [self.config.bos_token_id] + [
            self.config.streaming_pad_token_id
        ] * (n_left + n_delay)

        prompt_ids_mx = mx.array(prompt_ids)
        prompt_text_embeds = self.decoder.embed_tokens(prompt_ids_mx)
        prefix_embeds = adapter_out[:prompt_len] + prompt_text_embeds

        if verbose:
            print(f"Prompt: {prompt_len} tokens, Audio span: {n_audio} tokens")

        if prompt_len > 1:
            h, cache = self.decoder.forward(prefix_embeds[:-1], start_pos=0)
        else:
            cache = None

        h, cache = self.decoder.forward(
            prefix_embeds[-1:], start_pos=prompt_len - 1, cache=cache
        )
        logits = self.decoder.logits(h.squeeze(0))

        return adapter_out, n_audio, prompt_len, logits, cache, start_time

    def generate(
        self,
        audio: Union[str, Path, List[mx.array], mx.array],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        verbose: bool = False,
        stream: bool = False,
        **kwargs,
    ):
        """Transcribe audio. Returns STTOutput, or yields text deltas if stream=True."""
        audio_np = self._load_audio(audio)

        if stream:
            return self._generate_stream(audio_np, max_tokens, temperature, verbose)

        adapter_out, n_audio, prompt_len, logits, cache, start_time = \
            self._encode_and_prefill(audio_np, verbose)

        token = int(mx.argmax(logits).item()) if temperature == 0 else self._sample(logits, temperature)
        generated = [token]

        for pos in range(prompt_len, n_audio):
            if token == self.config.eos_token_id:
                break
            embed = adapter_out[pos] + self.decoder.embed_token(token)
            h, cache = self.decoder.forward(embed[None, :], start_pos=pos, cache=cache)
            logits = self.decoder.logits(h.squeeze(0))
            token = int(mx.argmax(logits).item()) if temperature == 0 else self._sample(logits, temperature)
            generated.append(token)
            if len(generated) > max_tokens:
                break

        if generated and generated[-1] == self.config.eos_token_id:
            generated = generated[:-1]

        text = self._tokenizer.decode(generated).strip()
        end_time = time.time()
        total_time = end_time - start_time

        mx.clear_cache()

        return STTOutput(
            text=text,
            prompt_tokens=prompt_len,
            generation_tokens=len(generated),
            total_tokens=prompt_len + len(generated),
            total_time=total_time,
            prompt_tps=prompt_len / total_time if total_time > 0 else 0,
            generation_tps=len(generated) / total_time if total_time > 0 else 0,
        )

    def _generate_stream(self, audio_np, max_tokens, temperature, verbose):
        """Generator that yields text deltas as tokens are decoded."""
        adapter_out, n_audio, prompt_len, logits, cache, start_time = \
            self._encode_and_prefill(audio_np, verbose)

        token = int(mx.argmax(logits).item()) if temperature == 0 else self._sample(logits, temperature)
        generated = [token]
        prev_text = ""

        # Yield first token delta
        text_so_far = self._tokenizer.decode(generated)
        if text_so_far != prev_text:
            yield text_so_far[len(prev_text):]
            prev_text = text_so_far

        for pos in range(prompt_len, n_audio):
            if token == self.config.eos_token_id:
                break
            embed = adapter_out[pos] + self.decoder.embed_token(token)
            h, cache = self.decoder.forward(embed[None, :], start_pos=pos, cache=cache)
            logits = self.decoder.logits(h.squeeze(0))
            token = int(mx.argmax(logits).item()) if temperature == 0 else self._sample(logits, temperature)
            generated.append(token)

            # Yield text delta
            text_so_far = self._tokenizer.decode(
                [t for t in generated if t != self.config.eos_token_id]
            )
            if text_so_far != prev_text:
                yield text_so_far[len(prev_text):]
                prev_text = text_so_far

            if len(generated) > max_tokens:
                break

        mx.clear_cache()

    def _sample(self, logits, temperature):
        probs = mx.softmax(logits / temperature, axis=-1)
        return int(mx.random.categorical(mx.log(probs)).item())

    def sanitize(self, weights):
        """Map weight names from consolidated.safetensors to our module structure."""
        new_weights = {}

        enc_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder"
        adapter_prefix = "mm_streams_embeddings.embedding_module"
        tok_emb_key = "mm_streams_embeddings.embedding_module.tok_embeddings.weight"

        for k, v in weights.items():
            new_key = None

            if k == tok_emb_key:
                new_key = "decoder.tok_embeddings.weight"

            elif k == "norm.weight":
                new_key = "decoder.norm.weight"

            elif k.startswith(f"{enc_prefix}.conv_layers."):
                # e.g., ...conv_layers.0.conv.weight -> encoder.conv_layers_0_conv.conv.weight
                rest = k[len(f"{enc_prefix}.conv_layers."):]
                # rest: "0.conv.weight" or "0.conv.bias"
                parts = rest.split(".", 2)  # ['0', 'conv', 'weight']
                layer_idx = parts[0]
                param = parts[2]  # 'weight' or 'bias'
                new_key = f"encoder.conv_layers_{layer_idx}_conv.conv.{param}"

                # Transpose conv weights from PyTorch [out, in, k] to MLX [out, k, in]
                if param == "weight" and v.ndim == 3:
                    v = v.transpose(0, 2, 1)

            elif k.startswith(f"{enc_prefix}.transformer.layers."):
                rest = k[len(f"{enc_prefix}.transformer.layers."):]
                # e.g., "0.attention.wq.weight"
                parts = rest.split(".", 1)
                layer_idx = parts[0]
                param_path = parts[1]

                # Map FFN weights
                param_path = param_path.replace("feed_forward.w1.", "feed_forward_w1.")
                param_path = param_path.replace("feed_forward.w2.", "feed_forward_w2.")
                param_path = param_path.replace("feed_forward.w3.", "feed_forward_w3.")

                new_key = f"encoder.transformer_layers.{layer_idx}.{param_path}"

            elif k.startswith(f"{enc_prefix}.transformer.norm."):
                rest = k[len(f"{enc_prefix}.transformer.norm."):]
                new_key = f"encoder.transformer_norm.{rest}"

            elif k.startswith(f"{adapter_prefix}.audio_language_projection."):
                rest = k[len(f"{adapter_prefix}.audio_language_projection."):]
                # "0.weight" -> "audio_language_projection_0.weight"
                # "2.weight" -> "audio_language_projection_2.weight"
                parts = rest.split(".", 1)
                idx = parts[0]
                param = parts[1]
                new_key = f"encoder.audio_language_projection_{idx}.{param}"

            elif k.startswith("layers."):
                rest = k[len("layers."):]
                # e.g., "0.attention.wq.weight"
                parts = rest.split(".", 1)
                layer_idx = parts[0]
                param_path = parts[1]

                # Map FFN
                param_path = param_path.replace("feed_forward.w1.", "feed_forward_w1.")
                param_path = param_path.replace("feed_forward.w2.", "feed_forward_w2.")
                param_path = param_path.replace("feed_forward.w3.", "feed_forward_w3.")
                # Map ada norm: ada_rms_norm_t_cond.0.weight -> ada_rms_norm_t_cond.ada_down.weight
                param_path = param_path.replace(
                    "ada_rms_norm_t_cond.0.", "ada_rms_norm_t_cond.ada_down."
                )
                param_path = param_path.replace(
                    "ada_rms_norm_t_cond.2.", "ada_rms_norm_t_cond.ada_up."
                )

                new_key = f"decoder.layers.{layer_idx}.{param_path}"

            if new_key is not None:
                new_weights[new_key] = v
            else:
                # Pass through any unrecognized weights as-is
                new_weights[k] = v

        return new_weights

    def model_quant_predicate(self, p, m):
        """Skip quantization on encoder norms, ada norms, embeddings."""
        skip_patterns = [
            "norm",
            "ada_rms_norm",
            "tok_embeddings",
            "conv_layers",
            "audio_language_projection",
        ]
        return not any(pat in p for pat in skip_patterns)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Initialize tokenizer and precompute ada scales after weight loading."""
        model_path = Path(model_path)

        # Load Tekken tokenizer
        model._tokenizer = TekkenTokenizer.from_model_path(model_path)

        # Precompute mel filters
        model._ensure_mel_filters()

        # Precompute ada scales from time conditioning
        n_delay = _num_delay_tokens(model.config.transcription_delay_ms)
        t_cond = compute_time_embedding(float(n_delay), model.config.decoder.dim)
        model.decoder.precompute_ada_scales(t_cond)
        # Evaluate ada scales eagerly
        for scale in model.decoder._ada_scales:
            if scale is not None:
                mx.eval(scale)

        return model

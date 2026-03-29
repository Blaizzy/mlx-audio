import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen2 import Model as Qwen2LM
from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelArgs

from mlx_audio.stt.models.base import STTOutput

from .config import EncoderConfig, ModelConfig


@dataclass
class StreamingResult:
    text: str
    is_final: bool
    start_time: float
    end_time: float
    language: str = "en"
    prompt_tokens: int = 0
    generation_tokens: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sinusoids(length: int, channels: int) -> mx.array:
    max_timescale = 10000
    log_timescale = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Qwen2AudioEncoderAttention(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class Qwen2AudioEncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen2AudioEncoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with pre-norm
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x

        # FFN with pre-norm
        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class Qwen2AudioEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        # MLX Conv1d takes (B, T, C); weights shape (out_channels, kW, in_channels)
        self.conv1 = nn.Conv1d(
            config.num_mel_bins, config.d_model, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=3, stride=2, padding=1
        )
        self.layers = [
            Qwen2AudioEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Fixed sinusoidal positional embeddings (generous buffer for off-by-one)
        self._embed_positions = sinusoids(
            config.max_source_positions + 1, config.d_model
        )

    @property
    def embed_positions(self) -> mx.array:
        return self._embed_positions

    def __call__(self, input_features: mx.array) -> mx.array:
        # input_features: (B, n_mels, T) -> transpose to (B, T, n_mels) for Conv1d
        x = input_features.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        # Add positional embeddings
        T = x.shape[1]
        x = x + self.embed_positions[:T]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # AvgPool1d with kernel=2, stride=2: reshape and mean
        B, seq_len, D = x.shape
        x = x[:, : (seq_len // 2) * 2, :].reshape(B, seq_len // 2, 2, D).mean(axis=2)

        x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class Qwen2AudioMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.audio_config.d_model,
            config.text_config.hidden_size,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.audio_tower = Qwen2AudioEncoder(config.audio_config)
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config)

        text_args = Qwen2ModelArgs.from_dict(
            config.text_config.__dict__
            if hasattr(config.text_config, "__dict__")
            else config.text_config
        )
        self.language_model = Qwen2LM(text_args)

        self.audio_token_id = config.audio_token_id
        self._processor = None

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in range(len(self.layers))]

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[KVCache]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.language_model.model.embed_tokens(input_ids)

        if cache is None:
            cache = [None] * len(self.language_model.model.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.language_model.model.layers, cache):
            h = layer(h, mask, cache=c)

        h = self.language_model.model.norm(h)

        if self.language_model.args.tie_word_embeddings:
            logits = self.language_model.model.embed_tokens.as_linear(h)
        else:
            logits = self.language_model.lm_head(h)

        return logits

    def get_audio_features(self, input_features: mx.array) -> mx.array:
        encoder_output = self.audio_tower(input_features)
        projected = self.multi_modal_projector(encoder_output)
        return projected

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return not (
            p.startswith("audio_tower") or p.startswith("multi_modal_projector")
        )

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        already_converted = any("scales" in k for k in weights)

        sanitized = {}
        for k, v in weights.items():
            # Skip stored sinusoidal positional embeddings (we compute them)
            if "embed_positions" in k:
                continue

            # Transpose Conv1d weights: PyTorch (out, in, kW) -> MLX (out, kW, in)
            if (
                not already_converted
                and "audio_tower" in k
                and "weight" in k
                and len(v.shape) == 3
            ):
                v = v.transpose(0, 2, 1)

            sanitized[k] = v
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        import transformers
        from transformers import AutoProcessor

        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model._processor = AutoProcessor.from_pretrained(
                str(model_path), trust_remote_code=True
            )
        finally:
            transformers.logging.set_verbosity(prev)

        return model

    def _get_feat_extract_output_lengths(self, input_length: int) -> int:
        # After conv2 (stride=2): (L - 1) // 2 + 1
        after_conv = (input_length - 1) // 2 + 1
        # After AvgPool1d (kernel=2, stride=2): floor(after_conv / 2)
        after_pool = after_conv // 2
        return after_pool

    def _extract_features(
        self, audio: Union[mx.array, np.ndarray]
    ) -> Tuple[mx.array, int]:
        n_mels = self.config.audio_config.num_mel_bins  # 128
        sample_rate = 16000
        n_fft = 400
        hop_length = 160
        max_source_positions = self.config.audio_config.max_source_positions  # 1500

        # Max audio length: max_source_positions * 2 * hop_length = 480000 samples
        max_samples = max_source_positions * 2 * hop_length

        if isinstance(audio, mx.array):
            audio_np = np.array(audio.reshape(-1), dtype=np.float32)
        else:
            audio_np = audio.flatten().astype(np.float32)

        # Pad or truncate to max_samples
        if len(audio_np) < max_samples:
            audio_np = np.pad(audio_np, (0, max_samples - len(audio_np)))
        else:
            audio_np = audio_np[:max_samples]

        # Hanning window
        window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

        # STFT with center padding (pad by n_fft // 2 on each side)
        pad = n_fft // 2
        audio_padded = np.pad(audio_np, (pad, pad), mode="reflect")

        n_frames = (len(audio_padded) - n_fft) // hop_length + 1
        frames = np.lib.stride_tricks.as_strided(
            audio_padded,
            shape=(n_frames, n_fft),
            strides=(audio_padded.strides[0] * hop_length, audio_padded.strides[0]),
        ).copy()

        windowed = frames * window[np.newaxis, :]
        spec = np.fft.rfft(windowed, n=n_fft)
        power = np.abs(spec) ** 2

        # Mel filterbank
        fmin = 0.0
        fmax = float(sample_rate) / 2.0
        n_freqs = n_fft // 2 + 1

        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        mel_spec = power @ filterbank.T  # (n_frames, n_mels)

        log_mel = np.log10(np.clip(mel_spec, 1e-10, None))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        # Shape: (1, n_mels, T) for encoder
        mel_tensor = mx.array(log_mel.T[np.newaxis], dtype=mx.float32)

        # Compute number of audio tokens after encoder processing
        mel_len = mel_tensor.shape[2]  # T dimension
        # After conv2 (stride=2): (mel_len - 1) // 2 + 1
        after_conv = (mel_len - 1) // 2 + 1
        # After avgpool (kernel=2, stride=2): floor(after_conv / 2)
        num_audio_tokens = after_conv // 2

        return mel_tensor, num_audio_tokens

    def _build_prompt(
        self,
        num_audio_tokens: int,
        user_prompt: str = None,
    ) -> mx.array:
        if user_prompt is None:
            user_prompt = "Please transcribe the speech."

        tokenizer = self._processor.tokenizer

        # Build the audio placeholder: <|audio_bos|> + N x <|AUDIO|> + <|audio_eos|>
        audio_content = (
            "<|audio_bos|>" + "<|AUDIO|>" * num_audio_tokens + "<|audio_eos|>"
        )
        content = audio_content + user_prompt

        chat = [{"role": "user", "content": content}]
        prompt_str = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = tokenizer.encode(prompt_str)
        return mx.array(prompt_ids)

    def _build_inputs_embeds(
        self, input_ids: mx.array, audio_features: mx.array
    ) -> mx.array:
        audio_token_id = self.audio_token_id
        is_audio = input_ids == audio_token_id
        llm_ids = mx.where(is_audio, 0, input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(llm_ids[None])

        is_audio_np = np.array(is_audio)
        audio_positions = np.where(is_audio_np)[0]

        orig_dtype = inputs_embeds.dtype
        embeds_np = np.array(inputs_embeds.astype(mx.float32))
        audio_np = np.array(audio_features.astype(mx.float32))

        num_audio = min(len(audio_positions), audio_np.shape[1])
        embeds_np[0, audio_positions[:num_audio]] = audio_np[0, :num_audio]

        return mx.array(embeds_np).astype(orig_dtype)

    def _load_audio(self, audio: Union[str, mx.array, np.ndarray]) -> mx.array:
        if isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            return load_audio(audio)
        elif isinstance(audio, np.ndarray):
            return mx.array(audio, dtype=mx.float32)
        elif isinstance(audio, mx.array):
            return audio
        elif isinstance(audio, list):
            audio_item = audio[0]
            if isinstance(audio_item, str):
                from mlx_audio.stt.utils import load_audio

                return load_audio(audio_item)
            return mx.array(np.array(audio_item), dtype=mx.float32)
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def generate(
        self,
        audio: Union[str, mx.array, np.ndarray],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        prompt: str = None,
        language: str = None,
        prefill_step_size: int = 2048,
        verbose: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[STTOutput, Generator[StreamingResult, None, None]]:
        if stream:
            return self._stream_generate(
                audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                prompt=prompt,
                prefill_step_size=prefill_step_size,
                verbose=verbose,
            )

        start_time = time.time()

        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        audio_data = self._load_audio(audio)
        input_features, num_audio_tokens = self._extract_features(audio_data)

        if verbose:
            print("Encoding audio...")
        audio_features = self.get_audio_features(input_features)
        mx.eval(audio_features)

        prompt_ids = self._build_prompt(num_audio_tokens, prompt)
        inputs_embeds = self._build_inputs_embeds(prompt_ids, audio_features)
        mx.eval(inputs_embeds)

        prompt_tokens = len(prompt_ids)

        sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        eos_token_id = self._processor.tokenizer.eos_token_id
        tokens = []

        for token, _logprobs in generate_step(
            prompt=prompt_ids,
            input_embeddings=inputs_embeds.squeeze(0),
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if token == eos_token_id:
                break
            tokens.append(token)

        text = self._processor.tokenizer.decode(tokens, skip_special_tokens=True)
        elapsed = time.time() - start_time
        gen_tokens = len(tokens)

        if verbose:
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Generation tokens: {gen_tokens}")
            print(f"Total time: {elapsed:.2f}s")
            if gen_tokens > 0:
                print(f"Generation TPS: {gen_tokens / elapsed:.1f}")

        return STTOutput(
            text=text,
            segments=[],
            prompt_tokens=prompt_tokens,
            generation_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
            total_time=elapsed,
            prompt_tps=prompt_tokens / elapsed if elapsed > 0 else 0,
            generation_tps=gen_tokens / elapsed if elapsed > 0 else 0,
        )

    def _stream_generate(
        self,
        audio: Union[str, mx.array, np.ndarray],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        prompt: str = None,
        prefill_step_size: int = 2048,
        verbose: bool = False,
    ) -> Generator[StreamingResult, None, None]:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        audio_data = self._load_audio(audio)
        input_features, num_audio_tokens = self._extract_features(audio_data)

        audio_features = self.get_audio_features(input_features)
        mx.eval(audio_features)

        prompt_ids = self._build_prompt(num_audio_tokens, prompt)
        inputs_embeds = self._build_inputs_embeds(prompt_ids, audio_features)
        mx.eval(inputs_embeds)

        prompt_token_count = len(prompt_ids)

        sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        eos_token_id = self._processor.tokenizer.eos_token_id
        gen_tokens = 0

        for token, _ in generate_step(
            prompt=prompt_ids,
            input_embeddings=inputs_embeds.squeeze(0),
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if token == eos_token_id:
                break
            gen_tokens += 1
            text = self._processor.tokenizer.decode([token], skip_special_tokens=True)
            yield StreamingResult(
                text=text,
                is_final=False,
                start_time=0.0,
                end_time=0.0,
                prompt_tokens=prompt_token_count,
                generation_tokens=gen_tokens,
            )

        yield StreamingResult(
            text="",
            is_final=True,
            start_time=0.0,
            end_time=0.0,
            prompt_tokens=prompt_token_count,
            generation_tokens=gen_tokens,
        )

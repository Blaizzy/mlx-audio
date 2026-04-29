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
from mlx_lm.models.granite import Model as GraniteLM
from mlx_lm.models.granite import ModelArgs as GraniteModelArgs

from mlx_audio.stt.models.base import STTOutput

from .config import EncoderConfig, ModelConfig, ProjectorConfig

LANGUAGE_CODES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
}


@dataclass
class StreamingResult:
    text: str
    is_final: bool
    start_time: float
    end_time: float
    language: str = "en"
    prompt_tokens: int = 0
    generation_tokens: int = 0


class BatchNorm1d(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.running_mean) / mx.sqrt(
            self.running_var + self.eps
        ) * self.weight + self.bias


class ConformerFeedForward(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.up_proj = nn.Linear(
            config.hidden_dim, config.hidden_dim * config.feedforward_mult
        )
        self.down_proj = nn.Linear(
            config.hidden_dim * config.feedforward_mult, config.hidden_dim
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_norm(x)
        x = nn.silu(self.up_proj(x))
        x = self.down_proj(x)
        return x


class ConformerAttention(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        inner_dim = config.dim_head * config.num_heads
        self.max_pos_emb = config.max_pos_emb
        self.context_size = config.context_size
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        self.scale = config.dim_head**-0.5
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.to_q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(config.hidden_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, config.hidden_dim)
        self.rel_pos_emb = nn.Embedding(2 * self.max_pos_emb + 1, self.dim_head)

    def __call__(self, x: mx.array, attention_dists: mx.array) -> mx.array:
        x = self.pre_norm(x)
        B, N, _ = x.shape

        num_blocks = math.ceil(N / self.context_size)
        remainder = N % self.context_size

        if remainder > 0:
            pad_len = self.context_size - remainder
            x = mx.pad(x, [(0, 0), (0, pad_len), (0, 0)])

        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, num_blocks, self.context_size, self.num_heads, -1)
        k = k.reshape(B, num_blocks, self.context_size, self.num_heads, -1)
        v = v.reshape(B, num_blocks, self.context_size, self.num_heads, -1)

        q = q.transpose(0, 1, 3, 2, 4)
        k = k.transpose(0, 1, 3, 2, 4)
        v = v.transpose(0, 1, 3, 2, 4)

        rel_pos_emb = self.rel_pos_emb(attention_dists)

        C = self.context_size
        pos_attn = (
            mx.sum(
                q[:, :, :, :, None, :] * rel_pos_emb[None, None, None, :, :, :],
                axis=-1,
            )
            * self.scale
        )

        if remainder > 0:
            row_valid = mx.arange(C)[:, None] < remainder
            col_valid = mx.arange(C)[None, :] < remainder
            mask = ~(row_valid & col_valid)
            mask_value = mx.array(mx.finfo(pos_attn.dtype).min)
            pos_attn_last = mx.where(
                mask[None, None, None], mask_value, pos_attn[:, -1:, :, :, :]
            )
            pos_attn = mx.concatenate(
                [pos_attn[:, :-1, :, :, :], pos_attn_last], axis=1
            )

        attn_weights = (q @ k.transpose(0, 1, 2, 4, 3)) * self.scale + pos_attn
        attn_weights = mx.softmax(attn_weights, axis=-1)

        out = attn_weights @ v
        out = out.transpose(0, 1, 3, 2, 4)
        out = out.reshape(B, -1, self.num_heads * self.dim_head)
        out = out[:, :N, :]
        out = self.to_out(out)
        return out


class DepthWiseConv1d(nn.Module):

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = (pad, pad - pad_offset)
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, groups=chan_in, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.padding[0], self.padding[1]), (0, 0)])
        return self.conv(x)


class ConformerConvModule(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(config.hidden_dim, inner_dim * 2, 1)
        self.depth_conv = DepthWiseConv1d(inner_dim, inner_dim, config.conv_kernel_size)
        self.batch_norm = BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, config.hidden_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.up_conv(x)
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * mx.sigmoid(x2)
        x = self.depth_conv(x)
        x = nn.silu(self.batch_norm(x))
        x = self.down_conv(x)
        return x


class ConformerBlock(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.ff1 = ConformerFeedForward(config)
        self.attn = ConformerAttention(config)
        self.conv = ConformerConvModule(config)
        self.ff2 = ConformerFeedForward(config)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def __call__(self, x: mx.array, attention_dists: mx.array) -> mx.array:
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x, attention_dists) + x
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        x = self.post_norm(x)
        return x


class CTCEncoder(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = [ConformerBlock(config) for _ in range(config.num_layers)]
        self.out = nn.Linear(config.hidden_dim, config.output_dim)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim)
        self.num_layers = config.num_layers
        self._attention_dists = None

        seq = mx.arange(config.context_size)
        relpos_dist = seq[:, None] - seq[None, :]
        self._attention_dists = (
            mx.clip(relpos_dist, -config.context_size, config.context_size)
            + config.max_pos_emb
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.input_linear(x)
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, attention_dists=self._attention_dists)
            if idx == self.num_layers // 2:
                x_mid = self.out(x)
                x = x + self.out_mid(mx.softmax(x_mid, axis=-1))
        return x


class QFormerMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, kv_hidden_size: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kv_dim = kv_hidden_size or hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(kv_dim, hidden_size)
        self.value = nn.Linear(kv_dim, hidden_size)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array = None
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        q = self.query(hidden_states)
        kv_input = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        k = self.key(kv_input)
        v = self.value(kv_input)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return out


class QFormerSelfOutput(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_hidden_size: int = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = QFormerMultiHeadAttention(
            hidden_size, num_heads, kv_hidden_size
        )
        self.output = QFormerSelfOutput(hidden_size, eps)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array = None
    ) -> mx.array:
        attn_out = self.attention(hidden_states, encoder_hidden_states)
        return self.output(attn_out, hidden_states)


class QFormerIntermediate(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.dense(x))


class QFormerOutput(nn.Module):

    def __init__(self, intermediate_size: int, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerLayer(nn.Module):

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.attention = QFormerAttention(
            config.hidden_size, config.num_attention_heads, eps=config.layer_norm_eps
        )
        self.crossattention = QFormerAttention(
            config.hidden_size,
            config.num_attention_heads,
            kv_hidden_size=config.encoder_hidden_size,
            eps=config.layer_norm_eps,
        )
        self.intermediate_query = QFormerIntermediate(
            config.hidden_size, config.intermediate_size
        )
        self.output_query = QFormerOutput(
            config.intermediate_size, config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        hidden_states = self.attention(hidden_states)
        hidden_states = self.crossattention(hidden_states, encoder_hidden_states)
        intermediate = self.intermediate_query(hidden_states)
        hidden_states = self.output_query(intermediate, hidden_states)
        return hidden_states


class QFormerEncoder(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layer = [QFormerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        for layer in self.layer:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return hidden_states


class QFormerModel(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = QFormerEncoder(config)

    def __call__(
        self, query_embeds: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        hidden_states = self.layernorm(query_embeds)
        return self.encoder(hidden_states, encoder_hidden_states)


class EncoderProjector(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.projector_config.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.window_size // config.downsample_rate

        self.query = mx.zeros(
            (1, self.num_queries, config.projector_config.hidden_size)
        )
        self.qformer = QFormerModel(config.projector_config)
        self.linear = nn.Linear(
            config.projector_config.hidden_size, config.text_config.hidden_size
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, L, D = hidden_states.shape
        nblocks = math.ceil(L / self.window_size)
        pad = nblocks * self.window_size - L
        if pad > 0:
            hidden_states = mx.pad(hidden_states, [(0, 0), (0, pad), (0, 0)])

        hidden_states = hidden_states.reshape(B * nblocks, self.window_size, D)

        query = mx.broadcast_to(
            self.query, (B * nblocks, self.num_queries, self.hidden_size)
        )

        query_output = self.qformer(query, hidden_states)
        query_proj = self.linear(
            query_output.reshape(B, nblocks * self.num_queries, -1)
        )
        return query_proj


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = CTCEncoder(config.encoder_config)
        self.projector = EncoderProjector(config)
        text_args = GraniteModelArgs.from_dict(
            config.text_config.__dict__
            if hasattr(config.text_config, "__dict__")
            else config.text_config
        )
        self.language_model = GraniteLM(text_args)

        self.audio_token_id = config.audio_token_index
        self._tokenizer = None

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

        h = h * self.language_model.model.embedding_multiplier

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

        return logits / self.language_model.logits_scaling

    def get_audio_features(self, input_features: mx.array) -> mx.array:
        encoder_output = self.encoder(input_features)
        projected = self.projector(encoder_output)
        return projected

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return not (p.startswith("encoder") or p.startswith("projector"))

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        already_converted = any("scales" in k for k in weights)

        sanitized = {}
        for k, v in weights.items():
            if "num_batches_tracked" in k:
                continue

            if (
                not already_converted
                and any(name in k for name in ["up_conv", "down_conv", "depth_conv"])
                and "weight" in k
                and len(v.shape) == 3
            ):
                # MLX Conv1d expects weights in (out_channels, kernel_size, in_channels)
                # layout, while PyTorch uses (out_channels, in_channels, kernel_size).
                # Models converted from PyTorch checkpoints need transposing; models
                # already saved in MLX-native layout (e.g. bf16 safetensors) do not.
                # Detect PyTorch layout: shape[-1] > shape[-2] indicates
                # (out, in, kernel) where kernel > in (true for depthwise convs).
                # When kernel_size == 1 the shapes are symmetric and no-op either way.
                if v.shape[-1] > v.shape[-2]:
                    v = v.transpose(0, 2, 1)

            sanitized[k] = v
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        import transformers
        from transformers import AutoTokenizer

        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
        finally:
            transformers.logging.set_verbosity(prev)

        return model

    def _extract_features(
        self, audio: Union[mx.array, np.ndarray]
    ) -> Tuple[mx.array, int]:
        from mlx_audio.dsp import hanning, mel_filters, stft

        n_fft = 512
        win_length = 400
        hop_length = 160
        n_mels = 80
        sample_rate = 16000

        if isinstance(audio, mx.array):
            audio_1d = audio.reshape(-1)
        else:
            audio_1d = mx.array(audio.flatten(), dtype=mx.float32)

        win = hanning(win_length, periodic=True)
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win_padded = mx.concatenate(
            [mx.zeros((pad_left,)), win, mx.zeros((pad_right,))]
        )

        spec = stft(
            audio_1d,
            n_fft=n_fft,
            hop_length=hop_length,
            window=win_padded,
            center=True,
            pad_mode="reflect",
        )

        power = mx.abs(spec) ** 2
        mel_fb = mel_filters(sample_rate, n_fft, n_mels, mel_scale="htk")
        mel_spec = power @ mel_fb.T

        logmel = mx.log10(mx.clip(mel_spec, 1e-10, None))
        mx_val = mx.max(logmel)
        logmel = mx.maximum(logmel, mx_val - 8.0) / 4.0 + 1.0

        if logmel.shape[0] % 2 == 1:
            logmel = logmel[:-1]

        encoder_input = logmel.reshape(-1, 2 * n_mels)

        encoder_length = encoder_input.shape[0]
        nblocks = math.ceil(encoder_length / self.config.window_size)
        num_audio_tokens = nblocks * (
            self.config.window_size // self.config.downsample_rate
        )

        input_features = encoder_input[None, :, :]
        return input_features, num_audio_tokens

    def _build_prompt(
        self,
        num_audio_tokens: int,
        user_prompt: str = None,
    ) -> mx.array:
        if user_prompt is None:
            user_prompt = "can you transcribe the speech into a written format?"

        audio_placeholder = "<|audio|>" * num_audio_tokens
        content = f"{audio_placeholder}{user_prompt}"

        if getattr(self._tokenizer, "chat_template", None):
            chat = [{"role": "user", "content": content}]
            prompt_str = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = f"USER: {content}\nASSISTANT:"

        prompt_ids = self._tokenizer.encode(prompt_str)

        return mx.array(prompt_ids)

    def _build_inputs_embeds(
        self, input_ids: mx.array, audio_features: mx.array
    ) -> mx.array:
        is_audio = input_ids == self.audio_token_id
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
        if prompt is None and language is not None:
            lang_name = LANGUAGE_CODES.get(language.lower(), language)
            prompt = f"Translate the speech to {lang_name}."

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

        eos_token_id = self._tokenizer.eos_token_id
        tokens = []

        for token, logprobs in generate_step(
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

        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
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

        eos_token_id = self._tokenizer.eos_token_id
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
            text = self._tokenizer.decode([token], skip_special_tokens=True)
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

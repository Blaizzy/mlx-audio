from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import TextModel, split_audio_into_chunks

from .audio import prepare_audio
from .config import AdaptorConfig, FunASRNanoConfig, SenseVoiceEncoderConfig

ISO_TO_PROMPT_LANGUAGE = {
    "zh": "中文",
    "zh-cn": "中文",
    "zh-hans": "中文",
    "zh-tw": "中文",
    "zh-hant": "中文",
    "cmn": "中文",
    "cjy": "中文",
    "gan": "中文",
    "hak": "中文",
    "hsn": "中文",
    "nan": "中文",
    "wuu": "中文",
    "yue": "中文",
    "en": "英文",
    "eng": "英文",
    "ja": "日文",
    "jpn": "日文",
    "jp": "日文",
}
SUPPORTED_ISO_LANGUAGES = ", ".join(
    sorted({"cjy", "cmn", "en", "gan", "hak", "hsn", "ja", "nan", "wuu", "yue", "zh"})
)


def _sequence_mask(lengths: mx.array, maxlen: int) -> mx.array:
    positions = mx.arange(maxlen)[None, :]
    return (positions < lengths[:, None]).astype(mx.float32)


class SinusoidalPositionEncoder(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        batch_size, timesteps, input_dim = x.shape
        positions = mx.arange(1, timesteps + 1, dtype=x.dtype)[None, :]
        half_dim = input_dim // 2
        log_timescale_increment = math.log(10000) / (half_dim - 1)
        inv_timescales = mx.exp(
            mx.arange(half_dim, dtype=x.dtype) * -log_timescale_increment
        )
        inv_timescales = mx.broadcast_to(
            inv_timescales[None, :], (batch_size, half_dim)
        )
        scaled_time = positions[:, :, None] * inv_timescales[:, None, :]
        encoding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=2)
        return x + encoding.astype(x.dtype)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(nn.relu(self.w_1(x)))


class MultiHeadedAttentionSANM(nn.Module):
    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        kernel_size: int = 11,
        sanm_shift: int = 0,
    ):
        super().__init__()
        if n_feat % n_head != 0:
            raise ValueError("n_feat must be divisible by n_head")
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.fsmn_block = nn.Conv1d(
            n_feat,
            n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=n_feat,
            bias=False,
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding += sanm_shift
        self.left_padding = left_padding
        self.right_padding = kernel_size - 1 - left_padding

    def _forward_fsmn(self, inputs: mx.array, mask: Optional[mx.array]) -> mx.array:
        if mask is not None:
            inputs = inputs * mask.transpose(0, 2, 1)
        x = mx.pad(
            inputs,
            pad_width=((0, 0), (self.left_padding, self.right_padding), (0, 0)),
        )
        x = self.fsmn_block(x)
        x = x + inputs
        if mask is not None:
            x = x * mask.transpose(0, 2, 1)
        return x

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, _ = x.shape
        q_k_v = self.linear_q_k_v(x)
        q, k, v = mx.split(q_k_v, 3, axis=-1)
        fsmn_memory = self._forward_fsmn(v, mask)

        q = q.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        v_h = v.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)

        scores = (q * (self.d_k**-0.5)) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            attn_mask = mask[:, None, :, :] == 0
            scores = mx.where(attn_mask, mx.array(-1e9, dtype=scores.dtype), scores)
        attn = mx.softmax(scores, axis=-1)
        if mask is not None:
            attn = mx.where(attn_mask, mx.zeros_like(attn), attn)

        att_out = attn @ v_h
        att_out = att_out.transpose(0, 2, 1, 3).reshape(B, T, self.n_feat)
        return self.linear_out(att_out) + fsmn_memory


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size: int,
        size: int,
        self_attn: MultiHeadedAttentionSANM,
        feed_forward: PositionwiseFeedForward,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(size)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        attn_out = self.self_attn(x, mask)
        x = residual + attn_out if self.in_size == self.size else attn_out

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        return residual + self.feed_forward(x)


class SenseVoiceEncoderSmall(nn.Module):
    def __init__(self, input_size: int, config: SenseVoiceEncoderConfig):
        super().__init__()
        self._output_size = config.output_size
        self.embed = SinusoidalPositionEncoder()
        self.encoders0 = [
            EncoderLayerSANM(
                input_size,
                config.output_size,
                MultiHeadedAttentionSANM(
                    config.attention_heads,
                    input_size,
                    config.output_size,
                    config.kernel_size,
                    config.sanm_shift,
                ),
                PositionwiseFeedForward(config.output_size, config.linear_units),
                normalize_before=config.normalize_before,
            )
        ]
        self.encoders = [
            EncoderLayerSANM(
                config.output_size,
                config.output_size,
                MultiHeadedAttentionSANM(
                    config.attention_heads,
                    config.output_size,
                    config.output_size,
                    config.kernel_size,
                    config.sanm_shift,
                ),
                PositionwiseFeedForward(config.output_size, config.linear_units),
                normalize_before=config.normalize_before,
            )
            for _ in range(config.num_blocks - 1)
        ]
        self.after_norm = nn.LayerNorm(config.output_size)
        self.tp_encoders = [
            EncoderLayerSANM(
                config.output_size,
                config.output_size,
                MultiHeadedAttentionSANM(
                    config.attention_heads,
                    config.output_size,
                    config.output_size,
                    config.kernel_size,
                    config.sanm_shift,
                ),
                PositionwiseFeedForward(config.output_size, config.linear_units),
                normalize_before=config.normalize_before,
            )
            for _ in range(config.tp_blocks)
        ]
        self.tp_norm = nn.LayerNorm(config.output_size)

    def output_size(self) -> int:
        return self._output_size

    def __call__(self, xs_pad: mx.array, ilens: mx.array) -> Tuple[mx.array, mx.array]:
        maxlen = xs_pad.shape[1]
        mask = _sequence_mask(ilens, maxlen=maxlen)[:, None, :]

        xs_pad = xs_pad * (self._output_size**0.5)
        xs_pad = self.embed(xs_pad)

        for layer in self.encoders0:
            xs_pad = layer(xs_pad, mask)
        for layer in self.encoders:
            xs_pad = layer(xs_pad, mask)
        xs_pad = self.after_norm(xs_pad)
        olens = mask.squeeze(1).sum(axis=1).astype(mx.int32)

        for layer in self.tp_encoders:
            xs_pad = layer(xs_pad, mask)
        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        if n_feat % n_head != 0:
            raise ValueError("n_feat must be divisible by n_head")
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, _ = x.shape
        q = self.linear_q(x).reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        k = self.linear_k(x).reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)

        scores = (q * (self.d_k**-0.5)) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            attn_mask = mask[:, None, :, :] == 0
            scores = mx.where(attn_mask, mx.array(-1e9, dtype=scores.dtype), scores)
        attn = mx.softmax(scores, axis=-1)
        if mask is not None:
            attn = mx.where(attn_mask, mx.zeros_like(attn), attn)
        y = attn @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, self.n_feat)
        return self.linear_out(y)


class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadedAttention):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = PositionwiseFeedForward(size, size // 4)
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, Optional[mx.array]]:
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, mask)
        residual = x
        x = self.norm2(x)
        return residual + self.feed_forward(x), mask


class AudioAdaptorTransformer(nn.Module):
    def __init__(self, config: AdaptorConfig):
        super().__init__()
        self.k = config.downsample_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, config.ffn_dim)
        self.linear2 = nn.Linear(config.ffn_dim, self.llm_dim)
        self.blocks = [
            EncoderLayer(
                self.llm_dim, MultiHeadedAttention(config.attention_heads, self.llm_dim)
            )
            for _ in range(config.n_layer)
        ]

    def __call__(self, x: mx.array, ilens: mx.array) -> Tuple[mx.array, mx.array]:
        max_len = int(ilens.max().item())
        x = x[:, :max_len, :]
        batch_size, seq_len, dim = x.shape
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        if pad_num > 0:
            x = mx.pad(x, [(0, 0), (0, pad_num), (0, 0)])
        x = x.reshape(batch_size, chunk_num, dim * self.k)
        x = self.linear2(nn.relu(self.linear1(x)))
        olens = ((ilens - 1) // self.k + 1).astype(mx.int32)
        mask = _sequence_mask(olens, maxlen=x.shape[1])[:, None, :]
        for block in self.blocks:
            x, mask = block(x, mask)
        return x, olens


class Qwen3CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TextModel(config)


def _normalise_text_for_join(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("/sil", " ")).strip()


class FunASRNano(nn.Module):
    def __init__(self, config: FunASRNanoConfig):
        super().__init__()
        if isinstance(config, dict):
            config = FunASRNanoConfig.from_dict(config)
        self.config = config
        self.audio_encoder = SenseVoiceEncoderSmall(
            config.input_size, config.audio_encoder_conf
        )
        self.audio_adaptor = AudioAdaptorTransformer(config.audio_adaptor_conf)
        self.llm = Qwen3CausalLM(config.text_config)
        self._tokenizer = None

    @property
    def sample_rate(self) -> int:
        return self.config.frontend_conf.fs

    @property
    def layers(self):
        return self.llm.model.layers

    def make_cache(self) -> List[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

    def __call__(
        self,
        input_ids: mx.array,
        input_embeddings: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if input_embeddings is None:
            input_embeddings = self.llm.model.embed_tokens(input_ids)
        hidden_states = self.llm.model(inputs_embeds=input_embeddings, cache=cache)
        return self.llm.model.embed_tokens.as_linear(hidden_states)

    @staticmethod
    def _map_language(language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        normalized = language.lower().replace("_", "-")
        if normalized in {"", "auto"}:
            return None
        if normalized in ISO_TO_PROMPT_LANGUAGE:
            return ISO_TO_PROMPT_LANGUAGE[normalized]
        if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]+)*", normalized):
            raise ValueError(
                "Unsupported ISO language for Fun-ASR-Nano-2512: "
                f"{language!r}. Supported ISO languages: {SUPPORTED_ISO_LANGUAGES}."
            )
        return language

    @staticmethod
    def _resolve_hotwords(
        hotwords: Optional[Iterable[str]], context: Optional[str]
    ) -> Optional[List[str]]:
        resolved_hotwords = []
        if hotwords is not None:
            for item in hotwords:
                word = item.strip()
                if word:
                    resolved_hotwords.append(word)
        context = context.strip() if context is not None else ""
        if resolved_hotwords and context:
            raise ValueError("Pass either hotwords or context, not both.")
        if resolved_hotwords:
            return resolved_hotwords
        return [context] if context else None

    @staticmethod
    def _prompt_text(
        hotwords: Optional[Iterable[str]] = None,
        language: Optional[str] = None,
        itn: bool = True,
    ) -> str:
        hotwords = list(hotwords or [])
        prompt = ""
        if hotwords:
            prompt += "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{', '.join(hotwords)}]\n"
        prompt += "语音转写" if language is None else f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    def _build_prompt_ids(
        self,
        fake_token_len: int,
        *,
        language: Optional[str],
        hotwords: Optional[Iterable[str]],
        itn: bool,
    ) -> Tuple[mx.array, int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call post_load_hook first.")
        language_label = self._map_language(language)
        user_prompt = self._prompt_text(hotwords, language_label, itn)
        before_audio = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}"
        )
        after_audio = "<|im_end|>\n<|im_start|>assistant\n"
        before_ids = self._tokenizer.encode(before_audio)
        after_ids = self._tokenizer.encode(after_audio)
        fbank_beg = len(before_ids)
        ids = before_ids + [0] * int(fake_token_len) + after_ids
        return mx.array([ids], dtype=mx.int32), fbank_beg

    def _build_inputs_embeds(
        self,
        audio,
        *,
        language: Optional[str],
        hotwords: Optional[Iterable[str]],
        itn: bool,
    ) -> Tuple[mx.array, mx.array]:
        feats, speech_lengths, fake_len = prepare_audio(
            audio, self.config.frontend_conf
        )
        encoder_out, encoder_out_lens = self.audio_encoder(feats, speech_lengths)
        adaptor_out, adaptor_out_lens = self.audio_adaptor(
            encoder_out, encoder_out_lens
        )
        del adaptor_out_lens
        input_ids, fbank_beg = self._build_prompt_ids(
            fake_len, language=language, hotwords=hotwords, itn=itn
        )
        inputs_embeds = self.llm.model.embed_tokens(input_ids)
        speech_token_len = min(int(fake_len), int(adaptor_out.shape[1]))
        speech_token = adaptor_out[:, :speech_token_len, :].astype(inputs_embeds.dtype)
        inputs_embeds = mx.concatenate(
            [
                inputs_embeds[:, :fbank_beg, :],
                speech_token,
                inputs_embeds[:, fbank_beg + speech_token_len :, :],
            ],
            axis=1,
        )
        return input_ids, inputs_embeds

    def stream_generate(
        self,
        audio,
        *,
        max_tokens: int = 512,
        sampler=None,
        logits_processors=None,
        language: Optional[str] = None,
        hotwords: Optional[Iterable[str]] = None,
        context: Optional[str] = None,
        itn: bool = True,
        prefill_step_size: int = 2048,
    ):
        from mlx_lm.generate import generate_step

        hotwords = self._resolve_hotwords(hotwords, context)
        input_ids, inputs_embeds = self._build_inputs_embeds(
            audio, language=language, hotwords=hotwords, itn=itn
        )
        eos_token_ids = {151643, 151645}
        for token, logprobs in generate_step(
            prompt=input_ids[0],
            input_embeddings=inputs_embeds[0],
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if int(token) in eos_token_ids:
                break
            yield token, logprobs

    def _generate_single_chunk(
        self,
        audio,
        *,
        max_tokens: int,
        sampler,
        logits_processors,
        language: Optional[str],
        hotwords: Optional[Iterable[str]],
        itn: bool,
        prefill_step_size: int,
    ) -> Tuple[str, int, int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call post_load_hook first.")

        input_ids, inputs_embeds = self._build_inputs_embeds(
            audio, language=language, hotwords=hotwords, itn=itn
        )
        prompt_tokens = int(input_ids.shape[1])
        generated_tokens = []
        eos_token_ids = {151643, 151645}

        from mlx_lm.generate import generate_step

        for token, _ in generate_step(
            prompt=input_ids[0],
            input_embeddings=inputs_embeds[0],
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            token_id = int(token)
            if token_id in eos_token_ids:
                break
            generated_tokens.append(token_id)

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _normalise_text_for_join(text), prompt_tokens, len(generated_tokens)

    def generate(
        self,
        audio: Union[
            str,
            Path,
            mx.array,
            np.ndarray,
            List[Union[str, Path, mx.array, np.ndarray]],
        ],
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        language: Optional[str] = None,
        hotwords: Optional[Iterable[str]] = None,
        context: Optional[str] = None,
        itn: bool = True,
        prefill_step_size: int = 2048,
        chunk_duration: float = 1200.0,
        min_chunk_duration: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        from mlx_audio.stt.utils import load_audio

        del verbose, kwargs
        start_time = time.time()
        max_tokens = int(max_tokens or self.config.default_max_tokens)
        hotwords = self._resolve_hotwords(hotwords, context)

        audio_input = audio[0] if isinstance(audio, list) else audio
        if isinstance(audio_input, (str, Path)):
            audio_input = load_audio(str(audio_input), sr=self.sample_rate)
        audio_np = (
            np.array(audio_input) if isinstance(audio_input, mx.array) else audio_input
        )
        chunks = split_audio_into_chunks(
            np.asarray(audio_np),
            sr=self.sample_rate,
            chunk_duration=chunk_duration,
            min_chunk_duration=min_chunk_duration,
        )

        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )
        logits_processors = (
            make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if repetition_penalty
            else None
        )

        texts = []
        segments = []
        prompt_tokens = 0
        generation_tokens = 0
        remaining_tokens = max_tokens
        for chunk_audio, offset_sec in chunks:
            if remaining_tokens <= 0:
                break
            text, p_toks, g_toks = self._generate_single_chunk(
                chunk_audio,
                max_tokens=remaining_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                language=language,
                hotwords=hotwords,
                itn=itn,
                prefill_step_size=prefill_step_size,
            )
            duration = len(chunk_audio) / self.sample_rate
            segment = {
                "text": text,
                "start": offset_sec,
                "end": offset_sec + duration,
                "language": language,
            }
            segments.append(segment)
            texts.append(text)
            prompt_tokens += p_toks
            generation_tokens += g_toks
            remaining_tokens -= g_toks
            mx.clear_cache()

        total_time = time.time() - start_time
        return STTOutput(
            text=" ".join(t for t in texts if t).strip(),
            segments=segments,
            language=language,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=prompt_tokens + generation_tokens,
            total_time=total_time,
            prompt_tps=prompt_tokens / total_time if total_time > 0 else 0.0,
            generation_tps=generation_tokens / total_time if total_time > 0 else 0.0,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if (
                key == "llm.lm_head.weight"
                and self.config.text_config.tie_word_embeddings
            ):
                continue
            if (
                key.endswith("fsmn_block.weight")
                and value.ndim == 3
                and value.shape[1] == 1
            ):
                value = value.transpose(0, 2, 1)
            sanitized[key] = value
        return sanitized

    def model_quant_predicate(self, path: str, module: nn.Module) -> bool:
        return path.startswith("llm.model")

    @staticmethod
    def post_load_hook(model: "FunASRNano", model_path: Path) -> "FunASRNano":
        from transformers import AutoTokenizer

        model_path = Path(model_path)
        tokenizer_path = model_path / model.config.qwen_tokenizer_path
        model._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True,
        )

        return model

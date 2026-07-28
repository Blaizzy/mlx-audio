import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args
from mlx_lm.models.qwen3 import Qwen3Model

from mlx_audio.stt.models.base import STTOutput

from .audio import AudioFeatureExtractor
from .config import AudioEncoderConfig, ModelConfig

DEFAULT_PROMPT = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."


class AudioAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class HiggsAudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, embed_dim)
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(embed_dim)

    def __call__(self, input_features: mx.array) -> mx.array:
        x = input_features.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        T = x.shape[1]
        x = x + self.embed_positions.weight[:T][None]

        for layer in self.layers:
            x = layer(x)

        B, seq_len, D = x.shape
        x = x[:, : (seq_len // 2) * 2, :].reshape(B, seq_len // 2, 2, D).mean(axis=2)
        x = self.layer_norm(x)
        return x


class HiggsAudioFeatureProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        audio_dim = config.audio_encoder_config.d_model
        llm_hidden = config.text_config.hidden_size
        self.stride = config.projector_temporal_downsample
        if self.stride > 1:
            self.temporal = nn.Conv1d(
                audio_dim,
                audio_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=audio_dim,
                bias=True,
            )
        self.linear1 = nn.Linear(audio_dim, 2048, bias=True)
        self.linear2 = nn.Linear(2048, llm_hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        if self.stride > 1:
            x = self.temporal(x)
        x = self.linear1(x)
        x = nn.relu(x)
        return self.linear2(x)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)
        self.audio_encoder_proj = HiggsAudioFeatureProjector(config)

        text_args = Qwen3Args.from_dict(
            {
                "model_type": "qwen3",
                "hidden_size": config.text_config.hidden_size,
                "num_hidden_layers": config.text_config.num_hidden_layers,
                "intermediate_size": config.text_config.intermediate_size,
                "num_attention_heads": config.text_config.num_attention_heads,
                "num_key_value_heads": config.text_config.num_key_value_heads,
                "head_dim": config.text_config.head_dim,
                "rms_norm_eps": config.text_config.rms_norm_eps,
                "vocab_size": config.text_config.vocab_size,
                "max_position_embeddings": config.text_config.max_position_embeddings,
                "rope_theta": config.text_config.rope_theta,
                "rope_scaling": config.text_config.rope_scaling,
                "tie_word_embeddings": False,
            }
        )
        self.model = Qwen3Model(text_args)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

        self.audio_in_token_idx = config.audio_in_token_idx
        self.feature_extractor = AudioFeatureExtractor(
            num_mel_bins=config.audio_encoder_config.num_mel_bins,
            sample_rate=config.sample_rate,
        )
        self._vad_backend = None

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in range(len(self.layers))]

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[KVCache]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        return self.lm_head(out)

    def get_audio_features(self, mel: mx.array) -> mx.array:
        encoded = self.audio_tower(mel)
        projected = self.audio_encoder_proj(encoded)
        lm_dtype = self.model.embed_tokens.weight.dtype
        return projected.astype(lm_dtype)

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return not (p.startswith("audio_tower") or p.startswith("audio_encoder_proj"))

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        already_converted = any("scales" in k for k in weights)
        sanitized = {}
        for k, v in weights.items():
            if k.startswith("audio_codebook_embeddings") or k.startswith(
                "audio_decoder_proj.audio_lm_head"
            ):
                continue

            if k == "audio_decoder_proj.text_lm_head.weight":
                sanitized["lm_head.weight"] = v
                continue

            if k == "embed_tokens.weight":
                sanitized["model.embed_tokens.weight"] = v
                continue
            if k == "norm.weight":
                sanitized["model.norm.weight"] = v
                continue
            if k.startswith("layers."):
                k = "model." + k

            if not already_converted and "weight" in k and v.ndim == 3:
                if "audio_tower.conv" in k:
                    v = v.transpose(0, 2, 1)
                elif "audio_encoder_proj.temporal" in k:
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

    def _extract_features(self, wav: mx.array) -> mx.array:
        dtype = self.audio_tower.conv1.weight.dtype
        return self.feature_extractor(wav).astype(dtype)

    def _get_vad_backend(self):
        if self._vad_backend is None:
            from .vad import SileroVADBackend

            self._vad_backend = SileroVADBackend(sample_rate=self.sample_rate)
        return self._vad_backend

    def _chunk_waveform(self, wav: np.ndarray) -> List[np.ndarray]:
        from .vad import vad_chunk_ranges

        chunk = int(self.config.chunk_size_seconds * self.sample_rate)
        backend = self._get_vad_backend() if self.config.vad_cut else None
        ranges = vad_chunk_ranges(
            wav, chunk, backend=backend, split_vads=self.config.split_vads
        )
        return [wav[s:e] for s, e in ranges]

    def get_input_embeddings(
        self,
        audio: Union[str, mx.array, np.ndarray],
        user_prompt: str = DEFAULT_PROMPT,
        verbose: bool = False,
    ) -> Tuple[List[int], mx.array, int]:
        wav = self._load_audio(audio)
        chunks = self._chunk_waveform(wav)

        max_frames = max(len(c) for c in chunks)
        feature_list = []
        for c in chunks:
            if len(c) < max_frames:
                c = np.pad(c, (0, max_frames - len(c)))
            mel = self._extract_features(mx.array(c, dtype=mx.float32))
            feature_list.append(self.get_audio_features(mel))

        tok = self._tokenizer

        def enc(s):
            return tok.encode(s, add_special_tokens=False)

        prefix = enc("<|im_start|>user\n") + enc(user_prompt) + enc("<|audio_bos|>")
        suffix = (
            enc("<|audio_eos|>") + enc("<|im_end|>\n") + enc("<|im_start|>assistant\n")
        )
        audio_ids = [self.audio_in_token_idx] * len(chunks)
        input_ids = prefix + audio_ids + suffix

        text_embeds = self.model.embed_tokens(mx.array(input_ids)[None])
        dtype = text_embeds.dtype

        segments = [text_embeds[:, : len(prefix), :]]
        for feat in feature_list:
            segments.append(feat.astype(dtype))
        segments.append(text_embeds[:, len(prefix) + len(chunks) :, :])
        inputs_embeds = mx.concatenate(segments, axis=1)
        mx.eval(inputs_embeds)

        prompt_len = inputs_embeds.shape[1]
        return input_ids, inputs_embeds, prompt_len

    def _load_single_audio(self, audio: Union[str, mx.array, np.ndarray]) -> np.ndarray:
        if isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            return np.array(load_audio(audio), dtype=np.float32)
        if isinstance(audio, mx.array):
            return np.array(audio, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32)

    def _load_audio(self, audio: Union[str, mx.array, np.ndarray, List]) -> np.ndarray:
        if isinstance(audio, list):
            audio = audio[0]
        wav = self._load_single_audio(audio)
        return wav.reshape(-1)

    @staticmethod
    def _parse_output(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in text:
            text = text[text.index("<think>") + len("<think>") :]
        text = re.sub(r"<\|.*?\|>", "", text)
        return text.strip()

    def generate(
        self,
        audio: Union[str, mx.array, np.ndarray, List],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        prompt: str = DEFAULT_PROMPT,
        prefill_step_size: int = 2048,
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if not hasattr(self, "_tokenizer"):
            raise RuntimeError("Tokenizer not initialized. Call post_load_hook first.")

        start_time = time.time()
        input_ids, inputs_embeds, prompt_tokens = self.get_input_embeddings(
            audio, prompt, verbose
        )
        prefill_time = time.time() - start_time

        sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = (
            make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if repetition_penalty
            else None
        )

        eos_token_ids = {151645, 151643}
        tokens = []
        gen_start = time.time()
        for token, _ in generate_step(
            prompt=mx.array([], dtype=mx.int32),
            input_embeddings=inputs_embeds.squeeze(0),
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if token in eos_token_ids:
                break
            tokens.append(int(token))

        full_text = self._tokenizer.decode(tokens, skip_special_tokens=False)
        text = self._parse_output(full_text)
        elapsed = time.time() - start_time
        gen_time = time.time() - gen_start
        gen_tokens = len(tokens)

        return STTOutput(
            text=text,
            segments=[{"start": 0.0, "end": elapsed, "text": text}],
            prompt_tokens=prompt_tokens,
            generation_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
            total_time=elapsed,
            prompt_tps=prompt_tokens / prefill_time if prefill_time > 0 else 0,
            generation_tps=gen_tokens / gen_time if gen_time > 0 else 0,
        )

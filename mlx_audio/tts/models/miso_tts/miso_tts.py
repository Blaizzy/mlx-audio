from __future__ import annotations

import copy
import re
import time
from typing import Callable, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

from mlx_audio.utils import load_audio

from ..base import GenerationResult
from ..sesame import sesame as sesame_module
from ..sesame.sesame import Segment
from ..sesame.sesame import SesameModel as SesameCoreModel
from ..sesame.sesame import load_llama3_tokenizer

MISO_TTS_WATERMARK = [0, 0, 0, 0, 0]

LLAMA3_ROPE_SCALING = {
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}

MISO_BACKBONE_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.1,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14_336,
    "max_position_embeddings": 2048,
    "mlp_bias": False,
    "num_attention_heads": 32,
    "num_codebooks": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-5,
    "rope_scaling": LLAMA3_ROPE_SCALING,
    "rope_theta": 500_000,
    "tie_codebooks_embeddings": False,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 128_256,
}

MISO_DECODER_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.1,
    "backbone_hidden_size": 4096,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 1536,
    "initializer_range": 0.02,
    "intermediate_size": 6912,
    "max_position_embeddings": 2048,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 24,
    "num_codebooks": 32,
    "num_hidden_layers": 8,
    "num_key_value_heads": 6,
    "rms_norm_eps": 1e-5,
    "rope_scaling": LLAMA3_ROPE_SCALING,
    "rope_theta": 500_000,
    "use_cache": True,
    "vocab_size": 128_256,
}

DEFAULT_CONFIG = {
    "model_type": "miso_tts",
    "backbone_flavor": "llama-8B",
    "decoder_flavor": "llama-300M",
    "text_tokenizer": "meta-llama/Llama-3.2-1B",
    "text_tokenizer_apply_template": True,
    "text_vocab_size": 128_256,
    "audio_vocab_size": 2051,
    "audio_num_codebooks": 32,
    "watermark_key": MISO_TTS_WATERMARK,
    "depth_decoder_config": MISO_DECODER_CONFIG,
    **MISO_BACKBONE_CONFIG,
}

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class Model(nn.Module):
    """MLX implementation of Miso TTS 8B.

    Miso TTS uses the same RVQ-transformer structure as Sesame CSM, but with an
    8B Llama 3.2-style backbone, a 300M depth decoder, no built-in speaker
    prompt, and a slightly different speaker/text prompt format.
    """

    def __init__(self, config: dict):
        merged_config = copy.deepcopy(DEFAULT_CONFIG)
        merged_config.update(config)
        nn.Module.__init__(self)
        self.model = SesameCoreModel(merged_config)
        self._frame_size = merged_config["audio_num_codebooks"] + 1
        self._sample_rate = merged_config.get("sample_rate", 24_000)
        self._watermark_key = merged_config.get("watermark_key", MISO_TTS_WATERMARK)
        self.tokenizer_repo = merged_config.get("text_tokenizer")
        self.config = merged_config
        self.model_type = "miso_tts"
        self._text_tokenizer = None
        self._audio_tokenizer = None
        self._streaming_decoder = None
        self._watermarker = None
        self._runtime_initialized = False

    def _ensure_runtime(self):
        if self._runtime_initialized:
            return

        self.model.setup_caches(1)
        if self.tokenizer_repo:
            if self.config.get("text_tokenizer_apply_template", True):
                self._text_tokenizer = load_llama3_tokenizer(self.tokenizer_repo)
            else:
                self._text_tokenizer = sesame_module.AutoTokenizer.from_pretrained(
                    self.tokenizer_repo
                )
        else:
            self._text_tokenizer = load_llama3_tokenizer(sesame_module.TOKENIZER_REPO)

        mimi = sesame_module.Mimi.from_pretrained(sesame_module.MIMI_REPO)
        mimi.eval()
        self._audio_tokenizer = mimi
        self._streaming_decoder = sesame_module.MimiStreamingDecoder(mimi)
        self._sample_rate = int(mimi.cfg.sample_rate)

        load_watermarker = getattr(sesame_module, "load_watermarker", None)
        try:
            self._watermarker = load_watermarker() if load_watermarker else None
        except Exception:
            self._watermarker = None
        self._runtime_initialized = True

    def model_quant_predicate(self, p, m):
        return not p.startswith("_audio_tokenizer")

    @property
    def layers(self):
        return self.model.backbone.layers

    @property
    def sample_rate(self):
        return self._sample_rate

    def _tokenize_text_segment(
        self, text: str, speaker: int
    ) -> tuple[mx.array, mx.array]:
        self._ensure_runtime()
        text_tokens = self._text_tokenizer.encode(
            f"[{speaker}] {text.lstrip()}", return_tensors="mlx"
        ).squeeze(0)

        text_frame = mx.zeros((len(text_tokens), self._frame_size)).astype(mx.int32)
        text_frame_mask = mx.zeros((len(text_tokens), self._frame_size)).astype(
            mx.bool_
        )
        text_frame[:, -1] = text_tokens
        text_frame_mask[:, -1] = True
        return text_frame, text_frame_mask

    def _tokenize_audio(
        self, audio: mx.array, add_eos: bool = True
    ) -> tuple[mx.array, mx.array]:
        self._ensure_runtime()
        audio_tokens = self._audio_tokenizer.encode(audio[None, None, ...])[0]

        if add_eos:
            eos_frame = mx.zeros((audio_tokens.shape[0], 1))
            audio_tokens = mx.concat([audio_tokens, eos_frame], axis=1)

        audio_frame = mx.zeros((audio_tokens.shape[1], self._frame_size)).astype(
            mx.int32
        )
        audio_frame_mask = mx.zeros((audio_tokens.shape[1], self._frame_size)).astype(
            mx.bool_
        )
        audio_frame[:, :-1] = audio_tokens.swapaxes(0, 1)
        audio_frame_mask[:, :-1] = True
        return audio_frame, audio_frame_mask

    def _tokenize_segment(
        self, segment: Segment, add_eos: bool = True
    ) -> tuple[mx.array, mx.array]:
        text_tokens, text_masks = self._tokenize_text_segment(
            segment.text, segment.speaker
        )
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio, add_eos=add_eos)

        return mx.concat([text_tokens, audio_tokens], axis=0), mx.concat(
            [text_masks, audio_masks], axis=0
        )

    def sanitize(self, weights):
        sanitized_weights = {}

        for key, value in weights.items():
            if not key.startswith("model."):
                key = "model." + key

            if "attn" in key and "self_attn" not in key:
                key = key.replace("attn", "self_attn")
                key = key.replace("output_proj", "o_proj")

            if "mlp" in key:
                key = key.replace("w1", "gate_proj")
                key = key.replace("w2", "down_proj")
                key = key.replace("w3", "up_proj")

            if "sa_norm" in key or "mlp_norm" in key:
                key = key.replace("sa_norm", "input_layernorm").replace(
                    "scale", "weight"
                )
                key = key.replace("mlp_norm", "post_attention_layernorm").replace(
                    "scale", "weight"
                )

            if "decoder.norm" in key or "backbone.norm" in key:
                key = key.replace("scale", "weight")

            sanitized_weights[key] = value

        return sanitized_weights

    def generate(
        self,
        text: List[str] | str,
        voice: Optional[str] = None,
        speaker: int = 0,
        context: Optional[List[Segment]] = None,
        split_pattern: Optional[str] = r"\n+",
        sampler: Optional[Callable[..., mx.array]] = None,
        max_audio_length_ms: float = 90_000,
        ref_audio: Optional[Union[str, mx.array, List[Union[str, mx.array]]]] = None,
        ref_text: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        streaming_interval: float = 0.5,
        temperature: float = 0.9,
        topk: Optional[int] = None,
        top_k: int = 50,
        verbose: bool = False,
        **kwargs,
    ):
        del voice, kwargs

        self._ensure_runtime()
        current_context = list(context or [])
        ref_audio_values = _as_list(ref_audio)
        ref_text_values = _as_list(ref_text)
        if ref_audio_values:
            if not ref_text_values:
                raise ValueError("ref_text is required when ref_audio is provided.")
            if len(ref_text_values) != len(ref_audio_values):
                raise ValueError(
                    "ref_audio and ref_text lists must have the same length."
                )
            current_context.extend(
                Segment(
                    speaker=speaker,
                    text=reference_text,
                    audio=load_audio(reference_audio, sample_rate=self.sample_rate),
                )
                for reference_audio, reference_text in zip(
                    ref_audio_values, ref_text_values
                )
            )

        sample_top_k = top_k if topk is None else topk
        sampler = sampler or make_sampler(temp=temperature, top_k=sample_top_k)
        max_audio_frames = int(max_audio_length_ms / 80)
        streaming_interval_tokens = max(1, int(streaming_interval * 12.5))

        if isinstance(text, str):
            prompts = re.split(split_pattern, text.strip()) if split_pattern else [text]
        else:
            prompts = text
        if split_pattern:
            prompts = [prompt for prompt in prompts if prompt]

        for segment_idx, prompt in enumerate(prompts):
            start_time = time.perf_counter()

            self.model.reset_caches()
            self._streaming_decoder.reset()

            tokens = []
            tokens_mask = []
            for segment in current_context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
                prompt, speaker
            )
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

            prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int32)
            prompt_tokens_mask = mx.concat(tokens_mask, axis=0).astype(mx.bool_)

            samples = []
            curr_tokens = mx.expand_dims(prompt_tokens, axis=0)
            curr_tokens_mask = mx.expand_dims(prompt_tokens_mask, axis=0)
            curr_pos = mx.expand_dims(
                mx.arange(0, prompt_tokens.shape[0]), axis=0
            ).astype(mx.int32)
            generated_frame_count = 0
            yielded_frame_count = 0

            max_context_len = 2048 - max_audio_frames
            if curr_tokens.shape[1] >= max_context_len:
                raise ValueError(
                    "Inputs too long, must be below "
                    f"max_seq_len - max_generation_len: {max_context_len}"
                )

            with tqdm(disable=not verbose) as pbar:
                for _ in range(max_audio_frames):
                    sample = self.model.generate_frame(
                        curr_tokens,
                        curr_tokens_mask,
                        curr_pos,
                        sampler,
                    )
                    if mx.all(sample == 0):
                        break

                    samples.append(sample)

                    curr_tokens = mx.expand_dims(
                        mx.concat(
                            [
                                sample,
                                mx.zeros((1, 1)).astype(mx.int32),
                            ],
                            axis=1,
                        ),
                        axis=1,
                    )
                    curr_tokens_mask = mx.expand_dims(
                        mx.concat(
                            [
                                mx.ones_like(sample).astype(mx.bool_),
                                mx.zeros((1, 1)).astype(mx.bool_),
                            ],
                            axis=1,
                        ),
                        axis=1,
                    )
                    curr_pos = curr_pos[:, -1:] + 1
                    generated_frame_count += 1
                    pbar.update()

                    if (
                        stream
                        and (generated_frame_count - yielded_frame_count)
                        >= streaming_interval_tokens
                    ):
                        yielded_frame_count = generated_frame_count
                        yield self.generate_result(
                            samples,
                            start_time,
                            stream=True,
                            segment_idx=segment_idx,
                        )
                        samples = []
                        start_time = time.perf_counter()

                if samples:
                    yield self.generate_result(
                        samples,
                        start_time,
                        stream=stream,
                        segment_idx=segment_idx,
                    )

                mx.clear_cache()

    def generate_result(
        self,
        samples,
        start_time: float,
        stream: bool = False,
        segment_idx: int = 0,
    ) -> GenerationResult:
        token_count = len(samples)
        transposed = mx.transpose(mx.stack(samples), axes=[1, 2, 0])

        tokens_per_batch = min(token_count, int(12.5 * 5))
        all_audio = []
        for i in range(0, transposed.shape[2], tokens_per_batch):
            batch_tokens = transposed[:, :, i : i + tokens_per_batch]
            audio = (
                self._streaming_decoder.decode_frames(batch_tokens)
                .squeeze(0)
                .squeeze(0)
            )
            all_audio.append(audio)
        audio = mx.concat(all_audio, axis=0)

        watermark = getattr(sesame_module, "watermark", None)
        if (
            self._watermarker is not None
            and self._watermark_key is not None
            and watermark is not None
        ):
            audio = watermark(
                self._watermarker,
                audio,
                self._sample_rate,
                self._watermark_key,
            )
            audio = mx.array(audio, dtype=mx.float32)

        mx.eval(audio)

        segment_time = time.perf_counter() - start_time
        samples_count = audio.shape[0] if audio is not None else 0
        assert samples_count > 0, "No audio generated"

        audio_duration_seconds = samples_count / self._sample_rate
        rtf = segment_time / audio_duration_seconds if audio_duration_seconds > 0 else 0

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = (
            f"{duration_hours:02d}:{duration_mins:02d}:"
            f"{duration_secs:02d}.{duration_ms:03d}"
        )

        return GenerationResult(
            audio=audio,
            samples=samples_count,
            sample_rate=self._sample_rate,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=round(rtf, 2),
            prompt={
                "tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / segment_time, 2) if segment_time > 0 else 0
                ),
            },
            audio_samples={
                "samples": samples_count,
                "samples-per-sec": (
                    round(samples_count / segment_time, 2) if segment_time > 0 else 0
                ),
            },
            processing_time_seconds=segment_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=stream,
        )

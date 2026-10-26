"""Native MLX implementation of Ming-3.0-asr-tiny."""

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.lm.models.bailing_moe_v3 import BailingMoeV3Model, sanitize_backbone
from mlx_audio.lm.sample_utils import make_sampler
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.utils import load_audio, resample_audio

from .audio import AudioEncoder, extract_features
from .config import ModelConfig

ASR_INSTRUCTION = (
    "Please recognize the language of this speech and transcribe it, "
    "placing a `\t` between the identified language and the transcribe."
)


def decode_transcript(text):
    text = text.strip()
    if "\t" in text:
        language, text = text.split("\t", 1)
        return text.strip(), language.strip()
    for language in ("Chinese", "English", "Canton", "Japanese", "Korean"):
        if text.startswith(language):
            return text[len(language) :].strip(), language
    return text, None


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = (
            ModelConfig.from_dict(config) if isinstance(config, dict) else config
        )
        self.model_type = self.config.model_type
        self.model = BailingMoeV3Model(self.config.text_config)
        self.model.audio = AudioEncoder(self.config.audio_config)
        width = self.config.audio_config.n_audio_state
        hidden = self.config.text_config.hidden_size
        self.linear_proj_audio = []
        for kernel, stride, padding in zip(
            self.config.audio_config.downsampler_kernel_size_list,
            self.config.audio_config.downsampler_stride_list,
            self.config.audio_config.downsampler_padding_list,
        ):
            self.linear_proj_audio.extend(
                [
                    nn.Conv1d(width, hidden, kernel, stride=stride, padding=padding),
                    nn.GELU(),
                ]
            )
            width = hidden
        # The checkpoint's transpose is unnecessary for MLX's channels-last Conv1d.
        self.linear_proj_audio.extend([nn.Identity(), nn.Linear(hidden, hidden)])
        self.lm_head = nn.Linear(hidden, self.config.text_config.vocab_size, bias=False)
        self.tokenizer = None

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self.model.make_cache()

    def __call__(self, inputs=None, cache=None, inputs_embeds=None):
        return self.lm_head(self.model(inputs, cache, inputs_embeds))

    def encode_audio(self, features, chunk_frames=None):
        config = self.config.audio_config
        chunk_frames = (
            config.whisper_chunk_size if chunk_frames is None else int(chunk_frames)
        )
        if not 0 < chunk_frames <= 2 * config.n_audio_ctx:
            raise ValueError(
                "Audio chunk length exceeds the encoder positional context"
            )
        if (
            features.ndim != 2
            or features.shape[-1] != config.n_mels
            or not features.shape[0]
        ):
            raise ValueError(
                "Audio features must have shape [positive frames, mel bins]"
            )
        parts = []
        dtype = self.model.audio.conv1.weight.dtype
        for start in range(0, features.shape[0], chunk_frames):
            chunk = features[None, start : start + chunk_frames].astype(dtype)
            encoded = self.model.audio(chunk)
            for layer in self.linear_proj_audio:
                encoded = layer(encoded)
            if encoded.shape[1] != config.embedding_length(chunk.shape[1]):
                raise ValueError(
                    "Audio encoder output violates the projector length contract"
                )
            if self.config.norm_query_embeds:
                norm = mx.sqrt(
                    mx.sum(encoded.astype(mx.float32) ** 2, axis=-1, keepdims=True)
                )
                encoded = (encoded / mx.maximum(norm, 1e-12)).astype(dtype)
            parts.append(encoded)
        return mx.concatenate(parts, axis=1)

    def prepare_audio(self, audio, sample_rate=16000):
        if isinstance(audio, (str, Path)):
            waveform = load_audio(str(audio), sample_rate=16000)
        else:
            waveform = np.asarray(audio, dtype=np.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=-1)
            if waveform.ndim != 1:
                raise ValueError(
                    "Audio arrays must have shape [samples] or [samples, channels]"
                )
            if sample_rate != 16000:
                waveform = resample_audio(waveform, sample_rate, 16000)
        return extract_features(mx.array(waveform), self.config.audio_config.n_mels)

    def inject_audio(self, ids, audio_embeds):
        ids = mx.array(ids, dtype=mx.int32)[None]
        positions = np.flatnonzero(np.asarray(ids[0]) == self.config.audio_token_id)
        if positions.size != audio_embeds.shape[1]:
            raise ValueError("Audio placeholder count does not match encoded features")
        embeds = self.model.word_embeddings(ids)
        embeds[0, mx.array(positions)] = audio_embeds[0].astype(embeds.dtype)
        return ids, embeds

    def decode_tokens(
        self, embeds, max_tokens=256, temperature=0.0, top_p=1.0, no_repeat_ngram_size=0
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if (
            embeds.shape[1] + max_tokens
            > self.config.text_config.max_position_embeddings
        ):
            raise ValueError("Prompt and generation exceed the model context budget")
        cache = self.make_cache()
        sampler = make_sampler(temperature, top_p=top_p)
        started = time.perf_counter()
        hidden = None
        for start in range(0, embeds.shape[1], 256):
            hidden = self.model(
                inputs_embeds=embeds[:, start : start + 256], cache=cache
            )
            mx.eval(hidden, [state.state for state in cache])
        prefill = time.perf_counter() - started
        tokens = []
        eos_ids = self.config.text_config.eos_token_id
        eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids]) | {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|role_end|>"),
        }
        for _ in range(max_tokens):
            logits = self.lm_head(hidden[:, -1]).astype(mx.float32)
            n = int(no_repeat_ngram_size)
            if n > 0 and len(tokens) >= n - 1:
                prefix = tuple(tokens[-(n - 1) :]) if n > 1 else ()
                banned = {
                    tokens[i + n - 1]
                    for i in range(len(tokens) - n + 1)
                    if tuple(tokens[i : i + n - 1]) == prefix
                }
                if banned:
                    logits[:, mx.array(sorted(banned))] = -float("inf")
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            token = int(sampler(logprobs)[0].item())
            if token in eos:
                break
            tokens.append(token)
            hidden = self.model(mx.array([[token]]), cache=cache)
        return tokens, prefill, time.perf_counter() - started

    def generate(
        self,
        audio,
        *,
        max_tokens=256,
        language=None,
        lang_code=None,
        prompt=None,
        hotwords=None,
        sample_rate=16000,
        temperature=0.0,
        top_p=1.0,
        audio_chunk_frames=None,
        no_repeat_ngram_size=10,
        **kwargs,
    ):
        if self.tokenizer is None:
            raise ValueError("Load the model with the STT loader before generating")
        features = self.prepare_audio(audio, sample_rate)
        audio_embeds = self.encode_audio(features, audio_chunk_frames)
        instruction = ASR_INSTRUCTION if prompt is None else prompt
        if (
            not isinstance(instruction, str)
            or not instruction.startswith(ASR_INSTRUCTION)
            or "\n" in instruction
            or "\r" in instruction
        ):
            raise ValueError(
                "Prompt must start with the ASR instruction and contain no newlines"
            )
        if hotwords:
            instruction += " Hotwords: " + ", ".join(hotwords) + "."
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        audio_start, audio_end = self.audio_tokens
        ids = encode(instruction + audio_start)
        ids += [self.config.audio_token_id] * audio_embeds.shape[1]
        ids += encode(audio_end)
        hint = language or lang_code
        if hint is not None:
            label = {
                "en": "English",
                "english": "English",
                "zh": "Chinese",
                "chinese": "Chinese",
            }.get(str(hint).lower())
            if label is None:
                raise ValueError("ASR language hints support en or zh")
            ids += encode(label + "\t")
        _, embeds = self.inject_audio(ids, audio_embeds)
        available = self.config.text_config.max_position_embeddings - len(ids)
        if available < 1:
            raise ValueError("Audio prompt exceeds the Ming ASR context budget")
        # The shared STT CLI defaults to 8192 new tokens; treat this as an upper
        # bound while reserving positions for the audio prompt.
        max_tokens = min(max_tokens, available)
        tokens, prefill, elapsed = self.decode_tokens(
            embeds, max_tokens, temperature, top_p, no_repeat_ngram_size
        )
        text, detected = decode_transcript(
            self.tokenizer.decode(tokens, skip_special_tokens=True)
        )
        return STTOutput(
            text=text,
            language=detected or hint,
            prompt_tokens=len(ids),
            generation_tokens=len(tokens),
            total_tokens=len(ids) + len(tokens),
            prompt_tps=len(ids) / max(prefill, 1e-6),
            generation_tps=len(tokens) / max(elapsed - prefill, 1e-6),
            total_time=elapsed,
        )

    def sanitize(self, weights):
        weights = sanitize_backbone(weights, self.config.text_config)
        # The released audio-only reference explicitly excludes these legacy vision tensors.
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith(("linear_proj.", "model.visual."))
        }
        shapes = {key: value.shape for key, value in tree_flatten(self.parameters())}
        for key in weights:
            if (
                (
                    key.startswith("model.audio.conv")
                    or key.startswith("linear_proj_audio.")
                )
                and key.endswith(".weight")
                and weights[key].ndim == 3
            ):
                # Original Torch Conv1d uses [out, in, kernel]; converted MLX uses [out, kernel, in].
                if weights[key].shape != shapes.get(key):
                    weights[key] = weights[key].transpose(0, 2, 1)
        return weights

    @staticmethod
    def model_quant_predicate(path, module):
        return not path.startswith("model.audio") and not path.endswith(".gate")

    @classmethod
    def post_load_hook(cls, model, model_path):
        from transformers import PreTrainedTokenizerFast

        model.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(model_path),
            trust_remote_code=False,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        for start, pad, end in (
            ("<audio>", "<audioPatch>", "</audio>"),
            ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"),
        ):
            if (
                model.tokenizer.convert_tokens_to_ids(pad)
                == model.config.audio_token_id
            ):
                model.audio_tokens = (start, end)
                break
        else:
            raise ValueError("Tokenizer audio token does not match the checkpoint")
        return model

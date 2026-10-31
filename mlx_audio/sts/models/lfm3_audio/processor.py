"""Native tokenization and speech placeholder expansion; no remote code."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import PreTrainedTokenizerFast

from .config import ModelConfig
from .features import SAMPLE_RATE, extract_features, num_encoder_tokens


def load_clip(audio, sample_rate=SAMPLE_RATE):
    if isinstance(audio, (str, Path)):
        from mlx_audio.utils import load_audio

        clip = np.asarray(load_audio(str(audio), sample_rate=SAMPLE_RATE))
    else:
        clip = np.asarray(audio, dtype=np.float32)
        if clip.ndim == 2:
            clip = clip.mean(axis=-1)
        if sample_rate != SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(sample_rate, SAMPLE_RATE)
            clip = resample_poly(clip, SAMPLE_RATE // divisor, sample_rate // divisor)
    if clip.ndim != 1 or not clip.size or not np.isfinite(clip).all():
        raise ValueError("Audio must be a nonempty mono waveform of finite samples")
    return clip.astype(np.float32)


class LFM3AudioProcessor:
    audio_token = "<|audio|>"
    audio_out_token = "<|audio_out|>"

    def __init__(self, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config or ModelConfig()

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            # This is LiquidAI's own tokenizer; keep its trained pre-tokenizer.
            model_path,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        template = Path(model_path) / "chat_template.jinja"
        if template.exists():
            tokenizer.chat_template = template.read_text()
        return cls(tokenizer, config)

    @staticmethod
    def _expand(text, token, counts):
        if text.count(token) != len(counts):
            raise ValueError(f"Expected {len(counts)} {token} placeholders")
        parts = text.split(token)
        return parts[0] + "".join(
            token * n + part for n, part in zip(counts, parts[1:])
        )

    def __call__(self, text, audio=None, audio_out_codes=None, sample_rate=SAMPLE_RATE):
        if not isinstance(text, str):
            raise ValueError("LFM3-Audio processes one prompt at a time")
        data = {}
        clips = [] if audio is None else audio
        if isinstance(clips, (str, Path, mx.array, np.ndarray)):
            clips = [clips]
        if len(clips):
            features, mask = extract_features(
                [load_clip(c, sample_rate) for c in clips]
            )
            data.update(
                input_features=mx.array(features), feature_attention_mask=mx.array(mask)
            )
            counts = num_encoder_tokens(mask.sum(-1)).tolist()
        else:
            counts = []
        text = self._expand(text, self.audio_token, counts)
        frames = []
        for codes in [] if audio_out_codes is None else audio_out_codes:
            codes = np.asarray(codes)
            if codes.ndim != 2 or codes.shape[-1] != self.config.num_codebooks:
                raise ValueError(
                    f"Speech history must have shape [frames, {self.config.num_codebooks}]"
                )
            if (
                not np.issubdtype(codes.dtype, np.integer)
                or np.any(codes < 0)
                or np.any(codes >= self.config.audio_eos_code)
            ):
                raise ValueError(
                    "Speech history codes must be integer codec IDs without EOS"
                )
            eos = np.full(
                (1, self.config.num_codebooks), self.config.audio_eos_code, np.int32
            )
            frames.append(np.concatenate([codes.astype(np.int32), eos]))
        text = self._expand(text, self.audio_out_token, [len(c) for c in frames])
        if frames:
            data["audio_out_codes"] = mx.array(np.concatenate(frames))
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Some tokenizer exports omit the placeholder special tokens. Refuse a
        # split placeholder rather than silently dropping the supplied features.
        for token, count in (
            (self.config.audio_token_id, sum(counts)),
            (self.config.audio_out_token_id, sum(map(len, frames))),
        ):
            if ids.count(token) != count:
                raise ValueError(
                    "Tokenizer speech placeholder IDs do not match the model configuration"
                )
        data["input_ids"] = mx.array([ids], mx.int32)
        return data

    def apply_chat_template(self, messages, *, add_generation_prompt=True, tools=None):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **({"tools": tools} if tools is not None else {}),
        )

    def prepare(self, messages, *, sample_rate=SAMPLE_RATE, tools=None):
        clips, history = [], []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                continue
            for block in content:
                if block["type"] == "audio":
                    clips.append(block.get("audio", block.get("path")))
                elif block["type"] == "audio_out":
                    if "codes" not in block:
                        raise ValueError(
                            "Earlier speech requires audio_out codes; waveform history is unsupported"
                        )
                    history.append(block["codes"])
        prompt = self.apply_chat_template(messages, tools=tools)
        return self(prompt, clips, history, sample_rate)

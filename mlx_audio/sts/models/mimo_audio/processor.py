"""Explicit text/audio segment packing and offline MiMo task prompts."""

import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import ModelConfig


class MiMoAudioProcessor:
    def __init__(self, tokenizer, config: ModelConfig, audio_tokenizer=None):
        self.tokenizer = tokenizer
        self.config = config
        self._audio_tokenizer = audio_tokenizer
        self.audio_tokenizer_path = config.audio_tokenizer_path
        self.audio_tokenizer_revision = config.audio_tokenizer_revision
        self.ids = {}
        for name in ("empty", "sosp", "eosp", "sostm", "eostm", "eot", "im_end"):
            token = f"<|{name}|>"
            value = tokenizer.convert_tokens_to_ids(token)
            if value is None or tokenizer.encode(token, add_special_tokens=False) != [
                value
            ]:
                raise ValueError(f"Tokenizer is missing MiMo's {token} token")
            self.ids[name] = value
        if self.ids["empty"] != config.empty_token_id:
            raise ValueError("Tokenizer's <|empty|> ID does not match the model")

    @property
    def audio_tokenizer(self):
        if self._audio_tokenizer is None:
            from mlx_audio.codec.models.mimo_audio_tokenizer import MiMoAudioTokenizer

            self._audio_tokenizer = MiMoAudioTokenizer.from_pretrained(
                self.audio_tokenizer_path, revision=self.audio_tokenizer_revision
            )
        return self._audio_tokenizer

    def text(self, text=None, *, ids=None):
        if ids is None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = np.asarray(ids, dtype=np.int32)
        group = self.config.group_size
        result = np.empty(
            (self.config.audio_channels + 1, len(ids) * group), dtype=np.int32
        )
        result[0] = -100
        result[0, ::group] = ids
        result[1:] = np.array(self.config.speech_empty_ids)[:, None]
        return result

    def audio_codes(self, codes, *, boundaries=True):
        codes = np.asarray(codes)
        channels = self.config.audio_channels
        if codes.ndim != 2 or codes.shape[0] < channels or codes.shape[1] == 0:
            raise ValueError("Audio codes must have shape (codebooks, nonempty frames)")
        codes = codes[:channels]
        limits = np.array(self.config.speech_empty_ids)[:, None]
        if not np.issubdtype(codes.dtype, np.integer) or np.any(
            (codes < 0) | (codes >= limits)
        ):
            raise ValueError("Invalid audio codes or speech padding in codec input")
        group = self.config.group_size
        padding = -codes.shape[1] % group
        if padding:
            codes = np.concatenate(
                [codes, np.repeat(codes[:, -1:], padding, axis=1)], axis=1
            )
        output = self.text(ids=[self.ids["empty"]] * (codes.shape[1] // group))
        output[1:] = codes
        if boundaries:
            output = np.concatenate(
                [self.text("<|sosp|>"), output, self.text("<|eosp|>")], axis=1
            )
        return output

    def encode_audio(self, audio, sample_rate=None):
        # A codes dictionary avoids waveform reconstruction when reusing history.
        if isinstance(audio, dict):
            if "codes" in audio:
                return audio["codes"]
            sample_rate = audio.get("sample_rate", sample_rate)
            audio = audio["audio"]
        return self.audio_tokenizer.encode(
            audio, sample_rate=sample_rate, num_quantizers=self.config.audio_channels
        )

    def audio(self, audio, sample_rate=None, *, boundaries=True):
        return self.audio_codes(
            self.encode_audio(audio, sample_rate), boundaries=boundaries
        )

    def interleaved(self, text, audio, sample_rate=None):
        """Reference history format: five text tokens, then five audio patches."""
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        text_chunks = [text_ids[i : i + 5] for i in range(0, len(text_ids), 5)] or [[]]
        text_chunks[-1] = text_chunks[-1] + [self.ids["eot"]]
        packed_audio = self.audio(audio, sample_rate, boundaries=False)
        step = 5 * self.config.group_size
        audio_chunks = [
            packed_audio[:, i : i + step] for i in range(0, packed_audio.shape[1], step)
        ]
        parts = [self.text("<|sostm|>")]
        for i in range(max(len(text_chunks), len(audio_chunks))):
            if i < len(text_chunks):
                parts.append(self.text(ids=text_chunks[i]))
            if i < len(audio_chunks):
                parts.append(audio_chunks[i])
        return np.concatenate(parts + [self.text("<|eostm|>")], axis=1)

    def segments(self, segments):
        """Pack explicit {'text': ...}, {'audio': ...}, or {'codes': ...} segments."""
        parts = []
        for segment in segments:
            if isinstance(segment, str):
                parts.append(self.text(segment))
            elif "codes" in segment:
                parts.append(
                    self.audio_codes(
                        segment["codes"], boundaries=segment.get("boundaries", True)
                    )
                )
            elif "audio" in segment:
                parts.append(
                    self.audio(
                        segment["audio"],
                        segment.get("sample_rate"),
                        boundaries=segment.get("boundaries", True),
                    )
                )
            elif "text" in segment:
                parts.append(self.text(segment["text"]))
            else:
                raise ValueError("A segment must contain text, audio, or codes")
        if not parts:
            raise ValueError("At least one prompt segment is required")
        return mx.array(np.concatenate(parts, axis=1))[None]

    @staticmethod
    def _normalize(text):
        return text.capitalize() if text.isupper() or text.islower() else text

    def tts(
        self,
        text,
        *,
        instruct=None,
        read_text_only=True,
        ref_audio=None,
        ref_sample_rate=None,
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("TTS requires nonempty text")
        template = (
            "请将这段文字转换为语音"
            if re.search(r"[\u4e00-\u9fff]", text)
            else "Please convert this text to speech"
        )
        text = self._normalize(text)
        plain = read_text_only and instruct is None and ref_audio is None
        parts = []
        if not plain:
            parts.append(self.text("<|im_start|>system\n"))
            if ref_audio is not None:
                parts.append(
                    self.text(
                        "你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是："
                    )
                )
                parts.append(self.audio(ref_audio, ref_sample_rate))
            else:
                parts.append(
                    self.text("你需要根据指定的风格指令和文本内容来生成语音。")
                )
            parts.append(self.text("<|im_end|>\n"))
        content = f"{template}: {text}" if read_text_only else text
        if read_text_only and instruct is not None:
            content += f"({instruct})"
        parts.append(self.text(f"<|im_start|>user\n{content}<|im_end|>\n"))
        parts.append(
            self.text(
                "<|im_start|>assistant\n" + ("<|sostm|>" if plain else "<think>\n")
            )
        )
        return mx.array(np.concatenate(parts, axis=1))[None]

    def understanding(self, audio, text, *, sample_rate=None, thinking=False):
        return self.segments(
            [
                "<|im_start|>user\n",
                {"audio": audio, "sample_rate": sample_rate},
                text,
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
                "<think>\n" if thinking else "<think>\n\n</think>\n",
            ]
        )

    def dialogue(
        self,
        messages,
        *,
        speech_input=False,
        speech_output=False,
        thinking=False,
        system_prompt=None,
        ref_audio=None,
        ref_sample_rate=None,
    ):
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Dialogue history must end with a user message")
        parts = []
        if ref_audio is not None:
            parts.extend(
                [
                    self.text("<|im_start|>system\nYour voice should be:"),
                    self.audio(ref_audio, ref_sample_rate),
                    self.text("<|im_end|>\n"),
                ]
            )
        for i, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role not in {"user", "assistant", "system"}:
                raise ValueError(f"Unsupported role: {role}")
            parts.append(self.text(f"<|im_start|>{role}\n"))
            if system_prompt and role == "user" and i == 0:
                parts.extend([self.text(system_prompt), self.text("\n\n")])
            if role == "assistant" and isinstance(content, dict):
                audio = (
                    {"codes": content["codes"]}
                    if "codes" in content
                    else content["audio"]
                )
                parts.append(
                    self.interleaved(content["text"], audio, content.get("sample_rate"))
                )
            elif role == "user" and speech_input:
                parts.append(self.audio(content, message.get("sample_rate")))
            elif isinstance(content, list):
                parts.append(np.asarray(self.segments(content)[0]))
            else:
                parts.append(self.text(content))
            parts.append(self.text("<|im_end|>\n"))
        suffix = (
            "<|sostm|>"
            if speech_output
            else ("<think>\n" if thinking else "<think>\n\n</think>\n")
        )
        parts.append(self.text("<|im_start|>assistant\n" + suffix))
        return mx.array(np.concatenate(parts, axis=1))[None]

    def few_shot(self, instruction, examples, audio, sample_rate=None):
        parts = [self.text(f"[Int]:{instruction}\n")]
        for example in examples:
            parts.extend(
                [
                    self.audio(example["input_audio"], example.get("sample_rate")),
                    self.text("\n"),
                    self.interleaved(
                        example["output_transcription"],
                        example["output_audio"],
                        example.get("sample_rate"),
                    ),
                    self.text(" \n\n"),
                ]
            )
        parts.extend(
            [self.audio(audio, sample_rate), self.text("\n"), self.text("<|sostm|>")]
        )
        return mx.array(np.concatenate(parts, axis=1))[None]

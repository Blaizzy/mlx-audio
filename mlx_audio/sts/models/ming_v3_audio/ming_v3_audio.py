"""Ming Audio Tiny: audio understanding and optional spoken answers."""

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import ASR_INSTRUCTION
from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import Model as AudioModel
from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import decode_transcript

from .config import ModelConfig


@dataclass
class GenerationResult:
    text: str
    audio: mx.array | None = None
    sample_rate: int = 44100
    reasoning: str | None = None
    raw_text: str = ""
    prompt_tokens: int = 0
    generation_tokens: int = 0
    processing_time: float = 0.0
    finish_reason: str = "stop"


class Model(AudioModel):
    def __init__(self, config):
        config = ModelConfig.from_dict(config) if isinstance(config, dict) else config
        super().__init__(config)
        self.model_type = "ming_v3_audio"
        self._talker = None

    def prepare_messages(self, messages, *, thinking=False, sample_rate=16000):
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Ming conversations must end with a user message")
        clips = []
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(
                    "Ming message roles must be system, user, or assistant"
                )
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "audio":
                        source = item.get("audio")
                        rate = sample_rate
                        if isinstance(source, tuple) and len(source) == 2:
                            source, rate = source
                        features = self.prepare_audio(source, rate)
                        clips.append(self.encode_audio(features))
                    elif item.get("type") != "text":
                        raise ValueError("Ming supports audio and text content only")
            elif not isinstance(content, str):
                raise ValueError(
                    "Message content must be text or a list of audio/text items"
                )
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        if rendered.count(placeholder) != len(clips):
            raise ValueError("Audio chat placeholders do not match the supplied clips")
        pieces = rendered.split(placeholder)
        rendered = pieces[0]
        for clip, suffix in zip(clips, pieces[1:]):
            expanded = (
                "<|audio_start|>" + "<|audio_pad|>" * clip.shape[1] + "<|audio_end|>"
            )
            rendered += expanded + suffix
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if clips:
            _, embeds = self.inject_audio(ids, mx.concatenate(clips, axis=1))
        else:
            embeds = self.model.word_embeddings(mx.array([ids], dtype=mx.int32))
        return ids, embeds

    def _get_talker(self):
        if self._talker is None:
            from .talker import Talker

            path = self.config.talker_path or str(
                Path(self.config.model_path) / "talker"
            )
            self._talker = Talker.from_pretrained(path)
        return self._talker

    def generate(
        self,
        audio=None,
        *,
        text=None,
        task="audio_understanding",
        messages=None,
        system_prompt=None,
        thinking=False,
        max_new_tokens=512,
        max_tokens=None,
        temperature=0.0,
        top_p=1.0,
        sample_rate=16000,
        voice="US_James",
        ref_audio=None,
        ref_text=None,
        spk_emb=None,
        max_talker_steps=1000,
        no_repeat_ngram_size=0,
        seed=None,
        stream=False,
        **kwargs,
    ):
        if stream:
            raise ValueError("Ming Audio currently supports offline generation only")
        if task not in {
            "asr",
            "audio_understanding",
            "speech_to_text",
            "spoken_dialogue",
            "text",
            "tts",
        }:
            raise ValueError(f"Unsupported Ming Audio task: {task}")
        if self.tokenizer is None:
            raise ValueError("Load the model with the STS loader before generating")
        if seed is not None:
            mx.random.seed(int(seed))
        started = time.perf_counter()
        limit = max_tokens if max_tokens is not None else max_new_tokens
        ids, tokens = [], []
        reasoning = None
        if task == "tts":
            if (
                messages is not None
                or audio is not None
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(
                    "TTS requires nonempty text without input audio or messages"
                )
            answer = raw = text
        else:
            if messages is not None and (audio is not None or text is not None):
                raise ValueError("Pass either messages or audio/text, not both")
            if messages is None:
                if audio is None and (not isinstance(text, str) or not text.strip()):
                    raise ValueError("Provide input audio, text, or messages")
                if task == "asr" and audio is None:
                    raise ValueError("ASR requires audio")
                content = []
                if task == "asr":
                    content = [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": text or ASR_INSTRUCTION},
                    ]
                else:
                    if text:
                        content.append({"type": "text", "text": text})
                    if audio is not None:
                        content.append({"type": "audio", "audio": audio})
                messages = [{"role": "user", "content": content}]
            if system_prompt:
                if messages[0].get("role") == "system":
                    raise ValueError(
                        "system_prompt conflicts with the messages system turn"
                    )
                messages = [{"role": "system", "content": system_prompt}, *messages]
            ids, embeds = self.prepare_messages(
                messages, thinking=thinking, sample_rate=sample_rate
            )
            tokens, _, _ = self.decode_tokens(
                embeds, limit, temperature, top_p, no_repeat_ngram_size
            )
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False).strip()
            answer = raw
            if thinking or raw.startswith("<think>"):
                reasoning, separator, answer = raw.removeprefix("<think>").partition(
                    "</think>"
                )
                reasoning = reasoning.strip()
                answer = answer.strip() if separator else ""
            if task == "asr":
                answer, _ = decode_transcript(answer)
        speech = None
        rate = 44100
        if task in {"spoken_dialogue", "tts"} and answer:
            generated = next(
                self._get_talker().generate(
                    answer,
                    voice=voice,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    spk_emb=spk_emb,
                    max_tokens=max_talker_steps,
                )
            )
            speech, rate = generated.audio, generated.sample_rate
            if not bool(mx.all(mx.isfinite(speech))):
                raise FloatingPointError("Ming Audio produced non-finite speech")
        return GenerationResult(
            text=answer,
            audio=speech,
            sample_rate=rate,
            reasoning=reasoning,
            raw_text=raw,
            prompt_tokens=len(ids),
            generation_tokens=len(tokens),
            processing_time=time.perf_counter() - started,
            finish_reason="length" if len(tokens) == limit else "stop",
        )

    @classmethod
    def post_load_hook(cls, model, model_path):
        model.config.model_path = str(model_path)
        return super().post_load_hook(model, model_path)

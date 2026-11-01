# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted from the mlx-vlm LFM3-Audio port and LiquidAI reference implementation.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.lfm3 import Model as LanguageModel
from mlx_audio.sts.models.base import STSOutput

from .audio import AudioEncoder, AudioProjector
from .codec import load_codec
from .depth import DepthDecoder, embed_frames, normalize_eos, sample, sample_frame
from .processor import LFM3AudioProcessor

TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"

_PREFIXES = (
    ("model.language_model.", "language_model.model."),
    ("model.audio_encoder.", "audio_tower."),
    ("model.audio_projector.", "audio_projector."),
    ("model.depth_decoder.", "depth_decoder."),
    ("model.codebook_embedding.", "codebook_embedding."),
    ("lfm.model.", "language_model.model."),
    ("encoder.downsample.", "audio_tower.downsample."),
    ("encoder.lfm.lfm.", "audio_tower.lfm."),
    ("mlp.model.0.", "audio_projector.layer_norm."),
    ("mlp.model.1.", "audio_projector.linear_1."),
    ("mlp.model.3.", "audio_projector.linear_2."),
    ("depthformer.model.", "depth_decoder.model."),
    ("depth_linear.", "depth_decoder.depth_linear."),
    ("mask_embedding", "depth_decoder.mask_embedding"),
    ("depth_embeddings.", "depth_decoder.depth_embeddings."),
)


@dataclass
class GenerationResult(STSOutput):
    audio_tokens: mx.array | None = None
    token_ids: list[int] = field(default_factory=list)
    generation_tokens: int = 0
    finish_reason: str = "stop"
    generation_time: float = 0.0
    reasoning: str | None = None


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.audio_tower = AudioEncoder(config.audio_encoder_config)
        self.audio_projector = AudioProjector(config)
        self.codebook_embedding = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.text_config.hidden_size
        )
        self.depth_decoder = DepthDecoder(
            config.depth_config,
            config.text_config.hidden_size,
            config.num_codebooks,
            config.codebook_size,
        )
        self._processor = None
        self._codec = None

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        from mlx_audio.sts.utils import load

        return load(model_path, **kwargs)

    @staticmethod
    def post_load_hook(model, model_path):
        model.config.model_path = str(model_path)
        object.__setattr__(
            model,
            "_processor",
            LFM3AudioProcessor.from_pretrained(model_path, model.config),
        )
        return model

    @property
    def processor(self):
        if self._processor is None:
            if self.config.model_path is None:
                raise ValueError("Load a checkpoint before preparing generation inputs")
            object.__setattr__(
                self,
                "_processor",
                LFM3AudioProcessor.from_pretrained(self.config.model_path, self.config),
            )
        return self._processor

    def codec(self):
        if self._codec is None:
            dtype = self.language_model.model.embedding_norm.weight.dtype
            object.__setattr__(
                self, "_codec", load_codec(self.config.model_path, dtype=dtype)
            )
        return self._codec

    def model_quant_predicate(self, path, module):
        return path.startswith("language_model.")

    def sanitize(self, weights):
        result = {}
        for key, value in weights.items():
            if ".to_logits." in key or key in {
                "audio_loss_weights",
                "codebook_offsets",
            }:
                continue
            if key in {"lm_head.weight", "lfm.lm_head.weight"}:
                if self.config.text_config.tie_word_embeddings:
                    continue
                key = "language_model.lm_head.weight"
            for old, new in _PREFIXES:
                if key.startswith(old):
                    key = new + key[len(old) :]
                    break
            if key.startswith("audio_tower.downsample."):
                if value.ndim == 4 and value.shape[2] == value.shape[3]:
                    value = value.transpose(0, 2, 3, 1)
                elif key.endswith("lin_out.weight") and value.ndim == 3:
                    value = value[..., 0]
            elif key.endswith("conv.conv.weight") and value.shape[1] == 1:
                value = value.transpose(0, 2, 1)
            result[key] = value
        return result

    def placeholder_token_ids(self):
        return [self.config.audio_token_id, self.config.audio_out_token_id]

    def get_input_embeddings(
        self,
        input_ids,
        input_features=None,
        feature_attention_mask=None,
        audio_out_codes=None,
    ):
        embeddings = self.language_model.model.embed_tokens(input_ids)
        config = self.config
        features = None
        if input_features is not None:
            input_features = mx.array(input_features)
            if feature_attention_mask is not None and feature_attention_mask.shape != (
                input_features.shape[0],
                input_features.shape[-1],
            ):
                raise ValueError(
                    "feature_attention_mask must match the feature batch and frame dimensions"
                )
            lengths = (
                mx.full((input_features.shape[0],), input_features.shape[-1])
                if feature_attention_mask is None
                else mx.array(feature_attention_mask).sum(-1)
            )
            features = self.audio_projector(self.audio_tower(input_features, lengths))
        frames = None
        if audio_out_codes is not None:
            codes = mx.array(audio_out_codes)
            if (
                codes.ndim != 2
                or codes.shape[-1] != config.num_codebooks
                or not mx.issubdtype(codes.dtype, mx.integer)
            ):
                raise ValueError(
                    "audio_out_codes must be integer [frames, num_codebooks]"
                )
            if codes.size and (codes.min() < 0 or codes.max() >= config.codebook_size):
                raise ValueError("audio_out_codes contain an invalid codebook ID")
            frames = embed_frames(self.codebook_embedding, codes)
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        ids = np.asarray(input_ids).reshape(-1)
        for token, values in (
            (config.audio_token_id, features),
            (config.audio_out_token_id, frames),
        ):
            indices = np.flatnonzero(ids == token)
            if len(indices) != (0 if values is None else values.shape[0]):
                raise ValueError(
                    f"Speech placeholders {token} do not match supplied frames"
                )
            if len(indices):
                flat[mx.array(indices)] = values.astype(embeddings.dtype)
        return flat.reshape(embeddings.shape)

    def __call__(self, input_ids, cache=None, **kwargs):
        embeddings = self.get_input_embeddings(input_ids, **kwargs)
        logits = self.language_model(cache=cache, input_embeddings=embeddings)
        logits[..., self.placeholder_token_ids()] = -mx.inf
        return logits

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.layers

    def generate(
        self,
        audio=None,
        *,
        text=None,
        task="sts",
        messages=None,
        system_prompt=None,
        language="English",
        voice="female",
        sample_rate=16000,
        thinking=False,
        max_new_tokens=2048,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        audio_temperature=1.0,
        audio_top_k=128,
        audio_repetition_penalty=1.1,
        audio_repetition_window=4096,
        text_repetition_window=32,
        generation_mode=None,
        seed=None,
        tools=None,
    ):
        """Generate a transcript, speech, or interleaved speech and text.

        Tasks: ``asr``, ``tts``, ``sts`` (voice chat), and ``chat`` (text-only
        audio questions). Audio arrays use ``sample_rate``; files are resampled.
        ``messages`` accepts ChatML messages, including typed ``audio`` blocks
        and previous ``audio_out`` blocks containing [frames, 12] codec IDs.
        """
        aliases = {
            "transcribe": "asr",
            "speech-to-text": "asr",
            "text-to-speech": "tts",
            "speech-to-speech": "sts",
        }
        task = aliases.get(task, task)
        if task not in {"asr", "tts", "sts", "chat"}:
            raise ValueError(f"Unknown LFM3-Audio task {task!r}")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if seed is not None:
            mx.random.seed(seed)
        if messages is None:
            if task == "asr" and audio is None:
                raise ValueError("ASR requires input audio")
            if task == "tts" and not text:
                raise ValueError("TTS requires text")
            if audio is None and not text:
                raise ValueError("Provide audio, text, or messages")
            prompts = {
                "asr": f"Perform ASR in {language}." if language else "Perform ASR.",
                "tts": f"Perform TTS in {language} using a {voice} voice.",
                "sts": "Respond with interleaved text and audio.",
                "chat": "Answer questions about the provided audio clip. Base every answer only on what can be heard.",
            }
            content = []
            if audio is not None:
                content.append({"type": "audio", "audio": audio})
            if text:
                content.append({"type": "text", "text": text})
            messages = [
                {"role": "system", "content": system_prompt or prompts[task]},
                {"role": "user", "content": content},
            ]
        else:
            if audio is not None or text is not None:
                raise ValueError(
                    "Put audio and text in messages when messages are provided"
                )
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}] + [
                    m for m in messages if m["role"] != "system"
                ]
        inputs = self.processor.prepare(messages, sample_rate=sample_rate, tools=tools)
        if thinking:
            if task != "chat":
                raise ValueError("thinking is supported for text-only chat")
            inputs["input_ids"] = mx.concatenate(
                [
                    inputs["input_ids"],
                    mx.array(
                        [
                            self.processor.tokenizer.encode(
                                "<think>", add_special_tokens=False
                            )
                        ]
                    ),
                ],
                axis=1,
            )
        if generation_mode is None:
            generation_mode = (
                TEXT if task in {"asr", "chat"} else AUDIO if task == "tts" else None
            )
        codec = self.codec() if generation_mode != TEXT else None
        tokens, frames = [], []
        started = time.perf_counter()
        for token, codes in self.stream_slots(
            **inputs,
            max_tokens=max_new_tokens,
            generation_mode=generation_mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window=audio_repetition_window,
            text_repetition_window=text_repetition_window,
        ):
            tokens.append(token)
            if codes is not None and codes[0].item() != self.config.audio_eos_code:
                frames.append(codes)
        codes = (
            mx.stack(frames, axis=-1)
            if frames
            else mx.zeros((self.config.num_codebooks, 0), mx.int32)
        )
        waveform = codec.decode(codes) if codec is not None and frames else None
        if waveform is not None:
            mx.eval(waveform)
        stops = np.atleast_1d(self.config.eos_token_id).tolist() + [
            self.config.text_end_token_id
        ]
        text = self.processor.tokenizer.decode(
            [t for t in tokens if t != self.config.audio_out_token_id],
            skip_special_tokens=True,
        )
        reasoning = None
        if thinking and "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            reasoning = reasoning.removeprefix("<think>").strip()
            text = text.strip()
        return GenerationResult(
            audio=waveform,
            sample_rate=self.sample_rate,
            text=text,
            audio_tokens=codes,
            token_ids=tokens,
            generation_tokens=len(tokens),
            finish_reason="stop" if tokens and tokens[-1] in stops else "length",
            generation_time=time.perf_counter() - started,
            reasoning=reasoning,
        )

    def stream_slots(
        self,
        input_ids,
        max_tokens=2048,
        generation_mode=None,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        repetition_penalty=None,
        audio_temperature=None,
        audio_top_k=None,
        audio_repetition_penalty=None,
        audio_repetition_window=64,
        text_repetition_window=32,
        suppress_mixed=False,
        suppress_audio_out=False,
        frame_order=None,
        **kwargs,
    ):
        """Yield one ``(token, codes)`` per generated slot, as the reference ``generate``.

        A text slot yields its token and ``None``; a speech slot yields
        ``audio_out_token_id`` and its 12 codes, an end-of-speech frame as all
        ``audio_eos_code``. ``generation_mode`` None starts from the prompt's last
        token; "audio" or "interleaved" appends the marker the mode needs.
        ``frame_order`` [frames, 12] fixes the
        MaskGIT reveal order, which is otherwise random even when greedy. Other
        keyword arguments (``input_features``, ``audio_out_codes``, ...) go to
        ``get_input_embeddings``. Batch size 1.
        """
        config = self.config
        ids = np.asarray(input_ids)
        if (
            ids.ndim not in (1, 2)
            or not ids.size
            or (ids.ndim == 2 and ids.shape[0] != 1)
        ):
            raise ValueError("LFM3-Audio generates one prompt at a time")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if min(audio_repetition_window, text_repetition_window) < 0:
            raise ValueError("Repetition windows must be nonnegative")
        for value in (temperature, audio_temperature):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Sampling temperatures must be finite and nonnegative")
        for value in (repetition_penalty, audio_repetition_penalty):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Repetition penalties must be finite and positive")
        ids = ids.reshape(-1).tolist()
        eos_code, frame_token = config.audio_eos_code, config.audio_out_token_id
        markers = {
            AUDIO: (config.audio_start_token_id, frame_token),
            INTERLEAVED: (config.mixed_start_token_id,),
        }
        mode = generation_mode
        if mode is None:
            mode = (
                AUDIO
                if ids[-1] in markers[AUDIO]
                else INTERLEAVED if ids[-1] == config.mixed_start_token_id else TEXT
            )
            # a replayed end-of-speech frame resumes the text
            codes = kwargs.get("audio_out_codes")
            if ids[-1] == frame_token and codes is not None:
                if (np.asarray(codes)[-1] == eos_code).any():
                    mode = TEXT
        elif mode not in (TEXT, AUDIO, INTERLEAVED):
            raise ValueError(f"Unknown generation_mode {mode!r}")
        elif mode in markers and ids[-1] not in markers[mode]:
            ids.append(markers[mode][0])
        eos_ids = config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
        suppress = self.placeholder_token_ids()
        # an explicit text mode keeps the reply in text, as the reference does
        if suppress_mixed or generation_mode == TEXT:
            suppress.append(config.mixed_start_token_id)
        if suppress_audio_out or generation_mode == TEXT:
            suppress.append(config.audio_start_token_id)

        backbone = self.language_model.model
        input_ids = mx.array([ids])
        cache = self.make_cache()
        embeds = self.get_input_embeddings(input_ids, **kwargs)
        hidden = backbone(cache=cache, input_embeddings=embeds)[:, -1]
        text_history = deque(maxlen=text_repetition_window)
        audio_history = deque(maxlen=audio_repetition_window)
        slots_left, frame = config.interleaved_n_text, 0
        speaking = text_done = False
        for step in range(max_tokens):
            interleaving, finished = mode == INTERLEAVED, False
            if mode == AUDIO or (interleaving and speaking):
                order = None
                if frame_order is not None:
                    order = mx.array(frame_order[frame]).reshape(1, -1)
                frame += 1
                history = mx.stack(list(audio_history))[None] if audio_history else None
                codes = sample_frame(
                    self.depth_decoder,
                    hidden,
                    config.maskgit_schedule,
                    audio_temperature,
                    audio_top_k,
                    audio_repetition_penalty,
                    history,
                    order,
                )[0]
                codes = normalize_eos(codes, eos_code)
                if codes[0].item() == eos_code:
                    mode = TEXT
                else:
                    audio_history.append(codes)
                embed = embed_frames(self.codebook_embedding, codes)
                yield frame_token, codes
            else:
                history = mx.array([list(text_history)]) if text_history else None
                token = sample(
                    (
                        backbone.embed_tokens.as_linear(hidden)
                        if config.text_config.tie_word_embeddings
                        else self.language_model.lm_head(hidden)
                    ),
                    temperature,
                    top_k,
                    repetition_penalty,
                    history,
                    suppress,
                    top_p,
                    min_p,
                ).item()
                text_history.append(token)
                if token == config.text_end_token_id:
                    # ends the text of an interleaved block, else the turn
                    if interleaving:
                        text_done = speaking = True
                    else:
                        finished = True
                elif token in eos_ids:
                    finished = True
                elif interleaving:
                    pass  # markers are plain text inside a block
                elif token == config.audio_start_token_id:
                    mode = AUDIO
                elif token == config.mixed_start_token_id:
                    mode = INTERLEAVED
                    slots_left = config.interleaved_n_text
                    speaking = text_done = False
                embed = backbone.embed_tokens(mx.array([token]))
                yield token, None
            if interleaving and not text_done:
                slots_left -= 1
                if slots_left == 0:
                    speaking = not speaking
                    slots_left = (
                        config.interleaved_n_audio
                        if speaking
                        else config.interleaved_n_text
                    )
            if finished or step == max_tokens - 1:
                return
            hidden = backbone(cache=cache, input_embeddings=embed.reshape(1, 1, -1))[
                :, -1
            ]

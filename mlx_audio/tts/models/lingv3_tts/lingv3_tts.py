"""Ming-3.0-tts-tiny speech generation with the native MLX hybrid backbone."""

import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.bailing_moe_v3 import Model as LanguageModel
from mlx_audio.lm.models.bailing_moe_v3 import sanitize_backbone
from mlx_audio.tts.models.bailingmm.bailingmm import (
    Aggregator,
    AudioVAE,
    format_duration,
)
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import load_audio

from .config import ModelConfig
from .flow import FlowLoss
from .prompt import (
    ALIASES,
    LANGUAGES,
    MASKED_TASKS,
    REFERENCE_TASKS,
    SPEAKER_TASKS,
    build_prompt,
    control_instruction,
    podcast_transcript,
    validate_podcast,
)
from .speaker import SPEAKER_SUBPATH, SpeakerEncoder


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = (
            ModelConfig.from_dict(config) if isinstance(config, dict) else config
        )
        c = self.config
        self.model_type = "lingv3_tts"
        self.model = LanguageModel(c.llm_config)
        self.audio = AudioVAE(c.audio_config)
        self.latent_dim = int(c.tts_config.get("latent_dim", 64))
        self.patch_size = int(c.tts_config.get("patch_size", 1))
        self.history_size = int(c.tts_config.get("history_size", 1))
        if not 0 < self.patch_size <= self.history_size:
            raise ValueError("Invalid Ming Tiny latent patch/history geometry")
        aggregator = {
            key: value
            for key, value in c.aggregator_config.items()
            if key not in {"in_channels", "llm_input_dim"}
        }
        self.linear_proj_audio = Aggregator(
            in_channels=self.latent_dim,
            llm_input_dim=c.llm_config.hidden_size,
            **aggregator,
        )
        self.flowloss = FlowLoss(
            self.latent_dim, c.llm_config.hidden_size, c.flow_config
        )
        self.spk_head = nn.Linear(
            int(c.tts_config.get("speaker_embedding_dim", 192)),
            c.llm_config.hidden_size,
        )
        self.stop_head = nn.Linear(c.llm_config.hidden_size, 2)
        self.tokenizer = None
        self._speaker_encoder = None

    @property
    def sample_rate(self):
        return self.audio.sample_rate

    @property
    def layers(self):
        return self.model.model.layers

    def _reference_waveform(self, reference):
        waveform = (
            load_audio(str(reference), sample_rate=self.sample_rate)
            if isinstance(reference, (str, Path))
            else mx.array(reference, dtype=mx.float32)
        )
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=-1)
        if (
            waveform.ndim != 1
            or waveform.size < 1
            or not bool(mx.all(mx.isfinite(waveform)))
        ):
            raise ValueError("References must contain finite nonempty audio")
        return waveform

    def _speaker_embeddings(self, references, spk_emb):
        if spk_emb is not None:
            embedding = mx.array(spk_emb, dtype=mx.float32)
            if embedding.size % 192:
                raise ValueError(
                    "Speaker embeddings must contain 192 values per speaker"
                )
            embedding = embedding.reshape(-1, 192)
        else:
            if self._speaker_encoder is None:
                self._speaker_encoder = SpeakerEncoder(
                    Path(self.config.model_path) / SPEAKER_SUBPATH,
                )
            embedding = mx.concatenate(
                [
                    self._speaker_encoder(reference, self.sample_rate)
                    for reference in references
                ],
                axis=0,
            )
        if (
            embedding.ndim != 2
            or embedding.shape[-1] != 192
            or not bool(mx.all(mx.isfinite(embedding)))
        ):
            raise ValueError("Invalid 192-dimensional speaker embedding")
        return self.spk_head(embedding.astype(self.spk_head.weight.dtype))

    def prepare_inputs(
        self,
        text,
        *,
        task,
        instruction,
        language=None,
        ref_audio=None,
        ref_text=None,
        spk_emb=None,
    ):
        references = (
            list(ref_audio)
            if task == "podcast" and isinstance(ref_audio, (list, tuple))
            else ([] if ref_audio is None else [ref_audio])
        )
        if task in REFERENCE_TASKS:
            if not references or ref_text is None:
                raise ValueError(f"{task} requires reference audio and its transcript")
        if task == "podcast":
            if (
                not isinstance(ref_text, (list, tuple))
                or len(ref_text) != len(references)
                or any(
                    not isinstance(text, str) or not text.strip() for text in ref_text
                )
            ):
                raise ValueError(
                    "Podcast requires one nonempty transcript per reference speaker"
                )
            validate_podcast(text, len(references))
            transcript = podcast_transcript(ref_text)
        elif task in REFERENCE_TASKS:
            if not isinstance(ref_text, str) or not ref_text.strip():
                raise ValueError("Reference transcript must be nonempty text")
            transcript = ref_text
        else:
            if ref_text is not None:
                raise ValueError(f"{task} does not accept a reference transcript")
            transcript = None
        if task in {"voice_design", "ip"} and (references or spk_emb is not None):
            raise ValueError(
                f"{task} uses an empty speaker slot and does not accept references"
            )
        if task in SPEAKER_TASKS and not references and spk_emb is None:
            raise ValueError(f"{task} requires a speaker reference or spk_emb=")
        masked = task in MASKED_TASKS and not references and spk_emb is None
        speakers = (
            mx.zeros(
                (1, self.config.llm_config.hidden_size),
                dtype=self.spk_head.weight.dtype,
            )
            if masked
            else self._speaker_embeddings(references, spk_emb)
        )
        expected_speakers = len(references) if task == "podcast" else 1
        if speakers.shape[0] != expected_speakers:
            raise ValueError(
                "Speaker embeddings must match the reference speaker count"
            )
        prompt_latents = None
        prompt_hidden = None
        if task in REFERENCE_TASKS:
            waveform = mx.concatenate(
                [self._reference_waveform(reference) for reference in references]
            )[None]
            # Tiny pads at the end, unlike the dense 0.5B checkpoint.
            alignment = (
                self.audio.config["enc_kwargs"]["hop_size"]
                * self.audio.patch_size
                * self.patch_size
            )
            waveform = mx.pad(
                waveform, [(0, 0), (0, (-waveform.shape[-1]) % alignment)]
            )
            prompt_latents, _ = self.audio.encode_latent(
                waveform, mx.array([waveform.shape[-1]])
            )
            prompt_hidden = self.linear_proj_audio(
                prompt_latents.reshape(-1, self.patch_size, self.latent_dim)
            ).reshape(1, -1, self.config.llm_config.hidden_size)
        ids, positions = build_prompt(
            self.tokenizer,
            self.config.tts_config,
            text,
            instruction=instruction,
            language=language,
            speakers=expected_speakers,
            prompt_text=transcript,
            prompt_steps=0 if prompt_hidden is None else prompt_hidden.shape[1],
        )
        embeds = self.model.model.word_embeddings(mx.array([ids], dtype=mx.int32))
        for position, speaker in zip(positions, speakers):
            embeds[0, position] = speaker.astype(embeds.dtype)
        if prompt_hidden is not None:
            embeds[:, -prompt_hidden.shape[1] :] = prompt_hidden.astype(embeds.dtype)
        return ids, embeds, prompt_latents

    def generate(
        self,
        text,
        voice=None,
        instruct=None,
        *,
        task=None,
        lang_code="en",
        ref_audio=None,
        ref_text=None,
        spk_emb=None,
        style=None,
        emotion=None,
        dialect=None,
        speed=1.0,
        pitch=None,
        volume=None,
        cfg_scale=2.0,
        ddpm_steps=10,
        temperature=0.0,
        sigma=0.25,
        max_tokens=None,
        seed=None,
        min_decode_steps=None,
        stop_threshold=None,
        stream=False,
        **kwargs,
    ):
        if stream:
            raise ValueError("Ming TTS Tiny currently supports offline generation only")
        if self.tokenizer is None:
            raise ValueError("Load the model with the TTS loader before generating")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be nonempty")
        # The shared TTS CLI sends a neutral numeric pitch to every model.
        # Ming's trained controls use categorical strings instead.
        if not isinstance(pitch, str):
            if pitch not in (None, 1.0):
                raise ValueError("Ming pitch control uses 低, 中 or 高")
            pitch = None
        if not isinstance(speed, str) and speed not in (None, 1.0):
            raise ValueError("Ming speed control uses 慢速, 中速 or 快速")
        if task is None:
            if ref_audio is not None and ref_text is not None:
                task = (
                    "multilingual"
                    if str(lang_code).lower() not in {"en", "english", "zh", "chinese"}
                    else "seed"
                )
            elif style is not None:
                task = "voice_design"
            elif instruct is not None:
                task = "freeform"
            elif emotion is not None:
                task = "emotion"
            elif dialect is not None:
                task = "dialect"
            else:
                task = "ip"
        task = ALIASES.get(task, task)
        if task not in REFERENCE_TASKS | SPEAKER_TASKS | MASKED_TASKS:
            raise ValueError(f"Unsupported Ming Tiny task: {task}")
        ip = kwargs.get("ip", voice or "US_James") if task == "ip" else None
        instruction = control_instruction(
            task,
            instruct=instruct,
            style=style,
            emotion=emotion,
            dialect=dialect,
            speed=speed if isinstance(speed, str) else None,
            pitch=pitch,
            volume=volume,
            ip=ip,
        )
        language = None
        if task in REFERENCE_TASKS:
            hint = kwargs.get(
                "target_language", None if task == "podcast" else lang_code
            )
            language = (
                LANGUAGES.get(str(hint).lower())
                if hint is not None
                else ("Chinese" if re.search(r"[一-鿿]", text) else "English")
            )
            allowed = (
                {"German", "Spanish", "French", "Korean", "Japanese"}
                if task == "multilingual"
                else {"Chinese", "English"}
            )
            if language not in allowed and task != "podcast":
                raise ValueError(f"{task} requires a supported target language")
        if seed is not None:
            mx.random.seed(int(seed))
        started = time.perf_counter()
        ids, embeds, prompt_latents = self.prepare_inputs(
            text,
            task=task,
            instruction=instruction,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            spk_emb=spk_emb,
        )
        tts = self.config.tts_config
        limit = int(
            kwargs.get(
                "max_decode_steps",
                (
                    max_tokens
                    if max_tokens is not None
                    else tts.get("maximum_decode_steps", 250)
                ),
            )
        )
        minimum = int(
            min_decode_steps
            if min_decode_steps is not None
            else tts.get("minimum_decode_steps", 13)
        )
        threshold = float(
            stop_threshold
            if stop_threshold is not None
            else tts.get("stop_threshold", 0.5)
        )
        if limit < 1 or not 0 <= minimum <= limit:
            raise ValueError(
                "Decode limits must satisfy 0 <= min_decode_steps <= max_tokens"
            )
        if len(ids) + limit > self.config.llm_config.max_position_embeddings:
            raise ValueError(
                "Prompt and generation exceed the Ming Tiny context budget"
            )
        cache = self.model.make_cache()
        history = mx.zeros((1, self.history_size, self.latent_dim), mx.float32)
        if prompt_latents is not None:
            available = min(self.history_size, prompt_latents.shape[1])
            history[:, -available:] = prompt_latents[:, -available:].astype(mx.float32)
        latents = []
        for _ in range(limit):
            hidden = self.model.model(inputs_embeds=embeds, cache=cache)[:, -1:]
            sampled = self.flowloss.sample(
                hidden,
                history,
                cfg=2.0 if cfg_scale is None else cfg_scale,
                patch_size=self.patch_size,
                sigma=sigma,
                temperature=temperature,
                steps=10 if ddpm_steps is None else ddpm_steps,
            )
            latents.append(sampled)
            probability = mx.softmax(
                self.stop_head(hidden).astype(mx.float32), axis=-1
            )[0, 0, 1]
            if len(latents) >= minimum and bool(probability > threshold):
                break
            embeds = self.linear_proj_audio(sampled.astype(hidden.dtype))
            history = mx.concatenate([history[:, self.patch_size :], sampled], axis=1)
            mx.eval(embeds, history, [state.state for state in cache])
        speech, _, _ = self.audio.decode(
            mx.concatenate(latents, axis=1).astype(self.audio.decoder.fc1.weight.dtype),
            use_cache=False,
            last_chunk=True,
        )
        speech = speech[0].astype(mx.float32)
        mx.eval(speech)
        if not bool(mx.all(mx.isfinite(speech))):
            raise FloatingPointError("Ming Tiny produced non-finite audio")
        elapsed = time.perf_counter() - started
        duration = speech.size / self.sample_rate
        yield GenerationResult(
            audio=speech,
            samples=speech.size,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(ids),
            audio_duration=format_duration(duration),
            real_time_factor=elapsed / max(duration, 1e-6),
            prompt={
                "tokens": len(ids),
                "tokens-per-sec": len(ids) / max(elapsed, 1e-6),
            },
            audio_samples={
                "samples": speech.size,
                "samples-per-sec": speech.size / max(elapsed, 1e-6),
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_final_chunk=True,
        )

    def sanitize(self, weights):
        return sanitize_backbone(weights, self.config.llm_config, prefix="model.model.")

    @staticmethod
    def model_quant_predicate(path, module):
        return path.startswith("model.") and not path.endswith(".gate")

    @classmethod
    def post_load_hook(cls, model, model_path):
        from transformers import PreTrainedTokenizerFast

        model.config.model_path = str(model_path)
        model.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(model_path),
            trust_remote_code=False,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        tokens = {
            "<audioPatch>": "audio_patch_token_id",
            "<audio>": "audio_start_token_id",
            "</audio>": "audio_end_token_id",
            "<spk>": "speaker_start_token_id",
            "</spk>": "speaker_end_token_id",
        }
        for token, field in tokens.items():
            expected = model.config.tts_config[field]
            if model.tokenizer.encode(token, add_special_tokens=False) != [expected]:
                raise ValueError(f"Tokenizer {token} does not match the checkpoint")
        if model.tokenizer.eos_token_id != model.config.tts_config.get(
            "prompt_eos_token_id", 156895
        ):
            raise ValueError("Tokenizer prompt EOS does not match Ming Tiny training")
        return model

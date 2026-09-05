from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_audio.stt.models.nemotron_asr import tokenizer as rnnt_tokenizer
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram
from mlx_audio.stt.utils import load_audio


@dataclass
class VoiceChatOutput:
    text: str
    audio: mx.array
    text_tokens: mx.array
    audio_tokens: mx.array
    function_tokens: mx.array
    sample_rate: int
    user_transcript: str | None = None

    @property
    def audio_codes(self) -> mx.array:
        return self.audio_tokens


class VoiceChatSession:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.tts_model.set_tokenizer(tokenizer)

    def _decode_text(self, tokens: mx.array) -> str:
        special = {
            self.model.config.pad_token_id,
            self.model.config.silence_token_id,
            self.model.config.bos_token_id,
            self.model.config.eos_token_id,
        }
        ids = [int(token) for token in tokens.tolist() if int(token) not in special]
        return self.tokenizer.decode(ids, skip_special_tokens=False).strip()

    def _rnnt_decode(self, encoded: mx.array, length: int) -> str | None:
        config = self.model.config
        if not config.rnnt_vocabulary:
            return None
        decoder = self.model.stt_model.rnnt_decoder
        joint = self.model.stt_model.rnnt_joint
        last_token = config.rnnt_blank_id
        hidden = None
        result: list[int] = []
        time = 0
        new_symbols = 0
        while time < length:
            token = (
                mx.array([[last_token]], dtype=mx.int32)
                if last_token != config.rnnt_blank_id
                else None
            )
            prediction, proposed_hidden = decoder(token, hidden)
            logits = joint(
                encoded[:, time : time + 1], prediction.astype(encoded.dtype)
            )
            next_token = int(mx.argmax(logits))
            if next_token == config.rnnt_blank_id:
                time += 1
                new_symbols = 0
                continue
            last_token = next_token
            hidden = tuple(value.astype(encoded.dtype) for value in proposed_hidden)
            if not rnnt_tokenizer.is_special_token(next_token, config.rnnt_vocabulary):
                result.append(next_token)
            new_symbols += 1
            if new_symbols >= config.rnnt_max_symbols:
                time += 1
                new_symbols = 0
        return rnnt_tokenizer.decode(result, config.rnnt_vocabulary).strip()

    def _tts_prompt(self, batch_size: int = 1):
        tts = self.model.tts_model
        config = self.model.config
        prompt_latent = tts.audio_prompt_latents[config.speaker_name]
        frames = prompt_latent.shape[1]
        total = frames + 1
        prompt_samples = total * tts.audio_codec.waveform_to_token_ratio
        codes = tts.audio_codec.encode(
            mx.zeros((batch_size, 1, prompt_samples), dtype=mx.float32)
        ).transpose(0, 2, 1)
        expected = (batch_size, total, config.tts.num_quantizers)
        if codes.shape != expected:
            raise RuntimeError(
                f"silent prompt codec produced {codes.shape}, expected {expected}"
            )
        pieces = [codes[:, index] for index in range(total)]
        mask_codes = mx.full(
            (batch_size, config.tts.num_quantizers),
            config.tts.codebook_size,
            dtype=mx.int32,
        )
        pieces[0] = mask_codes
        pieces[-2] = mask_codes
        codes = mx.stack(pieces, axis=1)
        subwords = mx.full((batch_size, frames), config.pad_token_id, dtype=mx.int32)
        subword_mask = mx.zeros((batch_size, frames), dtype=mx.bool_)
        subword_mask[:, -2:] = True
        audio_mask = mx.zeros((batch_size, frames), dtype=mx.bool_)
        audio_mask[:, -1] = True
        latent = mx.broadcast_to(
            prompt_latent[:1], (batch_size, frames, config.tts.hidden_size)
        )
        return codes[:, :-1], audio_mask, subwords, subword_mask, latent

    def _replace_control_codes(self, codes: mx.array) -> mx.array:
        mask = mx.zeros(codes.shape, dtype=mx.bool_)
        for token in self.model.tts_model.control_codes.tolist():
            mask = mask | (codes == int(token))
        silence = mx.broadcast_to(
            self.model.tts_model.codec_silence_tokens[None, None, :], codes.shape
        )
        return mx.where(mask, silence, codes)

    def create_streaming_session(
        self,
        *,
        system_prompt: str | None = None,
        seed: int = 0,
        max_streaming_seconds: float | None = None,
        use_language_cache: bool = True,
        use_perception_cache: bool = True,
    ):
        from .streaming import VoiceChatStreamingSession

        return VoiceChatStreamingSession(
            self,
            system_prompt=system_prompt,
            seed=seed,
            max_streaming_seconds=max_streaming_seconds,
            use_language_cache=use_language_cache,
            use_perception_cache=use_perception_cache,
        )

    def generate(
        self,
        audio: str | Path | mx.array,
        *,
        system_prompt: str | None = None,
        max_frames: int | None = None,
        extra_decoding_seconds: float = 0.0,
        seed: int = 0,
        use_language_cache: bool = False,
    ) -> VoiceChatOutput:
        config = self.model.config
        mx.random.seed(seed)
        if isinstance(audio, (str, Path)):
            waveform = load_audio(str(audio), sr=config.source_sample_rate)
        else:
            waveform = audio.astype(mx.float32)
        waveform = waveform.squeeze()
        if waveform.ndim != 1:
            raise ValueError("audio must be a mono waveform")
        if extra_decoding_seconds < 0:
            raise ValueError("extra_decoding_seconds must be non-negative")
        if extra_decoding_seconds:
            waveform = mx.pad(
                waveform,
                (
                    0,
                    round(extra_decoding_seconds * config.source_sample_rate),
                ),
            )

        mel = log_mel_spectrogram(waveform, config.preprocessor)
        mel_lengths = mx.array([mel.shape[1]], dtype=mx.int32)
        audio_embeds, lengths, asr_embeds = self.model.stt_model.perception(
            mel, mel_lengths
        )
        audio_frames = int(lengths[0])
        if max_frames is not None:
            if max_frames < 2:
                raise ValueError("max_frames must be at least 2")
            audio_frames = min(audio_frames, max_frames)
        audio_embeds = audio_embeds[:, :audio_frames]
        asr_embeds = asr_embeds[:, :audio_frames]

        prompt_text = (
            config.default_system_prompt if system_prompt is None else system_prompt
        )
        prompt_ids: list[int] = []
        if prompt_text.strip():
            prompt_ids = [
                config.bos_token_id,
                *self.tokenizer.encode(prompt_text, add_special_tokens=False),
                config.eos_token_id,
            ]
            prompt_embeds = self.model.stt_model.embed_tokens(
                mx.array([prompt_ids], dtype=mx.int32)
            ).astype(audio_embeds.dtype)
            audio_embeds = mx.concatenate([prompt_embeds, audio_embeds], axis=1)

        prompt_frames = len(prompt_ids)
        timeline_frames = prompt_frames + audio_frames
        mx.eval(audio_embeds, asr_embeds)
        text_tokens = [config.pad_token_id] * timeline_frames
        function_tokens = [config.pad_token_id] * timeline_frames
        language_cache = (
            self.model.stt_model.make_cache() if use_language_cache else None
        )
        input_history: list[mx.array] = []

        def language_step(inputs: mx.array):
            if language_cache is not None:
                return self.model.stt_model(inputs, cache=language_cache)
            input_history.append(inputs)
            return self.model.stt_model(mx.concatenate(input_history, axis=1))

        prompt = self._tts_prompt()
        previous_code, tts_cache = self.model.tts_model.tts_model.warmup(
            *prompt,
            guidance_enabled=True,
        )
        generated_codes = mx.zeros(
            (1, timeline_frames, config.tts.num_quantizers), dtype=mx.int32
        )

        for time in range(timeline_frames):
            previous_text_id = (
                config.pad_token_id if time == 0 else text_tokens[time - 1]
            )
            previous_function_id = (
                config.pad_token_id if time == 0 else function_tokens[time - 1]
            )
            previous_text = mx.array([[previous_text_id]], dtype=mx.int32)
            previous_function = mx.array([[previous_function_id]], dtype=mx.int32)
            fused = (
                config.text_channel_weight
                * self.model.stt_model.embed_tokens(previous_text)
                + config.audio_channel_weight * audio_embeds[:, time : time + 1]
                + config.function_channel_weight
                * self.model.stt_model.embed_tokens(previous_function)
            )
            output = language_step(fused)
            if time >= prompt_frames:
                text_tokens[time] = int(mx.argmax(output.text_logits[:, -1]))
                function_tokens[time] = int(mx.argmax(output.function_logits[:, -1]))
            if time == 0:
                continue
            current = mx.array([[text_tokens[time]]], dtype=mx.int32)
            previous_code, tts_cache = self.model.tts_model.tts_model.generate_step(
                current,
                previous_code,
                tts_cache,
                text_eos_id=config.eos_token_id,
                silence_codes=self.model.tts_model.codec_silence_tokens[None, None],
                guidance_enabled=True,
            )
            generated_codes[:, time : time + 1] = previous_code
            mx.eval(previous_code, output.text_logits, output.function_logits)

        text_array = mx.array(text_tokens[prompt_frames:], dtype=mx.int32)
        function_array = mx.array(function_tokens[prompt_frames:], dtype=mx.int32)
        generated_codes = self._replace_control_codes(
            generated_codes[:, prompt_frames:]
        )
        decoded = self.model.tts_model.audio_codec.decode(
            generated_codes.transpose(0, 2, 1)
        )
        mx.eval(decoded)
        return VoiceChatOutput(
            text=self._decode_text(text_array),
            audio=decoded[0, 0],
            text_tokens=text_array,
            audio_tokens=generated_codes[0],
            function_tokens=function_array,
            sample_rate=config.target_sample_rate,
            user_transcript=self._rnnt_decode(asr_embeds, audio_frames),
        )

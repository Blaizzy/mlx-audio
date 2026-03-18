"""Granite Speech model for speech-to-text transcription using MLX."""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_audio.stt.generate import wired_limit
from mlx_audio.stt.models.base import STTOutput

from .config import ModelConfig
from .conformer import CTCEncoder
from .qformer import EncoderProjector


class LanguageModel(nn.Module):
    """Wrapper around mlx_lm GraniteModel that supports input_embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        from mlx_lm.models.granite import GraniteModel, ModelArgs

        args = ModelArgs(
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            logits_scaling=config.logits_scaling,
            attention_multiplier=config.attention_multiplier,
            embedding_multiplier=config.embedding_multiplier,
            residual_multiplier=config.residual_multiplier,
            max_position_embeddings=config.max_position_embeddings,
            attention_bias=config.attention_bias,
            mlp_bias=config.mlp_bias,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            tie_word_embeddings=config.tie_word_embeddings,
        )

        self.model = GraniteModel(args)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings * self.model.embedding_multiplier
        else:
            h = self.model.embed_tokens(inputs) * self.model.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.model.layers)

        from mlx_lm.models.granite import create_attention_mask

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.model.layers, cache):
            h = layer(h, mask, cache=c)

        h = self.model.norm(h)
        out = self.lm_head(h)
        out = out / self.config.logits_scaling
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens


class Model(nn.Module):
    """Granite Speech model: CTC Conformer encoder + Q-Former projector + Granite LLM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = CTCEncoder(config.encoder_config)

        self.projector = EncoderProjector(
            config=config.projector_config,
            window_size=config.window_size,
            downsample_rate=config.downsample_rate,
            num_queries=config.window_size // config.downsample_rate,
            output_dim=config.text_config.hidden_size,
        )

        self.language_model = LanguageModel(config.text_config)

    @property
    def sample_rate(self) -> int:
        return 16000

    def _preprocess_audio(self, audio) -> mx.array:
        """Preprocess audio to stacked mel spectrogram (160-dim).

        Pipeline: 16kHz -> mel (80 mels, n_fft=512, hop=160, win=400)
                  -> log10 -> dynamic range compression -> stack x2 -> 160-dim
        Matches HF GraniteSpeechFeatureExtractor._extract_mel_spectrograms.
        """
        from mlx_audio.stt.utils import load_audio
        from mlx_audio.utils import hanning, mel_filters, stft

        if isinstance(audio, str):
            audio = load_audio(audio, sr=self.sample_rate)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        if audio.ndim == 3:
            return audio

        N_FFT = 512
        HOP_LENGTH = 160
        WIN_LENGTH = 400
        N_MELS = 80

        window = hanning(WIN_LENGTH, periodic=True)
        # Center-pad window to n_fft (PyTorch convention) before stft
        # stft() right-pads by default, but torchaudio centers the window
        if WIN_LENGTH < N_FFT:
            left = (N_FFT - WIN_LENGTH) // 2
            right = N_FFT - WIN_LENGTH - left
            window = mx.concatenate([mx.zeros(left), window, mx.zeros(right)])

        freqs = stft(audio, window=window, n_fft=N_FFT, hop_length=HOP_LENGTH)
        magnitudes = freqs.abs().square()

        filters = mel_filters(self.sample_rate, N_FFT, N_MELS)
        mel_spec = magnitudes @ filters.T  # (T, 80)

        # Log10 with dynamic range compression (matches HF)
        log_spec = mx.maximum(mel_spec, 1e-10)
        log_spec = mx.log10(log_spec)
        max_val = mx.max(log_spec)
        log_spec = mx.maximum(log_spec, max_val - 8.0) / 4.0 + 1.0

        # Drop last frame if odd count, then stack x2
        T = log_spec.shape[0]
        if T % 2 == 1:
            log_spec = log_spec[: T - 1]
            T = T - 1
        log_spec = log_spec.reshape(T // 2, 2 * N_MELS)

        return log_spec[None]  # (1, T//2, 160)

    def _merge_audio_text_embeddings(
        self,
        input_ids: mx.array,
        audio_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Replace audio placeholder tokens with projected audio embeddings."""
        text_embeds = self.language_model.embed_tokens(input_ids)

        if audio_embeds is None or (
            cache is not None and cache[0] is not None and cache[0].offset > 0
        ):
            return text_embeds

        B = text_embeds.shape[0]
        audio_token_idx = self.config.audio_token_index

        for b in range(B):
            token_ids_np = np.array(input_ids[b])
            audio_positions = np.where(token_ids_np == audio_token_idx)[0]

            if len(audio_positions) == 0:
                continue

            num_embeds = audio_embeds.shape[1]
            n = min(len(audio_positions), num_embeds)

            for i in range(n):
                pos = int(audio_positions[i])
                text_embeds[b, pos] = audio_embeds[b, i]

        return text_embeds

    def __call__(
        self,
        input_ids: mx.array,
        audio_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        input_embeds = self._merge_audio_text_embeddings(
            input_ids=input_ids,
            audio_embeds=audio_embeds,
            cache=cache,
        )
        return self.language_model(input_embeddings=input_embeds, cache=cache)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        for k, v in weights.items():
            # Skip num_batches_tracked (not needed for inference)
            if "num_batches_tracked" in k:
                continue

            # Transpose Conv1d weights: PyTorch [O,I,K] -> MLX [O,K,I]
            # Must be idempotent (called during both convert and load)
            if v.ndim == 3 and "conv" in k and "weight" in k:
                if "depth_conv" in k:
                    # Depthwise: PyTorch [C,1,K] -> MLX [C,K,1]
                    if v.shape[1] == 1 and v.shape[2] > 1:
                        v = v.transpose(0, 2, 1)
                else:
                    # Pointwise/regular: PyTorch [O,I,K] -> MLX [O,K,I]
                    # Heuristic: in PyTorch format, K <= I so shape[-1] < shape[-2]
                    if v.shape[-1] < v.shape[-2]:
                        v = v.transpose(0, 2, 1)

            sanitized[k] = v
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from transformers import AutoTokenizer

        if not hasattr(model, "_tokenizer") or model._tokenizer is None:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
        return model

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return p.startswith("language_model.")

    def stream_generate(
        self,
        input_ids: mx.array,
        *,
        audio_embeds: Optional[mx.array] = None,
        max_tokens: int = 4096,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        generation_stream: Optional[mx.Stream] = None,
        verbose: bool = False,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        from mlx_lm.generate import generate_step

        input_embeddings = self._merge_audio_text_embeddings(
            input_ids=input_ids,
            audio_embeds=audio_embeds,
        )

        # Remove batch dim for generate_step
        if input_embeddings.ndim == 3:
            input_embeddings = input_embeddings[0]

        streams = [generation_stream] if generation_stream is not None else None
        with wired_limit(self, streams):
            prompt = input_ids[0] if input_ids.ndim > 1 else input_ids
            for token, logprobs in tqdm(
                generate_step(
                    prompt=prompt,
                    input_embeddings=input_embeddings,
                    model=self.language_model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ),
                total=max_tokens,
                disable=not verbose,
                desc="Generating",
            ):
                eos = self.config.text_config.eos_token_id
                if isinstance(eos, list):
                    if token in eos:
                        break
                elif token == eos:
                    break

                yield token, logprobs

    def _build_prompt(self, audio_len: int) -> Tuple[mx.array, int]:
        """Build prompt token IDs with audio placeholders.

        Uses chat template: USER: <|audio|>...\n ASSISTANT:
        Returns (input_ids, num_audio_placeholder_tokens)
        """
        audio_token_id = self.config.audio_token_index

        prefix = "USER: "
        prefix_ids = self._tokenizer.encode(prefix, add_special_tokens=False)

        audio_placeholder = [audio_token_id] * audio_len

        suffix = "can you transcribe the speech into a written format?\n ASSISTANT:"
        suffix_ids = self._tokenizer.encode(suffix, add_special_tokens=False)

        all_ids = prefix_ids + audio_placeholder + suffix_ids
        return mx.array([all_ids]), len(audio_placeholder)

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        generation_stream: Optional[mx.Stream] = None,
        verbose: bool = False,
        stream: bool = False,
        chunk_duration: float = 30.0,
        min_chunk_duration: float = 1.0,
        **kwargs,
    ) -> Union[STTOutput, Generator]:
        from mlx_audio.stt.utils import load_audio

        if stream:
            return self.stream_transcribe(
                audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                min_tokens_to_keep=min_tokens_to_keep,
                chunk_duration=chunk_duration,
                min_chunk_duration=min_chunk_duration,
                verbose=verbose,
            )

        from mlx_lm.sample_utils import make_sampler

        start_time = time.time()

        if isinstance(audio, str):
            audio = load_audio(audio, sr=self.sample_rate)
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        chunks = _split_audio_into_chunks(
            audio,
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

        all_texts = []
        segments = []
        total_prompt_tokens = 0
        total_generation_tokens = 0
        remaining_tokens = max_tokens

        chunk_iter = tqdm(
            chunks,
            desc="Processing chunks",
            disable=not verbose or len(chunks) == 1,
        )
        for chunk_audio, offset_sec in chunk_iter:
            if remaining_tokens <= 0:
                break

            actual_duration = len(chunk_audio) / self.sample_rate

            text, prompt_toks, gen_toks = self._generate_single_chunk(
                chunk_audio,
                max_tokens=remaining_tokens,
                sampler=sampler,
                generation_stream=generation_stream,
                verbose=verbose and len(chunks) == 1,
            )
            all_texts.append(text)
            total_prompt_tokens += prompt_toks
            total_generation_tokens += gen_toks
            remaining_tokens -= gen_toks

            segments.append(
                {
                    "text": text,
                    "start": offset_sec,
                    "end": offset_sec + actual_duration,
                }
            )
            mx.clear_cache()

        end_time = time.time()
        full_text = " ".join(all_texts)

        return STTOutput(
            text=full_text.strip(),
            segments=segments,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=total_generation_tokens,
            total_tokens=total_prompt_tokens + total_generation_tokens,
            total_time=end_time - start_time,
            prompt_tps=(
                total_prompt_tokens / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
            generation_tps=(
                total_generation_tokens / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
        )

    def _generate_single_chunk(
        self,
        audio_chunk: np.ndarray,
        *,
        max_tokens: int = 4096,
        sampler: Optional[Callable] = None,
        generation_stream: Optional[mx.Stream] = None,
        verbose: bool = False,
    ) -> Tuple[str, int, int]:
        mel = self._preprocess_audio(audio_chunk)

        encoder_out = self.encoder(mel)
        audio_embeds = self.projector(encoder_out)
        mx.eval(audio_embeds)

        input_ids, _ = self._build_prompt(audio_embeds.shape[1])

        generated_tokens = []
        for token, _ in self.stream_generate(
            input_ids=input_ids,
            audio_embeds=audio_embeds,
            max_tokens=max_tokens,
            sampler=sampler,
            generation_stream=generation_stream,
            verbose=verbose,
        ):
            generated_tokens.append(token)

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text, input_ids.shape[1], len(generated_tokens)

    def stream_transcribe(
        self,
        audio,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        chunk_duration: float = 30.0,
        min_chunk_duration: float = 1.0,
        verbose: bool = False,
    ) -> Generator:
        from dataclasses import dataclass

        from mlx_lm.sample_utils import make_sampler

        from mlx_audio.stt.utils import load_audio

        @dataclass
        class StreamingResult:
            text: str
            is_final: bool
            start_time: float
            end_time: float
            language: str = "en"
            prompt_tokens: int = 0
            generation_tokens: int = 0

        if isinstance(audio, str):
            audio = load_audio(audio, sr=self.sample_rate)
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        chunks = _split_audio_into_chunks(
            audio,
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

        total_prompt_tokens = 0
        total_generation_tokens = 0
        remaining_tokens = max_tokens

        for chunk_idx, (chunk_audio, offset_sec) in enumerate(chunks):
            if remaining_tokens <= 0:
                break

            actual_duration = len(chunk_audio) / self.sample_rate
            is_last = chunk_idx == len(chunks) - 1

            mel = self._preprocess_audio(chunk_audio)
            encoder_out = self.encoder(mel)
            audio_embeds = self.projector(encoder_out)
            mx.eval(audio_embeds)  # noqa: S307

            input_ids, _ = self._build_prompt(audio_embeds.shape[1])
            chunk_prompt_tokens = input_ids.shape[1]
            token_count = 0

            for token, _ in self.stream_generate(
                input_ids=input_ids,
                audio_embeds=audio_embeds,
                max_tokens=remaining_tokens,
                sampler=sampler,
                verbose=verbose,
            ):
                text = self._tokenizer.decode([int(token)])
                token_count += 1

                yield StreamingResult(
                    text=text,
                    is_final=False,
                    start_time=offset_sec,
                    end_time=offset_sec + actual_duration,
                )

            total_prompt_tokens += chunk_prompt_tokens
            total_generation_tokens += token_count
            remaining_tokens -= token_count

            yield StreamingResult(
                text="",
                is_final=is_last and remaining_tokens > 0,
                start_time=offset_sec,
                end_time=offset_sec + actual_duration,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=total_generation_tokens,
            )
            mx.clear_cache()


def _split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    chunk_duration: float = 30.0,
    min_chunk_duration: float = 1.0,
) -> List[Tuple[np.ndarray, float]]:
    """Split audio into chunks at low-energy boundaries."""
    if wav.ndim > 1:
        wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)

    total_samples = len(wav)
    total_sec = total_samples / sr

    if total_sec <= chunk_duration:
        if total_sec < min_chunk_duration:
            min_samples = int(min_chunk_duration * sr)
            wav = np.pad(wav, (0, min_samples - len(wav)))
        return [(wav, 0.0)]

    chunks = []
    start_sample = 0
    max_chunk_samples = int(chunk_duration * sr)
    search_samples = int(2.0 * sr)
    min_window_samples = int(0.1 * sr)

    while start_sample < total_samples:
        end_sample = min(start_sample + max_chunk_samples, total_samples)

        if end_sample >= total_samples:
            chunk = wav[start_sample:total_samples]
            if len(chunk) < min_chunk_duration * sr:
                min_samples = int(min_chunk_duration * sr)
                chunk = np.pad(chunk, (0, min_samples - len(chunk)))
            chunks.append((chunk, start_sample / sr))
            break

        search_start = max(start_sample, end_sample - search_samples)
        search_end = min(total_samples, end_sample + search_samples)
        search_region = wav[search_start:search_end]

        if len(search_region) > min_window_samples:
            energy = np.convolve(
                search_region**2,
                np.ones(min_window_samples) / min_window_samples,
                mode="valid",
            )
            min_idx = np.argmin(energy) + min_window_samples // 2
            cut_sample = search_start + min_idx
        else:
            cut_sample = end_sample

        cut_sample = max(cut_sample, start_sample + sr)
        chunk = wav[start_sample:cut_sample]

        if len(chunk) < min_chunk_duration * sr:
            min_samples = int(min_chunk_duration * sr)
            chunk = np.pad(chunk, (0, min_samples - len(chunk)))

        chunks.append((chunk, start_sample / sr))
        start_sample = cut_sample

    return chunks

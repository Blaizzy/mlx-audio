import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

from mlx_audio.stt.models.parakeet import tokenizer
from mlx_audio.stt.models.parakeet.alignment import (
    AlignedResult,
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from mlx_audio.stt.models.parakeet.audio import PreprocessArgs, log_mel_spectrogram
from mlx_audio.stt.models.parakeet.conformer import Conformer, ConformerArgs
from mlx_audio.stt.models.parakeet.ctc import (
    AuxCTCArgs,
    ConvASRDecoder,
    ConvASRDecoderArgs,
)
from mlx_audio.stt.models.parakeet.rnnt import (
    JointArgs,
    JointNetwork,
    PredictArgs,
    PredictNetwork,
)
from mlx_audio.stt.utils import load_audio
from mlx_audio.utils import from_dict


@dataclass
class PreprocessArgs:
    sample_rate: int
    normalize: str
    window_size: float
    window_stride: float
    window: str
    features: int
    n_fft: int
    dither: float
    pad_to: int = 0
    pad_value: float = 0

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)


@dataclass
class TDTDecodingArgs:
    model_type: str
    durations: list[int]
    greedy: dict | None


@dataclass
class RNNTDecodingArgs:
    greedy: dict | None


@dataclass
class CTCDecodingArgs:
    greedy: dict | None


@dataclass
class ParakeetTDTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: TDTDecodingArgs


@dataclass
class ParakeetRNNTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: RNNTDecodingArgs


@dataclass
class ParakeetCTCArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: ConvASRDecoderArgs
    decoding: CTCDecodingArgs


@dataclass
class ParakeetTDTCTCArgs(ParakeetTDTArgs):
    aux_ctc: AuxCTCArgs


@dataclass
class StreamingResult:
    """Result from streaming transcription.

    Attributes:
        text: Decoded text for this emission.
        tokens: Token IDs that were decoded.
        is_final: True if this is a final (committed) result, False if partial.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        progress: Progress from 0.0 to 1.0 (percentage of audio processed).
        audio_position: Current position in audio (seconds).
        audio_duration: Total audio duration (seconds).
        language: Language code (e.g., "en").
    """

    text: str
    tokens: List[int]
    is_final: bool
    start_time: float
    end_time: float
    progress: float = 0.0
    audio_position: float = 0.0
    audio_duration: float = 0.0
    language: str = "en"


class ModelConfig:
    """Config wrapper for Parakeet models."""

    def __init__(self, config: dict):
        self._config = config

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(config)


class Model(nn.Module):
    def __new__(cls, config):
        # If config is a ModelConfig wrapper, use from_config factory
        if isinstance(config, ModelConfig):
            return cls.from_config(config._config)
        # Otherwise proceed with normal instantiation
        return super().__new__(cls)

    def __init__(self, preprocess_args: PreprocessArgs):
        # Skip init if already initialized via from_config
        if hasattr(self, "preprocessor_config"):
            return
        super().__init__()

        self.preprocessor_config = preprocess_args

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """
        Decode mel spectrograms to produce transcriptions with the Parakeet model.
        Handles batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        raise NotImplementedError

    def decode_chunk(
        self, audio_data: mx.array, verbose: bool = False
    ) -> AlignedResult:
        mel = log_mel_spectrogram(audio_data, self.preprocessor_config)
        result = self.decode(mel)[0]
        if verbose:
            print(result.text)
        return result

    def generate(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
        stream: bool = False,
        **kwargs,
    ) -> AlignedResult | Generator[StreamingResult, None, None]:
        """
        Transcribe an audio file, with optional chunking for long files.

        Args:
            path: Path to the audio file
            dtype: Data type for processing
            chunk_duration: If provided, splits audio into chunks of this length for processing
            overlap_duration: Overlap between chunks (only used when chunking)
            chunk_callback: A function to call back when chunk is processed, called with (current_position, total_position)
            stream: If True, yields streaming results instead of returning final result

        Returns:
            Transcription result with aligned tokens and sentences, or a generator of StreamingResult if stream=True
        """

        kwargs.pop("max_tokens", None)
        kwargs.pop("generation_stream", None)
        verbose = kwargs.pop("verbose", False)

        if stream:
            return self.stream_generate(
                path,
                dtype=dtype,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                verbose=verbose,
                **kwargs,
            )

        audio_path = Path(path)
        audio_data = load_audio(
            audio_path, self.preprocessor_config.sample_rate, dtype=dtype
        )

        if chunk_duration is None:
            return self.decode_chunk(audio_data, verbose)

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate

        if audio_length_seconds <= chunk_duration:
            return self.decode_chunk(audio_data, verbose)

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)

        all_tokens = []

        for start in tqdm(
            range(0, len(audio_data), chunk_samples - overlap_samples),
            disable=not verbose,
        ):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            chunk_audio = audio_data[start:end]
            chunk_mel = log_mel_spectrogram(chunk_audio, self.preprocessor_config)

            chunk_result = self.decode(chunk_mel)[0]

            chunk_offset = start / self.preprocessor_config.sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            chunk_tokens = []
            for sentence in chunk_result.sentences:
                chunk_tokens.extend(sentence.tokens)

            if all_tokens:
                try:
                    all_tokens = merge_longest_contiguous(
                        all_tokens, chunk_tokens, overlap_duration=overlap_duration
                    )
                except RuntimeError:
                    all_tokens = merge_longest_common_subsequence(
                        all_tokens, chunk_tokens, overlap_duration=overlap_duration
                    )
            else:
                all_tokens = chunk_tokens

        result = sentences_to_result(tokens_to_sentences(all_tokens))

        # Clear cache after each segment to avoid memory leaks
        mx.clear_cache()

        if verbose:
            print(result.text)
            print("\n\033[94mSegments:\033[0m")
            if hasattr(result, "segments"):
                print(result.segments)
            elif hasattr(result, "tokens"):
                print(result.tokens)
            else:
                print(result)

        return result

    def stream_generate(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: float = 5.0,
        overlap_duration: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[StreamingResult, None, None]:
        """
        Transcribe an audio file with streaming output, yielding results as chunks are processed.

        Args:
            path: Path to the audio file
            dtype: Data type for processing
            chunk_duration: Duration of each chunk in seconds (default 5.0)
            overlap_duration: Overlap between chunks in seconds (default 1.0)
            verbose: Show progress bar

        Yields:
            StreamingResult objects with partial/final transcriptions.

        Example:
            >>> for result in model.stream_generate("audio.wav"):
            ...     print(f"[{'FINAL' if result.is_final else 'partial'}] {result.text}")
        """

        audio_path = Path(path)
        audio_data = load_audio(
            audio_path, self.preprocessor_config.sample_rate, dtype=dtype
        )

        sample_rate = self.preprocessor_config.sample_rate
        total_samples = len(audio_data)
        audio_duration = total_samples / sample_rate

        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        step_samples = chunk_samples - overlap_samples

        all_tokens = []
        previous_text = ""

        for start in tqdm(
            range(0, total_samples, step_samples),
            disable=not verbose,
            desc="Streaming",
        ):
            end = min(start + chunk_samples, total_samples)
            is_last = end >= total_samples

            chunk_audio = audio_data[start:end]
            chunk_mel = log_mel_spectrogram(chunk_audio, self.preprocessor_config)

            chunk_result = self.decode(chunk_mel)[0]

            # Adjust timestamps for chunk offset
            chunk_offset = start / sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            chunk_tokens = []
            for sentence in chunk_result.sentences:
                chunk_tokens.extend(sentence.tokens)

            # Merge with previous tokens
            if all_tokens:
                try:
                    all_tokens = merge_longest_contiguous(
                        all_tokens, chunk_tokens, overlap_duration=overlap_duration
                    )
                except RuntimeError:
                    all_tokens = merge_longest_common_subsequence(
                        all_tokens, chunk_tokens, overlap_duration=overlap_duration
                    )
            else:
                all_tokens = chunk_tokens

            # Build current result
            current_result = sentences_to_result(tokens_to_sentences(all_tokens))
            accumulated_text = current_result.text

            # Calculate new text since last emission
            new_text = accumulated_text[len(previous_text) :]
            previous_text = accumulated_text

            # Calculate progress
            progress = end / total_samples
            audio_position = end / sample_rate

            # Get start and end times from tokens
            start_time = all_tokens[0].start if all_tokens else 0.0
            end_time = all_tokens[-1].end if all_tokens else audio_position

            # Get token IDs
            token_ids = [t.id for t in all_tokens]

            yield StreamingResult(
                text=new_text,
                tokens=token_ids,
                is_final=is_last,
                start_time=start_time,
                end_time=end_time,
                progress=progress,
                audio_position=audio_position,
                audio_duration=audio_duration,
                language="en",
            )

            # Clear cache after each chunk
            mx.clear_cache()

            if is_last:
                break

    @classmethod
    def from_config(cls, config: dict):
        """Loads model from config (randomized weights)"""
        if (
            config.get("target")
            == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
            and config.get("model_defaults", {}).get("tdt_durations") is not None
        ):
            cfg = from_dict(ParakeetTDTArgs, config)
            model = ParakeetTDT(cfg)
        elif (
            config.get("target")
            == "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel"
            and config.get("model_defaults", {}).get("tdt_durations") is not None
        ):
            cfg = from_dict(ParakeetTDTCTCArgs, config)
            model = ParakeetTDTCTC(cfg)
        elif (
            config.get("target")
            == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
            and config.get("model_defaults", {}).get("tdt_durations") is None
        ):
            cfg = from_dict(ParakeetRNNTArgs, config)
            model = ParakeetRNNT(cfg)
        elif (
            config.get("target")
            == "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE"
        ):
            cfg = from_dict(ParakeetCTCArgs, config)
            model = ParakeetCTC(cfg)
        else:
            raise ValueError("Model is not supported yet!")

        model.eval()  # prevents layernorm not computing correctly on inference!

        return model

    @classmethod
    def from_pretrained(cls, path_or_hf_repo: str, *, dtype: mx.Dtype = mx.bfloat16):
        """
        Loads model from Hugging Face or local directory.

        .. deprecated::
            Use `mlx_audio.stt.load()` instead. This method will be removed in a future version.
        """
        warnings.warn(
            "Model.from_pretrained() is deprecated. Use mlx_audio.stt.load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            config = json.load(
                open(hf_hub_download(path_or_hf_repo, "config.json"), "r")
            )
            weight = hf_hub_download(path_or_hf_repo, "model.safetensors")
        except Exception:
            config = json.load(open(Path(path_or_hf_repo) / "config.json", "r"))
            weight = str(Path(path_or_hf_repo) / "model.safetensors")

        model = cls.from_config(config)
        model.load_weights(weight)

        # cast dtype
        curr_weights = dict(tree_flatten(model.parameters()))
        curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
        model.update(tree_unflatten(curr_weights))

        return model


class ParakeetTDT(Model):
    def __init__(self, args: ParakeetTDTArgs):
        # Skip init if already initialized via from_config
        if hasattr(self, "preprocessor_config"):
            return
        super().__init__(args.preprocessor)

        assert args.decoding.model_type == "tdt", "Model must be a TDT model"

        self.encoder_config = args.encoder

        self.vocabulary = args.joint.vocabulary
        self.durations = args.decoding.durations
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.encoder = Conformer(args.encoder)
        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """
        Generate with skip token logic for the Parakeet model, handling batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel)
        mx.eval(batch_features, lengths)

        results = []
        for b in range(batch_size):
            features = batch_features[b : b + 1]
            max_length = int(lengths[b])

            last_token = len(
                self.vocabulary
            )  # In TDT, space token is always len(vocab)
            hypothesis = []

            time = 0
            new_symbols = 0
            decoder_hidden = None

            while time < max_length:
                feature = features[:, time : time + 1]

                current_token = (
                    mx.array([[last_token]], dtype=mx.int32)
                    if last_token != len(self.vocabulary)
                    else None
                )
                decoder_output, (hidden, cell) = self.decoder(
                    current_token, decoder_hidden
                )

                # cast
                decoder_output = decoder_output.astype(feature.dtype)
                proposed_decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                joint_output = self.joint(feature, decoder_output)

                pred_token = mx.argmax(
                    joint_output[0, 0, :, : len(self.vocabulary) + 1]
                )
                decision = mx.argmax(joint_output[0, 0, :, len(self.vocabulary) + 1 :])

                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=time
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=self.durations[int(decision)]
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([int(pred_token)], self.vocabulary),
                        )
                    )
                    last_token = int(pred_token)
                    decoder_hidden = proposed_decoder_hidden

                time += self.durations[int(decision)]
                new_symbols += 1

                if self.durations[int(decision)] != 0:
                    new_symbols = 0
                else:
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        time += 1
                        new_symbols = 0

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetRNNT(Model):
    def __init__(self, args: ParakeetRNNTArgs):
        # Skip init if already initialized via from_config
        if hasattr(self, "preprocessor_config"):
            return
        super().__init__(args.preprocessor)

        self.encoder_config = args.encoder

        self.vocabulary = args.joint.vocabulary
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.encoder = Conformer(args.encoder)
        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """
        Generate with skip token logic for the Parakeet model, handling batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel)
        mx.eval(batch_features, lengths)

        results = []
        for b in range(batch_size):
            features = batch_features[b : b + 1]
            max_length = int(lengths[b])

            last_token = len(self.vocabulary)
            hypothesis = []

            time = 0
            new_symbols = 0
            decoder_hidden = None

            while time < max_length:
                feature = features[:, time : time + 1]

                current_token = (
                    mx.array([[last_token]], dtype=mx.int32)
                    if last_token != len(self.vocabulary)
                    else None
                )
                decoder_output, (hidden, cell) = self.decoder(
                    current_token, decoder_hidden
                )

                # cast
                decoder_output = decoder_output.astype(feature.dtype)
                proposed_decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                joint_output = self.joint(feature, decoder_output)

                pred_token = mx.argmax(joint_output)

                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=time
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=1
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([int(pred_token)], self.vocabulary),
                        )
                    )
                    last_token = int(pred_token)
                    decoder_hidden = proposed_decoder_hidden

                    new_symbols += 1
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        time += 1
                        new_symbols = 0
                else:
                    time += 1
                    new_symbols = 0

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetCTC(Model):
    def __init__(self, args: ParakeetCTCArgs):
        # Skip init if already initialized via from_config
        if hasattr(self, "preprocessor_config"):
            return
        super().__init__(args.preprocessor)

        self.encoder_config = args.encoder

        self.vocabulary = args.decoder.vocabulary

        self.encoder = Conformer(args.encoder)
        self.decoder = ConvASRDecoder(args.decoder)

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """
        Generate with CTC decoding for the Parakeet model, handling batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel)
        logits = self.decoder(batch_features)
        mx.eval(logits, lengths)

        results = []
        for b in range(batch_size):
            features_len = int(lengths[b])
            predictions = logits[b, :features_len]
            best_tokens = mx.argmax(predictions, axis=1)

            hypothesis = []
            token_boundaries = []
            prev_token = -1

            for t, token_id in enumerate(best_tokens):
                token_idx = int(token_id)

                if token_idx == len(self.vocabulary):
                    continue

                if token_idx == prev_token:
                    continue

                if prev_token != -1:
                    token_start_time = (
                        token_boundaries[-1][0]
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_end_time = (
                        t
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_duration = token_end_time - token_start_time

                    hypothesis.append(
                        AlignedToken(
                            prev_token,
                            start=token_start_time,
                            duration=token_duration,
                            text=tokenizer.decode([prev_token], self.vocabulary),
                        )
                    )

                token_boundaries.append((t, None))
                prev_token = token_idx

            if prev_token != -1:
                last_non_blank = features_len - 1
                for t in range(features_len - 1, token_boundaries[-1][0], -1):
                    if int(best_tokens[t]) != len(self.vocabulary):
                        last_non_blank = t
                        break

                token_start_time = (
                    token_boundaries[-1][0]
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_end_time = (
                    (last_non_blank + 1)
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_duration = token_end_time - token_start_time

                hypothesis.append(
                    AlignedToken(
                        prev_token,
                        start=token_start_time,
                        duration=token_duration,
                        text=tokenizer.decode([prev_token], self.vocabulary),
                    )
                )

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetTDTCTC(ParakeetTDT):
    """Has ConvASRDecoder decoder in `.ctc_decoder` but `.generate` uses TDT decoder all the times (Please open an issue if you need CTC decoder use-case!)"""

    def __init__(self, args: ParakeetTDTCTCArgs):
        # Skip init if already initialized via from_config
        if hasattr(self, "preprocessor_config"):
            return
        super().__init__(args)

        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)

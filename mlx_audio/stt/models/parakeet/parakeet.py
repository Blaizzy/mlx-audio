import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from dacite import from_dict
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten

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


class Model(nn.Module):
    def __init__(self, preprocess_args: PreprocessArgs):
        super().__init__()

        self.preprocessor_config = preprocess_args

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """
        Decode mel spectrograms to produce transcriptions with the Parakeet model.
        Handles batches and single input. Uses greedy decoding.
        mel: [batch, sequence, mel_dim] or [sequence, mel_dim]
        """
        raise NotImplementedError

    def generate(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
    ) -> AlignedResult:
        """
        Transcribe an audio file, with optional chunking for long files.

        Args:
            path: Path to the audio file
            dtype: Data type for processing
            chunk_duration: If provided, splits audio into chunks of this length for processing
            overlap_duration: Overlap between chunks (only used when chunking)
            chunk_callback: A function to call back when chunk is processed, called with (current_position, total_position)

        Returns:
            Transcription result with aligned tokens and sentences
        """
        audio_path = Path(path)
        audio_data = load_audio(
            audio_path, self.preprocessor_config.sample_rate, dtype=dtype
        )

        if chunk_duration is None:
            mel = log_mel_spectrogram(audio_data, self.preprocessor_config)

            return self.decode(mel)[0]

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate

        if audio_length_seconds <= chunk_duration:
            mel = log_mel_spectrogram(audio_data, self.preprocessor_config)

            return self.decode(mel)[0]

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)

        all_tokens = []

        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
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

        return result

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
        """Loads model from Hugging Face or local directory"""

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
        super().__init__(args)

        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)

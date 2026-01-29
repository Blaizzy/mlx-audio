# Copyright 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .qwen3_asr import (
    AudioEncoder,
    TextModel,
    _get_feat_extract_output_lengths,
)


class ForceAlignProcessor:
    """Processor for forced alignment text tokenization and timestamp parsing."""

    def __init__(self):
        self.ko_score = None
        self.ko_tokenizer = None
        self._nagisa = None

    def is_kept_char(self, ch: str) -> bool:
        """Check if character should be kept (letters, numbers, apostrophe)."""
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return True
        return False

    def clean_token(self, token: str) -> str:
        """Remove non-kept characters from token."""
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        """Check if character is CJK (Chinese/Japanese/Korean ideograph)."""
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # Extension A
            or 0x20000 <= code <= 0x2A6DF  # Extension B
            or 0x2A700 <= code <= 0x2B73F  # Extension C
            or 0x2B740 <= code <= 0x2B81F  # Extension D
            or 0x2B820 <= code <= 0x2CEAF  # Extension E
            or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
        )

    def tokenize_chinese_mixed(self, text: str) -> List[str]:
        """Tokenize text with Chinese characters (each char is a token)."""
        tokens: List[str] = []
        current_latin: List[str] = []

        def flush_latin():
            nonlocal current_latin
            if current_latin:
                token = "".join(current_latin)
                cleaned = self.clean_token(token)
                if cleaned:
                    tokens.append(cleaned)
                current_latin = []

        for ch in text:
            if self.is_cjk_char(ch):
                flush_latin()
                tokens.append(ch)
            else:
                if self.is_kept_char(ch):
                    current_latin.append(ch)
                else:
                    flush_latin()

        flush_latin()
        return tokens

    def tokenize_japanese(self, text: str) -> List[str]:
        """Tokenize Japanese text using nagisa."""
        if self._nagisa is None:
            try:
                import nagisa

                self._nagisa = nagisa
            except ImportError:
                raise ImportError(
                    "Japanese tokenization requires nagisa. "
                    "Install with: pip install nagisa"
                )
        words = self._nagisa.tagging(text).words
        tokens: List[str] = []
        for w in words:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def tokenize_korean(self, text: str) -> List[str]:
        """Tokenize Korean text using soynlp."""
        if self.ko_tokenizer is None:
            try:
                from soynlp.tokenizer import LTokenizer

                # Simple frequency-based tokenizer
                self.ko_tokenizer = LTokenizer()
            except ImportError:
                raise ImportError(
                    "Korean tokenization requires soynlp. "
                    "Install with: pip install soynlp"
                )
        raw_tokens = self.ko_tokenizer.tokenize(text)
        tokens: List[str] = []
        for w in raw_tokens:
            w_clean = self.clean_token(w)
            if w_clean:
                tokens.append(w_clean)
        return tokens

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        """Split segment containing Chinese characters."""
        tokens: List[str] = []
        buf: List[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        """Tokenize space-separated languages (English, etc.)."""
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def fix_timestamp(self, data: np.ndarray) -> List[int]:
        """Fix non-monotonic timestamps using Longest Increasing Subsequence."""
        data = data.tolist()
        n = len(data)

        if n == 0:
            return []

        # Find LIS using dynamic programming
        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_length = max(dp)
        max_idx = dp.index(max_length)

        # Reconstruct LIS indices
        lis_indices = []
        idx = max_idx
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for idx in lis_indices:
            is_normal[idx] = True

        result = data.copy()
        i = 0

        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]:
                    j += 1

                anomaly_count = j - i

                if anomaly_count <= 2:
                    # For small anomalies, use nearest valid neighbor
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break

                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break

                    for k in range(i, j):
                        if left_val is None:
                            result[k] = right_val
                        elif right_val is None:
                            result[k] = left_val
                        else:
                            result[k] = (
                                left_val if (k - (i - 1)) <= ((j) - k) else right_val
                            )

                else:
                    # For large anomalies, interpolate linearly
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break

                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break

                    if left_val is not None and right_val is not None:
                        step = (right_val - left_val) / (anomaly_count + 1)
                        for k in range(i, j):
                            result[k] = left_val + step * (k - i + 1)
                    elif left_val is not None:
                        for k in range(i, j):
                            result[k] = left_val
                    elif right_val is not None:
                        for k in range(i, j):
                            result[k] = right_val

                i = j
            else:
                i += 1

        return [int(res) for res in result]

    def encode_timestamp(self, text: str, language: str) -> Tuple[List[str], str]:
        """Tokenize text and create input with timestamp tokens.

        Args:
            text: The transcript text to align.
            language: Language name (e.g., "Chinese", "Japanese", "Korean", "English").

        Returns:
            Tuple of (word_list, input_text with timestamp tokens).
        """
        language = language.lower()

        if language == "japanese":
            word_list = self.tokenize_japanese(text)
        elif language == "korean":
            word_list = self.tokenize_korean(text)
        elif language == "chinese":
            word_list = self.tokenize_chinese_mixed(text)
        else:
            # Space-separated languages (English, etc.)
            word_list = self.tokenize_space_lang(text)

        # Build input text with timestamp tokens between words
        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text

        return word_list, input_text

    def parse_timestamp(
        self, word_list: List[str], timestamp: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Parse timestamps into word-level alignments.

        Args:
            word_list: List of words/tokens.
            timestamp: Raw timestamp values in milliseconds.

        Returns:
            List of dicts with text, start_time, end_time.
        """
        timestamp_output = []
        timestamp_fixed = self.fix_timestamp(timestamp)

        for i, word in enumerate(word_list):
            start_time = timestamp_fixed[i * 2]
            end_time = timestamp_fixed[i * 2 + 1]
            timestamp_output.append(
                {"text": word, "start_time": start_time, "end_time": end_time}
            )

        return timestamp_output


@dataclass(frozen=True)
class ForcedAlignItem:
    """One aligned item span.

    Attributes:
        text: The aligned unit (CJK character or word).
        start_time: Start time in seconds.
        end_time: End time in seconds.
    """

    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ForcedAlignResult:
    """Forced alignment output for one sample.

    Attributes:
        items: Aligned token spans.
    """

    items: List[ForcedAlignItem]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]


@dataclass
class ForcedAlignerConfig:
    """Configuration for Qwen3-ForcedAligner model."""

    audio_config: Any = None
    text_config: Any = None
    model_type: str = "qwen3_forced_aligner"
    model_repo: str = None
    audio_token_id: int = 151676
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    timestamp_token_id: int = 151705
    timestamp_segment_time: float = 80.0
    classify_num: int = 5000  # Number of timestamp classes
    support_languages: List[str] = None

    def __post_init__(self):
        from .config import AudioEncoderConfig, TextConfig

        if self.audio_config is None:
            self.audio_config = AudioEncoderConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig.from_dict(self.audio_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

        if self.support_languages is None:
            self.support_languages = []

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ForcedAlignerConfig":
        from .config import AudioEncoderConfig, TextConfig

        params = params.copy()

        # Extract top-level timestamp params (HF config has these at root level)
        top_level_timestamp_token_id = params.get("timestamp_token_id")
        top_level_timestamp_segment_time = params.get("timestamp_segment_time")

        # Handle nested thinker_config (from HF config)
        if "thinker_config" in params:
            thinker = params.pop("thinker_config")
            if "audio_config" in thinker:
                params["audio_config"] = thinker["audio_config"]
            if "text_config" in thinker:
                params["text_config"] = thinker["text_config"]
            if "audio_token_id" in thinker:
                params["audio_token_id"] = thinker["audio_token_id"]
            if "audio_start_token_id" in thinker:
                params["audio_start_token_id"] = thinker["audio_start_token_id"]
            if "audio_end_token_id" in thinker:
                params["audio_end_token_id"] = thinker["audio_end_token_id"]
            # Use thinker values if present, otherwise use top-level
            if "timestamp_token_id" in thinker:
                params["timestamp_token_id"] = thinker["timestamp_token_id"]
            if "timestamp_segment_time" in thinker:
                params["timestamp_segment_time"] = thinker["timestamp_segment_time"]
            if "classify_num" in thinker:
                params["classify_num"] = thinker["classify_num"]

        # Use top-level timestamp params if not set from thinker_config
        if top_level_timestamp_token_id is not None and "timestamp_token_id" not in params:
            params["timestamp_token_id"] = top_level_timestamp_token_id
        if top_level_timestamp_segment_time is not None and "timestamp_segment_time" not in params:
            params["timestamp_segment_time"] = top_level_timestamp_segment_time

        # Handle nested configs
        if "audio_config" in params and isinstance(params["audio_config"], dict):
            params["audio_config"] = AudioEncoderConfig.from_dict(params["audio_config"])
        elif "audio_config" not in params:
            params["audio_config"] = AudioEncoderConfig()

        if "text_config" in params and isinstance(params["text_config"], dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])
        elif "text_config" not in params:
            params["text_config"] = TextConfig()

        import inspect

        # Always set model_type to qwen3_forced_aligner for this config class
        params["model_type"] = "qwen3_forced_aligner"

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class ForcedAlignerModel(nn.Module):
    """Qwen3-ForcedAligner Model for word-level audio alignment.

    This model takes audio and text as input and returns word-level
    timestamps showing when each word/character is spoken.
    """

    def __init__(self, config: ForcedAlignerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextModel(config.text_config)
        self.aligner_processor = ForceAlignProcessor()

        # ForcedAligner always uses lm_head for timestamp classification
        # The output size is classify_num (number of timestamp classes), not vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.classify_num,
            bias=False,
        )

    def get_audio_features(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode audio features."""
        return self.audio_tower(input_features, feature_attention_mask)

    def __call__(
        self,
        input_ids: mx.array,
        input_features: Optional[mx.array] = None,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass returning logits.

        Args:
            input_ids: Token IDs [batch, seq_len].
            input_features: Audio features [batch, mel_bins, time].
            feature_attention_mask: Attention mask for audio.

        Returns:
            Logits tensor [batch, seq_len, classify_num] for timestamp classification.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask
            )
            audio_features = audio_features.astype(inputs_embeds.dtype)

            audio_token_mask = input_ids == self.config.audio_token_id

            if audio_token_mask.any():
                batch_size, seq_len, hidden_dim = inputs_embeds.shape
                flat_mask = audio_token_mask.flatten()
                flat_mask_np = np.array(flat_mask)
                audio_indices = np.where(flat_mask_np)[0]

                if len(audio_indices) > 0 and audio_features.shape[0] > 0:
                    num_to_replace = min(len(audio_indices), audio_features.shape[0])
                    flat_embeds = inputs_embeds.reshape(-1, hidden_dim)

                    result_list = []
                    audio_idx = 0
                    for i in range(flat_embeds.shape[0]):
                        if audio_idx < num_to_replace and i == audio_indices[audio_idx]:
                            result_list.append(audio_features[audio_idx])
                            audio_idx += 1
                        else:
                            result_list.append(flat_embeds[i])

                    inputs_embeds = mx.stack(result_list, axis=0).reshape(
                        batch_size, seq_len, hidden_dim
                    )

        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=None)

        # ForcedAligner always uses lm_head for timestamp classification
        logits = self.lm_head(hidden_states)

        return logits

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self) -> int:
        """Sample rate for audio input."""
        return 16000

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from HuggingFace/PyTorch format to MLX format."""
        sanitized = {}
        is_formatted = not any(k.startswith("thinker.") for k in weights.keys())

        for k, v in weights.items():
            if k.startswith("thinker."):
                k = k[len("thinker.") :]

            # ForcedAligner uses lm_head, don't skip it

            if (
                not is_formatted
                and "conv2d" in k
                and "weight" in k
                and len(v.shape) == 4
            ):
                v = v.transpose(0, 2, 3, 1)

            sanitized[k] = v

        return sanitized

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        """Determine which layers to quantize."""
        return not p.startswith("audio_tower")

    @classmethod
    def post_load_hook(cls, model: "ForcedAlignerModel", model_path: Path) -> "ForcedAlignerModel":
        """Hook called after model weights are loaded."""
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        model._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        model._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            str(model_path)
        )

        if not hasattr(model.config, "model_repo") or model.config.model_repo is None:
            try:
                index = model_path.parts.index("hub")
                model.config.model_repo = (
                    model_path.parts[index + 1]
                    .replace("models--", "")
                    .replace("--", "/")
                )
            except (ValueError, IndexError):
                model.config.model_repo = str(model_path)

        return model

    def _preprocess_audio(
        self,
        audio: Union[str, mx.array, np.ndarray],
    ) -> Tuple[mx.array, mx.array, int]:
        """Preprocess audio for the model."""
        from mlx_audio.stt.utils import load_audio

        if isinstance(audio, str):
            audio = load_audio(audio)

        audio_np = np.array(audio) if isinstance(audio, mx.array) else audio

        audio_inputs = self._feature_extractor(
            audio_np,
            sampling_rate=16000,
            return_attention_mask=True,
            truncation=False,
            padding=True,
            return_tensors="np",
        )
        input_features = mx.array(audio_inputs["input_features"])
        feature_attention_mask = mx.array(audio_inputs["attention_mask"])

        audio_lengths = feature_attention_mask.sum(axis=-1)
        aftercnn_lens = _get_feat_extract_output_lengths(audio_lengths)
        num_audio_tokens = int(aftercnn_lens[0].item())

        return input_features, feature_attention_mask, num_audio_tokens

    def align(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "English",
    ) -> Union[ForcedAlignResult, List[ForcedAlignResult]]:
        """Run forced alignment for audio and text.

        Args:
            audio: Audio input(s) - file path, array, or list of these.
            text: Transcript(s) for alignment.
            language: Language(s) for each sample (e.g., "Chinese", "English", "Japanese", "Korean").

        Returns:
            ForcedAlignResult or list of results with word-level timestamps.
        """
        from mlx_audio.stt.utils import load_audio

        if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
            raise RuntimeError(
                "Tokenizer/FeatureExtractor not initialized. Call post_load_hook first."
            )

        # Normalize inputs to lists
        single_input = not isinstance(audio, list)
        audios = [audio] if single_input else audio
        texts = [text] if isinstance(text, str) else text
        languages = [language] if isinstance(language, str) else language

        if len(languages) == 1 and len(audios) > 1:
            languages = languages * len(audios)

        if not (len(audios) == len(texts) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: audio={len(audios)}, text={len(texts)}, language={len(languages)}"
            )

        results: List[ForcedAlignResult] = []

        # Process each sample
        for audio_input, txt, lang in zip(audios, texts, languages):
            # Load audio if needed
            if isinstance(audio_input, str):
                audio_input = load_audio(audio_input)
            audio_np = (
                np.array(audio_input)
                if isinstance(audio_input, mx.array)
                else audio_input
            )

            # Preprocess audio
            input_features, feature_attention_mask, num_audio_tokens = (
                self._preprocess_audio(audio_np)
            )

            # Encode text with timestamp tokens
            word_list, aligner_input_text = self.aligner_processor.encode_timestamp(
                txt, lang
            )

            # Replace single audio_pad with correct number
            aligner_input_text = aligner_input_text.replace(
                "<|audio_pad|>", "<|audio_pad|>" * num_audio_tokens
            )

            # Tokenize input
            input_ids = self._tokenizer.encode(
                aligner_input_text, return_tensors="np", add_special_tokens=False
            )
            input_ids = mx.array(input_ids)

            # Forward pass
            logits = self(
                input_ids,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )
            mx.eval(logits)

            # Get predicted tokens
            output_ids = mx.argmax(logits, axis=-1)

            # Extract timestamps at timestamp token positions
            timestamp_token_id = self.config.timestamp_token_id
            timestamp_segment_time = self.config.timestamp_segment_time

            input_ids_flat = input_ids[0] if input_ids.ndim > 1 else input_ids
            output_ids_flat = output_ids[0] if output_ids.ndim > 1 else output_ids

            # Convert to numpy for boolean indexing (not supported in MLX)
            input_ids_np = np.array(input_ids_flat)
            output_ids_np = np.array(output_ids_flat)

            # Find positions where input has timestamp token
            timestamp_mask = input_ids_np == timestamp_token_id
            masked_output = output_ids_np[timestamp_mask]

            # Convert to milliseconds
            timestamp_ms = masked_output * timestamp_segment_time

            # Parse timestamps
            timestamp_output = self.aligner_processor.parse_timestamp(
                word_list, timestamp_ms
            )

            # Convert to seconds
            items = []
            for it in timestamp_output:
                items.append(
                    ForcedAlignItem(
                        text=str(it["text"]),
                        start_time=round(it["start_time"] / 1000.0, 3),
                        end_time=round(it["end_time"] / 1000.0, 3),
                    )
                )

            results.append(ForcedAlignResult(items=items))

            # Clear cache between samples
            mx.clear_cache()

        return results[0] if single_input else results

    def get_supported_languages(self) -> Optional[List[str]]:
        """List supported language names for the current model.

        Returns:
            Sorted list of supported languages, or None.
        """
        if hasattr(self.config, "support_languages") and self.config.support_languages:
            return sorted({str(x).lower() for x in self.config.support_languages})
        return None


# Alias for model loading compatibility
Model = ForcedAlignerModel

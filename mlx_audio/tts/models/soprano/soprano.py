"""
Soprano TTS Model for MLX.
Ultra-lightweight text-to-speech with ~80M parameters.
Ported from https://github.com/ekwek1/soprano
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelConfig
from mlx_lm.models.qwen3 import Qwen3Model
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from ..base import BaseModelArgs, GenerationResult
from .decoder import SopranoDecoder
from .text import clean_text


@dataclass
class DecoderConfig(BaseModelArgs):
    """Configuration for Soprano decoder."""

    # Decoder config
    decoder_num_layers: int = 8
    decoder_dim: int = 512
    decoder_intermediate_dim: int = 1536
    hop_length: int = 512
    n_fft: int = 2048
    upscale: int = 4
    dw_kernel: int = 3

    token_size: int = 2048  # Samples per audio token
    receptive_field: int = 4  # Decoder receptive fiel


@dataclass
class ModelConfig(Qwen3ModelConfig):
    sample_rate: int = 32000
    decoder_config: DecoderConfig = None

    def __post_init__(self):
        if self.decoder_config is None:
            self.decoder_config = DecoderConfig()


class SopranoModel(Qwen3Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        out = super().__call__(input_ids, cache)
        if self.config.tie_word_embeddings:
            out = self.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out


class Model(nn.Module):
    """Main Soprano TTS model.

    Combines a Qwen3-based language model with a Vocos-based decoder
    for ultra-fast text-to-speech synthesis.
    """

    def __init__(self, config: ModelConfig, tokenizer=None):
        super().__init__()
        self.config = (
            ModelConfig.from_dict(config) if isinstance(config, dict) else config
        )
        self.tokenizer = tokenizer
        self._stop_token_id = None

        # Initialize LM
        self.language_model = SopranoModel(self.config)

        # Initialize Decoder
        self.decoder = SopranoDecoder(
            num_input_channels=self.config.hidden_size,
            decoder_num_layers=self.config.decoder_config.decoder_num_layers,
            decoder_dim=self.config.decoder_config.decoder_dim,
            decoder_intermediate_dim=self.config.decoder_config.decoder_intermediate_dim,
            hop_length=self.config.decoder_config.hop_length,
            n_fft=self.config.decoder_config.n_fft,
            upscale=self.config.decoder_config.upscale,
            dw_kernel=self.config.decoder_config.dw_kernel,
        )

    def post_load_hook(self, model_path: Path) -> "Model":
        """Post-load hook to initialize tokenizer."""

        if self.tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.tokenizer = tokenizer

        # Set stop token ID (encode [STOP] and get the token ID)
        stop_tokens = self.tokenizer.encode("[STOP]", add_special_tokens=False)

        if self.tokenizer.pad_token_id is not None:
            self._stop_token_id = self.tokenizer.pad_token_id
        elif stop_tokens:
            self._stop_token_id = stop_tokens[0]
        else:
            raise ValueError("Stop token not found in tokenizer")
        return self

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "ekwek/Soprano-80M",
    ) -> "Model":
        """Load pre-trained Soprano model.

        Args:
            model_name: HuggingFace model name or local path.

        Returns:
            Loaded Soprano model.
        """
        path = Path(model_name)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=model_name,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "decoder.pth",
                        "tokenizer*",
                        "special_tokens*",
                        "vocab*",
                    ],
                )
            )

        # Load tokenizer

        # Load config
        import json

        config_path = path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        # Map HF config to our config
        config = ModelConfig.from_dict(config_dict)

        model = cls(config)
        model.post_load_hook(path)

        # Load LM weights
        lm_weights_path = path / "model.safetensors"
        if lm_weights_path.exists():
            lm_weights = mx.load(str(lm_weights_path))
            # Map weights to our structure
            mapped_weights = model.sanitize(lm_weights)
            model.language_model.load_weights(
                list(mapped_weights.items()), strict=False
            )

        # Load decoder weights
        decoder_weights_path = path / "decoder.pth"
        if decoder_weights_path.exists():
            import torch

            decoder_weights_pt = torch.load(
                str(decoder_weights_path), map_location="cpu", weights_only=True
            )
            decoder_weights = model._convert_decoder_weights(decoder_weights_pt)
            model.decoder.load_weights(list(decoder_weights.items()), strict=False)

        model.eval()
        return model

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}
        for k, v in weights.items():
            k = k.replace("model.", "") if k.startswith("model.") else k
            sanitized[k] = v
        return sanitized

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @property
    def layers(self):
        return self.language_model.layers

    def _convert_decoder_weights(self, pt_weights: dict) -> dict:
        """Convert PyTorch decoder weights to MLX format."""
        mlx_weights = {}
        for k, v in pt_weights.items():
            # Convert tensor to numpy then to mlx
            np_v = v.cpu().numpy()
            mx_v = mx.array(np_v)

            # Handle Conv1d weight transposition (PyTorch: out, in, kernel -> MLX: in, kernel, out)
            if "dwconv.weight" in k or "embed.weight" in k:
                # Conv1d weights need to be transposed
                # PyTorch: (out_channels, in_channels, kernel_size)
                # MLX: (out_channels, kernel_size, in_channels)
                if len(mx_v.shape) == 3:
                    mx_v = mx.transpose(mx_v, (0, 2, 1))

            mlx_weights[k] = mx_v
        return mlx_weights

    def _preprocess_text(
        self, texts: List[str], min_length: int = 30
    ) -> List[Tuple[str, int, int]]:
        """Preprocess text for generation.

        Args:
            texts: List of input texts.
            min_length: Minimum sentence length.

        Returns:
            List of (prompt, text_idx, sentence_idx) tuples.
        """
        res = []
        for text_idx, text in enumerate(texts):
            text = text.strip()
            cleaned_text = clean_text(text)
            sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
            processed = [{"text": s, "text_idx": text_idx} for s in sentences]

            if min_length > 0 and len(processed) > 1:
                merged = []
                i = 0
                while i < len(processed):
                    cur = processed[i]
                    if len(cur["text"]) < min_length:
                        if merged:
                            merged[-1]["text"] = (
                                merged[-1]["text"] + " " + cur["text"]
                            ).strip()
                        else:
                            if i + 1 < len(processed):
                                processed[i + 1]["text"] = (
                                    cur["text"] + " " + processed[i + 1]["text"]
                                ).strip()
                            else:
                                merged.append(cur)
                    else:
                        merged.append(cur)
                    i += 1
                processed = merged

            sentence_idxes = {}
            for item in processed:
                if item["text_idx"] not in sentence_idxes:
                    sentence_idxes[item["text_idx"]] = 0
                res.append(
                    (
                        f'[STOP][TEXT]{item["text"]}[START]',
                        item["text_idx"],
                        sentence_idxes[item["text_idx"]],
                    )
                )
                sentence_idxes[item["text_idx"]] += 1
        return res

    def _tokenize(self, text: str) -> mx.array:
        """Tokenize text using the HuggingFace tokenizer."""
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer not initialized. Use from_pretrained() to load the model."
            )
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return mx.array(tokens, dtype=mx.int32)

    def _forward_with_hidden_states(
        self,
        input_ids: mx.array,
        cache: Optional[List[KVCache]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass that returns both logits and hidden states.

        Args:
            input_ids: Input token IDs.
            cache: KV cache for incremental decoding.

        Returns:
            Tuple of (logits, hidden_states).
        """
        # Access the internal model components (Qwen3Model has .model attribute)
        model = self.language_model

        h = model.embed_tokens(input_ids)

        if cache is None:
            cache = [None] * len(model.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(model.layers, cache):
            h = layer(h, mask=mask, cache=c)

        # Get hidden states before lm_head
        hidden_states = model.norm(h)

        # Compute logits
        logits = self.language_model.lm_head(hidden_states)

        return logits, hidden_states

    def stream_generate(
        self,
        input_ids: mx.array,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        **kwargs,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """Stream generate tokens and hidden states.

        Args:
            input_ids: Input token IDs of shape (seq_len,) or (1, seq_len).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.

        Yields:
            Tuple of (token, hidden_state) for each generated token.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]  # Add batch dimension

        # Create KV cache
        cache = [KVCache() for _ in range(self.config.num_hidden_layers)]

        # Prefill - get both logits and hidden states
        logits, hidden_states = self._forward_with_hidden_states(input_ids, cache)
        mx.eval(logits, hidden_states)

        # Yield the last hidden state from prefill
        yield None, hidden_states[:, -1:, :]

        sampler = make_sampler(temperature, top_p)

        # Generate tokens
        for _ in range(max_tokens):
            # Sample next token
            next_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            else:
                next_token = sampler(next_logits)
                if next_token.ndim == 1:
                    next_token = next_token[:, None]

            # Check for stop token
            token_id = int(next_token[0, 0])
            if self._stop_token_id is not None and token_id == self._stop_token_id:
                break
            if self.tokenizer is not None and token_id == self.tokenizer.eos_token_id:
                break

            # Forward pass with new token - get both logits and hidden states
            logits, hidden_states = self._forward_with_hidden_states(next_token, cache)
            mx.eval(logits, hidden_states)

            yield next_token, hidden_states[:, -1:, :]

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.95,
        split_pattern: str = "\n",
        max_tokens: int = 512,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        Args:
            text: Input text to synthesize.
            voice: Voice name (not used in base Soprano).
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            split_pattern: Pattern to split text into segments.
            max_tokens: Maximum tokens per segment.
            verbose: Whether to print progress.

        Yields:
            GenerationResult for each segment.
        """
        _ = voice  # Unused in base Soprano

        prompt = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = prompt.split(split_pattern)

        for segment_idx, segment_text in enumerate(prompts):
            if not segment_text.strip():
                continue

            time_start = time.perf_counter()

            # Preprocess text
            sentence_data = self._preprocess_text([segment_text])

            # Generate for each sentence
            audio_parts = []
            total_tokens = 0

            for prompt_text, _, _ in sentence_data:
                # Tokenize the prompt
                input_ids = self._tokenize(prompt_text)

                # Collect hidden states using stream_generate
                all_hidden_states = []
                token_count = 0

                for token, hidden_state in self.stream_generate(
                    input_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                ):
                    all_hidden_states.append(hidden_state)
                    token_count += 1

                total_tokens += token_count

                if token_count >= max_tokens and verbose:
                    print("Warning: Generation hit max tokens, possible hallucination.")

                # Stack hidden states: list of (1, 1, C) -> (1, N, C)
                hidden_states = mx.concatenate(all_hidden_states, axis=1)

                # Decode hidden states to audio
                audio = self.decoder(hidden_states)

                # Trim based on token count
                token_size = self.config.decoder_config.token_size
                audio_length = token_count * token_size - token_size
                if audio_length > 0:
                    audio = audio[0, -audio_length:]
                else:
                    audio = audio[0]
                audio_parts.append(audio)

            # Concatenate audio parts
            if len(audio_parts) > 1:
                audio = mx.concatenate(audio_parts)
            else:
                audio = audio_parts[0]

            time_end = time.perf_counter()

            samples = audio.shape[0]
            audio_duration_seconds = samples / self.sample_rate
            elapsed_time = time_end - time_start
            rtf = (
                elapsed_time / audio_duration_seconds
                if audio_duration_seconds > 0
                else 0
            )

            yield GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=total_tokens,
                audio_duration=self._format_duration(audio_duration_seconds),
                real_time_factor=rtf,
                prompt={
                    "tokens": total_tokens,
                    "tokens-per-sec": (
                        round(total_tokens / elapsed_time, 2) if elapsed_time > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                    ),
                },
                processing_time_seconds=elapsed_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            mx.clear_cache()

    def _format_duration(self, seconds: float) -> str:
        """Format duration in HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

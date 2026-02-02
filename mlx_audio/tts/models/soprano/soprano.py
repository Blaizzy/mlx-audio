# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import re
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Qwen3Model
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from ..base import GenerationResult
from .config import ModelConfig
from .decoder import SopranoDecoder
from .text import clean_text


class SopranoModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        out = self.model(input_ids, cache)
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def norm(self):
        return self.model.norm


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
            input_kernel_size=self.config.decoder_config.input_kernel_size,
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

        # Load config
        import json

        config_path = path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        # Map HF config to our config
        config = ModelConfig.from_dict(config_dict)

        model = cls(config)
        model.post_load_hook(path)

        weights_path = path / "model.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            # Map weights to our structure
            mapped_weights = model.sanitize(weights)
            model.load_weights(list(mapped_weights.items()), strict=False)
            mx.eval(model.parameters())

        model.eval()
        return model

    def sanitize(self, weights: dict) -> dict:
        m_path = getattr(self.config, "model_path", None)
        d_path = getattr(self.config.decoder_config, "decoder_path", "decoder.pth")

        # Load auxiliary decoder weights if present
        if m_path and (Path(m_path) / d_path).exists():
            import torch

            decoder_path = Path(m_path) / d_path
            print(f"[INFO] Loading decoder weights from {decoder_path}")

            for k, v in torch.load(decoder_path, map_location="cpu").items():
                if "window" in k:
                    continue
                v = mx.array(v.numpy().astype("float32"))
                # Align torch Conv1d (B, C, L) weights with MLX Conv1d (B, L, C)
                if "weight" in k and ("embed" in k or "dwconv" in k):
                    v = v.transpose(0, 2, 1)

                weights[f"decoder.{k}"] = v

        sanitized = {}
        for k, v in weights.items():
            if "window" in k:
                continue

            # Map weights to unified structure: LLM under 'language_model' and Vocos under 'decoder'
            if k.startswith("model."):
                new_key = f"language_model.{k}"
            elif k.startswith("lm_head."):
                new_key = f"language_model.{k}"
            elif k.startswith("decoder."):
                new_key = k
            elif k.startswith("language_model."):
                new_key = k
            elif any(x in k for x in ["embed_tokens", "layers.", "norm.", "lm_head"]):
                # Handle weights saved without 'model.' prefix in some HF repos
                if "lm_head" in k:
                    new_key = f"language_model.{k}"
                else:
                    new_key = f"language_model.model.{k}"
            else:
                new_key = k

            # Decoder weights must be float32 for DSP operations (ISTFT)
            if new_key.startswith("decoder.") and v.dtype != mx.uint32:
                v = v.astype(mx.float32)

            sanitized[new_key] = v

        return sanitized

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @property
    def layers(self):
        return self.language_model.layers

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

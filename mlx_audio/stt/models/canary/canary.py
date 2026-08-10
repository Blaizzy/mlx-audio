import base64
import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import STTOutput
from .config import DecoderConfig, EncoderConfig, ModelConfig, PreprocessorConfig
from .decoder import CanaryDecoder
from .tokenizer import CanaryTokenizer


class CanaryEncoder(nn.Module):
    """FastConformer encoder for Canary model.

    Reuses the Conformer implementation from parakeet.
    For canary-1b-v2, encoder output dim matches decoder dim (1024)
    so no projection is needed (Identity).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        from mlx_audio.stt.models.parakeet.conformer import Conformer, ConformerArgs

        enc_cfg = config.encoder
        conformer_args = ConformerArgs(
            feat_in=enc_cfg.feat_in,
            n_layers=enc_cfg.n_layers,
            d_model=enc_cfg.d_model,
            n_heads=enc_cfg.n_heads,
            ff_expansion_factor=enc_cfg.ff_expansion_factor,
            subsampling_factor=enc_cfg.subsampling_factor,
            self_attention_model=enc_cfg.self_attention_model,
            subsampling=enc_cfg.subsampling,
            conv_kernel_size=enc_cfg.conv_kernel_size,
            subsampling_conv_channels=enc_cfg.subsampling_conv_channels,
            pos_emb_max_len=enc_cfg.pos_emb_max_len,
            causal_downsampling=enc_cfg.causal_downsampling,
            use_bias=enc_cfg.use_bias,
            xscaling=enc_cfg.xscaling,
            subsampling_conv_chunking_factor=enc_cfg.subsampling_conv_chunking_factor,
        )
        self.conformer = Conformer(conformer_args)

        if enc_cfg.d_model != config.enc_output_dim:
            self.projection = nn.Linear(enc_cfg.d_model, config.enc_output_dim)
        else:
            self.projection = None

    def __call__(self, mel: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        """Encode mel spectrogram."""
        enc_out, enc_len = self.conformer(mel, lengths)
        if self.projection is not None:
            enc_out = self.projection(enc_out)
        return enc_out, enc_len


class Model(nn.Module):
    """Canary-1B-v2 model for multilingual ASR.

    Supports transcription and translation between 25+ languages.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config

        self.encoder = CanaryEncoder(config)
        self.decoder = CanaryDecoder(
            config=config.transf_decoder,
            vocab_size=config.vocab_size,
            d_model=config.enc_output_dim,
        )
        self._tokenizer: Optional[CanaryTokenizer] = None

    @property
    def sample_rate(self) -> int:
        return self.config.preprocessor.sample_rate

    def _preprocess_audio(self, audio) -> mx.array:
        """Preprocess audio to mel spectrogram.

        Uses the parakeet audio preprocessing which matches NeMo's
        per-feature normalization.
        """
        from mlx_audio.stt.models.parakeet.audio import (
            PreprocessArgs as PPreprocessArgs,
        )
        from mlx_audio.stt.models.parakeet.audio import log_mel_spectrogram
        from mlx_audio.stt.utils import load_audio

        pp = self.config.preprocessor

        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio), sr=pp.sample_rate)
        elif isinstance(audio, np.ndarray):
            audio = mx.array(audio)

        if audio.ndim == 3:
            return audio

        preprocess_args = PPreprocessArgs(
            sample_rate=pp.sample_rate,
            normalize=pp.normalize,
            window_size=pp.window_size,
            window_stride=pp.window_stride,
            window=pp.window,
            features=pp.features,
            n_fft=pp.n_fft,
            dither=pp.dither,
            pad_to=pp.pad_to,
            pad_value=pp.pad_value,
            preemph=pp.preemph,
        )

        mel = log_mel_spectrogram(audio, preprocess_args)
        return mel

    def _encode_audio(self, mel: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Run encoder on mel spectrogram.

        Returns:
            enc_out: Encoder output, shape (B, T, D)
            enc_len: Output lengths, shape (B,)
            enc_mask: Encoder mask, shape (B, T)
        """
        B, T, _ = mel.shape
        lengths = mx.array([T] * B, dtype=mx.int32)
        enc_out, enc_len = self.encoder(mel, lengths)
        mx.eval(enc_out, enc_len)

        max_len = enc_out.shape[1]
        arange = mx.arange(max_len)
        enc_mask = (arange[None, :] < enc_len[:, None]).astype(mx.float32)

        return enc_out, enc_len, enc_mask

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 200,
        source_lang: str = "en",
        target_lang: str = "en",
        use_pnc: bool = True,
        temperature: float = 0.0,
        verbose: bool = False,
        stream: bool = False,
        dtype: mx.Dtype = mx.bfloat16,
        **kwargs,
    ) -> STTOutput:
        """Generate transcription/translation from audio.

        Args:
            audio: Audio path, waveform array, or mel spectrogram
            max_tokens: Maximum tokens to generate
            source_lang: Source language code (e.g., "en", "de", "fr")
            target_lang: Target language code
            use_pnc: Enable punctuation and capitalization
            temperature: Sampling temperature (0 = greedy)
            verbose: Print during generation
            stream: Not supported yet (returns STTOutput)
            dtype: Data type for computation

        Returns:
            STTOutput with transcription text
        """
        kwargs.pop("generation_stream", None)
        language = kwargs.pop("language", None)
        if language is not None:
            source_lang = language
            target_lang = language

        start_time = time.time()

        mel = self._preprocess_audio(audio)
        if mel.dtype != dtype:
            mel = mel.astype(dtype)

        enc_out, enc_len, enc_mask = self._encode_audio(mel)

        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Use post_load_hook or set _tokenizer."
            )

        prompt_tokens = self._tokenizer.build_prompt_tokens(
            source_lang=source_lang,
            target_lang=target_lang,
            use_pnc=use_pnc,
        )

        if verbose:
            print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

        eos_id = self._tokenizer.eos_id
        generated_tokens = []
        cache = None

        prompt_ids = mx.array([prompt_tokens], dtype=mx.int32)
        logits, cache = self.decoder(
            prompt_ids, enc_out, encoder_mask=enc_mask, cache=cache, start_pos=0
        )
        mx.eval(logits)

        if temperature > 0:
            next_token = int(mx.random.categorical(logits[:, -1, :] / temperature))
        else:
            next_token = int(logits[:, -1, :].argmax())

        if next_token == eos_id:
            generated_tokens = []
        else:
            generated_tokens.append(next_token)

            for step in range(max_tokens - 1):
                token_ids = mx.array([[next_token]], dtype=mx.int32)
                logits, cache = self.decoder(
                    token_ids,
                    enc_out,
                    encoder_mask=enc_mask,
                    cache=cache,
                    start_pos=len(prompt_tokens) + step,
                )
                mx.eval(logits)

                if temperature > 0:
                    next_token = int(
                        mx.random.categorical(logits[:, -1, :] / temperature)
                    )
                else:
                    next_token = int(logits[:, -1, :].argmax())

                if next_token == eos_id:
                    break
                generated_tokens.append(next_token)

        text = self._tokenizer.decode(generated_tokens)

        end_time = time.time()
        total_time = end_time - start_time

        if verbose:
            print(f"Generated {len(generated_tokens)} tokens")
            print(f"Text: {text}")

        return STTOutput(
            text=text.strip(),
            segments=[{"text": text.strip(), "start": 0.0, "end": 0.0}],
            language=source_lang,
            prompt_tokens=len(prompt_tokens),
            generation_tokens=len(generated_tokens),
            total_tokens=len(prompt_tokens) + len(generated_tokens),
            total_time=total_time,
            prompt_tps=len(prompt_tokens) / total_time if total_time > 0 else 0,
            generation_tps=len(generated_tokens) / total_time if total_time > 0 else 0,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map checkpoint weight names to this model's MLX parameter names.

        Three on-disk layouts are supported:

        * **Already-sanitized** – internal MLX parameter names produced after a
          previous sanitize pass (e.g. a re-saved or convert-script output). Keys
          start with ``decoder.blocks.``; no remapping needed.
        * **NeMo-native** – raw NeMo names (``transf_decoder._decoder.layers.*``,
          ``log_softmax.mlp.layer0.*``) with PyTorch conv layout that needs to be
          transposed to MLX's ``(out, *kernel, in)`` order.
        * **MLX-native** – community conversions that already use MLX tensor layouts
          and flattened names (``transf_decoder.layers.N.first_sub_layer.linear_q``,
          ``head.classifier``). These must *not* be transposed again.
        """
        is_already_sanitized = any(k.startswith("decoder.blocks.") for k in weights)
        if is_already_sanitized:
            return dict(weights)

        is_mlx_native = "head.classifier.weight" in weights or any(
            k.startswith("transf_decoder.layers.") for k in weights
        )
        if is_mlx_native:
            return self._sanitize_mlx_native(weights)
        return self._sanitize_nemo(weights)

    def _sanitize_mlx_native(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap an already-MLX-layout Canary checkpoint (no conv transpose)."""

        def map_sublayer(sub: str) -> str:
            attn = (
                ("linear_q.", "q_proj."),
                ("linear_k.", "k_proj."),
                ("linear_v.", "v_proj."),
                ("linear_out.", "out_proj."),
            )
            if sub.startswith("first_sub_layer."):
                inner = sub[len("first_sub_layer.") :]
                for a, b in attn:
                    if inner.startswith(a):
                        inner = b + inner[len(a) :]
                        break
                return "self_attn." + inner
            if sub.startswith("second_sub_layer."):
                inner = sub[len("second_sub_layer.") :]
                for a, b in attn:
                    if inner.startswith(a):
                        inner = b + inner[len(a) :]
                        break
                return "cross_attn." + inner
            if sub.startswith("third_sub_layer."):
                inner = sub[len("third_sub_layer.") :]
                if inner.startswith("linear1."):
                    inner = "ff1." + inner[len("linear1.") :]
                elif inner.startswith("linear2."):
                    inner = "ff2." + inner[len("linear2.") :]
                else:
                    warnings.warn(
                        f"Unrecognized third_sub_layer key {inner!r}; passing through unchanged. "
                        "The resulting weight key may not match any model parameter.",
                        RuntimeWarning,
                        stacklevel=4,
                    )
                return inner
            if sub.startswith("layer_norm_1."):
                return "self_attn_norm." + sub[len("layer_norm_1.") :]
            if sub.startswith("layer_norm_2."):
                return "cross_attn_norm." + sub[len("layer_norm_2.") :]
            if sub.startswith("layer_norm_3."):
                return "ff_norm." + sub[len("layer_norm_3.") :]
            warnings.warn(
                f"Unrecognized sub-layer key {sub!r} in MLX-native checkpoint; "
                "passing through unchanged. The resulting weight key may not match any model parameter.",
                RuntimeWarning,
                stacklevel=4,
            )
            return sub

        sanitized = {}
        for key, value in weights.items():
            if key.startswith("encoder."):
                new_key = "encoder.conformer." + key[len("encoder.") :]
            elif key.startswith("transf_decoder.token_embedding."):
                new_key = (
                    "decoder.embedding." + key[len("transf_decoder.token_embedding.") :]
                )
            elif key.startswith("transf_decoder.embedding_layer_norm."):
                new_key = (
                    "decoder.embedding_layer_norm."
                    + key[len("transf_decoder.embedding_layer_norm.") :]
                )
            elif key.startswith("transf_decoder.final_layer_norm."):
                new_key = (
                    "decoder.final_norm."
                    + key[len("transf_decoder.final_layer_norm.") :]
                )
            elif key.startswith("transf_decoder.layers."):
                rest = key[len("transf_decoder.layers.") :]
                layer_idx, sub_rest = rest.split(".", 1)
                new_key = f"decoder.blocks.{layer_idx}.{map_sublayer(sub_rest)}"
            elif key.startswith("head.classifier."):
                new_key = "decoder.output_proj." + key[len("head.classifier.") :]
            else:
                # Unknown / auxiliary tensors (e.g. encoder_decoder_proj) are dropped.
                continue
            sanitized[new_key] = value
        return sanitized

    def _sanitize_nemo(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map raw NeMo weight names to MLX weight names (PyTorch conv layout)."""
        sanitized = {}

        for key, value in weights.items():
            new_key = key

            # Encoder weights: encoder.* -> encoder.conformer.*
            if key.startswith("encoder.") and not key.startswith("encoder_decoder"):
                new_key = "encoder.conformer." + key[len("encoder.") :]

            elif key.startswith("encoder_decoder_proj."):
                continue

            elif key.startswith("transf_decoder._embedding.token_embedding."):
                new_key = key.replace(
                    "transf_decoder._embedding.token_embedding.", "decoder.embedding."
                )

            elif key.startswith("transf_decoder._embedding.position_embedding."):
                new_key = key.replace(
                    "transf_decoder._embedding.position_embedding.",
                    "decoder.position_embedding.",
                )

            elif key.startswith("transf_decoder._embedding.layer_norm."):
                new_key = key.replace(
                    "transf_decoder._embedding.layer_norm.",
                    "decoder.embedding_layer_norm.",
                )

            elif key.startswith("transf_decoder._decoder.layers."):
                rest = key[len("transf_decoder._decoder.layers.") :]
                parts = rest.split(".", 1)
                layer_idx = parts[0]
                sub_rest = parts[1]

                if sub_rest.startswith("first_sub_layer."):
                    inner = sub_rest[len("first_sub_layer.") :]
                    inner = inner.replace("query_net.", "self_attn.q_proj.")
                    inner = inner.replace("key_net.", "self_attn.k_proj.")
                    inner = inner.replace("value_net.", "self_attn.v_proj.")
                    inner = inner.replace("out_projection.", "self_attn.out_proj.")
                    new_key = f"decoder.blocks.{layer_idx}.{inner}"

                elif sub_rest.startswith("second_sub_layer."):
                    inner = sub_rest[len("second_sub_layer.") :]
                    inner = inner.replace("query_net.", "cross_attn.q_proj.")
                    inner = inner.replace("key_net.", "cross_attn.k_proj.")
                    inner = inner.replace("value_net.", "cross_attn.v_proj.")
                    inner = inner.replace("out_projection.", "cross_attn.out_proj.")
                    new_key = f"decoder.blocks.{layer_idx}.{inner}"

                elif sub_rest.startswith("third_sub_layer."):
                    inner = sub_rest[len("third_sub_layer.") :]
                    inner = inner.replace("dense_in.", "ff1.")
                    inner = inner.replace("dense_out.", "ff2.")
                    new_key = f"decoder.blocks.{layer_idx}.{inner}"

                elif sub_rest.startswith("layer_norm_1."):
                    new_key = f"decoder.blocks.{layer_idx}.{sub_rest.replace('layer_norm_1.', 'self_attn_norm.')}"
                elif sub_rest.startswith("layer_norm_2."):
                    new_key = f"decoder.blocks.{layer_idx}.{sub_rest.replace('layer_norm_2.', 'cross_attn_norm.')}"
                elif sub_rest.startswith("layer_norm_3."):
                    new_key = f"decoder.blocks.{layer_idx}.{sub_rest.replace('layer_norm_3.', 'ff_norm.')}"

                else:
                    new_key = f"decoder.blocks.{layer_idx}.{sub_rest}"

            elif key.startswith("transf_decoder._decoder.final_layer_norm."):
                new_key = key.replace(
                    "transf_decoder._decoder.final_layer_norm.", "decoder.final_norm."
                )

            elif key.startswith("log_softmax.mlp.layer0."):
                new_key = key.replace("log_softmax.mlp.layer0.", "decoder.output_proj.")

            if "attn_dropout" in key or "layer_dropout" in key:
                continue
            if key == "log_softmax.mlp.log_softmax":
                continue
            if "num_batches_tracked" in key:
                continue

            if "conv" in new_key and "weight" in new_key and value.ndim >= 3:
                if value.ndim == 3:
                    value = mx.transpose(value, (0, 2, 1))
                elif value.ndim == 4:
                    value = mx.transpose(value, (0, 2, 3, 1))

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Load tokenizer after weights are loaded."""
        model_path = Path(model_path)

        sp_path = model_path / "tokenizer.model"
        tokens_path = model_path / "tokens.txt"

        if sp_path.exists():
            model._tokenizer = CanaryTokenizer(
                str(sp_path),
                str(tokens_path) if tokens_path.exists() else None,
            )
        elif tokens_path.exists():
            model._tokenizer = CanaryTokenizer.__new__(CanaryTokenizer)
            model._tokenizer.token2id = {}
            model._tokenizer.id2token = {}
            with open(tokens_path, encoding="utf-8") as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) == 2:
                        token, idx = fields[0], int(fields[1])
                        if line[0] == " ":
                            token = " " + token
                    elif len(fields) == 1:
                        token = " "
                        idx = int(fields[0])
                    else:
                        continue
                    model._tokenizer.token2id[token] = idx
                    model._tokenizer.id2token[idx] = token
        else:
            # Some conversions embed the SentencePiece model in config.json
            # (e.g. base64) rather than shipping a separate tokenizer.model file.
            proto = cls._load_embedded_tokenizer_proto(model_path)
            if proto is not None:
                model._tokenizer = CanaryTokenizer(model_proto=proto)
            else:
                warnings.warn(
                    f"No tokenizer found for model at '{model_path}'. "
                    "Looked for tokenizer.model, tokens.txt, and tokenizer.model_base64 in config.json. "
                    "Calls to generate() will raise RuntimeError.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return model

    @staticmethod
    def _load_embedded_tokenizer_proto(model_path: Path) -> Optional[bytes]:
        """Return raw SentencePiece bytes embedded in config.json, if present."""
        config_path = model_path / "config.json"
        if not config_path.exists():
            return None
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        tok = config.get("tokenizer")
        if not isinstance(tok, dict):
            return None
        b64 = tok.get("model_base64")
        if not b64:
            return None
        try:
            return base64.b64decode(b64)
        except ValueError as exc:
            import warnings

            warnings.warn(
                f"Failed to decode tokenizer.model_base64 from {config_path}: {exc}. "
                "Tokenizer will not be loaded from config.json.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None

    @classmethod
    def from_pretrained(cls, path_or_repo: str, *, dtype: mx.Dtype = mx.bfloat16):
        """Load model from a local directory or HuggingFace repo.

        .. deprecated::
            Use `mlx_audio.stt.load()` instead.
        """
        warnings.warn(
            "Model.from_pretrained() is deprecated. Use mlx_audio.stt.load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from mlx_audio.stt.utils import load

        return load(path_or_repo)

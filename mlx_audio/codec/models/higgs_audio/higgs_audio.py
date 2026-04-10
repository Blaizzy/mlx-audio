import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.utils import resample_audio

from .config import HiggsAudioConfig
from .dac import AcousticDecoder, AcousticEncoder, ResidualVectorQuantizer


class HiggsAudioTokenizer(nn.Module):
    """
    HiggsAudioV2 acoustic tokenizer.

    Decode path (tokens → waveform): quantizer → fc2 → acoustic_decoder  [MLX]
    Encode path (waveform → tokens): pure MLX semantic + acoustic fusion
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config: HiggsAudioConfig = config
        self.acoustic_encoder: nn.Module = AcousticEncoder()
        self.quantizer: nn.Module = ResidualVectorQuantizer()
        self.acoustic_decoder: nn.Module = AcousticDecoder()
        # Decode path: quantizer (1024-dim) → fc2 → decoder (256-dim)
        self.fc2: nn.Linear = nn.Linear(1024, 256, bias=True)
        self.semantic_model: nn.Module | None = None
        self.encoder_semantic: nn.Module | None = None
        self.fc: nn.Linear | None = None
        # Backward-compat attribute retained, but no longer used for encode().
        self._pt_tokenizer: Any | None = None

    def _init_encode_modules(self):
        from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig, Wav2Vec2Model

        if self.config.semantic_model_config is None:
            raise RuntimeError(
                "semantic_model_config is required to initialize encode modules"
            )

        from .semantic import SemanticEncoder

        semantic_config = ModelConfig.from_dict(self.config.semantic_model_config)
        hidden_size = semantic_config.hidden_size

        self.semantic_model = Wav2Vec2Model(semantic_config)
        self.encoder_semantic = SemanticEncoder(
            hidden_size=hidden_size,
            strides=self.config.strides,
            dilations=self.config.block_dilations,
            channel_ratios=self.config.channel_ratios,
            kernel_size=self.config.kernel_size,
            unit_kernel_size=self.config.unit_kernel_size,
        )
        fusion_dim = hidden_size + 256
        self.fc = nn.Linear(fusion_dim, fusion_dim, bias=True)

    def decode(self, tokens: mx.array) -> mx.array:
        """
        tokens: [T, 8] or [B, T, 8] int32
        Returns: [T*960] (1D) if 2D input, or [B, T*960, 1] if 3D input
        """
        squeeze = tokens.ndim == 2
        if squeeze:
            tokens = tokens[None]  # [1, T, 8]
        quantizer = cast(Any, self.quantizer)
        acoustic_decoder = cast(nn.Module, self.acoustic_decoder)

        z = quantizer.decode(tokens)  # [B, T, 1024]
        z = self.fc2(z)  # [B, T, 256]
        wav = acoustic_decoder(z)  # [B, T*960, 1]
        if squeeze:
            return wav[0, :, 0]  # [T*960]
        return wav  # [B, T*960, 1]

    def encode(self, waveform: mx.array) -> mx.array:
        """
        waveform: [B, T, 1] float32 at 24kHz
        Returns: [B, T', 8] int32 codebook tokens
        """
        if self.semantic_model is None:
            raise RuntimeError(
                "Encode modules are not initialized. Call _init_encode_modules() or "
                "load via HiggsAudioTokenizer.from_pretrained()."
            )

        waveform_np = np.asarray(waveform.astype(mx.float32))
        if waveform_np.ndim != 3 or waveform_np.shape[-1] != 1:
            raise ValueError("waveform must have shape [B, T, 1]")

        audio_24k = waveform_np[..., 0]
        resampled = [
            resample_audio(
                sample, self.config.sample_rate, self.config.semantic_sample_rate
            )
            for sample in audio_24k
        ]
        target_len = min(len(r) for r in resampled)
        audio_16k = np.stack([r[:target_len] for r in resampled], axis=0).astype(
            np.float32
        )
        hubert_pad = self.config.downsample_factor // 2
        audio_16k = np.pad(
            audio_16k, ((0, 0), (hubert_pad, hubert_pad)), mode="constant"
        )
        audio_16k = mx.array(audio_16k)

        semantic_model = cast(nn.Module, self.semantic_model)
        encoder_semantic = cast(nn.Module, self.encoder_semantic)
        fc = cast(nn.Linear, self.fc)

        semantic_outputs = cast(
            Any, semantic_model(audio_16k, output_hidden_states=True, return_dict=True)
        )
        hidden_states = mx.stack(list(semantic_outputs.hidden_states), axis=0)
        semantic_features = mx.mean(hidden_states, axis=0)
        dsf = self.config.semantic_downsample_factor
        if dsf > 1:
            semantic_features = semantic_features[:, ::dsf, :]
        semantic_features = encoder_semantic(semantic_features)

        acoustic_features = self.acoustic_encoder(waveform.astype(mx.float32))
        time_steps = min(semantic_features.shape[1], acoustic_features.shape[1])
        semantic_features = semantic_features[:, :time_steps, :]
        acoustic_features = acoustic_features[:, :time_steps, :]

        embeddings = mx.concatenate([acoustic_features, semantic_features], axis=-1)
        embeddings = fc(embeddings)
        quantizer = cast(Any, self.quantizer)
        return quantizer.encode(embeddings).astype(mx.int32)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Filter and transform checkpoint weights for MLX.

        Keeps: acoustic_encoder, acoustic_decoder, quantizer, fc2 (decode path)
               semantic_model, encoder_semantic, fc (encode path)
        Drops: decoder_semantic, fc1 (unused decode-semantic path)
               .embed_avg, .cluster_size, .inited (VQ bookkeeping)
        """
        keep_prefixes = (
            "acoustic_encoder.",
            "acoustic_decoder.",
            "quantizer.",
            "fc2.",
            "semantic_model.",
            "encoder_semantic.",
        )
        keep_exact = ("fc.weight", "fc.bias")
        drop_prefixes = ("decoder_semantic.", "fc1.")
        drop_suffixes = (".embed_avg", ".cluster_size", ".inited")

        result = {}
        for k, v in weights.items():
            # Explicit drops first
            if any(k.startswith(p) for p in drop_prefixes):
                continue
            if not (any(k.startswith(p) for p in keep_prefixes) or k in keep_exact):
                continue
            if any(k.endswith(s) for s in drop_suffixes):
                continue

            # === Semantic model (HuBERT/Wav2Vec2) weight transforms ===
            if k.startswith("semantic_model."):
                # Remap parametrized weight norm keys
                if ".parametrizations.weight.original0" in k:
                    k = k.replace(".parametrizations.weight.original0", ".weight_g")
                elif ".parametrizations.weight.original1" in k:
                    k = k.replace(".parametrizations.weight.original1", ".weight_v")
                # Transpose 3D conv weights: PyTorch [C_out, C_in, K] -> MLX [C_out, K, C_in]
                if v.ndim == 3 and (
                    k.endswith(".weight")
                    or k.endswith(".weight_g")
                    or k.endswith(".weight_v")
                ):
                    v = v.transpose(0, 2, 1)

            # === Encoder semantic (SemanticEncoder CNN) weight transforms ===
            elif k.startswith("encoder_semantic."):
                if v.ndim == 3 and k.endswith(".weight"):
                    v = v.transpose(0, 2, 1)

            # === Acoustic path weight transforms (existing logic) ===
            elif k.startswith(
                ("acoustic_encoder.", "acoustic_decoder.", "quantizer.", "fc2.")
            ):
                if k.endswith(".codebook.embed"):
                    k = k[: -len("embed")] + "weight"
                if k.endswith(".alpha") and v.ndim == 3:
                    v = v.transpose(0, 2, 1)
                elif v.ndim == 3 and k.endswith(".weight"):
                    if "conv_t" in k:
                        v = v.transpose(1, 2, 0)
                    else:
                        v = v.transpose(0, 2, 1)

            result[k] = v
        return result

    @classmethod
    def from_pretrained(cls, model_path: str) -> "HiggsAudioTokenizer":
        """
        Load from k2-fsa/OmniVoice local directory.
        Expects: <model_path>/audio_tokenizer/config.json
                 <model_path>/audio_tokenizer/model.safetensors

        Initializes encode-path modules (HuBERT, SemanticEncoder, fc) when
        semantic_model_config is present in the checkpoint config.
        """
        config_path = Path(model_path) / "audio_tokenizer" / "config.json"
        weights_path = Path(model_path) / "audio_tokenizer" / "model.safetensors"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        config = HiggsAudioConfig.from_dict(json.loads(config_path.read_text()))
        inst = cls(config)
        if config.semantic_model_config is not None:
            inst._init_encode_modules()
        raw = cast(dict[str, mx.array], mx.load(str(weights_path)))
        sanitized = inst.sanitize(raw)
        inst.load_weights(list(sanitized.items()))
        mx.eval(inst.parameters())

        return inst

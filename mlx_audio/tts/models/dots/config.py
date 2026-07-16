"""Config dataclasses for the dots.tts MLX port.

Pure stdlib only — NO torch, NO mlx. This module is imported by ``convert.py``
(which runs under the torch *oracle* venv) as well as by the MLX runtime, so it
must stay framework-agnostic.

Values are the locked constants derived from the checkpoint's ``config.json`` /
``llm_config.json`` (Task 0). ``ModelConfig.from_checkpoint`` re-reads those JSON
files so the runtime config tracks the checkpoint, with these dataclass defaults
serving as the cross-check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiTConfig:
    """Velocity-field predictor (flow-matching DiT)."""

    num_layers: int = 18
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    head_dim: int = 64
    modulation: bool = True
    qkv_bias: bool = False
    qk_norm: bool = True
    norm_layer: str = "RMSNorm"
    alibi_bias: bool = False
    rotary_bias: bool = True
    rotary_theta: float = 10000.0
    attn_dropout: float = 0.0
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "DiTConfig":
        hidden = int(d.get("hidden_size", 1024))
        heads = int(d.get("num_heads", 16))
        return cls(
            num_layers=int(d.get("num_layers", 18)),
            num_heads=heads,
            hidden_size=hidden,
            ffn_hidden_size=int(d.get("ffn_hidden_size", 4096)),
            head_dim=hidden // heads,
            modulation=bool(d.get("modulation", True)),
            qkv_bias=bool(d.get("qkv_bias", False)),
            qk_norm=bool(d.get("qk_norm", True)),
            norm_layer=str(d.get("norm_layer", "RMSNorm")),
            alibi_bias=bool(d.get("alibi_bias", False)),
            rotary_bias=bool(d.get("rotary_bias", True)),
            rotary_theta=float(d.get("rotary_theta", 10000.0)),
            attn_dropout=float(d.get("attn_dropout", 0.0)),
            dropout=float(d.get("dropout", 0.0)),
        )


@dataclass
class EncoderConfig:
    """Semantic patch encoder.

    NOTE: the checkpoint's ``config.json`` carries ``qk_norm``/``rotary_*`` flags
    for the encoder, but the upstream encoder code does NOT consume them — they
    are forced OFF here (Task 0 finding). ``causal=True``.
    """

    num_layers: int = 24
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    input_dim: int = 128
    modulation: bool = False
    qkv_bias: bool = False
    norm_layer: str = "RMSNorm"
    causal: bool = True
    # Flags present in config.json but UNUSED by the encoder code -> OFF.
    qk_norm: bool = False
    rotary_bias: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        return cls(
            num_layers=int(d.get("num_layers", 24)),
            num_heads=int(d.get("num_heads", 16)),
            hidden_size=int(d.get("hidden_size", 1024)),
            ffn_hidden_size=int(d.get("ffn_hidden_size", 4096)),
            input_dim=int(d.get("input_dim", 128)),
            modulation=bool(d.get("modulation", False)),
            qkv_bias=bool(d.get("qkv_bias", False)),
            norm_layer=str(d.get("norm_layer", "RMSNorm")),
            causal=bool(d.get("causal", True)),
            # Forced OFF regardless of what config.json says.
            qk_norm=False,
            rotary_bias=False,
        )


@dataclass
class VocoderConfig:
    """AudioVAE vocoder (BigVGAN-style decoder + AudioVAE encoder)."""

    sample_rate: int = 48000
    causal: bool = True
    activation: str = "snakebeta"
    snake_logscale: bool = True
    latent_dim: int = 128
    upsample_rates: list[int] = field(default_factory=lambda: [10, 6, 4, 2, 2, 2])
    upsample_kernel_sizes: list[int] = field(
        default_factory=lambda: [20, 12, 8, 4, 4, 4]
    )
    upsample_initial_channel: int = 1536
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    downsample_rates: list[int] = field(default_factory=lambda: [2, 2, 2, 4, 6, 10])
    downsample_channels: list[int] = field(
        default_factory=lambda: [12, 24, 48, 96, 192, 384, 768]
    )
    use_bias_at_final: bool = False
    use_tanh_at_final: bool = False
    mi_num_layers: int = 4
    num_decoder_lookahead: int = 2
    # Encode-path (audio_encoder) settings.
    causal_encoder: bool = True
    num_encoder_lookahead: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "VocoderConfig":
        return cls(
            sample_rate=int(d.get("sample_rate", 48000)),
            causal=bool(d.get("causal", True)),
            activation=str(d.get("activation", "snakebeta")),
            snake_logscale=bool(d.get("snake_logscale", True)),
            latent_dim=int(d.get("latent_dim", 128)),
            upsample_rates=list(d.get("upsample_rates", [10, 6, 4, 2, 2, 2])),
            upsample_kernel_sizes=list(
                d.get("upsample_kernel_sizes", [20, 12, 8, 4, 4, 4])
            ),
            upsample_initial_channel=int(d.get("upsample_initial_channel", 1536)),
            resblock=str(d.get("resblock", "1")),
            resblock_kernel_sizes=list(d.get("resblock_kernel_sizes", [3, 7, 11])),
            resblock_dilation_sizes=[
                list(x) for x in d.get("resblock_dilation_sizes", [[1, 3, 5]] * 3)
            ],
            downsample_rates=list(d.get("downsample_rates", [2, 2, 2, 4, 6, 10])),
            downsample_channels=list(
                d.get("downsample_channels", [12, 24, 48, 96, 192, 384, 768])
            ),
            use_bias_at_final=bool(d.get("use_bias_at_final", False)),
            use_tanh_at_final=bool(d.get("use_tanh_at_final", False)),
            mi_num_layers=int(d.get("mi_num_layers", 4)),
            num_decoder_lookahead=int(d.get("num_decoder_lookahead", 2)),
            causal_encoder=bool(d.get("causal_encoder", True)),
            num_encoder_lookahead=int(d.get("num_encoder_lookahead", 2)),
        )


@dataclass
class LLMConfig:
    """Qwen2.5 backbone."""

    vocab_size: int = 151672
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 131072
    bos_token_id: int = 151643
    eos_token_id: int = 151643

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        hidden = int(d.get("hidden_size", 1536))
        heads = int(d.get("num_attention_heads", 12))
        return cls(
            vocab_size=int(d.get("vocab_size", 151672)),
            hidden_size=hidden,
            intermediate_size=int(d.get("intermediate_size", 8960)),
            num_hidden_layers=int(d.get("num_hidden_layers", 28)),
            num_attention_heads=heads,
            num_key_value_heads=int(d.get("num_key_value_heads", 2)),
            head_dim=int(d.get("head_dim", hidden // heads)),
            rope_theta=float(d.get("rope_theta", 1e6)),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            hidden_act=str(d.get("hidden_act", "silu")),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", True)),
            max_position_embeddings=int(d.get("max_position_embeddings", 131072)),
            bos_token_id=int(d.get("bos_token_id", 151643)),
            eos_token_id=int(d.get("eos_token_id", 151643)),
        )


@dataclass
class QuantizationConfig:
    """Records which submodules are quantized + the mlx quant params.

    Absent from config.json ⇒ unquantized (``ModelConfig.quantization is None``).
    Present ⇒ the loader rebuilds the quantized skeleton (``nn.quantize``) before
    binding weights. Stage-1 scope: ``components == ["llm"]``.
    """

    bits: int
    group_size: int = 64
    components: list[str] = field(default_factory=lambda: ["llm"])

    @classmethod
    def from_dict(cls, d: dict) -> "QuantizationConfig":
        return cls(
            bits=int(d["bits"]),
            group_size=int(d.get("group_size", 64)),
            components=list(d.get("components", ["llm"])),
        )


@dataclass
class MeanFlowConfig:
    """MeanFlow distilled-sampler config (the ``mf`` checkpoint).

    Absent from config.json ⇒ flow-matching mode (``ModelConfig.meanflow is None``).
    Present with ``enabled=True`` ⇒ the DiT carries a ``duration_embedder`` and the
    decode loop uses the few-step ``meanflow_sample`` solver (NFE, no CFG).
    """

    enabled: bool = False
    use_duration_embedding: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MeanFlowConfig":
        return cls(
            enabled=bool(d.get("enabled", False)),
            use_duration_embedding=bool(d.get("use_duration_embedding", True)),
        )


@dataclass
class ModelConfig:
    """Top-level dots.tts config holding all submodule configs + shared constants."""

    latent_dim: int = 128
    patch_size: int = 4
    campplus_embedding_size: int = 512
    xvec_max_audio_seconds: float = 10.0
    fm_sigma: float = 0.0
    cfg_droprate: float = 0.2
    xvec_drop_rate: float = 0.2
    sample_rate: int = 48000

    dit: DiTConfig = field(default_factory=DiTConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    quantization: "QuantizationConfig | None" = None
    meanflow: "MeanFlowConfig | None" = None

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ModelConfig":
        """Build a ModelConfig from ``<path>/config.json`` + ``<path>/llm_config.json``."""
        path = Path(path)
        with open(path / "config.json") as f:
            cfg = json.load(f)
        with open(path / "llm_config.json") as f:
            llm_cfg = json.load(f)

        voc = cfg.get("vocoder", {})
        q = cfg.get("quantization")
        mf = cfg.get("meanflow")
        return cls(
            latent_dim=int(cfg.get("latent_dim", 128)),
            patch_size=int(cfg.get("patch_size", 4)),
            campplus_embedding_size=int(cfg.get("campplus_embedding_size", 512)),
            xvec_max_audio_seconds=float(cfg.get("xvec_max_audio_seconds", 10.0)),
            fm_sigma=float(cfg.get("fm_sigma", 0.0)),
            cfg_droprate=float(cfg.get("cfg_droprate", 0.2)),
            xvec_drop_rate=float(cfg.get("xvec_drop_rate", 0.2)),
            sample_rate=int(voc.get("sample_rate", 48000)),
            dit=DiTConfig.from_dict(cfg.get("DiT", {})),
            encoder=EncoderConfig.from_dict(cfg.get("PatchEncoder", {})),
            vocoder=VocoderConfig.from_dict(voc),
            llm=LLMConfig.from_dict(llm_cfg),
            quantization=QuantizationConfig.from_dict(q) if q else None,
            meanflow=MeanFlowConfig.from_dict(mf) if mf else None,
        )

    @property
    def mode(self) -> str:
        """``"meanflow"`` iff a meanflow block is present and enabled, else ``"flow_matching"``."""
        return (
            "meanflow"
            if (self.meanflow is not None and self.meanflow.enabled)
            else "flow_matching"
        )

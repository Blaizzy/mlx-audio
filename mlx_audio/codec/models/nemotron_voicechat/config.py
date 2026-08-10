from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NemotronVoiceChatCodecConfig:
    """Configuration for the codec embedded in NemotronLabs VoiceChat."""

    sample_rate: int = 22_050
    base_channels: int = 384
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    downsample_rates: tuple[int, ...] = (7, 7, 9)
    blocks_per_stage: int = 3
    block_kernel_size: int = 7
    latent_dim: int = 512
    n_fft: int = 16
    hop_length: int = 4
    num_quantizers: int = 31
    codebook_size: int = 1024

    @property
    def waveform_to_token_ratio(self) -> int:
        ratio = self.hop_length
        for rate in self.downsample_rates:
            ratio *= rate
        return ratio

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.waveform_to_token_ratio

    @property
    def stft_channels(self) -> int:
        return 2 * (self.n_fft // 2 + 1)

    @classmethod
    def from_dict(cls, config: dict | None) -> "NemotronVoiceChatCodecConfig":
        values = dict(config or {})
        for name in ("channel_multipliers", "downsample_rates"):
            if name in values:
                values[name] = tuple(values[name])
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass
class NemotronVoiceChatCodecArtifactConfig:
    """Standalone artifact wrapper used by codec-focused tooling/tests."""

    codec_config: NemotronVoiceChatCodecConfig = field(
        default_factory=NemotronVoiceChatCodecConfig
    )
    model_type: str = "nemotron_voicechat_codec"

    @classmethod
    def from_dict(cls, config: dict) -> "NemotronVoiceChatCodecArtifactConfig":
        return cls(
            codec_config=NemotronVoiceChatCodecConfig.from_dict(
                config.get("codec_config", config)
            ),
            model_type=config.get("model_type", "nemotron_voicechat_codec"),
        )

from dataclasses import dataclass, field

from mlx_audio.base import BaseModelArgs
from mlx_audio.lm.models.lfm3 import ModelArgs as LFMConfig


@dataclass
class EncoderConfig(BaseModelArgs):
    num_mel_bins: int = 128
    conv_channels: int = 256
    hidden_size: int = 768
    sliding_window: int = 32
    text_config: LFMConfig | dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for old, new in (
            ("dim_in", "num_mel_bins"),
            ("dim_conv", "conv_channels"),
            ("dim_out", "hidden_size"),
            ("lfm", "text_config"),
        ):
            if old in values:
                values[new] = values.pop(old)
        args = values.get("text_config", {})
        if isinstance(args, dict):
            args = {
                "model_type": "lfm2",
                "vocab_size": 2,
                "hidden_size": 768,
                "intermediate_size": 2560,
                "num_hidden_layers": 17,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "conv_L_cache": 3,
                "use_sandwich_norm": False,
                "layer_types": ["conv", "conv", "full_attention"] * 5
                + ["full_attention"] * 2,
                **args,
            }
            values["text_config"] = LFMConfig.from_dict(args)
        return super().from_dict(values)


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "lfm3_audio"
    model_path: str | None = None
    text_config: LFMConfig | dict = field(default_factory=dict)
    audio_encoder_config: EncoderConfig | dict = field(default_factory=dict)
    depth_config: LFMConfig | dict = field(default_factory=dict)
    num_codebooks: int = 12
    codebook_size: int = 2017
    audio_eos_code: int = 2016
    maskgit_schedule: list[int] | None = None
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    audio_token_id: int = 125031
    audio_out_token_id: int = 125032
    audio_start_token_id: int = 125012
    text_end_token_id: int = 125014
    mixed_start_token_id: int = 125015
    eos_token_id: int | list[int] = 124900
    sample_rate: int = 24000
    preprocessor: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for old, new in (
            ("lfm", "text_config"),
            ("encoder", "audio_encoder_config"),
            ("depthformer", "depth_config"),
            ("codebooks", "num_codebooks"),
            ("audio_vocab_size", "codebook_size"),
        ):
            if old in values:
                values[new] = values.pop(old)
        if "audio_eos_code" not in values:
            values["audio_eos_code"] = values.get("codebook_size", 2017) - 1
        steps = values.pop("maskgit_steps", 3)
        if values.get("maskgit_schedule") is None:
            count = values.get("num_codebooks", 12)
            if not 1 <= steps <= count:
                raise ValueError("maskgit_steps must be between 1 and num_codebooks")
            values["maskgit_schedule"] = [count // steps] * steps
            values["maskgit_schedule"][-1] += count % steps
        return super().from_dict(values)

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = LFMConfig.from_dict(self.text_config)
        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config = EncoderConfig.from_dict(
                self.audio_encoder_config
            )
        if isinstance(self.depth_config, dict):
            defaults = {
                "vocab_size": 1,
                "hidden_size": 1024,
                "intermediate_size": 2816,
                "num_hidden_layers": 3,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 32,
                "layer_types": ["full_attention"] * 3,
                "use_sandwich_norm": False,
                "rope_parameters": {"full_attention": {"rope_theta": 1000000.0}},
            }
            self.depth_config = LFMConfig.from_dict({**defaults, **self.depth_config})
        if self.maskgit_schedule is None:
            self.maskgit_schedule = [4, 4, 4]
        if (
            min(self.maskgit_schedule, default=0) < 1
            or sum(self.maskgit_schedule) != self.num_codebooks
        ):
            raise ValueError(
                "maskgit_schedule must be positive and sum to num_codebooks"
            )
        if not 0 <= self.audio_eos_code < self.codebook_size:
            raise ValueError("audio_eos_code must lie within the codebook")
        if min(self.interleaved_n_text, self.interleaved_n_audio) < 1:
            raise ValueError("Interleaving block sizes must be positive")

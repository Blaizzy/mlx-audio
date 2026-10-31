"""Configuration for NVIDIA NemotronLabs RE-USE."""

from dataclasses import dataclass, fields


@dataclass
class ModelConfig:
    model_type: str = "nemotron_reuse"
    hid_feature: int = 64
    num_tfmamba: int = 30
    d_state: int = 16
    d_conv: int = 4
    expand: int = 4
    input_channel: int = 2
    output_channel: int = 1
    compress_factor: str | float = "relu_log1p"
    n_fft: int = 320
    hop_size: int = 40
    win_size: int = 320
    sampling_rate: int = 8000

    @classmethod
    def from_dict(cls, config):
        values = {**config, **config.get("model_cfg", {}), **config.get("stft_cfg", {})}
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})

    def stft_sizes(self, sample_rate):
        # Match upstream integer scaling followed by rounding up to an even size.
        def scale(value):
            value = value * sample_rate // self.sampling_rate
            return value + value % 2

        return tuple(scale(v) for v in (self.n_fft, self.hop_size, self.win_size))

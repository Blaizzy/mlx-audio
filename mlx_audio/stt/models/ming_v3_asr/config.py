from dataclasses import dataclass, field

from mlx_audio.base import BaseModelArgs
from mlx_audio.lm.models.bailing_moe_v3 import ModelArgs


@dataclass
class AudioConfig(BaseModelArgs):
    n_mels: int = 128
    n_audio_ctx: int = 3000
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 32
    downsampler_kernel_size_list: list[int] = field(default_factory=lambda: [3])
    downsampler_stride_list: list[int] = field(default_factory=lambda: [2])
    downsampler_padding_list: list[int] | None = None
    whisper_chunk_size: int = 3000

    def __post_init__(self):
        if not self.downsampler_kernel_size_list or len(
            self.downsampler_kernel_size_list
        ) != len(self.downsampler_stride_list):
            raise ValueError(
                "Audio projector kernels and strides must have equal positive lengths"
            )
        if self.downsampler_padding_list is None:
            self.downsampler_padding_list = [
                k // 2 for k in self.downsampler_kernel_size_list
            ]
        if len(self.downsampler_padding_list) != len(self.downsampler_kernel_size_list):
            raise ValueError("Audio projector paddings must match its kernels")
        if min(self.downsampler_kernel_size_list + self.downsampler_stride_list) < 1:
            raise ValueError("Audio projector kernels and strides must be positive")
        if not 0 < self.whisper_chunk_size <= 2 * self.n_audio_ctx:
            raise ValueError(
                "Audio chunk length exceeds the Whisper positional context"
            )

    def embedding_length(self, frames):
        length = (frames + 1) // 2
        for kernel, stride, padding in zip(
            self.downsampler_kernel_size_list,
            self.downsampler_stride_list,
            self.downsampler_padding_list,
        ):
            length = (length + 2 * padding - kernel) // stride + 1
        return length

    def token_count(self, frames):
        return sum(
            self.embedding_length(min(self.whisper_chunk_size, frames - i))
            for i in range(0, frames, self.whisper_chunk_size)
        )


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "ming_v3_asr"
    text_config: dict = field(default_factory=dict)
    audio_config: dict = field(default_factory=dict)
    audio_token_id: int = 157168
    norm_query_embeds: bool = True
    model_path: str | None = None

    def __post_init__(self):
        self.text_config = (
            ModelArgs.from_dict(self.text_config)
            if isinstance(self.text_config, dict)
            else self.text_config
        )
        self.audio_config = (
            AudioConfig.from_dict(self.audio_config)
            if isinstance(self.audio_config, dict)
            else self.audio_config
        )

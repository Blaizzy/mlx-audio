"""Config parsing for Nemotron 3.5 ASR (NeMo model_config -> typed args)."""

from mlx_audio.stt.models.nemotron_asr.audio import PreprocessArgs
from mlx_audio.stt.models.nemotron_asr.encoder import EncoderArgs


class ModelConfig:
    """Thin wrapper over the NeMo config dict (matches Parakeet convention)."""

    def __init__(self, config: dict):
        self._config = config

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        if "model_type" not in config:
            config = {**config, "model_type": "nemotron_asr"}
        return cls(config)


def _first_att_context(enc: dict) -> tuple:
    acs = enc.get("att_context_size", [[56, 3]])
    first = acs[0] if acs and isinstance(acs[0], (list, tuple)) else acs
    return (int(first[0]), int(first[1]))


def parse_preprocess(config: dict) -> PreprocessArgs:
    p = config.get("preprocessor", {})
    sr = int(p.get("sample_rate", 16000))
    return PreprocessArgs(
        sample_rate=sr,
        n_fft=int(p.get("n_fft", 512)),
        win_length=int(p.get("window_size", 0.025) * sr),
        hop_length=int(p.get("window_stride", 0.01) * sr),
        features=int(p.get("features", 128)),
        preemph=float(p.get("preemph", 0.97) or 0.0),
    )


def parse_encoder(config: dict) -> EncoderArgs:
    e = config.get("encoder", {})
    return EncoderArgs(
        feat_in=int(e.get("feat_in", 128)),
        n_layers=int(e.get("n_layers", 24)),
        d_model=int(e.get("d_model", 1024)),
        n_heads=int(e.get("n_heads", 8)),
        ff_expansion_factor=int(e.get("ff_expansion_factor", 4)),
        subsampling_factor=int(e.get("subsampling_factor", 8)),
        subsampling_conv_channels=int(e.get("subsampling_conv_channels", 256)),
        conv_kernel_size=int(e.get("conv_kernel_size", 9)),
        pos_emb_max_len=int(e.get("pos_emb_max_len", 5000)),
        att_context_size=_first_att_context(e),
    )

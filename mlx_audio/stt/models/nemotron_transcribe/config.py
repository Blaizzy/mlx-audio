"""Configuration for NVIDIA's consolidated Nemotron 3.5 Transcribe checkpoint."""

from dataclasses import dataclass, field

from mlx_audio.base import BaseModelArgs
from mlx_audio.lm.models.nemotron_h import ModelArgs as TextConfig
from mlx_audio.vad.models.nemotron_diarization.config import (
    DiarizationModulesConfig,
    EncoderConfig,
    MelConfig,
)
from mlx_audio.vad.models.nemotron_diarization.config import (
    ModelConfig as DiarizationConfig,
)


def mel_config(source):
    for key, expected in {
        "window": "hann",
        "frame_splicing": 1,
        "exact_pad": False,
        "mag_power": 2.0,
        "log_zero_guard_type": "add",
        "log_zero_guard_value": 2**-24,
    }.items():
        if source.get(key, expected) != expected:
            raise ValueError(f"Unsupported preprocessor {key}: {source[key]}")
    if source.get("normalize") not in (None, "NA") or not source.get("log", True):
        raise ValueError("Nemotron Transcribe requires unnormalized log-mels")
    rate = source.get("sample_rate", 16000)
    return MelConfig(
        sampling_rate=rate,
        feature_size=source.get("features", 128),
        n_fft=source.get("n_fft", 512),
        hop_length=int(rate * source.get("window_stride", 0.01)),
        win_length=int(rate * source.get("window_size", 0.025)),
        preemphasis=source.get("preemph", 0.97),
        pad_to=source.get("pad_to", 0),
    )


def encoder_config(source):
    for key, expected in {
        "subsampling": "feature_stacking",
        "self_attention_model": "rope",
        "attn_mode": "full",
        "feat_out": -1,
    }.items():
        if source.get(key, expected) != expected:
            raise ValueError(f"Unsupported encoder {key}: {source[key]}")
    return EncoderConfig.from_dict(source)


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "nemotron_transcribe"
    llm_config: dict = field(default_factory=dict)
    pe_encoder_config: dict = field(default_factory=dict)
    perception: dict = field(default_factory=dict)
    prompt_format: str = "nemotron3p5"
    audio_locator_tag: str = "<|audio|>"
    eos_token_id: list[int] = field(default_factory=lambda: [11])
    pad_token_id: int = 0

    def __post_init__(self):
        if not self.llm_config or not self.pe_encoder_config or not self.perception:
            raise ValueError(
                "Expected inline llm_config, pe_encoder_config and perception"
            )
        if self.llm_config.get("model_type") != "nemotron_h":
            raise ValueError(
                "Only the Nemotron-H speech language backbone is supported"
            )
        if self.prompt_format != "nemotron3p5":
            raise ValueError(f"Unsupported prompt format: {self.prompt_format}")
        pe = self.pe_encoder_config
        if pe.get("asr_encoder_type", "transformer") != "transformer":
            raise ValueError("Expected the feature-stacking Transformer speech encoder")
        if pe.get("speaker_feature_config_version") != 1:
            raise ValueError("Expected speaker_feature_config_version=1")
        mode = pe.get("speaker_feature_mode")
        threshold = pe.get("speaker_activity_threshold")
        if mode == "thresholded":
            if threshold is None or not 0 <= threshold <= 1:
                raise ValueError(
                    "Thresholded speaker features need a threshold in [0, 1]"
                )
        elif mode != "continuous" or threshold is not None:
            raise ValueError("Unsupported speaker feature contract")
        if pe.get("asr_normalize_type", "per_feature") != "per_feature":
            raise ValueError("Expected per-feature ASR normalization")
        if pe.get("diar_normalize_type", "NA") not in (None, "NA"):
            raise ValueError("Expected unnormalized diarization features")
        if pe.get("online_inference_length", 500) < 1:
            raise ValueError("online_inference_length must be positive")
        if (
            self.perception.get("modality_adapter", {})
            .get("_target_", "")
            .split(".")[-1]
            != "IdentityConnector"
        ):
            raise ValueError(
                "Expected IdentityConnector with a linear output projection"
            )
        if self.perception.get("rote"):
            raise ValueError(
                "Rotary time embeddings are not supported by this checkpoint"
            )
        if self.perception["output_dim"] != self.llm_config["hidden_size"]:
            raise ValueError("Perception output must match the language model width")
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]

    @property
    def text_config(self):
        config = dict(self.llm_config)
        if config.get("time_step_limit") is not None:
            # NeMo's JSON serializer encodes nonfinite floats as tagged objects.
            config["time_step_limit"] = tuple(
                (
                    float(value["__float__"])
                    if isinstance(value, dict) and set(value) == {"__float__"}
                    else value
                )
                for value in config["time_step_limit"]
            )
        return TextConfig.from_dict(config)

    @property
    def asr_config(self):
        return encoder_config(self.pe_encoder_config["asr_encoder_cfg"])

    @property
    def processor_config(self):
        return mel_config(self.perception["preprocessor"])

    @property
    def diarization_config(self):
        pe = self.pe_encoder_config
        source = pe["diarization_model_cfg"]
        if not source.get("high_resolution") or source.get("transformer_encoder"):
            raise ValueError("Expected a high-resolution single-encoder speaker model")
        enc = encoder_config(source["encoder"])
        if enc.subsampling_factor != self.asr_config.subsampling_factor:
            raise ValueError("Speech and speaker encoders must use the same stride")
        modules = dict(source["sortformer_modules"])
        modules.update(
            num_speakers=modules.pop("num_spks"),
            subsampling_factor=enc.subsampling_factor,
            use_activity_head=False,
            use_aosc=True,
            # SALMAutomodel.generate uses the PE encoder's online settings.
            chunk_len=pe.get("online_inference_length", 500),
            fifo_len=pe.get("diar_fifo_len", 40),
            spkcache_len=pe.get("diar_spkcache_len", 188),
            spkcache_update_period=pe.get("diar_spkcache_update_period", 300),
        )
        return DiarizationConfig(
            num_speakers=modules["num_speakers"],
            encoder_config=enc,
            modules_config=DiarizationModulesConfig.from_dict(modules),
            processor_config=mel_config(source["preprocessor"]),
            output_subsampling_factor=enc.subsampling_factor,
        )

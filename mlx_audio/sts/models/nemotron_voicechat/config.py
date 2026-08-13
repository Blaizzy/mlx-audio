from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_audio.codec.models.nemotron_voicechat import NemotronVoiceChatCodecConfig
from mlx_audio.lm.models.nemotron_h import ModelArgs as NemotronHArgs
from mlx_audio.stt.models.nemotron_asr.config import (
    ConformerArgs,
    JointArgs,
    PredictArgs,
    PreprocessArgs,
)


@dataclass
class VoiceChatTTSConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    sliding_window: int
    latent_size: int
    codebook_size: int
    num_quantizers: int
    exponent: float
    num_iterations: int
    guidance_scale: float
    top_p: float
    noise_scale: float
    char_vocab_size: int
    text_vocab_size: int
    mog_intermediate_size: int
    mog_num_layers: int
    mog_num_predictions: int
    mog_low_rank: int
    mog_min_log_std: float


@dataclass
class NemotronVoiceChatConfig:
    llm: NemotronHArgs
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    codec: NemotronVoiceChatCodecConfig
    tts: VoiceChatTTSConfig
    pretrained_llm: str
    rnnt_vocabulary: list[str]
    source_sample_rate: int = 16_000
    target_sample_rate: int = 22_050
    frame_duration: float = 0.08
    audio_prompt_frames: int = 37
    output_dim: int = 4_480
    text_channel_weight: float = 1.0
    audio_channel_weight: float = 1.0
    function_channel_weight: float = 2.0
    use_function_head: bool = True
    speaker_name: str = "Aria"
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 12
    silence_token_id: int = 11
    rnnt_blank_id: int = 1024
    rnnt_max_symbols: int = 10
    default_system_prompt: str = ""
    prepared_weights: bool = False
    model_type: str = "nemotron_voicechat"


def _codec_config(config: dict[str, Any]) -> NemotronVoiceChatCodecConfig:
    return NemotronVoiceChatCodecConfig.from_dict(
        {
            "sample_rate": config.get("sample_rate", 22_050),
            "base_channels": config.get("base_hidden_size", 384),
            "channel_multipliers": config.get("channel_mult", (1, 2, 4)),
            "downsample_rates": config.get("rates", (7, 7, 9)),
            "blocks_per_stage": config.get("num_blocks", 3),
            "block_kernel_size": config.get("kernel_size", 7),
            "latent_dim": config.get("latent_size", 512),
            "n_fft": config.get("n_fft", 16),
            "hop_length": config.get("hop_length", 4),
            "num_quantizers": config.get("num_quantizers", 31),
            "codebook_size": config.get("codebook_size", 1024),
        }
    )


def _llm_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    explicit = config.get("mlx_audio", {}).get("llm_config")
    if explicit is not None:
        return explicit

    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name).to_dict()


class ModelConfig:
    def __init__(self, config: NemotronVoiceChatConfig):
        self.config = config

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelConfig":
        data = config.get("data", {})
        root = config.get("model", {})
        stt_root = root.get("stt", {})
        stt = stt_root.get("model", stt_root)
        perception = stt.get("perception", {})
        pre = perception.get("preprocessor", {})
        enc = perception.get("encoder", {})
        rnnt = config.get("_rnnt_merge_info", {})
        decoder_config = rnnt.get("decoder_config", {})
        decoder = decoder_config.get("prednet", {})
        joint_config = rnnt.get("joint_config", {})
        joint = joint_config.get("jointnet", {})

        speech_root = root.get("speech_generation", {})
        speech = speech_root.get("model", speech_root)
        speech_data = speech_root.get("data", {})
        tts_config = speech.get("tts_config", {})
        backbone = tts_config.get("backbone_config", {})
        mog = tts_config.get("mog_head_config", {})
        codec_raw = speech.get("codec_config", {})
        pretrained_llm = stt.get("pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
        llm = NemotronHArgs.from_dict(_llm_config(config, pretrained_llm))
        source_sample_rate = data.get(
            "source_sample_rate",
            stt_root.get("data", {}).get("source_sample_rate", 16_000),
        )
        target_sample_rate = data.get(
            "target_sample_rate",
            speech_root.get("data", {}).get("target_sample_rate", 22_050),
        )
        codec_raw = {"sample_rate": target_sample_rate, **codec_raw}

        parsed = NemotronVoiceChatConfig(
            llm=llm,
            preprocessor=PreprocessArgs(
                sample_rate=pre.get("sample_rate", source_sample_rate),
                features=pre.get("features", 128),
                n_fft=pre.get("n_fft", 512),
                window_size=pre.get("window_size", 0.025),
                window_stride=pre.get("window_stride", 0.01),
                window=pre.get("window", "hann"),
                preemph=pre.get("preemph", 0.97),
                dither=pre.get("dither", 1.0e-5),
                normalize=str(pre.get("normalize", "NA")),
                log_zero_guard_value=float(pre.get("log_zero_guard_value", 2.0**-24)),
                pad_to=pre.get("pad_to", 0),
                pad_value=pre.get("pad_value", 0.0),
            ),
            encoder=ConformerArgs(
                feat_in=enc.get("feat_in", 128),
                n_layers=enc.get("n_layers", 24),
                d_model=enc.get("d_model", 1024),
                n_heads=enc.get("n_heads", 8),
                ff_expansion_factor=enc.get("ff_expansion_factor", 4),
                subsampling_factor=enc.get("subsampling_factor", 8),
                subsampling_conv_channels=enc.get("subsampling_conv_channels", 256),
                conv_kernel_size=enc.get("conv_kernel_size", 9),
                causal_downsampling=enc.get("causal_downsampling", True),
                conv_context_size=enc.get("conv_context_size", "causal"),
                conv_norm_type=enc.get("conv_norm_type", "layer_norm"),
                self_attention_model=enc.get("self_attention_model", "rel_pos"),
                att_context_style=enc.get("att_context_style", "chunked_limited"),
                att_context_size=[enc.get("att_context_size", [70, 0])],
                pos_emb_max_len=enc.get("pos_emb_max_len", 5000),
                use_bias=enc.get("use_bias", False),
                xscaling=enc.get("xscaling", False),
            ),
            decoder=PredictArgs(
                pred_hidden=decoder.get("pred_hidden", 640),
                pred_rnn_layers=decoder.get("pred_rnn_layers", 2),
                vocab_size=decoder_config.get("vocab_size", 1024),
                blank_as_pad=decoder_config.get("blank_as_pad", True),
            ),
            joint=JointArgs(
                joint_hidden=joint.get("joint_hidden", 640),
                activation=joint.get("activation", "relu"),
                encoder_hidden=joint.get("encoder_hidden", enc.get("d_model", 1024)),
                pred_hidden=joint.get("pred_hidden", 640),
                num_classes=joint_config.get("num_classes", 1024),
            ),
            codec=_codec_config(codec_raw),
            tts=VoiceChatTTSConfig(
                hidden_size=backbone.get("hidden_size", 1152),
                intermediate_size=backbone.get("intermediate_size", 4608),
                num_hidden_layers=backbone.get("num_hidden_layers", 28),
                num_attention_heads=backbone.get("num_attention_heads", 16),
                num_key_value_heads=backbone.get("num_key_value_heads", 16),
                head_dim=backbone.get("head_dim", 72),
                sliding_window=backbone.get("sliding_window", 7500),
                latent_size=tts_config.get("latent_size", 512),
                codebook_size=tts_config.get("codebook_size", 1024),
                num_quantizers=tts_config.get("num_quantizers", 31),
                exponent=tts_config.get("exponent", 3.0),
                num_iterations=speech.get("inference_num_iterations", 8),
                guidance_scale=speech.get("inference_guidance_scale", 0.2),
                top_p=speech.get("inference_top_p_or_k", 0.95),
                noise_scale=speech.get("inference_noise_scale", 0.001),
                char_vocab_size=config.get("mlx_audio", {}).get("char_vocab_size", 256),
                text_vocab_size=llm.vocab_size,
                mog_intermediate_size=mog.get("intermediate_size", 4608),
                mog_num_layers=mog.get("num_layers", 3),
                mog_num_predictions=mog.get("num_predictions", 1024),
                mog_low_rank=mog.get("low_rank", 64),
                mog_min_log_std=mog.get("min_log_std", -4.0),
            ),
            pretrained_llm=pretrained_llm,
            rnnt_vocabulary=joint_config.get("vocabulary", []),
            source_sample_rate=source_sample_rate,
            target_sample_rate=target_sample_rate,
            frame_duration=data.get("frame_length", 0.08),
            audio_prompt_frames=int(
                speech_data.get("audio_prompt_duration", 3.0)
                / speech_data.get("frame_length", 0.08)
            ),
            output_dim=perception.get("output_dim", llm.hidden_size),
            text_channel_weight=stt.get("duplex_text_channel_weight", 1.0),
            audio_channel_weight=stt.get("duplex_user_channel_weight", 1.0),
            function_channel_weight=stt.get("duplex_function_channel_weight", 2.0),
            use_function_head=stt.get("use_function_head", True),
            speaker_name=root.get("inference_speaker_name", "Aria"),
            rnnt_blank_id=joint_config.get(
                "blank_id", joint_config.get("num_classes", 1024)
            ),
            rnnt_max_symbols=stt.get("max_symbols", 10),
            prepared_weights=config.get("mlx_audio", {}).get("prepared_weights", False),
            model_type=config.get("model_type", "nemotron_voicechat"),
        )
        return cls(parsed)

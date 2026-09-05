"""CosyVoice3 configuration.

CosyVoice3 architecture (three stages):
  1. LLM  : Qwen2 backbone that autoregressively predicts speech tokens from
            text tokens.
  2. Flow : an UpsampleConformer encoder + a DiT-based conditional
            flow-matching decoder that maps speech tokens -> mel spectrogram.
  3. HiFT : vocoder (mel -> waveform, NSF + ISTFT).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelArgs


@dataclass
class LLMConfig(BaseModelArgs):
    """Qwen2 backbone config for the speech-token language model.

    Defaults follow the CosyVoice2/3 0.5B Qwen2 config; override from the
    model's ``config.json`` on load.
    """

    model_type: str = "qwen2"
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    # CosyVoice3 speech-token LM head params
    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 6561  # FSQ codebook size (same as v2 s3 tokenizer)
    # speech LM special tokens live in the speech_embedding table.
    # CosyVoice3LM:  sos=speech_token_size+0, eos=+1, task_id=+2, fill=+3
    # llm_decoder projects to speech_token_size + 200 (extra reserved ids).
    speech_vocab_extra: int = 200
    mix_ratio: List[int] = field(default_factory=lambda: [5, 15])
    endofprompt_token: int = 151646  # <|endofprompt|> — required in v3 text


@dataclass
class FlowConfig(BaseModelArgs):
    """Flow-matching module: PreLookahead + repeat_interleave + DiT decoder.

    No conformer encoder: token -> input_embedding(80) -> PreLookaheadLayer
    -> repeat_interleave(token_mel_ratio) -> DiT flow-matching decoder.
    """

    input_size: int = 80  # input_embedding dim
    output_size: int = 80  # mel bins
    spk_embed_dim: int = 192
    vocab_size: int = 6561
    output_type: str = "mel"

    # token path
    pre_lookahead_len: int = 3
    pre_lookahead_channels: int = 1024
    token_mel_ratio: int = 2  # repeat_interleave upsample factor

    # DiT flow-matching decoder
    dit_hidden_size: int = 1024  # DiT dim
    dit_depth: int = 22
    dit_num_heads: int = 16
    dit_head_dim: int = 64
    dit_mlp_ratio: float = 2.0  # ff_mult
    dit_mel_dim: int = 80
    dit_mu_dim: int = 80
    dit_spk_dim: int = 80  # spk projected to output_size before DiT
    dit_static_chunk_size: int = 50  # chunk_size(25) * token_mel_ratio(2)
    dit_num_decoding_left_chunks: int = -1

    # CFM sampling
    n_timesteps: int = 10
    inference_cfg_rate: float = 0.7


@dataclass
class HiFTConfig(BaseModelArgs):
    """HiFT-GAN vocoder (mel -> waveform, NSF source + ISTFT)."""

    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10.0
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 11, 7])
    istft_params: Dict[str, int] = field(
        default_factory=lambda: {"n_fft": 16, "hop_len": 4}
    )
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    source_resblock_kernel_sizes: List[int] = field(
        default_factory=lambda: [7, 7, 11]
    )
    source_resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    # right-context (in mel frames) consumed by conv_pre's causal_type='right'
    # padding; conv_pre's kernel size is conv_pre_look_right + 1.
    conv_pre_look_right: int = 4


@dataclass
class ModelConfig(BaseModelArgs):
    """Top-level CosyVoice3 config consumed by ``Model``."""

    model_type: str = "cosyvoice3"
    sample_rate: int = 24000
    token_frame_rate: int = 25  # speech tokens per second
    token_mel_ratio: int = 2  # mel frames per speech token (flow up_rate)

    # runs of these FSQ silent/breath tokens longer than max_silent_token_num
    # are dropped before the flow stage, to avoid overlong pauses in the audio
    silent_tokens: tuple = (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)
    max_silent_token_num: int = 5

    llm: LLMConfig = field(default_factory=LLMConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    hift: HiFTConfig = field(default_factory=HiFTConfig)

    # tokenizer / frontend
    speech_tokenizer_name: str = "speech_tokenizer_v2_25hz"
    use_onnx_speech_tokenizer: bool = True

    @classmethod
    def from_dict(cls, params: dict) -> "ModelConfig":
        params = dict(params)
        for key, sub in (("llm", LLMConfig), ("flow", FlowConfig), ("hift", HiFTConfig)):
            if isinstance(params.get(key), dict):
                params[key] = sub.from_dict(params[key])
        return super().from_dict(params)

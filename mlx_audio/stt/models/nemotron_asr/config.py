"""Configuration dataclasses for the Nemotron 3.5 ASR (streaming) model.

The schema mirrors the relevant pieces of NeMo's ``model_config.yaml`` for
``EncDecRNNTBPEModelWithPrompt`` (a cache-aware streaming FastConformer-RNNT with
language-ID prompt conditioning), flattened into the ``config.json`` produced by
``convert.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PreprocessArgs:
    """Mel-spectrogram featurizer settings (NeMo AudioToMelSpectrogramPreprocessor)."""

    sample_rate: int = 16000
    features: int = 128
    n_fft: int = 512
    window_size: float = 0.025
    window_stride: float = 0.01
    window: str = "hann"
    preemph: float = 0.97
    dither: float = 1.0e-05
    # NeMo ``normalize: NA`` => no normalization. Kept for parity / future configs.
    normalize: str = "NA"
    log_zero_guard_value: float = 2.0**-24
    pad_to: int = 0
    pad_value: float = 0.0

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)


@dataclass
class ConformerArgs:
    """Cache-aware FastConformer encoder settings."""

    feat_in: int = 128
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion_factor: int = 4
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    conv_kernel_size: int = 9
    causal_downsampling: bool = True
    # "causal" or [left, right]; resolved against conv_kernel_size at build time.
    conv_context_size: object = "causal"
    conv_norm_type: str = "layer_norm"
    self_attention_model: str = "rel_pos"
    att_context_style: str = "chunked_limited"
    # All look-ahead contexts the model was trained with (list of [left, right]).
    att_context_size: List[List[int]] = field(default_factory=lambda: [[56, 13]])
    pos_emb_max_len: int = 5000
    use_bias: bool = False
    xscaling: bool = False


@dataclass
class PromptArgs:
    """Language-ID prompt conditioning (PromptStreamingMixin)."""

    num_prompts: int = 128
    # Hidden size of the prompt_kernel MLP (NeMo uses enc_hidden * 2).
    prompt_hidden: int = 2048
    prompt_dictionary: Dict[str, int] = field(default_factory=dict)


@dataclass
class PredictArgs:
    """RNN-T prediction (decoder) network."""

    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    vocab_size: int = 13087
    blank_as_pad: bool = True


@dataclass
class JointArgs:
    """RNN-T joint network."""

    joint_hidden: int = 640
    activation: str = "relu"
    encoder_hidden: int = 1024
    pred_hidden: int = 640
    num_classes: int = 13087


@dataclass
class NemotronASRConfig:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    prompt: PromptArgs
    decoder: PredictArgs
    joint: JointArgs
    vocabulary: List[str]
    model_type: str = "nemotron_asr"
    target: str = (
        "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt"
    )
    # Default language prompt key (e.g. "auto", "en-US"). "auto" lets the model
    # detect the language and emit a leading <lang> tag.
    default_language: str = "auto"
    # Look-ahead [left, right] used at inference. [56, 13] = best offline accuracy.
    default_att_context_size: List[int] = field(default_factory=lambda: [56, 13])
    max_symbols: int = 10

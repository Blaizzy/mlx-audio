import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "wav2vec2"
    vocab_size: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    feat_proj_dropout: float = 0.0
    feat_quantizer_dropout: float = 0.0
    final_dropout: float = 0.1
    layerdrop: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    feat_extract_norm: str = "group"
    feat_extract_activation: str = "gelu"
    conv_dim: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_stride: Tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    num_feat_extract_layers: int = 7
    do_stable_layer_norm: bool = False
    apply_spec_augment: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    num_codevectors_per_group: int = 320
    num_codevector_groups: int = 2
    contrastive_logits_temperature: float = 0.1
    num_negatives: int = 100
    codevector_dim: int = 256
    proj_codevector_dim: int = 256
    diversity_loss_weight: float = 0.1
    ctc_loss_reduction: str = "sum"
    ctc_zero_infinity: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = nn.GELU()

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim)
        self.activation = nn.GELU()

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states.swapaxes(-2, -1))

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.swapaxes(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = nn.GELU()

        self.layer_norm = nn.GroupNorm(
            num_groups=self.out_conv_dim,
            dims=self.out_conv_dim,
            affine=True,
            pytorch_compatible=True,
        )

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


def normalize_weight(x, except_dim=0):
    if x.ndim != 3:
        raise ValueError("Input tensor must have 3 dimensions")

    axes = tuple(i for i in range(x.ndim) if i != except_dim)
    return mx.sqrt(mx.sum(mx.power(x, 2), axis=axes, keepdims=True))


class WNConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()

        if bias:
            self.bias = mx.zeros((out_channels,))

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        scale = math.sqrt(1 / (in_channels * kernel_size))
        weight_init = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        self.weight_g = normalize_weight(weight_init, except_dim=1)
        self.weight_v = weight_init / (self.weight_g + 1e-12)

    def _extra_repr(self):
        return (
            f"in_channels={self.weight_v.shape[2]}, out_channels={self.weight_v.shape[0]}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        weight = (
            self.weight_g
            * self.weight_v
            / normalize_weight(self.weight_v, except_dim=1)
        )
        y = mx.conv1d(x, weight, self.stride, self.padding, self.dilation, self.groups)
        if "bias" in self:
            y = y + self.bias
        return y


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = WNConv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = nn.GELU()

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def __call__(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i)
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers

    def __call__(self, input_values):
        hidden_states = input_values[:, None]

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def __call__(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class Wav2Vec2Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: mx.array, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[Any] = None,
        past_key_value: Optional[Tuple[Any]] = None,
        attention_mask: Optional[Any] = None,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[Tuple[mx.array]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = mx.concatenate([past_key_value[0], key_states], axis=2)
            value_states = mx.concatenate([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)

        attn_output = mx.fast.scaled_dot_product_attention(
            q=query_states,
            k=key_states,
            v=value_states,
            scale=1.0,
            mask=attention_mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value


class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )

        self.intermediate_act_fn = nn.GELU()

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def __call__(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, hidden_states, attention_mask=None):
        attn_residual = hidden_states
        hidden_states, _ = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, _ = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states)
        )

        outputs = (hidden_states,)

        return outputs


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = [
            Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask[..., None]
            expand_attention_mask = mx.repeat(
                expand_attention_mask, 1, 1, hidden_states.shape[2]
            )
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].astype(
                hidden_states.dtype
            )
            attention_mask = attention_mask * mx.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = [
            Wav2Vec2EncoderLayerStableLayerNorm(config)
            for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask[..., None]
            expand_attention_mask = mx.repeat(
                expand_attention_mask, 1, 1, hidden_states.shape[2]
            )
            hidden_states = hidden_states * expand_attention_mask.astype(
                hidden_states.dtype
            )

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].astype(
                hidden_states.dtype
            )
            attention_mask = attention_mask * mx.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


@dataclass
class Wav2Vec2BaseModelOutput:
    last_hidden_state: Optional[mx.array] = None
    extract_features: Optional[mx.array] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


class Wav2Vec2Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

    def _mask_hidden_states(
        self,
        hidden_states: mx.array,
        mask_time_indices: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        return hidden_states

    def __call__(
        self,
        input_values: Optional[mx.array],
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(0, 2, 1)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs.last_hidden_state

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("wav2vec2."):
                k = k.replace("wav2vec2.", "")
            if k.endswith(".conv.weight"):
                v = v.swapaxes(1, 2)
            if k.endswith(".conv.weight_v") or k.endswith(".conv.weight_g"):
                v = v.swapaxes(1, 2)
            if k.endswith(".parametrizations.weight.original0"):
                k = k.replace(".parametrizations.weight.original0", ".weight_g")
                v = v.swapaxes(1, 2)
            if k.endswith(".parametrizations.weight.original1"):
                k = k.replace(".parametrizations.weight.original1", ".weight_v")
                v = v.swapaxes(1, 2)
            if (
                "lm_head." in k
                or k.startswith("quantizer.")
                or k.startswith("project_")
                or k == "masked_spec_embed"
            ):
                continue

            sanitized_weights[k] = v
        return sanitized_weights

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs):
        path = fetch_from_hub(repo_id)

        if path is None:
            raise ValueError(f"Could not find model {path}")

        config_path = path / "config.json"
        model_path = path / "model.safetensors"

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
        model = Wav2Vec2Model(config)

        weights = mx.load(model_path.as_posix(), format="safetensors")
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        return model


# fetch model from hub


def fetch_from_hub(model_path: str) -> Path:
    if not Path(model_path).exists():
        model_path = Path(
            snapshot_download(
                repo_id=model_path,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )
    return Path(model_path)

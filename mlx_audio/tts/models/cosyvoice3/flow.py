"""CosyVoice3 flow decoder from speech tokens to mel spectrograms.

Speech tokens are embedded, passed through a causal look-ahead layer, upsampled
in time, and decoded with conditional DiT flow matching.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import FlowConfig
from .dit import DiT
from .flow_matching import CausalConditionalCFM


def make_pad_mask(lengths: mx.array, max_len: int) -> mx.array:
    """True where padded. lengths: (B,) -> (B, max_len)."""
    batch_size = lengths.shape[0]
    seq_range = mx.arange(max_len)
    seq_range = mx.broadcast_to(mx.expand_dims(seq_range, 0), (batch_size, max_len))
    return seq_range >= mx.expand_dims(lengths, -1)


class PreLookaheadLayer(nn.Module):
    """Causal look-ahead convolution block.

    conv1: Conv1d(in->channels, k=pre_lookahead_len+1), right-pad by lookahead.
    conv2: Conv1d(channels->in, k=3), left-pad by 2. Residual add.
    Supports an optional ``context`` tail for chunked streaming inference.
    """

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3)

    def __call__(
        self, inputs: mx.array, context: Optional[mx.array] = None
    ) -> mx.array:
        # inputs: (B, T, C) channels-last (MLX conv convention)
        if context is None or context.shape[1] == 0:
            outputs = mx.pad(inputs, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        else:
            assert context.shape[1] == self.pre_lookahead_len
            outputs = mx.concatenate([inputs, context], axis=1)
            pad = self.pre_lookahead_len - context.shape[1]
            if pad > 0:
                outputs = mx.pad(outputs, [(0, 0), (0, pad), (0, 0)])
        outputs = nn.leaky_relu(self.conv1(outputs))
        outputs = mx.pad(outputs, [(0, 0), (2, 0), (0, 0)])  # conv2 kernel=3 causal
        outputs = self.conv2(outputs)
        return outputs + inputs


class CausalMaskedDiffWithDiT(nn.Module):
    """token + prompt + xvector -> mel (v3 flow: PreLookahead + repeat + DiT)."""

    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.vocab_size = config.vocab_size
        self.token_mel_ratio = config.token_mel_ratio
        self.pre_lookahead_len = config.pre_lookahead_len

        self.input_embedding = nn.Embedding(config.vocab_size, config.input_size)
        self.spk_embed_affine_layer = nn.Linear(
            config.spk_embed_dim, config.output_size
        )
        self.pre_lookahead_layer = PreLookaheadLayer(
            in_channels=config.input_size,
            channels=config.pre_lookahead_channels,
            pre_lookahead_len=config.pre_lookahead_len,
        )
        self.decoder = CausalConditionalCFM(
            estimator=DiT(
                dim=config.dit_hidden_size,
                depth=config.dit_depth,
                heads=config.dit_num_heads,
                dim_head=config.dit_head_dim,
                ff_mult=config.dit_mlp_ratio,
                mel_dim=config.dit_mel_dim,
                mu_dim=config.dit_mu_dim,
                spk_dim=config.dit_spk_dim,
                out_channels=config.output_size,
                static_chunk_size=config.dit_static_chunk_size,
                num_decoding_left_chunks=config.dit_num_decoding_left_chunks,
            ),
            inference_cfg_rate=config.inference_cfg_rate,
        )

    @property
    def up_rate(self) -> int:
        return self.token_mel_ratio

    def inference(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: Optional[mx.array],
        embedding: mx.array,
        n_timesteps: int = 10,
    ) -> mx.array:
        if token.shape[0] != 1:
            raise ValueError("CosyVoice3 flow inference supports batch size 1 only")

        # xvec projection
        embedding = embedding / (
            mx.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        # concat prompt + target tokens, embed, mask
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len
        mask = mx.logical_not(make_pad_mask(token_len, token.shape[1]))
        mask = mx.expand_dims(mask, -1).astype(embedding.dtype)
        token = mx.clip(token, 0, self.input_embedding.weight.shape[0] - 1)
        token = self.input_embedding(token) * mask

        # pre-lookahead + repeat-interleave upsample (x token_mel_ratio)
        h = self.pre_lookahead_layer(token)
        h = mx.repeat(h, self.token_mel_ratio, axis=1)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]

        # conditions: prompt mel on the prompt region, zeros on the target region
        conds = mx.concatenate(
            [
                prompt_feat,
                mx.zeros((h.shape[0], mel_len2, self.output_size), dtype=h.dtype),
            ],
            axis=1,
        )
        conds = mx.transpose(conds, (0, 2, 1))  # (B, mel, T)

        total_len = mel_len1 + mel_len2
        dmask = mx.logical_not(make_pad_mask(mx.array([total_len]), total_len))
        dmask = mx.expand_dims(dmask.astype(h.dtype), 1)  # (B, 1, T)

        feat = self.decoder(
            mu=mx.transpose(h, (0, 2, 1)),  # (B, mel, T)
            mask=dmask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )
        feat = feat[:, :, mel_len1:]  # drop prompt region
        if feat.shape[2] != mel_len2:
            raise RuntimeError(f"mel length mismatch: {feat.shape[2]} != {mel_len2}")
        return feat

    def sanitize(self, weights: dict) -> dict:
        """Map checkpoint keys to this module's tree."""
        from .convert import convert_flow_weights

        return convert_flow_weights(weights)

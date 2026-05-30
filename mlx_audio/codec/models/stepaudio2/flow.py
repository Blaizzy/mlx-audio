import mlx.core as mx
import mlx.nn as nn

from .decoder_dit import DiT
from .flow_matching import CausalConditionalCFM
from .upsample_encoder_v2 import UpsampleConformerEncoderV2, make_pad_mask


class CausalMaskedDiffWithXvec(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        encoder: UpsampleConformerEncoderV2 | None = None,
        decoder: CausalConditionalCFM | None = None,
        input_embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        if encoder is None:
            encoder = UpsampleConformerEncoderV2(
                input_size=input_size,
                output_size=input_size,
                input_layer="linear",
                pre_lookahead_len=3,
                num_blocks=6,
                num_up_blocks=4,
                up_stride=2,
                up_scale_factor=2,
                attention_heads=8,
                pos_enc_layer_type="rel_pos_espnet",
                selfattention_layer_type="rel_selfattn",
                key_bias=True,
                linear_units=2048,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
            )
        if decoder is None:
            decoder = CausalConditionalCFM(
                estimator=DiT(
                    in_channels=320,
                    out_channels=output_size,
                    mlp_ratio=4.0,
                    depth=16,
                    num_heads=8,
                    head_dim=64,
                    hidden_size=512,
                ),
                inference_cfg_rate=0.7,
            )
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        self.input_embedding = (
            nn.Embedding(vocab_size, input_size)
            if input_embedding is None
            else input_embedding
        )
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder

    def inference(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: mx.array | None,
        embedding: mx.array,
        n_timesteps: int = 10,
    ) -> mx.array:
        if token.shape[0] != 1:
            raise ValueError(
                "StepAudio2 flow inference currently supports batch size 1"
            )

        embedding = embedding / (
            mx.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len
        mask = mx.logical_not(make_pad_mask(token_len, token.shape[1]))
        mask = mx.expand_dims(mask, -1).astype(embedding.dtype)
        token = mx.clip(token, 0, self.input_embedding.weight.shape[0] - 1)
        token = self.input_embedding(token) * mask

        h, _ = self.encoder(token, token_len)
        h = self.encoder_proj(h)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        conds = mx.concatenate(
            [
                prompt_feat,
                mx.zeros((h.shape[0], mel_len2, self.output_size), dtype=h.dtype),
            ],
            axis=1,
        )
        conds = mx.transpose(conds, (0, 2, 1))

        total_len = mel_len1 + mel_len2
        mask = mx.logical_not(make_pad_mask(mx.array([total_len]), total_len))
        mask = mx.expand_dims(mask.astype(h.dtype), 1)
        feat = self.decoder(
            mu=mx.transpose(h, (0, 2, 1)),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )
        feat = feat[:, :, mel_len1:]
        if feat.shape[2] != mel_len2:
            raise RuntimeError(
                f"Unexpected generated mel length: {feat.shape[2]} != {mel_len2}"
            )
        return feat

# Ported from https://github.com/resemble-ai/chatterbox

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn


class CausalMaskedDiffWithXvec(nn.Module):
    """Causal masked diffusion model with speaker embeddings for streaming TTS."""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        decoder_conf: Dict = None,
        mel_feat_conf: Dict = None,
    ):
        """
        Args:
            input_size: Size of input token embeddings
            output_size: Size of output mel features
            spk_embed_dim: Dimension of speaker embeddings
            output_type: Type of output ("mel")
            vocab_size: Size of speech token vocabulary
            input_frame_rate: Frame rate of input tokens (Hz)
            only_mask_loss: Whether to use only masked loss
            token_mel_ratio: Ratio of tokens to mel frames (after upsampling)
            pre_lookahead_len: Length of lookahead window
            encoder: UpsampleConformerEncoder instance
            decoder: ConditionalDecoder instance
            decoder_conf: Decoder configuration dict
            mel_feat_conf: Mel feature configuration dict
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf or {}
        self.mel_feat_conf = mel_feat_conf or {}
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    def inference(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: mx.array,
        embedding: mx.array,
        finalize: bool,
    ):
        """
        Inference for streaming TTS.

        Args:
            token: Speech tokens (1, T)
            token_len: Token lengths (1,)
            prompt_token: Prompt tokens (1, T_p)
            prompt_token_len: Prompt token lengths (1,)
            prompt_feat: Prompt mel features (1, T_p_mel, D)
            prompt_feat_len: Prompt feature lengths (1,)
            embedding: Speaker embedding (1, spk_embed_dim)
            finalize: Whether this is the final chunk

        Returns:
            feat: Generated mel features (1, D, T_mel)
            flow_cache: Flow cache (None for causal version)
        """
        assert token.shape[0] == 1

        # Speaker embedding projection
        embedding = embedding / mx.linalg.norm(
            embedding, axis=1, keepdims=True
        )  # Normalize
        embedding = self.spk_embed_affine_layer(embedding)

        # Concatenate prompt and new tokens
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len

        # Create mask
        batch_size = token_len.shape[0]
        max_len = int(mx.max(token_len).item())
        seq_range = mx.arange(max_len)
        seq_range_expand = mx.expand_dims(seq_range, 0)
        seq_range_expand = mx.broadcast_to(seq_range_expand, (batch_size, max_len))
        seq_length_expand = mx.expand_dims(token_len, -1)
        mask = seq_range_expand < seq_length_expand
        mask = mx.expand_dims(mask, -1).astype(embedding.dtype)

        # Embed tokens
        # MLX Embedding doesn't have num_embeddings attr, use weight shape instead
        num_embeddings = self.input_embedding.weight.shape[0]
        token = mx.clip(token, 0, num_embeddings - 1)
        token = self.input_embedding(token) * mask

        # Encode
        h, h_lengths = self.encoder(token, token_len)

        # Trim lookahead for streaming (unless finalizing)
        if not finalize:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # Get conditions (prompt mel features)
        conds = mx.zeros([1, mel_len1 + mel_len2, self.output_size], dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = mx.transpose(conds, [0, 2, 1])  # (1, D, T)

        # Create mask for decoder (float dtype for multiplication)
        total_len = mel_len1 + mel_len2
        mask = mx.ones([1, 1, total_len], dtype=h.dtype)

        # Generate mel features
        feat, _ = self.decoder(
            mu=mx.transpose(h, [0, 2, 1]),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=10,
        )

        # Extract only the new portion (after prompt)
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2

        return feat, None

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import TransformerConfig
from .layers import Attention, FeedForward, rms_norm
from .rope import LTXRopeType, precompute_freqs_cis
from .scheduler import to_denoised
from .timestep import AdaLayerNormSingle, adaln_embedding_coefficient


@dataclass(frozen=True)
class Modality:
    latent: mx.array
    sigma: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: mx.array | None = None
    attention_mask: mx.array | None = None


@dataclass(frozen=True)
class TransformerArgs:
    x: mx.array
    context: mx.array
    context_mask: mx.array | None
    timesteps: mx.array
    embedded_timestep: mx.array
    positional_embeddings: tuple[mx.array, mx.array]
    enabled: bool
    prompt_timestep: mx.array | None = None
    self_attention_mask: mx.array | None = None


class TransformerArgsPreprocessor:
    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[float],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType | str,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ):
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.prompt_adaln = prompt_adaln

    def _prepare_timestep(
        self,
        timestep: mx.array,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype,
    ) -> tuple[mx.array, mx.array]:
        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep_out, embedded = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        timestep_out = timestep_out.reshape(batch_size, -1, timestep_out.shape[-1])
        embedded = embedded.reshape(batch_size, -1, embedded.shape[-1])
        return timestep_out, embedded

    @staticmethod
    def _prepare_context_mask(mask: mx.array | None, x_dtype) -> mx.array | None:
        if mask is None or mx.issubdtype(mask.dtype, mx.floating):
            return mask
        return (mask.astype(mx.int64) - 1).astype(x_dtype).reshape(
            mask.shape[0], 1, -1, mask.shape[-1]
        ) * mx.finfo(x_dtype).max

    @staticmethod
    def _prepare_self_attention_mask(
        mask: mx.array | None,
        x_dtype,
    ) -> mx.array | None:
        if mask is None:
            return None
        finfo = mx.finfo(x_dtype)
        eps = finfo.eps
        positive = mask > 0
        bias = mx.full(mask.shape, finfo.min, dtype=x_dtype)
        safe_log = mx.log(mx.maximum(mask.astype(x_dtype), eps))
        return mx.where(positive, safe_log, bias)[:, None, :, :]

    def prepare(self, modality: Modality) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps,
            self.adaln,
            batch_size,
            modality.latent.dtype,
        )
        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep(
                modality.sigma,
                self.prompt_adaln,
                batch_size,
                modality.latent.dtype,
            )
        pe = precompute_freqs_cis(
            modality.positions,
            dim=self.inner_dim,
            out_dtype=modality.latent.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            double_precision=self.double_precision_rope,
        )
        return TransformerArgs(
            x=x,
            context=modality.context.reshape(batch_size, -1, x.shape[-1]),
            context_mask=self._prepare_context_mask(
                modality.context_mask,
                modality.latent.dtype,
            ),
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self._prepare_self_attention_mask(
                modality.attention_mask,
                modality.latent.dtype,
            ),
        )


class BasicAudioTransformerBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        dim: int,
        heads: int,
        dim_head: int,
        context_dim: int,
        norm_eps: float,
        rope_type: LTXRopeType | str,
        cross_attention_adaln: bool = True,
        apply_gated_attention: bool = True,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps
        self.cross_attention_adaln = cross_attention_adaln
        self.audio_attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.audio_attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.audio_ff = FeedForward(dim, dim_out=dim)
        self.audio_scale_shift_table = mx.zeros(
            (adaln_embedding_coefficient(cross_attention_adaln), dim)
        )
        if cross_attention_adaln:
            self.audio_prompt_scale_shift_table = mx.zeros((2, dim))

    def _ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        timestep: mx.array,
        start: int,
        stop: int,
    ) -> tuple[mx.array, ...]:
        num = scale_shift_table.shape[0]
        values = (
            scale_shift_table[start:stop][None, None]
            + timestep.reshape(batch_size, timestep.shape[1], num, -1)[
                :, :, start:stop, :
            ]
        )
        return tuple(mx.split(values, stop - start, axis=2))

    def _cross_attention(
        self,
        x: mx.array,
        args: TransformerArgs,
    ) -> mx.array:
        if not self.cross_attention_adaln:
            return self.audio_attn2(
                rms_norm(x, eps=self.norm_eps),
                context=args.context,
                mask=args.context_mask,
            )
        shift_q, scale_q, gate = (
            v.squeeze(2)
            for v in self._ada_values(
                self.audio_scale_shift_table,
                x.shape[0],
                args.timesteps,
                6,
                9,
            )
        )
        if args.prompt_timestep is None:
            raise ValueError("cross_attention_adaln requires prompt_timestep")
        prompt_values = self.audio_prompt_scale_shift_table[
            None, None
        ] + args.prompt_timestep.reshape(
            x.shape[0], args.prompt_timestep.shape[1], 2, -1
        )
        shift_kv, scale_kv = (v.squeeze(2) for v in mx.split(prompt_values, 2, axis=2))
        attn_input = rms_norm(x, eps=self.norm_eps) * (1 + scale_q) + shift_q
        context = args.context * (1 + scale_kv) + shift_kv
        return (
            self.audio_attn2(attn_input, context=context, mask=args.context_mask) * gate
        )

    def __call__(
        self,
        args: TransformerArgs,
        skip_audio_self_attn: bool = False,
    ) -> TransformerArgs:
        x = args.x
        shift_msa, scale_msa, gate_msa = (
            v.squeeze(2)
            for v in self._ada_values(
                self.audio_scale_shift_table,
                x.shape[0],
                args.timesteps,
                0,
                3,
            )
        )
        norm_x = rms_norm(x, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
        x = (
            x
            + self.audio_attn1(
                norm_x,
                pe=args.positional_embeddings,
                mask=args.self_attention_mask,
                all_perturbed=skip_audio_self_attn,
            )
            * gate_msa
        )
        x = x + self._cross_attention(x, args)
        shift_mlp, scale_mlp, gate_mlp = (
            v.squeeze(2)
            for v in self._ada_values(
                self.audio_scale_shift_table,
                x.shape[0],
                args.timesteps,
                3,
                6,
            )
        )
        x = (
            x
            + self.audio_ff(
                rms_norm(x, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
            )
            * gate_mlp
        )
        return TransformerArgs(**{**args.__dict__, "x": x})


class AudioOnlyLTXModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.audio_inner_dim = (
            config.audio_num_attention_heads * config.audio_attention_head_dim
        )
        self.audio_patchify_proj = nn.Linear(
            config.audio_in_channels,
            self.audio_inner_dim,
            bias=True,
        )
        coefficient = adaln_embedding_coefficient(config.cross_attention_adaln)
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=coefficient,
        )
        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2)
            if config.cross_attention_adaln
            else None
        )
        self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim,
            eps=config.norm_eps,
            affine=False,
        )
        self.audio_proj_out = nn.Linear(
            self.audio_inner_dim,
            config.audio_out_channels,
            bias=True,
        )
        self.audio_args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            inner_dim=self.audio_inner_dim,
            max_pos=config.audio_positional_embedding_max_pos,
            num_attention_heads=config.audio_num_attention_heads,
            use_middle_indices_grid=config.use_middle_indices_grid,
            timestep_scale_multiplier=config.timestep_scale_multiplier,
            double_precision_rope=config.frequencies_precision == "float64",
            positional_embedding_theta=config.positional_embedding_theta,
            rope_type=LTXRopeType(config.rope_type),
            prompt_adaln=self.audio_prompt_adaln_single,
        )
        self.transformer_blocks = [
            BasicAudioTransformerBlock(
                idx=i,
                dim=self.audio_inner_dim,
                heads=config.audio_num_attention_heads,
                dim_head=config.audio_attention_head_dim,
                context_dim=config.audio_cross_attention_dim,
                norm_eps=config.norm_eps,
                rope_type=LTXRopeType(config.rope_type),
                cross_attention_adaln=config.cross_attention_adaln,
                apply_gated_attention=config.apply_gated_attention,
            )
            for i in range(config.num_layers)
        ]

    def _process_output(self, x: mx.array, embedded_timestep: mx.array) -> mx.array:
        values = (
            self.audio_scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = (v.squeeze(2) for v in mx.split(values, 2, axis=2))
        x = self.audio_norm_out(x)
        return self.audio_proj_out(x * (1 + scale) + shift)

    def __call__(
        self,
        audio: Modality,
        stg_blocks: set[int] | None = None,
    ) -> mx.array:
        args = self.audio_args_preprocessor.prepare(audio)
        stg_blocks = stg_blocks or set()
        for block in self.transformer_blocks:
            args = block(args, skip_audio_self_attn=block.idx in stg_blocks)
        return self._process_output(args.x, args.embedded_timestep)


class X0Model(nn.Module):
    def __init__(self, velocity_model: AudioOnlyLTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        audio: Modality,
        stg_blocks: set[int] | None = None,
    ) -> mx.array:
        velocity = self.velocity_model(audio, stg_blocks=stg_blocks)
        return to_denoised(audio.latent, velocity, audio.timesteps[..., None])

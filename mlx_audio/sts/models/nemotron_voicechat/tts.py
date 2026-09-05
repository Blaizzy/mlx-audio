from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models import gemma3_text
from mlx_audio.lm.models.cache import KVCache, RotatingKVCache
from mlx_audio.lm.sample_utils import apply_top_p

from . import t5gemma
from .config import VoiceChatTTSConfig


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, inputs: mx.array) -> mx.array:
        return mx.fast.rms_norm(inputs, 1.0 + self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.down_proj(
            nn.gelu_approx(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class MLPLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size)

    def __call__(self, inputs: mx.array) -> mx.array:
        return inputs + self.post_norm(self.mlp(self.pre_norm(inputs)))


class MoGHead(nn.Module):
    def __init__(self, config: VoiceChatTTSConfig):
        super().__init__()
        self.low_rank = config.mog_low_rank
        self.num_predictions = config.mog_num_predictions
        self.min_log_std = config.mog_min_log_std
        self.mlp_stack = [
            MLPLayer(config.hidden_size, config.mog_intermediate_size)
            for _ in range(config.mog_num_layers)
        ] + [RMSNorm(config.hidden_size)]
        self.proj_logits = nn.Linear(
            config.hidden_size, config.mog_num_predictions, bias=False
        )
        self.proj_mus = nn.Linear(
            config.hidden_size,
            config.mog_num_predictions * config.mog_low_rank,
            bias=False,
        )
        self.proj_logs = nn.Linear(config.hidden_size, 1, bias=False)
        self.proj_else = nn.Linear(config.hidden_size, config.latent_size, bias=False)
        self.low_mat = mx.zeros(
            (
                config.mog_num_predictions,
                config.latent_size,
                config.mog_low_rank,
            )
        )

    def infer(
        self,
        inputs: mx.array,
        *,
        guidance_scale: float,
        top_p: float,
    ) -> tuple[mx.array, mx.array]:
        for layer in self.mlp_stack:
            inputs = layer(inputs)

        if guidance_scale > 0:
            conditional, unconditional = mx.split(inputs, 2, axis=0)
            inputs = conditional + guidance_scale * (conditional - unconditional)

        logits = self.proj_logits(inputs)
        log_probabilities = mx.log(mx.softmax(logits, axis=-1))
        if 0 < top_p < 1:
            log_probabilities = apply_top_p(log_probabilities, top_p)
        mixture_indices = mx.random.categorical(log_probabilities)

        batch, length, _ = inputs.shape
        low_rank_means = self.proj_mus(inputs).reshape(
            batch, length, self.num_predictions, self.low_rank
        )
        selected_means = mx.take_along_axis(
            low_rank_means,
            mixture_indices[..., None, None],
            axis=2,
        ).squeeze(2)
        selected_projection = self.low_mat[mixture_indices]
        means = mx.einsum("btol,btl->bto", selected_projection, selected_means)
        residual = self.proj_else(inputs)
        log_stds = mx.maximum(self.proj_logs(inputs), self.min_log_std)
        return means * mx.exp(log_stds) + residual, log_stds


class SubwordFlagEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.is_continuation = mx.zeros((vocab_size + 1,), dtype=mx.int64)
        self.pad_tensor = mx.array(vocab_size, dtype=mx.int64)
        self.cont_emb = nn.Embedding(2, hidden_size)

    def __call__(self, inputs: mx.array, token_ids: mx.array) -> mx.array:
        safe_ids = mx.where(
            token_ids >= self.is_continuation.shape[0] - 1,
            self.pad_tensor,
            token_ids,
        )
        return inputs + self.cont_emb(self.is_continuation[safe_ids])


class BOSEOSEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.special_flags = mx.zeros((vocab_size,), dtype=mx.int64)
        self.pad_tensor = mx.array(vocab_size - 1, dtype=mx.int64)
        self.special_emb = nn.Embedding(3, hidden_size)

    def __call__(self, inputs: mx.array, token_ids: mx.array) -> mx.array:
        safe_ids = mx.where(
            token_ids >= self.special_flags.shape[0], self.pad_tensor, token_ids
        )
        return inputs + self.special_emb(self.special_flags[safe_ids])


class CharAwareSubwordEncoder(nn.Module):
    def __init__(self, config: VoiceChatTTSConfig):
        super().__init__()
        self.backbone = t5gemma.Model(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        self.embed_tokens = nn.Embedding(config.char_vocab_size + 1, config.hidden_size)
        self.proj_embedding = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.subword_flag_emb = SubwordFlagEmbedding(
            config.text_vocab_size, config.hidden_size
        )
        self.bos_eos_emb = BOSEOSEmbedding(config.text_vocab_size, config.hidden_size)
        self.char_padding_idx = config.char_vocab_size
        self.subword_id_to_char_ids: dict[int, tuple[int, ...]] = {}

    def set_tokenizer(self, tokenizer) -> None:
        vocabulary = tokenizer.get_vocab()
        single_characters = {
            token: token_id for token, token_id in vocabulary.items() if len(token) == 1
        }
        characters = sorted(single_characters, key=single_characters.get)
        char_vocabulary = {
            character: index for index, character in enumerate(characters)
        }
        if len(char_vocabulary) != self.char_padding_idx:
            raise ValueError(
                "Tokenizer character vocabulary does not match VoiceChat weights: "
                f"expected {self.char_padding_idx}, got {len(char_vocabulary)}"
            )
        self.subword_id_to_char_ids = {
            token_id: tuple(
                char_vocabulary[character]
                for character in token
                if character in char_vocabulary
            )
            for token, token_id in vocabulary.items()
        }
        self.subword_id_to_char_ids = {
            token_id: character_ids
            for token_id, character_ids in self.subword_id_to_char_ids.items()
            if character_ids
        }
        tokens = {token_id: token for token, token_id in vocabulary.items()}
        continuation_flags = [0] * self.subword_flag_emb.is_continuation.shape[0]
        for token_id in range(len(continuation_flags) - 1):
            token = tokens.get(token_id, "")
            continuation_flags[token_id] = int(
                bool(token) and not token.startswith(("Ġ", "▁", "<"))
            )
        self.subword_flag_emb.is_continuation = mx.array(
            continuation_flags, dtype=mx.int64
        )

        special_flags = [0] * self.bos_eos_emb.special_flags.shape[0]
        bos_id = getattr(tokenizer, "bos_token_id", vocabulary.get("<s>"))
        eos_id = getattr(tokenizer, "eos_token_id", vocabulary.get("</s>"))
        if bos_id is not None and bos_id < len(special_flags):
            special_flags[bos_id] = 1
        if eos_id is not None and eos_id < len(special_flags):
            special_flags[eos_id] = 2
        self.bos_eos_emb.special_flags = mx.array(special_flags, dtype=mx.int64)

    def __call__(
        self, token_ids: mx.array, subword_mask: mx.array | None = None
    ) -> mx.array:
        if not self.subword_id_to_char_ids:
            raise RuntimeError("VoiceChat tokenizer has not been initialized")
        if subword_mask is None:
            subword_mask = mx.ones(token_ids.shape, dtype=mx.bool_)

        positions: list[tuple[int, int]] = []
        character_sequences: list[tuple[int, ...]] = []
        mask_values = subword_mask.tolist()
        token_values = token_ids.tolist()
        for batch_index, row in enumerate(mask_values):
            for time_index, enabled in enumerate(row):
                if enabled:
                    positions.append((batch_index, time_index))
                    character_sequences.append(
                        self.subword_id_to_char_ids.get(
                            int(token_values[batch_index][time_index]), ()
                        )
                    )

        output = mx.zeros(
            token_ids.shape + (self.proj_embedding.weight.shape[0],),
            dtype=self.proj_embedding.weight.dtype,
        )
        if positions:
            max_length = max(max(map(len, character_sequences)), 1)
            character_ids = mx.full(
                (len(positions), max_length),
                self.char_padding_idx,
                dtype=mx.int32,
            )
            character_mask = mx.zeros((len(positions), max_length), dtype=mx.bool_)
            for index, sequence in enumerate(character_sequences):
                if sequence:
                    character_ids[index, : len(sequence)] = mx.array(sequence)
                    character_mask[index, : len(sequence)] = True

            hidden = self.backbone(
                self.embed_tokens(character_ids), attention_mask=character_mask
            )
            lengths = mx.maximum(character_mask.sum(axis=1, keepdims=True), 1)
            pooled = (hidden * character_mask[..., None]).sum(axis=1) / lengths
            encoded = self.proj_embedding(pooled)
            for index, (batch_index, time_index) in enumerate(positions):
                output[batch_index, time_index] = encoded[index]

        output = self.subword_flag_emb(output, token_ids)
        return self.bos_eos_emb(output, token_ids)


class GatedProjectedSumRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, num_quantizers: int):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = mx.zeros((hidden_size,), dtype=mx.float32)
        self.residual_scale = mx.array(0.5, dtype=mx.float32)
        self.final_norm = RMSNorm(hidden_size)

    def __call__(self, audio: mx.array, text: mx.array) -> mx.array:
        audio = self.audio_proj(audio / self.num_quantizers)
        text = self.text_proj(text)
        gate = mx.sigmoid(self.gate).astype(audio.dtype)
        residual_scale = mx.sigmoid(self.residual_scale).astype(audio.dtype)
        return self.final_norm(residual_scale * (gate * audio + (1 - gate) * text))


class EARTTSModel(nn.Module):
    def __init__(self, config: VoiceChatTTSConfig):
        super().__init__()
        self.config = config
        backbone_args = gemma3_text.ModelArgs(
            model_type="gemma3_text",
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            sliding_window=config.sliding_window,
            sliding_window_pattern=min(6, config.num_hidden_layers),
            max_position_embeddings=131_072,
            vocab_size=1,
        )
        self.backbone = gemma3_text.Gemma3Model(backbone_args)
        self.backbone.pop("embed_tokens")
        self.bos_emb = mx.zeros((config.hidden_size,))
        self.null_emb = mx.zeros((config.hidden_size,))
        self.embed_code = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.embed_subword = CharAwareSubwordEncoder(config)
        self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
            config.hidden_size, config.num_quantizers
        )
        self.mog_head = MoGHead(config)
        self.rvq_embs = mx.zeros(
            (config.num_quantizers, config.codebook_size, config.latent_size)
        )

    def make_cache(self):
        caches = []
        pattern = self.backbone.sliding_window_pattern
        for index in range(self.config.num_hidden_layers):
            if index % pattern == pattern - 1:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window))
        return caches

    def depthsum_embedding(self, codes: mx.array) -> mx.array:
        embeddings = mx.pad(self.rvq_embs, ((0, 0), (0, 1), (0, 0)))
        output = mx.zeros(
            codes.shape[:2] + (self.config.latent_size,),
            dtype=embeddings.dtype,
        )
        for index in range(codes.shape[-1]):
            output = output + embeddings[index][codes[..., index]]
        return output

    def _conditioning(
        self,
        subword_ids: mx.array,
        subword_mask: mx.array,
        guidance_enabled: bool,
    ) -> mx.array:
        conditioning = self.embed_subword(subword_ids, subword_mask)
        if guidance_enabled:
            unconditional = mx.broadcast_to(self.null_emb, conditioning.shape)
            conditioning = mx.concatenate([conditioning, unconditional], axis=0)
        return conditioning

    def warmup(
        self,
        codes: mx.array,
        audio_mask: mx.array,
        subword_ids: mx.array,
        subword_mask: mx.array,
        audio_prompt_latent: mx.array,
        *,
        guidance_enabled: bool,
    ):
        shifted_codes = mx.pad(codes[:, :-1], ((0, 0), (1, 0), (0, 0)))
        code_embeddings = self.embed_code(self.depthsum_embedding(shifted_codes))
        bos_mask = audio_mask & ~mx.pad(audio_mask[:, :-1], ((0, 0), (1, 0)))
        pre_bos_mask = mx.cumsum(bos_mask, axis=1) == 0
        code_embeddings = mx.where(
            pre_bos_mask[..., None], audio_prompt_latent, code_embeddings
        )
        code_embeddings = code_embeddings + bos_mask[..., None] * self.bos_emb
        conditioning = self._conditioning(subword_ids, subword_mask, guidance_enabled)
        if guidance_enabled:
            code_embeddings = mx.concatenate([code_embeddings] * 2, axis=0)
        inputs = self.gated_fusion_audio_text(code_embeddings, conditioning)
        cache = self.make_cache()
        self.backbone(None, cache=cache, input_embeddings=inputs)
        return codes[:, -1:], cache

    def generate_step(
        self,
        current_subword_id: mx.array,
        previous_codes: mx.array,
        cache,
        *,
        text_eos_id: int,
        silence_codes: mx.array,
        guidance_enabled: bool,
    ) -> tuple[mx.array, list]:
        previous_codes = mx.where(
            (current_subword_id == text_eos_id)[..., None],
            mx.broadcast_to(silence_codes, previous_codes.shape),
            previous_codes,
        )
        code_embeddings = self.embed_code(self.depthsum_embedding(previous_codes))
        subword_mask = mx.ones(current_subword_id.shape, dtype=mx.bool_)
        conditioning = self._conditioning(
            current_subword_id, subword_mask, guidance_enabled
        )
        if guidance_enabled:
            code_embeddings = mx.concatenate([code_embeddings] * 2, axis=0)
        inputs = self.gated_fusion_audio_text(code_embeddings, conditioning)
        hidden = self.backbone(None, cache=cache, input_embeddings=inputs)
        codes = self._generate_codes(hidden, guidance_enabled=guidance_enabled)
        return codes, cache

    def _generate_codes(self, hidden: mx.array, *, guidance_enabled: bool) -> mx.array:
        if guidance_enabled:
            conditional, _ = mx.split(hidden, 2, axis=0)
        else:
            conditional = hidden
        batch, length, _ = conditional.shape
        codes = mx.full(
            (batch, length, self.config.num_quantizers),
            self.config.codebook_size,
            dtype=mx.int32,
        )
        rates = [
            index / self.config.num_iterations
            for index in range(self.config.num_iterations)
        ]
        masked = [
            math.ceil(
                ((1 - rate**self.config.exponent) ** (1 / self.config.exponent))
                * self.config.num_quantizers
            )
            for rate in rates
        ] + [0]

        completed = 0
        for current, following in zip(masked, masked[1:]):
            count = current - following
            if count <= 0:
                continue
            embedded = self.embed_code(self.depthsum_embedding(codes))
            if guidance_enabled:
                conditional_hidden, unconditional_hidden = mx.split(hidden, 2, axis=0)
                mog_inputs = mx.concatenate(
                    [
                        embedded + conditional_hidden,
                        embedded + unconditional_hidden,
                    ],
                    axis=0,
                )
            else:
                mog_inputs = embedded + hidden
            means, log_stds = self.mog_head.infer(
                mog_inputs,
                guidance_scale=(
                    self.config.guidance_scale if guidance_enabled else 0.0
                ),
                top_p=self.config.top_p,
            )
            residual = (
                means
                + mx.exp(log_stds)
                * mx.random.normal(means.shape)
                * self.config.noise_scale
            )
            for quantizer in range(completed, completed + count):
                codebook = self.rvq_embs[quantizer]
                distances = mx.sum(codebook * codebook, axis=-1)[None, None, :] - 2 * (
                    residual @ codebook.T
                )
                indices = mx.argmin(distances, axis=-1)
                codes[..., quantizer] = indices
                residual = residual - codebook[indices]
            completed += count
        return codes

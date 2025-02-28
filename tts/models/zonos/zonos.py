import json
from typing import Callable, Dict, Any, Optional, Union, Tuple

import safetensors
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import find_multiple, pad_weight_

# Default backbone needs to be adapted for MLX
DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_multiple(n: int, k: int) -> int:
    if k == 0 or n % k == 0:
        return n
    return n + k - (n % k)


def pad_weight_(w: nn.Embedding | nn.Linear, multiple: int):
    """Pad the weight of an embedding or linear layer to a multiple of `multiple`."""
    if isinstance(w, nn.Embedding):
        # Pad input dim
        if w.weight.shape[1] % multiple == 0:
            return
        w.weight = mx.pad(w.weight, (0, 0, 0, w.weight.shape[1] % multiple))
        w.num_embeddings, w.embedding_dim = w.weight.shape
    elif isinstance(w, nn.Linear):
        # Pad output dim
        if w.weight.shape[0] % multiple == 0:
            return
        w.weight = mx.pad(w.weight, (0, 0, 0, w.weight.shape[0] % multiple))
        w.out_features, w.in_features = w.weight.shape
    else:
        raise ValueError(f"Unsupported weight type: {type(w)}")

class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # MLX embedding layers
        self.embeddings = [nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)]
        self.heads = [nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)]

        # MLX doesn't need CUDA graph-related attributes
        # Removed: self._cg_graph, self._cg_batch_size, etc.

        if config.pad_vocab_to_multiple_of:
            # Will need custom implementation for MLX
            self._pad_embeddings_and_heads_flag = True

    def _pad_embeddings_and_heads(self):
        # Custom implementation for MLX
        for w in [*self.embeddings, *self.heads]:
            # Adapt pad_weight_ for MLX
            # This would need to be reimplemented for MLX
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)


    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: Optional[str] = None, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, backbone: Optional[str] = None
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # MLX would have different optimized backbones
            if is_transformer and "mlx" in BACKBONES:
                backbone_cls = BACKBONES["mlx"]

        model = cls(config, backbone_cls)
        # No need to move to device in MLX

        # Load state dict
        sd = model.parameters()
        with safetensors.safe_open(model_path, framework="pt") as f:
            # This needs to be adapted for MLX tensors
            for k in f.keys():
                # Convert PyTorch tensor to MLX array
                tensor = f.get_tensor(k)
                # This would need a custom conversion from PyTorch to MLX
                # sd[k] = mx.array(tensor.numpy(), dtype=mx.bfloat16)

        # For MLX, we would need a different state dict loading mechanism
        # model.update(sd)

        return model

    def make_speaker_embedding(self, wav: mx.array, sr: int) -> mx.array:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        # This would need adaptation for MLX
        _, spk_embedding = self.spk_clone_model(wav, sr)
        return mx.expand_dims(spk_embedding, axis=0)

    def embed_codes(self, codes: mx.array) -> mx.array:
        # Sum embeddings from each codebook
        result = self.embeddings[0](codes[:, 0])
        for i in range(1, len(self.embeddings)):
            result = result + self.embeddings[i](codes[:, i])
        return result

    def apply_heads(self, hidden_states: mx.array) -> mx.array:
        # Apply each head and stack results
        heads_output = [head(hidden_states) for head in self.heads]
        return mx.stack(heads_output, axis=1)

    def _compute_logits(
        self, hidden_states: mx.array, inference_params: InferenceParams, cfg_scale: float
    ) -> mx.array:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        # Get last hidden states
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :]
        last_hidden_states = mx.expand_dims(last_hidden_states, axis=1)

        # Apply heads and convert to float
        logits = self.apply_heads(last_hidden_states)
        logits = mx.squeeze(logits, axis=2)
        logits = mx.cast(logits, mx.float32)

        # Apply classifier-free guidance if needed
        if cfg_scale != 1.0:
            # Split logits for conditional and unconditional
            cond_logits = logits[:logits.shape[0]//2]
            uncond_logits = logits[logits.shape[0]//2:]
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        # Mask padding tokens
        # MLX equivalent of filling with -inf
        # This would need custom implementation
        # logits[..., 1025:] = -float('inf')

        return logits

    def _decode_one_token(
        self,
        input_ids: mx.array,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> mx.array:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        MLX doesn't use CUDA graphs, so the implementation is simplified.
        """
        # For cfg_scale == 1.0
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        # For classifier-free guidance
        hidden_states = self.embed_codes(input_ids)
        # Repeat for conditional and unconditional
        hidden_states = mx.repeat(hidden_states, 2, axis=0)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def _prefill(
        self,
        prefix_hidden_states: mx.array,
        input_ids: mx.array,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> mx.array:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            # MLX equivalent of expand
            input_ids = mx.repeat(input_ids, prefix_hidden_states.shape[0], axis=0)

        # Concatenate prefix hidden states with new embeddings
        hidden_states = mx.concatenate([prefix_hidden_states, self.embed_codes(input_ids)], axis=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype=mx.bfloat16) -> InferenceParams:
        max_seqlen = find_multiple(max_seqlen, 8)
        # MLX equivalent of allocating cache
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        # Create lengths per sample tensor
        lengths_per_sample = mx.zeros((batch_size,), dtype=mx.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: Optional[dict] = None) -> mx.array:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}

        # Generate and concatenate conditionings
        cond = self.prefix_conditioner(cond_dict)
        uncond = self.prefix_conditioner(uncond_dict)
        return mx.concatenate([cond, uncond], axis=0)

    def generate(
        self,
        prefix_conditioning: mx.array,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: Optional[mx.array] = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
        progress_bar: bool = True,
        callback: Optional[Callable[[mx.array, int, int], bool]] = None,
    ):
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]

        # MLX doesn't use CUDA graphs or torch.compile
        decode_one_token = self._decode_one_token

        unknown_token = -1
        audio_seq_len = prefix_audio_len + max_new_tokens
        seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

        # Setup for generation
        inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
        codes = mx.full((batch_size, 9, audio_seq_len), unknown_token, dtype=mx.int32)

        if audio_prefix_codes is not None:
            # Set prefix codes
            # MLX equivalent of tensor assignment
            codes = mx.array_scatter(codes, audio_prefix_codes, axis_indices=(None, None, slice(0, prefix_audio_len)))

        # Apply delay pattern
        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

        # Get prefix audio codes with delay pattern
        delayed_prefix_audio_codes = delayed_codes[..., :prefix_audio_len + 1]

        # Initial prefill
        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params)

        # Update codes with first token
        offset = delayed_prefix_audio_codes.shape[2]
        frame = delayed_codes[..., offset:offset + 1]
        # MLX equivalent of masked_scatter_
        mask = frame == unknown_token
        frame = mx.array_scatter(frame, next_token, axis_indices=(mask,))

        # Update inference params
        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample = inference_params.lengths_per_sample + prefix_length

        # Setup logit bias to only allow EOS in codebook 0
        logit_bias = mx.zeros_like(logits)
        # MLX equivalent of setting -inf
        # This would need custom implementation
        # logit_bias[:, 1:, self.eos_token_id] = -float('inf')

        # Setup stopping conditions
        stopping = mx.zeros((batch_size,), dtype=mx.bool_)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = mx.full((batch_size,), max_steps, dtype=mx.int32)
        progress = tqdm(total=max_steps, desc="Generating", disable=not progress_bar)
        cfg_scale_tensor = mx.array(cfg_scale)

        # Generation loop
        step = 0
        while mx.max(remaining_steps) > 0:
            offset += 1
            input_ids = delayed_codes[..., offset - 1:offset]
            logits = decode_one_token(input_ids, inference_params, cfg_scale_tensor)
            logits = logits + logit_bias

            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)
            eos_in_cb0 = next_token[:, 0] == self.eos_token_id

            # Update remaining steps and stopping condition
            # MLX equivalent of minimum
            remaining_steps = mx.where(eos_in_cb0[:, 0], mx.minimum(remaining_steps, mx.array(9)), remaining_steps)
            stopping = stopping | eos_in_cb0[:, 0]

            # Handle EOS tokens
            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx = mx.clip(eos_codebook_idx, 0, 9 - 1)

            # MLX doesn't have direct indexing like PyTorch
            # This would require a custom implementation
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = int(eos_codebook_idx[i])
                    # Set tokens before idx to masked_token_id
                    next_token = mx.array_scatter(
                        next_token,
                        mx.full((idx,), self.masked_token_id),
                        axis_indices=(mx.array([i]), mx.arange(idx))
                    )
                    # Set token at idx to eos_token_id
                    next_token = mx.array_scatter(
                        next_token,
                        mx.array([self.eos_token_id]),
                        axis_indices=(mx.array([i]), mx.array([idx]))
                    )

            # Update codes with next token
            frame = delayed_codes[..., offset:offset + 1]
            # MLX equivalent of masked_scatter_
            mask = frame == unknown_token
            frame = mx.array_scatter(frame, next_token, axis_indices=(mask,))

            # Update inference parameters
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample = inference_params.lengths_per_sample + 1

            # Decrement remaining steps
            remaining_steps = remaining_steps - 1

            progress.update()
            step += 1

            # Check callback
            if callback is not None and not callback(frame, step, max_steps):
                break

        # Revert delay pattern and prepare output codes
        out_codes = revert_delay_pattern(delayed_codes)
        # MLX equivalent of masked_fill_
        out_codes = mx.where(out_codes >= 1024, mx.zeros_like(out_codes), out_codes)
        out_codes = out_codes[..., :offset - 9]

        return out_codes
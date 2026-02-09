import glob
import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_audio.stt.generate import wired_limit
from mlx_audio.stt.utils import get_model_path

from ..base import STTOutput
from .config import AudioConfig, ModelConfig


class Attention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = config.d_model // config.encoder_attention_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        bsz, tgt_len, _ = x.shape

        query_states = self.q_proj(x) * self.scaling
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=1.0, mask=mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, tgt_len, self.embed_dim
        )

        return self.out_proj(attn_output)


class VoxtralEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        r = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = r + x

        r = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = r + x

        return x


class Encoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.layers = [
            VoxtralEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:

        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        embed_pos = self.embed_positions.weight

        x = (x + embed_pos).astype(x.dtype)

        for encoder_layer in self.layers:
            x = encoder_layer(x, mask)

        return self.layer_norm(x)


class MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.intermediate_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def __call__(self, audio_features: mx.array) -> mx.array:
        hidden_states = self.linear_1(audio_features)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        from mlx_lm.models.llama import LlamaModel

        self.model = LlamaModel(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.text_config.vocab_size

        self.language_model = LanguageModel(config.text_config)

        self.audio_tower = Encoder(config.audio_config)
        self.multi_modal_projector = MultiModalProjector(config)

    def get_audio_embeds(self, x: mx.array) -> mx.array:
        audio_embeds = self.audio_tower(x).reshape(
            -1, self.config.audio_config.intermediate_size
        )
        audio_embeds = self.multi_modal_projector(audio_embeds)
        return audio_embeds

    def _merge_input_embeddings(
        self,
        input_ids: mx.array,
        input_features: mx.array,
        cache: Optional[mx.array] = None,
    ) -> mx.array:
        if input_ids is not None:
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = None

        if input_features is not None and (cache is None or cache[0].offset == 0):
            audio_embeds = self.get_audio_embeds(input_features)

            if inputs_embeds is not None:
                # Replace audio token placeholders with audio embeddings
                audio_token_mask = input_ids == self.config.audio_token_id
                # Expand audio_token_mask to match inputs_embeds shape
                audio_token_mask_expanded = audio_token_mask[..., None]
                audio_token_mask_expanded = mx.broadcast_to(
                    audio_token_mask_expanded, inputs_embeds.shape
                )
                # Expand audio_embeds to match the number of audio tokens
                audio_token_positions = np.where(audio_token_mask.flatten())[0].tolist()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                inputs_embeds_flat[audio_token_positions] = audio_embeds
                inputs_embeds = inputs_embeds_flat.reshape(inputs_embeds.shape)
            else:
                inputs_embeds = audio_embeds

        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        input_features: mx.array = None,
        cache: Optional[mx.array] = None,
    ) -> mx.array:

        inputs_embeds = self._merge_input_embeddings(
            input_ids=input_ids,
            input_features=input_features,
            cache=cache,
        )

        logits = self.language_model(input_embeddings=inputs_embeds, cache=cache)

        return logits

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "conv" in k and "weight" in k:
                if v.shape[-1] < v.shape[-2]:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights

    def model_quant_predicate(self, p, m):
        return not p.startswith("audio_tower")

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """
        Hook called after model weights are loaded.
        Used to initialize the processor which is required for audio/text input.
        """
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_path))
        model._processor = processor
        model._processor.tokenizer.eos_token_ids = getattr(
            model._processor.tokenizer, "eos_token_ids", [2, 4, 32000]
        )

        # Store model_repo for transcription requests
        if not hasattr(model.config, "model_repo") or model.config.model_repo is None:
            # Try to extract from model_path
            try:
                index = model_path.parts.index("hub")
                model.config.model_repo = (
                    model_path.parts[index + 1]
                    .replace("models--", "")
                    .replace("--", "/")
                )
            except (ValueError, IndexError):
                model.config.model_repo = str(model_path)

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        """
        Load a pretrained Voxtral model.

        .. deprecated::
            Use `mlx_audio.stt.load()` instead. This method will be removed in a future version.

        Args:
            model_path: Path to the model or HuggingFace repo ID
            config: Optional model configuration
            **kwargs: Additional arguments (revision, force_download)

        Returns:
            Model: The loaded model
        """
        warnings.warn(
            "Model.from_pretrained() is deprecated. Use mlx_audio.stt.load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from mlx_audio.stt.utils import load

        return load(model_path, **kwargs)

    def stream_generate(
        self,
        input_ids: Optional[mx.array] = None,
        *,
        input_features: Optional[mx.array] = None,
        max_tokens: int = 128,
        sampler: Optional[Callable[mx.array, mx.array]] = None,
        generation_stream: bool = False,
        verbose: bool = False,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:

        from mlx_lm.generate import generate_step

        input_embeddings = self._merge_input_embeddings(
            input_ids=input_ids,
            input_features=input_features,
        )[0]

        with wired_limit(self, None):
            for n, (token, logprobs) in tqdm(
                enumerate(
                    generate_step(
                        prompt=mx.array([]),
                        input_embeddings=input_embeddings,
                        model=self.language_model,
                        max_tokens=max_tokens,
                        sampler=sampler,
                    )
                ),
                total=max_tokens,
                disable=not verbose,
                desc="Streaming",
            ):
                if token in self._processor.tokenizer.eos_token_ids:
                    break

                yield token, logprobs

    def generate(
        self,
        audio: List[mx.array],
        *,
        message: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        language: str = "en",
        verbose: bool = False,
        generation_stream: bool = False,
    ) -> mx.array:

        start_time = time.time()

        if message is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": audio,
                        },
                    ],
                }
            ]

        inputs = self._processor.apply_transcription_request(
            language=language, audio=audio, model_id=self.config.model_repo
        )
        input_ids = mx.array(inputs["input_ids"])
        input_features = mx.array(inputs["input_features"]).transpose(0, 2, 1)

        generated = []

        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
            xtc_special_tokens=self._processor.tokenizer.encode("\n")
            + list(self._processor.tokenizer.eos_token_ids),
        )

        for token, _ in self.stream_generate(
            input_ids=input_ids,
            input_features=input_features,
            max_tokens=max_tokens,
            sampler=sampler,
            generation_stream=generation_stream,
            verbose=verbose,
        ):
            generated.append(token)

        end_time = time.time()

        # Clear cache after each segment to avoid memory leaks
        mx.clear_cache()

        return STTOutput(
            text=self._processor.decode(generated),
            prompt_tokens=input_ids.shape[1],
            generation_tokens=len(generated),
            total_tokens=input_ids.shape[1] + len(generated),
            total_time=end_time - start_time,
            prompt_tps=input_ids.shape[1] / (end_time - start_time),
            generation_tps=len(generated) / (end_time - start_time),
        )

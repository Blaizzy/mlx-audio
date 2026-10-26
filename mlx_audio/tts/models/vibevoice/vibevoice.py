# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
import re
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from mlx_audio.lm.sample_utils import apply_top_k, apply_top_p
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import load_audio

from .acoustic_tokenizer import AcousticTokenizer, TokenizerEncoder
from .config import ModelConfig, Qwen2DecoderConfig
from .diffusion_head import DiffusionHead
from .language_model import BinaryClassifier, Qwen2Model, SpeechConnector
from .scheduler import DPMSolverMultistepScheduler

TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6
NON_STREAMING_SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)


class SemanticTokenizer(nn.Module):
    """Waveform encoder for semantic feedback."""

    def __init__(self, config):
        super().__init__()
        self.encoder = TokenizerEncoder(config)

    def encode(self, audio, cache=None):
        return self.encoder(audio, cache=cache)


class Model(nn.Module):
    """VibeVoice TTS with voice-cache streaming or reference-audio conditioning."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        decoder_config = config.decoder_config

        if self.is_non_streaming:
            self.language_model = Qwen2Model(decoder_config, use_norm=True)
            self.lm_head = nn.Linear(
                decoder_config.hidden_size,
                decoder_config.vocab_size,
                bias=False,
            )

            self.acoustic_tokenizer = AcousticTokenizer(
                config.acoustic_tokenizer_config, include_encoder=True
            )
            self.semantic_tokenizer = SemanticTokenizer(
                config.semantic_tokenizer_config
            )
            self.acoustic_connector = SpeechConnector(
                input_dim=config.acoustic_vae_dim,
                output_dim=decoder_config.hidden_size,
            )
            self.semantic_connector = SpeechConnector(
                input_dim=config.semantic_vae_dim,
                output_dim=decoder_config.hidden_size,
            )
            self.prediction_head = DiffusionHead(config.diffusion_head_config)
        else:
            # Streaming architecture uses a split LM (text LM + TTS LM).
            tts_layers = config.tts_backbone_num_hidden_layers
            lm_layers = decoder_config.num_hidden_layers - tts_layers

            lm_config = Qwen2DecoderConfig(
                model_type=decoder_config.model_type,
                hidden_size=decoder_config.hidden_size,
                intermediate_size=decoder_config.intermediate_size,
                num_attention_heads=decoder_config.num_attention_heads,
                num_key_value_heads=decoder_config.num_key_value_heads,
                num_hidden_layers=lm_layers,
                rms_norm_eps=decoder_config.rms_norm_eps,
                vocab_size=decoder_config.vocab_size,
                max_position_embeddings=decoder_config.max_position_embeddings,
                rope_theta=decoder_config.rope_theta,
                head_dim=decoder_config.head_dim,
            )

            tts_lm_config = Qwen2DecoderConfig(
                model_type=decoder_config.model_type,
                hidden_size=decoder_config.hidden_size,
                intermediate_size=decoder_config.intermediate_size,
                num_attention_heads=decoder_config.num_attention_heads,
                num_key_value_heads=decoder_config.num_key_value_heads,
                num_hidden_layers=tts_layers,
                rms_norm_eps=decoder_config.rms_norm_eps,
                vocab_size=decoder_config.vocab_size,  # Reuse embeddings
                max_position_embeddings=decoder_config.max_position_embeddings,
                rope_theta=decoder_config.rope_theta,
                head_dim=decoder_config.head_dim,
            )

            # Base LM doesn't have final norm (it continues into tts_language_model)
            self.language_model = Qwen2Model(lm_config, use_norm=False)
            self.tts_language_model = Qwen2Model(tts_lm_config, use_norm=True)
            self.tts_input_types = nn.Embedding(2, decoder_config.hidden_size)
            self.acoustic_tokenizer = AcousticTokenizer(
                config.acoustic_tokenizer_config
            )
            self.acoustic_connector = SpeechConnector(
                input_dim=config.acoustic_vae_dim,
                output_dim=decoder_config.hidden_size,
            )
            self.prediction_head = DiffusionHead(config.diffusion_head_config)
            self.tts_eos_classifier = BinaryClassifier(decoder_config.hidden_size)

        # Noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
        )

        # Scaling factors (will be loaded from weights)
        self.speech_scaling_factor = mx.array(1.0)
        self.speech_bias_factor = mx.array(0.0)

        # Inference settings
        self.ddpm_inference_steps = (
            config.diffusion_head_config.ddpm_num_inference_steps
        )
        self._cached_ddpm_steps: Optional[int] = None
        self._cached_timesteps: Optional[List[int]] = None

        # Tokenizer placeholder
        self.tokenizer = None
        self.speech_start_id: Optional[int] = None
        self.speech_end_id: Optional[int] = None
        self.speech_diffusion_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None

        # Optional voice cache state (loaded via load_voice_cache)
        self._voice_path: Optional[str] = None
        self._voice_lm_hidden: Optional[mx.array] = None
        self._voice_tts_hidden: Optional[mx.array] = None
        self._voice_neg_tts_hidden: Optional[mx.array] = None
        self._voice_lm_cache: Optional[list] = None
        self._voice_tts_cache: Optional[list] = None
        self._voice_neg_tts_cache: Optional[list] = None
        self._voice_neg_lm_cache: Optional[list] = None

    @property
    def is_non_streaming(self) -> bool:
        return getattr(self.config, "model_type", "vibevoice_streaming") == "vibevoice"

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self.config.sample_rate

    def _resolve_non_streaming_token_ids(self) -> None:
        """Resolve required special token IDs for non-streaming generation."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")

        def _id(token: str) -> int:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0:
                raise ValueError(
                    f"Tokenizer does not contain required token `{token}` for VibeVoice."
                )
            return int(token_id)

        self.speech_start_id = _id("<|vision_start|>")
        self.speech_end_id = _id("<|vision_end|>")
        self.speech_diffusion_id = _id("<|vision_pad|>")
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        self.bos_id = int(bos_id) if bos_id is not None and bos_id >= 0 else None
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None or eos_id < 0:
            eos_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if eos_id is None or eos_id < 0:
            raise ValueError("Tokenizer has no valid EOS token id for VibeVoice.")
        self.eos_id = int(eos_id)

    def _parse_non_streaming_script(self, text: str) -> List[Tuple[int, str]]:
        """Parse a non-streaming VibeVoice script into normalized speaker turns.

        Supported forms:
        - Plain text -> single speaker 0
        - Speaker-labelled script:
          Speaker 1: Hello
          Speaker 2: Hi
        - Multi-line turns where unlabeled lines continue the previous speaker
        """
        script = text.strip()
        if not script:
            raise ValueError("Text is empty.")

        lines = [line.strip() for line in script.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Text is empty.")

        speaker_pattern = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
        saw_speaker_labels = False
        raw_entries: List[Tuple[int, str]] = []
        current_speaker: Optional[int] = None
        current_parts: List[str] = []

        def _flush() -> None:
            nonlocal current_speaker, current_parts
            if current_speaker is None:
                return
            merged = " ".join(part for part in current_parts if part).strip()
            if merged:
                raw_entries.append((current_speaker, f" {merged}"))
            current_speaker = None
            current_parts = []

        for line in lines:
            match = speaker_pattern.match(line)
            if match:
                saw_speaker_labels = True
                _flush()
                current_speaker = int(match.group(1))
                initial_text = match.group(2).strip()
                current_parts = [initial_text] if initial_text else []
                continue

            if saw_speaker_labels:
                if current_speaker is None:
                    current_speaker = 0
                current_parts.append(line)
            else:
                current_parts.append(line)

        if saw_speaker_labels:
            _flush()
            if not raw_entries:
                raise ValueError("No valid speaker lines found in script.")
        else:
            merged = " ".join(part for part in current_parts if part).strip()
            if not merged:
                raise ValueError("Text is empty.")
            return [(0, f" {merged}")]

        speaker_ids = [speaker_id for speaker_id, _ in raw_entries]
        if speaker_ids and min(speaker_ids) > 0:
            unique_ids = sorted(set(speaker_ids))
            expected_ids = list(range(1, len(unique_ids) + 1))
            if unique_ids == expected_ids:
                raw_entries = [
                    (speaker_id - 1, speaker_text)
                    for speaker_id, speaker_text in raw_entries
                ]
            else:
                normalized_map = {
                    speaker_id: idx for idx, speaker_id in enumerate(unique_ids)
                }
                raw_entries = [
                    (normalized_map[speaker_id], speaker_text)
                    for speaker_id, speaker_text in raw_entries
                ]

        return raw_entries

    def _get_non_streaming_speaker_order(
        self, parsed_script: List[Tuple[int, str]]
    ) -> List[int]:
        return list(dict.fromkeys(speaker for speaker, _ in parsed_script))

    def _reset_negative_cache_on_speech_start(self, neg_cache):
        """Retain only the initial negative state for the next speech segment."""
        if neg_cache is not None:
            for layer_cache in neg_cache:
                if layer_cache.offset > 0:
                    keys, values = layer_cache.state
                    layer_cache.state = (keys[..., :1, :], values[..., :1, :])
        return neg_cache

    def _build_non_streaming_prompt_tokens(
        self,
        parsed_script: List[Tuple[int, str]],
        ref_prompt_steps: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[bool]]:
        """Build non-streaming prompt tokens and speech input mask."""
        if (
            self.speech_start_id is None
            or self.speech_end_id is None
            or self.speech_diffusion_id is None
        ):
            raise ValueError("Special token ids are not initialized.")
        if not parsed_script:
            raise ValueError("Parsed script is empty.")

        prompt_tokens = []
        speech_input_mask = []

        def _append(tokens: List[int], is_speech: bool = False) -> None:
            prompt_tokens.extend(tokens)
            speech_input_mask.extend([is_speech] * len(tokens))

        speaker_order = self._get_non_streaming_speaker_order(parsed_script)

        system_tokens = self.tokenizer.encode(NON_STREAMING_SYSTEM_PROMPT)
        _append(system_tokens, is_speech=False)

        if ref_prompt_steps is not None:
            _append(self.tokenizer.encode(" Voice input:\n", add_special_tokens=False))
            for speaker_id, vae_tok_len in zip(speaker_order, ref_prompt_steps):
                _append(
                    self.tokenizer.encode(
                        f" Speaker {speaker_id}:", add_special_tokens=False
                    )
                )
                _append([self.speech_start_id], is_speech=False)
                _append([self.speech_diffusion_id] * vae_tok_len, is_speech=True)
                _append([self.speech_end_id], is_speech=False)
                _append(
                    self.tokenizer.encode("\n", add_special_tokens=False),
                    is_speech=False,
                )

        _append(self.tokenizer.encode(" Text input:\n", add_special_tokens=False))
        for speaker_id, speaker_text in parsed_script:
            _append(
                self.tokenizer.encode(
                    f" Speaker {speaker_id}:{speaker_text}\n",
                    add_special_tokens=False,
                )
            )
        _append(self.tokenizer.encode(" Speech output:\n", add_special_tokens=False))
        _append([self.speech_start_id], is_speech=False)
        return prompt_tokens, speech_input_mask

    def _prepare_single_non_streaming_ref_embedding(
        self, ref_audio: Union[str, Path, mx.array]
    ) -> mx.array:
        """Encode mono (T,) or multichannel (T, C) reference audio."""
        if isinstance(ref_audio, (str, Path)):
            ref_audio = load_audio(str(ref_audio), sample_rate=self.sample_rate)
        elif not isinstance(ref_audio, mx.array):
            ref_audio = mx.array(ref_audio, dtype=mx.float32)

        if ref_audio.ndim == 2:
            ref_audio = mx.mean(ref_audio, axis=-1)
        if ref_audio.ndim != 1:
            raise ValueError(
                "Reference audio must have shape (samples,) or (samples, channels)."
            )
        if ref_audio.size == 0:
            raise ValueError("Reference audio is empty.")
        ref_audio = ref_audio.astype(mx.float32)
        # Normalize reference audio to -25 dBFS.
        target = mx.array(10 ** (-25.0 / 20.0), dtype=mx.float32)
        eps = mx.array(1e-6, dtype=mx.float32)
        rms = mx.sqrt(mx.mean(ref_audio * ref_audio) + eps)
        scale = target / mx.maximum(rms, eps)
        ref_audio = ref_audio * scale
        max_abs = mx.max(mx.abs(ref_audio)) + eps
        ref_audio = ref_audio / mx.maximum(max_abs, mx.array(1.0, dtype=mx.float32))
        ref_audio_3d = ref_audio[None, None, :]

        acoustic_output = self.acoustic_tokenizer.encode(ref_audio_3d)
        acoustic_latents, _ = acoustic_output.sample(
            dist_type=self.acoustic_tokenizer.std_dist_type
        )
        acoustic_features = (
            acoustic_latents + self.speech_bias_factor
        ) * self.speech_scaling_factor
        # Voice prompts use acoustic embeddings only.
        acoustic_embed = self.acoustic_connector(acoustic_features)
        return acoustic_embed[0]

    def _inject_prefill_speech_embeddings(
        self,
        input_embeds: mx.array,
        speech_input_mask: List[bool],
        speech_embeds: List[mx.array],
    ) -> mx.array:
        """Replace voice-prompt slots with acoustic embeddings."""
        positions = mx.array(
            [i for i, speech in enumerate(speech_input_mask) if speech]
        )
        result = mx.array(input_embeds)
        result[0, positions, :] = mx.concatenate(speech_embeds).astype(
            input_embeds.dtype
        )
        return result

    def _select_non_streaming_next_token(
        self,
        logits: mx.array,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> int:
        """Select a speech control token."""
        candidates = [
            self.speech_start_id,
            self.speech_end_id,
            self.speech_diffusion_id,
            self.eos_id,
        ]
        if self.bos_id is not None and self.bos_id not in candidates:
            candidates.append(self.bos_id)
        scores = logits[:, mx.array(candidates)]
        if not do_sample:
            return candidates[int(mx.argmax(scores, axis=-1).item())]
        scores = scores.astype(mx.float32) / max(float(temperature), 1e-5)
        if top_k is not None:
            k = max(1, min(int(top_k), len(candidates)))
            if k < len(candidates):
                scores = apply_top_k(scores, k)
        if top_p is not None and 0.0 < top_p < 1.0:
            scores = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
            scores = apply_top_p(scores, top_p)
        return candidates[int(mx.random.categorical(scores).item())]

    def load_voice(self, voice: str) -> None:
        """Load a VibeVoice voice-cache (.safetensors) for conditioning.


        Expected keys (per layer):
          - lm_hidden
          - lm_key_{i}, lm_value_{i}
          - tts_lm_hidden
          - tts_lm_key_{i}, tts_lm_value_{i}
          - neg_tts_lm_hidden
          - neg_lm_key_{i}, neg_lm_value_{i}   (optional for our inference)
          - neg_tts_lm_key_{i}, neg_tts_lm_value_{i}
        """
        voice_path = Path(self.config.model_path) / f"voices/{voice}.safetensors"

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice cache not found: {voice_path}")

        tensors = mx.load(str(voice_path))

        lm_layers = self.language_model.config.num_hidden_layers
        tts_layers = self.tts_language_model.config.num_hidden_layers

        def _mx(name: str) -> mx.array:
            if name not in tensors:
                raise KeyError(f"Voice cache missing key: {name}")
            return mx.array(tensors[name])

        def _load_kv(prefix: str, i: int) -> Tuple[mx.array, mx.array]:
            k = _mx(f"{prefix}_key_{i}")
            v = _mx(f"{prefix}_value_{i}")
            # Swift caches are stored as (B, kv_heads, seq, head_dim).
            # Our attention cache expects (B, seq, kv_heads, head_dim).
            if k.ndim == 4:
                k = mx.transpose(k, (0, 2, 1, 3))
            if v.ndim == 4:
                v = mx.transpose(v, (0, 2, 1, 3))
            return (k, v)

        # Store caches/hidden states for generation
        self._voice_path = str(voice_path)
        self._voice_lm_hidden = _mx("lm_hidden")
        self._voice_tts_hidden = _mx("tts_lm_hidden")
        self._voice_neg_tts_hidden = _mx("neg_tts_lm_hidden")

        self._voice_lm_cache = [_load_kv("lm", i) for i in range(lm_layers)]
        self._voice_tts_cache = [_load_kv("tts_lm", i) for i in range(tts_layers)]
        self._voice_neg_tts_cache = [
            _load_kv("neg_tts_lm", i) for i in range(tts_layers)
        ]

        if all(
            f"neg_lm_key_{i}" in tensors and f"neg_lm_value_{i}" in tensors
            for i in range(lm_layers)
        ):
            self._voice_neg_lm_cache = [_load_kv("neg_lm", i) for i in range(lm_layers)]
        else:
            self._voice_neg_lm_cache = None

        voice_arrays = [
            self._voice_lm_hidden,
            self._voice_tts_hidden,
            self._voice_neg_tts_hidden,
            *(x for pair in self._voice_lm_cache for x in pair),
            *(x for pair in self._voice_tts_cache for x in pair),
            *(x for pair in self._voice_neg_tts_cache for x in pair),
        ]
        if self._voice_neg_lm_cache is not None:
            voice_arrays.extend(x for pair in self._voice_neg_lm_cache for x in pair)
        mx.eval(*voice_arrays)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.language_model.embed_tokens

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights for loading from HuggingFace format."""
        import re

        from mlx.utils import tree_flatten

        new_weights = {}
        curr_shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}

        def transform_key(key: str) -> str:
            """Transform HuggingFace key to MLX key format."""
            # Remove "model." prefix
            if key.startswith("model."):
                key = key[6:]

            # Prediction head transformations
            # t_embedder.mlp.0 -> t_embedder.mlp.layers.0
            key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.", r".t_embedder.mlp.layers.\1.", key
            )
            # Handle weight at end: t_embedder.mlp.0.weight -> t_embedder.mlp.layers.0.weight
            key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.weight$",
                r".t_embedder.mlp.layers.\1.weight",
                key,
            )

            # adaLN_modulation.1 -> adaLN_modulation.layers.1
            key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.", r".adaLN_modulation.layers.\1.", key
            )
            key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.weight$",
                r".adaLN_modulation.layers.\1.weight",
                key,
            )

            return key

        for k, v in weights.items():
            new_key = transform_key(k)

            # Handle scaling factors specially
            if "speech_scaling_factor" in new_key:
                new_weights["speech_scaling_factor"] = v
                continue
            if "speech_bias_factor" in new_key:
                new_weights["speech_bias_factor"] = v
                continue

            # Skip rotary embedding inv_freq (computed at init)
            if "rotary_emb.inv_freq" in new_key:
                continue

            # Check if key exists in model
            if new_key not in curr_shapes:
                # Preserve quantization metadata -- model isn't quantized yet at sanitize time
                if new_key.endswith((".scales", ".biases")):
                    new_weights[new_key] = v
                continue

            target_shape = curr_shapes[new_key]

            # Handle shape mismatches
            if v.shape == target_shape:
                new_weights[new_key] = v
            elif len(v.shape) == 2 and v.T.shape == target_shape:
                # Transpose Linear weights (PyTorch vs MLX layout)
                new_weights[new_key] = v.T
            elif len(v.shape) == 3:
                # Check if it's a transposed conv weight
                is_convtr = "convtr" in new_key

                if is_convtr:
                    # ConvTranspose1d: PyTorch (C_in, C_out, K) -> MLX (C_out, K, C_in)
                    # Need to transpose (0, 1, 2) -> (1, 2, 0)
                    if (
                        v.shape[1] == target_shape[0]
                        and v.shape[2] == target_shape[1]
                        and v.shape[0] == target_shape[2]
                    ):
                        new_weights[new_key] = mx.transpose(v, (1, 2, 0))
                    else:
                        new_weights[new_key] = v
                else:
                    # Conv1d weights: PyTorch (C_out, C_in, K) -> MLX (C_out, K, C_in)
                    if (
                        v.shape[0] == target_shape[0]
                        and v.shape[1] == target_shape[2]
                        and v.shape[2] == target_shape[1]
                    ):
                        # Swap last two dimensions
                        new_weights[new_key] = mx.transpose(v, (0, 2, 1))
                    else:
                        new_weights[new_key] = v
            else:
                # Keep as is - might be a different kind of mismatch
                new_weights[new_key] = v

        return new_weights

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Hook called after model weights are loaded."""
        import json

        if model.tokenizer is None:
            # Try to load tokenizer from preprocessor_config
            preprocessor_config_path = model_path / "preprocessor_config.json"
            tokenizer_name = "Qwen/Qwen2.5-0.5B"  # Default

            if preprocessor_config_path.exists():
                with open(preprocessor_config_path, encoding="utf-8") as f:
                    preprocessor_config = json.load(f)
                    tokenizer_name = preprocessor_config.get(
                        "language_model_pretrained_name", tokenizer_name
                    )

            model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if model.is_non_streaming:
            model._resolve_non_streaming_token_ids()
        return model

    def sample_speech_tokens(
        self,
        condition: mx.array,
        neg_condition: mx.array,
        cfg_scale: float = 3.0,
        ddpm_steps: Optional[int] = None,
    ) -> mx.array:
        """Denoise speech latents with classifier-free guidance."""
        if not self.is_non_streaming:
            condition = condition.astype(mx.float32)
            neg_condition = neg_condition.astype(mx.float32)
        num_steps = int(ddpm_steps or self.ddpm_inference_steps)
        self.noise_scheduler.reset()
        if self._cached_ddpm_steps != num_steps:
            self.noise_scheduler.set_timesteps(num_steps)
            self._cached_ddpm_steps = num_steps
            self._cached_timesteps = self.noise_scheduler.timesteps.tolist()

        batch_size = condition.shape[0]
        speech = mx.random.normal(
            (batch_size, self.config.acoustic_vae_dim), dtype=condition.dtype
        )
        use_cfg = cfg_scale != 1.0
        conditions = (
            mx.concatenate([condition, neg_condition], axis=0) if use_cfg else condition
        )
        projected_condition = self.prediction_head.project_condition(conditions)
        prev_x0 = None
        for t in self._cached_timesteps:
            timestep = mx.array([float(t)], dtype=condition.dtype)
            c = projected_condition + self.prediction_head.embed_timesteps(timestep)
            noisy = mx.concatenate([speech, speech], axis=0) if use_cfg else speech
            prediction = self.prediction_head.forward_with_condition(noisy, c)
            if use_cfg:
                positive, negative = mx.split(prediction, 2, axis=0)
                prediction = negative + cfg_scale * (positive - negative)
            output = self.noise_scheduler.step(prediction, t, speech, prev_x0=prev_x0)
            speech, prev_x0 = output.prev_sample, output.x0_pred
        return speech

    def generate(
        self,
        text: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        voice: Optional[Union[str, Path, List[Tuple[str, str]]]] = None,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        Args:
            text: Input text to synthesize (must be a list if voice is a list of tuples)
            max_tokens: Maximum generation steps; defaults depend on the model variant.
            cfg_scale: Classifier-free guidance scale
            ddpm_steps: Override diffusion inference steps (higher = better quality, slower)
            voice: Either a single voice name/path, or a list
            verbose: Whether to show progress

        Yields:
            GenerationResult containing audio and metadata
        """
        if self.is_non_streaming:
            yield from self._generate_non_streaming(
                text=text if isinstance(text, str) else "\n".join(text),
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                ddpm_steps=ddpm_steps,
                verbose=verbose,
                **kwargs,
            )
            return
        max_tokens = 512 if max_tokens is None else max_tokens

        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        # Handle multi-speaker dialogue mode
        if isinstance(text, list) and isinstance(voice, list):
            if len(text) != len(voice):
                raise ValueError(
                    f"text and voice lists must have the same length. "
                    f"Got {len(text)} texts and {len(voice)} voices."
                )

            dialogue = list(zip(voice, text))
            yield from self._generate_multi_speaker(
                dialogue=dialogue,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                ddpm_steps=ddpm_steps,
                verbose=verbose,
                stream=stream,
                streaming_interval=streaming_interval,
                **kwargs,
            )
            return

        # Single voice mode - load voice and delegate to single speaker generator
        if voice is not None:
            # Only reload if different
            if not hasattr(self, "_voice_path") or str(voice) != getattr(
                self, "_voice_path"
            ):
                self.load_voice(voice)

        yield from self._generate_single_speaker(
            text=text,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
            verbose=verbose,
            stream=stream,
            streaming_interval=streaming_interval,
            **kwargs,
        )

    def _generate_non_streaming(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        verbose: bool = False,
        ref_audio=None,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")
        self._resolve_non_streaming_token_ids()
        start_time = time.perf_counter()
        parsed_script = self._parse_non_streaming_script(text)
        speaker_order = self._get_non_streaming_speaker_order(parsed_script)
        ref_embeds = None
        ref_prompt_steps = None
        if ref_audio is not None:
            references = (
                list(ref_audio) if isinstance(ref_audio, (list, tuple)) else [ref_audio]
            )
            if len(references) != len(speaker_order):
                raise ValueError("Provide one reference audio sample per speaker.")
            ref_embeds = [
                self._prepare_single_non_streaming_ref_embedding(audio)
                for audio in references
            ]
            ref_prompt_steps = [int(embed.shape[0]) for embed in ref_embeds]
        prompt_tokens, speech_input_mask = self._build_non_streaming_prompt_tokens(
            parsed_script, ref_prompt_steps
        )
        context_budget = self.config.decoder_config.max_position_embeddings - len(
            prompt_tokens
        )
        if context_budget <= 0:
            raise ValueError("The speech prompt exceeds the model context length.")
        max_steps = min(
            context_budget,
            max_tokens if max_tokens is not None else 2 * len(prompt_tokens),
        )
        input_ids = mx.array([prompt_tokens], dtype=mx.int32)
        input_embeds = self.language_model.embed_tokens(input_ids)
        if ref_embeds is not None:
            input_embeds = self._inject_prefill_speech_embeddings(
                input_embeds, speech_input_mask, ref_embeds
            )
        cache = self.language_model.make_cache()
        hidden, cache = self.language_model(inputs_embeds=input_embeds, cache=cache)
        neg_cache = None
        prev_step_embed = None
        audio_chunks = []
        acoustic_cache = {}
        semantic_cache = {}
        generated_steps = 0
        for _ in range(max_steps):
            next_token_id = self._select_non_streaming_next_token(
                self.lm_head(hidden[:, -1, :]),
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated_steps += 1
            if next_token_id == self.eos_id:
                break
            if next_token_id == self.speech_diffusion_id:
                positive_condition = hidden[:, -1, :]
                negative_condition = positive_condition
                if cfg_scale != 1.0:
                    if neg_cache is None:
                        neg_cache = self.language_model.make_cache()
                    neg_embed = prev_step_embed
                    if neg_embed is None:
                        neg_ids = mx.array([[self.speech_start_id]], dtype=mx.int32)
                        neg_embed = self.language_model.embed_tokens(neg_ids)
                    neg_hidden, neg_cache = self.language_model(
                        inputs_embeds=neg_embed, cache=neg_cache
                    )
                    negative_condition = neg_hidden[:, -1, :]
                latent = self.sample_speech_tokens(
                    positive_condition, negative_condition, cfg_scale, ddpm_steps
                )
                latent = latent[:, None, :]
                scaled_latent = (
                    latent / self.speech_scaling_factor - self.speech_bias_factor
                )
                decoded = self.acoustic_tokenizer.decode(
                    scaled_latent,
                    cache=acoustic_cache,
                )
                audio_chunks.append(decoded[0, 0, :])
                semantic = self.semantic_tokenizer.encode(
                    decoded,
                    cache=semantic_cache,
                )
                acoustic_embed = self.acoustic_connector(latent)
                semantic_embed = self.semantic_connector(semantic)
                next_embed = acoustic_embed + semantic_embed
            else:
                next_embed = self.language_model.embed_tokens(
                    mx.array([[next_token_id]], dtype=mx.int32)
                )
                if next_token_id == self.speech_start_id:
                    neg_cache = self._reset_negative_cache_on_speech_start(neg_cache)
                elif next_token_id == self.speech_end_id:
                    acoustic_cache.clear()
                    semantic_cache.clear()
            hidden, cache = self.language_model(inputs_embeds=next_embed, cache=cache)
            prev_step_embed = next_embed
            if verbose and generated_steps % 20 == 0:
                print(f"Generated {generated_steps}/{max_steps} control steps")

        final_audio = mx.concatenate(audio_chunks) if audio_chunks else mx.array([])
        mx.eval(final_audio)
        elapsed_time = time.perf_counter() - start_time
        samples = final_audio.size
        duration = samples / self.sample_rate
        duration_str = f"{int(duration // 3600):02d}:{int(duration // 60) % 60:02d}:{int(duration) % 60:02d}.{int(duration % 1 * 1000):03d}"
        yield GenerationResult(
            audio=final_audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(prompt_tokens) + generated_steps,
            audio_duration=duration_str,
            real_time_factor=duration / elapsed_time,
            prompt={
                "tokens": len(prompt_tokens),
                "tokens-per-sec": round(len(prompt_tokens) / elapsed_time, 2),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed_time, 2),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=float(mx.get_peak_memory() / 1e9),
        )

    def _generate_multi_speaker(
        self,
        dialogue: List[Tuple[str, str]],
        max_tokens: int = 512,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from multiple speakers.

        Args:
            dialogue: List of (voice_name, text) tuples for each speaker segment
            max_tokens: Maximum tokens per segment
            cfg_scale: Classifier-free guidance scale
            ddpm_steps: Override diffusion inference steps
            verbose: Whether to show progress

        Yields:
            GenerationResult containing combined audio from all speakers
        """
        start_time = time.perf_counter()
        all_audio_segments = []
        total_tokens = 0
        total_prompt_tokens = 0

        for segment_idx, (voice_name, segment_text) in enumerate(dialogue):
            if verbose:
                print(
                    f"Generating segment {segment_idx + 1}/{len(dialogue)}: {voice_name}"
                )

            # Load the voice for this segment
            self.load_voice(voice_name)

            # Generate audio for this segment (single speaker path)
            for result in self._generate_single_speaker(
                text=segment_text,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                ddpm_steps=ddpm_steps,
                verbose=verbose,
                stream=stream,
                streaming_interval=streaming_interval,
                **kwargs,
            ):
                if stream:
                    result.segment_idx = segment_idx
                    result.is_final_chunk = (
                        result.is_final_chunk and segment_idx == len(dialogue) - 1
                    )
                    yield result
                    continue
                all_audio_segments.append(result.audio)
                total_tokens += result.token_count
                total_prompt_tokens += result.prompt["tokens"]

        if stream:
            return

        # Combine all audio segments
        if all_audio_segments:
            final_audio = mx.concatenate(all_audio_segments)
        else:
            final_audio = mx.array([])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Calculate statistics
        samples = final_audio.shape[0] if final_audio.size > 0 else 0
        audio_duration_seconds = samples / self.sample_rate if samples > 0 else 0

        # Format duration
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=final_audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=total_tokens,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": total_prompt_tokens,
                "tokens-per-sec": (
                    round(total_prompt_tokens / elapsed_time, 2)
                    if elapsed_time > 0
                    else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _generate_single_speaker(
        self,
        text: str,
        max_tokens: int = 512,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech for a single speaker segment (internal method).

        This contains the core generation logic, used by both generate() and
        _generate_multi_speaker().
        """
        start_time = time.perf_counter()

        def decode_latents(latents, cache=None):
            latent_seq = mx.concatenate(latents, axis=1)
            scaled_latents = (
                latent_seq / self.speech_scaling_factor - self.speech_bias_factor
            )
            audio = self.acoustic_tokenizer.decode(scaled_latents, cache=cache)
            audio = audio.squeeze(1).squeeze(0)
            mx.eval(audio)
            return audio

        def make_result(
            audio,
            token_count,
            segment_idx,
            chunk_start,
            *,
            is_streaming_chunk=False,
            is_final_chunk=False,
        ):
            elapsed = time.perf_counter() - chunk_start
            samples = audio.shape[0] if audio.size > 0 else 0
            duration_seconds = samples / self.sample_rate if samples > 0 else 0
            duration_mins = int(duration_seconds // 60)
            duration_secs = int(duration_seconds % 60)
            duration_ms = int((duration_seconds % 1) * 1000)
            duration = (
                f"{int(duration_seconds // 3600):02d}:{duration_mins:02d}:"
                f"{duration_secs:02d}.{duration_ms:03d}"
            )
            return GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=token_count,
                audio_duration=duration,
                real_time_factor=(duration_seconds / elapsed if elapsed > 0 else 0),
                prompt={
                    "tokens": prompt_token_count,
                    "tokens-per-sec": (
                        round(prompt_token_count / elapsed, 2) if elapsed > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / elapsed, 2) if elapsed > 0 else 0
                    ),
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=is_streaming_chunk,
                is_final_chunk=is_final_chunk,
            )

        # Tokenize input
        text_token_ids = self.tokenizer.encode(
            text.strip() + "\n", add_special_tokens=False
        )
        input_ids = mx.array([text_token_ids], dtype=mx.int32)
        prompt_token_count = len(text_token_ids)

        batch_size = 1
        seq_len = input_ids.shape[1]

        # Use voice cache if available
        use_voice_cache = hasattr(self, "_voice_lm_cache") and hasattr(
            self, "_voice_tts_cache"
        )

        if use_voice_cache:
            lm_cache = self._voice_lm_cache
            tts_cache = self._voice_tts_cache
            tts_hidden = self._voice_tts_hidden
            neg_hidden = self._voice_neg_tts_hidden
            neg_cache = self._voice_neg_tts_cache
        else:
            lm_cache = None
            tts_cache = None
            tts_hidden = None
            neg_hidden = None
            neg_cache = None

        speech_latents = []
        stream_latents = []
        stream_cache = {} if stream else None
        stream_chunk_start = start_time
        if stream:
            if streaming_interval <= 0:
                stream_latents_per_chunk = 1
            else:
                acoustic_config = self.config.acoustic_tokenizer_config
                ratios = (
                    acoustic_config.decoder_ratios
                    if acoustic_config.decoder_ratios
                    else acoustic_config.encoder_ratios
                )
                samples_per_latent = math.prod(ratios)
                stream_latents_per_chunk = max(
                    1,
                    int(streaming_interval * self.sample_rate / samples_per_latent),
                )
        finished = False
        step = 0
        text_pos = 0

        while not finished and step < max_tokens:
            if text_pos < seq_len:
                cur_text_ids = input_ids[
                    :, text_pos : min(seq_len, text_pos + TTS_TEXT_WINDOW_SIZE)
                ]
                cur_window = cur_text_ids.shape[1]
                text_pos += cur_window

                text_embeds = self.language_model.embed_tokens(cur_text_ids)
                lm_out, lm_cache = self.language_model(
                    inputs_embeds=text_embeds, cache=lm_cache
                )

                text_type = mx.ones((batch_size, cur_window), dtype=mx.int32)
                type_embed = self.tts_input_types(text_type)
                tts_in = lm_out + type_embed
                tts_out, tts_cache = self.tts_language_model(
                    inputs_embeds=tts_in, cache=tts_cache
                )

                if tts_hidden is None:
                    tts_hidden = tts_out
                else:
                    tts_hidden = mx.concatenate([tts_hidden, tts_out], axis=1)

                if neg_hidden is None or not use_voice_cache:
                    neg_embed = mx.zeros(
                        (batch_size, cur_window, self.config.decoder_config.hidden_size)
                    )
                    neg_type_embed = self.tts_input_types(
                        mx.ones((batch_size, cur_window), dtype=mx.int32)
                    )
                    neg_in = neg_embed + neg_type_embed
                    neg_out, neg_cache = self.tts_language_model(
                        inputs_embeds=neg_in, cache=neg_cache
                    )
                    if neg_hidden is None:
                        neg_hidden = neg_out
                    else:
                        neg_hidden = mx.concatenate([neg_hidden, neg_out], axis=1)

            if tts_hidden is None or neg_hidden is None:
                break

            for _ in range(TTS_SPEECH_WINDOW_SIZE):
                positive_condition = tts_hidden[:, -1, :]
                negative_condition = neg_hidden[:, -1, :]

                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                    ddpm_steps=ddpm_steps,
                )
                speech_latent = mx.expand_dims(speech_latent, 1)

                if stream:
                    stream_latents.append(speech_latent)
                    if len(stream_latents) >= stream_latents_per_chunk:
                        audio_chunk = decode_latents(stream_latents, cache=stream_cache)
                        yield make_result(
                            audio_chunk,
                            len(stream_latents),
                            0,
                            stream_chunk_start,
                            is_streaming_chunk=True,
                        )
                        stream_latents.clear()
                        stream_chunk_start = time.perf_counter()
                else:
                    speech_latents.append(speech_latent)

                acoustic_embed = self.acoustic_connector(speech_latent)

                type_embed = self.tts_input_types(
                    mx.zeros((batch_size, 1), dtype=mx.int32)
                )
                tts_input = acoustic_embed + type_embed

                tts_out, tts_cache = self.tts_language_model(
                    inputs_embeds=tts_input,
                    cache=tts_cache,
                )
                tts_hidden = mx.concatenate([tts_hidden, tts_out], axis=1)

                neg_type_embed = self.tts_input_types(
                    mx.zeros((batch_size, 1), dtype=mx.int32)
                )
                neg_input = acoustic_embed + neg_type_embed
                neg_out, neg_cache = self.tts_language_model(
                    inputs_embeds=neg_input,
                    cache=neg_cache,
                )
                neg_hidden = mx.concatenate([neg_hidden, neg_out], axis=1)

                eos_logits = mx.sigmoid(self.tts_eos_classifier(tts_out[:, -1, :]))
                if eos_logits[0].item() > 0.5:
                    finished = True
                    break

                step += 1
                if step >= max_tokens:
                    finished = True
                    break

        if stream:
            if stream_latents:
                final_audio = decode_latents(stream_latents, cache=stream_cache)
                yield make_result(
                    final_audio,
                    len(stream_latents),
                    0,
                    stream_chunk_start,
                    is_streaming_chunk=True,
                    is_final_chunk=True,
                )
            else:
                yield make_result(
                    mx.array([]),
                    0,
                    0,
                    stream_chunk_start,
                    is_streaming_chunk=True,
                    is_final_chunk=True,
                )
            return

        if speech_latents:
            final_audio = decode_latents(speech_latents)
        else:
            final_audio = mx.array([])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        samples = final_audio.shape[0] if final_audio.size > 0 else 0
        audio_duration_seconds = samples / self.sample_rate if samples > 0 else 0

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=final_audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(speech_latents),
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": prompt_token_count,
                "tokens-per-sec": (
                    round(prompt_token_count / elapsed_time, 2)
                    if elapsed_time > 0
                    else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

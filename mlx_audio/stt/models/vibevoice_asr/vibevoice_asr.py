# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm


from ..base import STTOutput
from .audio_encoder import AcousticTokenizerEncoder, SemanticTokenizerEncoder
from .config import ModelConfig


class SpeechConnector(nn.Module):
    """
    MLP connector to project speech features to LM hidden dimension.

    Structure: Linear -> RMSNorm -> Linear
    """

    def __init__(self, input_dim: int, output_dim: int, eps: float = 1e-6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = nn.RMSNorm(output_dim, eps=eps)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class LanguageModel(nn.Module):
    """
    Qwen2-based language model wrapper using mlx_lm.

    This wraps the LlamaModel from mlx_lm (which supports Qwen2 architecture).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # Import and use mlx_lm's Qwen2Model
        try:
            from mlx_lm.models.qwen2 import Qwen2Model

            self.model = Qwen2Model(config)
        except ImportError:
            # Fallback to llama if qwen2 not available
            from mlx_lm.models.llama import LlamaModel

            self.model = LlamaModel(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
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

    @property
    def embed_tokens(self):
        return self.model.embed_tokens


class Model(nn.Module):
    """
    VibeVoice-ASR model for speech-to-text transcription.

    Architecture:
    - Acoustic tokenizer encoder: Encodes raw audio to acoustic features
    - Semantic tokenizer encoder: Encodes raw audio to semantic features
    - Acoustic connector: Projects acoustic features to LM dimension
    - Semantic connector: Projects semantic features to LM dimension
    - Language model (Qwen2): Generates text from combined features

    The model supports:
    - Long-form audio transcription (up to 60 minutes)
    - Speaker diarization
    - Timestamp generation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Audio tokenizer encoders
        self.acoustic_tokenizer = AcousticTokenizerEncoder(
            config.acoustic_tokenizer_config
        )
        self.semantic_tokenizer = SemanticTokenizerEncoder(
            config.semantic_tokenizer_config
        )

        # Speech connectors
        self.acoustic_connector = SpeechConnector(
            config.acoustic_vae_dim, config.decoder_config.hidden_size
        )
        self.semantic_connector = SpeechConnector(
            config.semantic_vae_dim, config.decoder_config.hidden_size
        )

        # Language model (includes lm_head for logit computation)
        self.language_model = LanguageModel(config.decoder_config)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings from the language model."""
        return self.language_model.embed_tokens

    def model_quant_predicate(self, p, m):
        """Only quantize language model layers."""
        return p.startswith("language_model")

    def encode_speech(self, speech_tensors: mx.array) -> mx.array:
        """
        Encode speech input to features for the language model.

        Args:
            speech_tensors: Audio waveform [B, T] at 24kHz

        Returns:
            Combined acoustic + semantic features [B, T', hidden_size]
        """
        # Ensure speech has batch and channel dimensions
        if speech_tensors.ndim == 1:
            speech_tensors = speech_tensors[None, :]
        if speech_tensors.ndim == 2:
            speech_tensors = speech_tensors[:, None, :]  # [B, 1, T]

        # Encode through acoustic tokenizer (use mean directly, no sampling for inference)
        acoustic_tokens = self.acoustic_tokenizer.encode(speech_tensors)
        acoustic_features = self.acoustic_connector(acoustic_tokens)

        # Encode through semantic tokenizer
        semantic_tokens = self.semantic_tokenizer.encode(speech_tensors)
        semantic_features = self.semantic_connector(semantic_tokens)

        # Combine features
        combined_features = acoustic_features + semantic_features

        return combined_features

    def _merge_speech_text_embeddings(
        self,
        input_ids: mx.array,
        speech_features: Optional[mx.array] = None,
        acoustic_input_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """
        Merge speech features into text embeddings at masked positions.

        Args:
            input_ids: Token IDs [B, L]
            speech_features: Pre-computed speech features [B, T', hidden_size]
            acoustic_input_mask: Boolean mask indicating where to insert speech [B, L]
            cache: KV cache (if populated, skip merging as it's done)

        Returns:
            Combined embeddings [B, L, hidden_size]
        """
        # Get text embeddings
        text_embeds = self.get_input_embeddings()(input_ids)

        # Skip if no speech features or cache already populated
        if speech_features is None or (cache is not None and cache[0].offset > 0):
            return text_embeds

        # Insert speech features at masked positions
        if acoustic_input_mask is not None:
            # acoustic_input_mask is [B, L] boolean
            # speech_features is [B, T', hidden_size]
            # We use mx.where with 3 args to conditionally select values
            batch_size, seq_len, hidden_size = text_embeds.shape
            num_speech_tokens = speech_features.shape[1]

            # Expand mask to match embedding shape [B, L, hidden_size]
            mask_expanded = acoustic_input_mask[:, :, None]
            mask_expanded = mx.broadcast_to(mask_expanded, text_embeds.shape)

            # Create padded speech features matching text_embeds shape
            # We need to place speech_features at the mask positions
            # First, find the indices where mask is True for each batch
            for b in range(batch_size):
                # Get flat indices where mask is true
                mask_flat = acoustic_input_mask[b]
                # Count cumulative True values to map to speech feature indices
                cumsum = mx.cumsum(mask_flat.astype(mx.int32))
                # speech_idx maps each position to its speech feature index
                # (or 0 if not a speech position, but we'll use the mask anyway)
                speech_idx = cumsum - 1
                speech_idx = mx.clip(speech_idx, 0, num_speech_tokens - 1)

                # Gather speech features for this batch
                # speech_features[b] is [T', hidden_size]
                # We need to expand it to [L, hidden_size] using speech_idx
                expanded_speech = speech_features[b][speech_idx]  # [L, hidden_size]

                # Use where to select between speech and text embeddings
                text_embeds = mx.where(
                    mask_expanded[b : b + 1],
                    expanded_speech[None],
                    text_embeds[b : b + 1],
                )

        return text_embeds

    def __call__(
        self,
        input_ids: mx.array,
        speech_tensors: Optional[mx.array] = None,
        acoustic_input_mask: Optional[mx.array] = None,
        speech_features: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [B, L]
            speech_tensors: Raw audio [B, T] at 24kHz
            acoustic_input_mask: Mask for where to insert speech in text sequence
            speech_features: Pre-computed speech features (optional)
            cache: KV cache for generation

        Returns:
            Logits [B, L, vocab_size]
        """
        # Encode speech if raw audio provided
        if speech_tensors is not None and speech_features is None:
            speech_features = self.encode_speech(speech_tensors)

        # Merge speech and text embeddings
        input_embeds = self._merge_speech_text_embeddings(
            input_ids=input_ids,
            speech_features=speech_features,
            acoustic_input_mask=acoustic_input_mask,
            cache=cache,
        )

        # Forward through language model (returns logits including lm_head)
        logits = self.language_model(
            inputs=None, cache=cache, input_embeddings=input_embeds
        )

        return logits

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize weights from PyTorch checkpoint to MLX model structure.

        This handles:
        - Key name remapping (model.* prefix removal)
        - Conv weight transposition (PyTorch [O,I,K] -> MLX [O,K,I])
        - Tokenizer encoder path remapping

        PyTorch tokenizer encoder key patterns:
        - downsample_layers.N.0.conv.conv.X -> downsample_layers.N.conv.X
        - head.conv.conv.X -> head.conv.X
        - stages.N.M.mixer.conv.conv.conv.X -> stages.N.M.mixer.conv.conv.X
        - stages.N.M.{ffn,norm,ffn_norm,gamma,ffn_gamma}.* -> unchanged
        """
        sanitized = {}

        already_converted = not any(k.startswith("model.") for k in weights.keys())

        for key, value in weights.items():
            new_key = key

            # Remove 'model.' prefix if present
            if new_key.startswith("model."):
                new_key = new_key[6:]

            # Skip decoder weights (we only use encoders for ASR)
            if "acoustic_tokenizer.decoder" in new_key:
                continue

            # Handle tokenizer encoder weights (acoustic and semantic)
            if (
                "acoustic_tokenizer.encoder." in new_key
                or "semantic_tokenizer.encoder." in new_key
            ):
                # PyTorch path already matches our wrapper structure:
                # Model.acoustic_tokenizer (AcousticTokenizerEncoder)
                #   .encoder (TokenizerEncoder)
                # So acoustic_tokenizer.encoder.* maps directly.

                # Fix downsample_layers: remove .0. wrapper and one .conv. level
                # PyTorch: downsample_layers.N.0.conv.conv.X -> MLX: downsample_layers.N.conv.X
                if ".downsample_layers." in new_key:
                    new_key = re.sub(
                        r"\.downsample_layers\.(\d+)\.0\.conv\.conv\.",
                        r".downsample_layers.\1.conv.",
                        new_key,
                    )

                # Fix head: remove one .conv. level
                # PyTorch: head.conv.conv.X -> MLX: head.conv.X
                elif ".head.conv.conv." in new_key:
                    new_key = new_key.replace(".head.conv.conv.", ".head.conv.")

                # Fix mixer in stages: remove one .conv. level
                # PyTorch: mixer.conv.conv.conv.X -> MLX: mixer.conv.conv.X
                elif ".mixer.conv.conv.conv." in new_key:
                    new_key = new_key.replace(
                        ".mixer.conv.conv.conv.", ".mixer.conv.conv."
                    )

                # Other stage keys (ffn, norm, gamma) map directly

            # Handle language model weights
            # PyTorch: language_model.{layers,embed_tokens,norm}.* -> language_model.model.*
            if new_key.startswith("language_model.layers."):
                new_key = "language_model.model." + new_key[len("language_model.") :]
            elif new_key.startswith("language_model.embed_tokens"):
                new_key = (
                    "language_model.model.embed_tokens"
                    + new_key[len("language_model.embed_tokens") :]
                )
            elif new_key.startswith("language_model.norm"):
                new_key = (
                    "language_model.model.norm" + new_key[len("language_model.norm") :]
                )

            # Map lm_head to language_model.lm_head
            if new_key.startswith("lm_head."):
                new_key = "language_model." + new_key

            # Handle Conv1d weight transposition 
            # PyTorch: [out_channels, in_channels/groups, kernel_size]
            # MLX: [out_channels, kernel_size, in_channels/groups]
            if not already_converted and "conv" in new_key.lower() and "weight" in new_key:
                if value.ndim == 3:
                    value = value.transpose(0, 2, 1)

            # Skip position_ids as MLX doesn't use them
            if "position_ids" in new_key:
                continue

            # Skip fix_std buffer - we handle it in config
            if "fix_std" in new_key:
                continue

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """
        Hook called after model weights are loaded.
        Initializes the tokenizer for text input/output.
        """
        from transformers import AutoTokenizer

        # Load tokenizer (Qwen2.5-7B base, which has the special tokens we need)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B", trust_remote_code=True
            )

        # Set the chat template used by VibeVoice ASR
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

        model.tokenizer = tokenizer

        # VibeVoice ASR repurposes existing Qwen2.5 special tokens for speech:
        # <|object_ref_start|> = speech_start
        # <|object_ref_end|> = speech_end
        # <|box_start|> = speech_pad (placeholder for audio features)
        model._speech_start_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
        model._speech_end_id = tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
        model._speech_pad_id = tokenizer.convert_tokens_to_ids("<|box_start|>")

        return model

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Model":
        """
        Load model from pretrained weights.

        .. deprecated::
            Use `mlx_audio.stt.load()` instead.
        """
        warnings.warn(
            "Model.from_pretrained() is deprecated. Use mlx_audio.stt.load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from mlx_audio.stt.utils import load

        return load(model_path)

    def _preprocess_audio(self, audio) -> mx.array:
        """
        Preprocess audio for the model.

        Args:
            audio: Audio path (str), waveform (np.ndarray/mx.array)

        Returns:
            Audio tensor ready for encoding [B, T]
        """
        from mlx_audio.stt.utils import load_audio

        SAMPLE_RATE = 24000

        if isinstance(audio, str):
            audio = load_audio(audio, sr=SAMPLE_RATE)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # Ensure 1D or 2D
        if audio.ndim == 3:
            audio = audio.squeeze()
        if audio.ndim == 1:
            audio = audio[None, :]  # Add batch dim

        return audio

    def stream_generate(
        self,
        input_ids: Optional[mx.array] = None,
        *,
        speech_features: Optional[mx.array] = None,
        acoustic_input_mask: Optional[mx.array] = None,
        max_tokens: int = 8192,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        generation_stream: bool = False,
        verbose: bool = False,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """
        Stream generate tokens.

        Args:
            input_ids: Input token IDs
            speech_features: Pre-computed speech features
            acoustic_input_mask: Mask for speech token positions
            max_tokens: Maximum tokens to generate
            sampler: Sampling function
            generation_stream: Enable streaming generation
            verbose: Print progress

        Yields:
            Tuple of (token, logprobs)
        """
        from mlx_lm.generate import generate_step

        # Get input embeddings with speech merged in
        input_embeddings = self._merge_speech_text_embeddings(
            input_ids=input_ids,
            speech_features=speech_features,
            acoustic_input_mask=acoustic_input_mask,
        )[
            0
        ]  # Remove batch dim

        # with wired_limit(self, [generation_stream]):
        prompt = input_ids[0] if input_ids.ndim > 1 else input_ids

        for token, logprobs in tqdm(
            generate_step(
                prompt=prompt,
                input_embeddings=input_embeddings,
                model=self.language_model,
                max_tokens=max_tokens,
                sampler=sampler,
            ),
            total=max_tokens,
            disable=not verbose,
            desc="Generating",
        ):
            # Check for EOS tokens
            eos_token_ids = (
                self.tokenizer.eos_token_id
                if hasattr(self.tokenizer, "eos_token_id")
                else [151643, 151645]
            )
            # Normalize to list if it's a single integer
            if isinstance(eos_token_ids, int):
                eos_token_ids = [eos_token_ids]
            if token in eos_token_ids:
                break

            yield token, logprobs

    def generate(
        self,
        audio,
        *,
        context: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        generation_stream: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        """
        Generate transcription from audio.

        Args:
            audio: Audio path (str) or waveform (mx.array/np.array)
            context: Optional context string (hotwords, metadata)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling
            top_k: Top-k sampling
            min_p: Min-p sampling
            min_tokens_to_keep: Min tokens for sampling
            generation_stream: Enable streaming
            verbose: Print progress

        Returns:
            STTOutput with transcription text and segments
        """
        from mlx_lm.sample_utils import make_sampler

        start_time = time.time()

        # Preprocess audio
        audio_tensor = self._preprocess_audio(audio)

        # Encode speech
        speech_features = self.encode_speech(audio_tensor)
        mx.eval(speech_features)

        # Build prompt
        audio_duration = audio_tensor.shape[1] / 24000
        input_ids, acoustic_input_mask = self._build_prompt_tokens(
            speech_features, audio_duration, context
        )

        # Create sampler
        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )

        # Generate tokens
        generated_tokens = []

        for token, _ in self.stream_generate(
            input_ids=input_ids,
            speech_features=speech_features,
            acoustic_input_mask=acoustic_input_mask,
            max_tokens=max_tokens,
            sampler=sampler,
            generation_stream=generation_stream,
            verbose=verbose,
        ):
            generated_tokens.append(token)

        end_time = time.time()

        if verbose:
            print()

        # Clear cache
        mx.clear_cache()

        # Decode output
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Parse structured output
        segments = self.parse_transcription(text)

        return STTOutput(
            text=text.strip(),
            segments=segments,
            prompt_tokens=input_ids.shape[1],
            generation_tokens=len(generated_tokens),
            total_tokens=input_ids.shape[1] + len(generated_tokens),
            total_time=end_time - start_time,
            prompt_tps=input_ids.shape[1] / (end_time - start_time),
            generation_tps=len(generated_tokens) / (end_time - start_time),
        )

    def stream_transcribe(
        self,
        audio,
        *,
        context: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        verbose: bool = False,
    ) -> Generator[str, None, None]:
        """
        Stream transcription token-by-token from audio.

        Args:
            audio: Audio path (str) or waveform (mx.array/np.array)
            context: Optional context string (hotwords, metadata)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling
            top_k: Top-k sampling
            min_p: Min-p sampling
            min_tokens_to_keep: Min tokens for sampling
            verbose: Print progress

        Yields:
            Decoded text chunks as they are generated.
        """
        from mlx_lm.sample_utils import make_sampler

        # Preprocess audio
        audio_tensor = self._preprocess_audio(audio)

        # Encode speech
        speech_features = self.encode_speech(audio_tensor)
        mx.eval(speech_features)

        # Build prompt (same as generate)
        audio_duration = audio_tensor.shape[1] / 24000
        input_ids, acoustic_input_mask = self._build_prompt_tokens(
            speech_features, audio_duration, context
        )

        # Create sampler
        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )

        # Stream tokens
        for token, _ in self.stream_generate(
            input_ids=input_ids,
            speech_features=speech_features,
            acoustic_input_mask=acoustic_input_mask,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=verbose,
        ):
            text = self.tokenizer.decode([token])
            yield text

        mx.clear_cache()

    def _build_prompt_tokens(
        self,
        speech_features: mx.array,
        audio_duration: float,
        context: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Build input_ids and acoustic_input_mask for generation.

        Returns:
            Tuple of (input_ids [1, L], acoustic_input_mask [1, L])
        """
        system_prompt = (
            "You are a helpful assistant that transcribes audio input into text "
            "output in JSON format."
        )

        vae_tok_len = speech_features.shape[1]

        # VibeVoice ASR uses repurposed Qwen2.5 special tokens:
        speech_start_token = "<|object_ref_start|>"
        speech_pad_token = "<|box_start|>"
        speech_end_token = "<|object_ref_end|>"

        show_keys = ["Start time", "End time", "Speaker ID", "Content"]
        if context and context.strip():
            user_suffix = (
                f"This is a {audio_duration:.2f} seconds audio, "
                f"with extra info: {context.strip()}\n\n"
                f"Please transcribe it with these keys: " + ", ".join(show_keys)
            )
        else:
            user_suffix = (
                f"This is a {audio_duration:.2f} seconds audio, "
                f"please transcribe it with these keys: " + ", ".join(show_keys)
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": speech_start_token
                + speech_pad_token * vae_tok_len
                + speech_end_token
                + "\n"
                + user_suffix,
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(prompt_text)

        input_ids = mx.array([tokens])
        acoustic_input_mask = mx.array(
            [[t == self._speech_pad_id for t in tokens]]
        ).astype(mx.bool_)

        return input_ids, acoustic_input_mask

    def parse_transcription(self, text: str) -> List[Dict[str, Any]]:
        """Parse structured JSON output from model."""
        import json

        try:
            # Try to extract JSON from text
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
            else:
                # Find JSON array or object
                json_start = text.find("[")
                if json_start == -1:
                    json_start = text.find("{")
                if json_start != -1:
                    bracket_count = 0
                    json_end = json_start
                    for i in range(json_start, len(text)):
                        if text[i] in "[{":
                            bracket_count += 1
                        elif text[i] in "]}":
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i + 1
                                break
                    json_str = text[json_start:json_end]
                else:
                    json_str = text

            result = json.loads(json_str)

            if isinstance(result, dict):
                result = [result]

            # Clean up keys
            segments = []
            key_mapping = {
                "Start time": "start",
                "Start": "start",
                "End time": "end",
                "End": "end",
                "Speaker ID": "speaker_id",
                "Speaker": "speaker_id",
                "Content": "text",
            }

            for item in result:
                if isinstance(item, dict):
                    segment = {}
                    for old_key, new_key in key_mapping.items():
                        if old_key in item:
                            segment[new_key] = item[old_key]
                    if segment:
                        segments.append(segment)

            return segments

        except (json.JSONDecodeError, Exception):
            return []

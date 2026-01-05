"""
CosyVoice3 LLM module for speech token generation.

Based on: https://github.com/FunAudioLLM/CosyVoice
"""

from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.models.qwen2 import ModelArgs as Qwen2Config
from mlx_lm.models.qwen2 import create_attention_mask


class CosyVoice3LM(nn.Module):
    """
    CosyVoice3 Language Model for speech token generation.

    This wraps Qwen2 and adds speech token embedding/decoding.
    Architecture based on weights from FunAudioLLM/Fun-CosyVoice3-0.5B-2512.
    """

    # Base speech token vocabulary (actual speech codes)
    SPEECH_TOKEN_SIZE = 6561

    # Special tokens (offset from SPEECH_TOKEN_SIZE)
    SOS_TOKEN = 6561  # Start of sequence = speech_token_size + 0
    EOS_TOKEN = 6562  # End of sequence = speech_token_size + 1
    TASK_ID_TOKEN = 6563  # Task identifier = speech_token_size + 2
    FILL_TOKEN = 6564  # Fill token = speech_token_size + 3

    def __init__(
        self,
        llm_input_size: int = 896,
        llm_output_size: int = 896,
        speech_token_size: int = 6761,  # Full embedding size (includes special tokens)
        text_vocab_size: int = 151936,
        qwen2_config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # Speech token embedding (includes special tokens)
        self.speech_embedding = nn.Embedding(speech_token_size, llm_input_size)

        # LLM decoder for speech tokens (includes special tokens for EOS prediction)
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size, bias=False)

        # Qwen2 model - architecture from weights analysis
        # q_proj: (896, 896) -> 14 heads, k/v_proj: (128, 896) -> 2 KV heads
        if qwen2_config is None:
            qwen2_config = Qwen2Config(
                model_type="qwen2",
                hidden_size=llm_input_size,
                intermediate_size=4864,
                num_hidden_layers=24,
                num_attention_heads=14,  # 896 / 64 = 14 heads
                num_key_value_heads=2,  # 128 / 64 = 2 KV heads (GQA)
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
                vocab_size=text_vocab_size,
            )
        self.llm = Qwen2Model(qwen2_config)

    def encode_text(self, input_ids: mx.array) -> mx.array:
        """
        Encode text tokens using Qwen2 embeddings.

        Args:
            input_ids: Text token IDs (B, T)

        Returns:
            Text embeddings (B, T, D)
        """
        return self.llm.model.embed_tokens(input_ids)

    def encode_speech(self, speech_tokens: mx.array) -> mx.array:
        """
        Embed speech tokens.

        Args:
            speech_tokens: Speech token IDs (B, T)

        Returns:
            Speech embeddings (B, T, D)
        """
        return self.speech_embedding(speech_tokens)

    def forward(
        self,
        embeddings: mx.array,
        cache: Optional[List] = None,
    ) -> Tuple[mx.array, List]:
        """
        Forward pass through the LLM.

        Args:
            embeddings: Input embeddings (B, T, D)
            cache: KV cache list (objects are modified in-place)

        Returns:
            Tuple of (hidden_states, cache)
        """
        h = embeddings

        # Create cache if not provided
        if cache is None:
            cache = [None] * len(self.llm.model.layers)

        # Create attention mask
        mask = create_attention_mask(h, cache[0] if cache else None)

        # Forward through layers (cache is modified in-place)
        for layer, c in zip(self.llm.model.layers, cache):
            h = layer(h, mask=mask, cache=c)

        h = self.llm.model.norm(h)
        return h, cache

    def decode_to_speech(self, hidden_states: mx.array) -> mx.array:
        """
        Decode hidden states to speech token logits.

        Args:
            hidden_states: LLM hidden states (B, T, D)

        Returns:
            Speech token logits (B, T, speech_vocab_size)
        """
        return self.llm_decoder(hidden_states)

    def sample_next_token(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
    ) -> mx.array:
        """
        Sample next token from logits.

        Args:
            logits: Token logits (B, 1, V) or (B, V)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling (not implemented yet)

        Returns:
            Sampled token IDs (B, 1)
        """
        # Squeeze if needed: (B, 1, V) -> (B, V)
        if logits.ndim == 3:
            logits = logits.squeeze(1)

        B = logits.shape[0]

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy
            return mx.argmax(logits, axis=-1, keepdims=True)

        # Apply top-k filtering
        if top_k > 0:
            # Get indices of top-k elements using argpartition
            # argpartition returns indices that would partition the array
            # such that the k largest elements are at the end
            top_k_indices = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[
                :, :top_k
            ]  # Take first k (largest after negation)

            # Get the corresponding logits
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)

            # Sample from top-k
            probs = mx.softmax(top_k_logits, axis=-1)
            sampled_idx = mx.random.categorical(probs)

            # Map back to vocabulary
            next_token = mx.take_along_axis(
                top_k_indices, sampled_idx[:, None], axis=-1
            )
        else:
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs)
            next_token = next_token[:, None]

        return next_token

    def generate(
        self,
        text_tokens: mx.array,
        prompt_speech_tokens: Optional[mx.array] = None,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        min_tokens: int = 0,
    ) -> Generator[mx.array, None, None]:
        """
        Generate speech tokens autoregressively from text tokens.

        Input sequence format: [SOS, text_tokens, TASK_ID, prompt_speech_tokens...]

        Args:
            text_tokens: Text token IDs (B, T)
            prompt_speech_tokens: Optional prompt speech tokens for voice cloning (B, T_prompt)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling probability
            min_tokens: Minimum tokens to generate before allowing EOS

        Yields:
            Generated speech token IDs (B, 1)
        """
        B = text_tokens.shape[0]

        # Create KV cache for incremental generation
        cache = make_prompt_cache(self.llm)

        # Build input sequence: [SOS, text, TASK_ID, prompt_speech]
        # 1. SOS embedding
        sos_token = mx.full((B, 1), self.SOS_TOKEN, dtype=mx.int32)
        sos_emb = self.encode_speech(sos_token)

        # 2. Text embeddings
        text_emb = self.encode_text(text_tokens)

        # 3. Task ID embedding
        task_id_token = mx.full((B, 1), self.TASK_ID_TOKEN, dtype=mx.int32)
        task_id_emb = self.encode_speech(task_id_token)

        # 4. Prompt speech embeddings (if provided)
        if prompt_speech_tokens is not None and prompt_speech_tokens.shape[1] > 0:
            prompt_emb = self.encode_speech(prompt_speech_tokens)
            embeddings = mx.concatenate(
                [sos_emb, text_emb, task_id_emb, prompt_emb], axis=1
            )
        else:
            embeddings = mx.concatenate([sos_emb, text_emb, task_id_emb], axis=1)

        # Initial forward pass (prompt processing)
        hidden, cache = self.forward(embeddings, cache)

        # Get logits for speech tokens (last position)
        logits = self.decode_to_speech(hidden[:, -1:, :])

        # Sample first token
        current_token = self.sample_next_token(logits, temperature, top_k, top_p)
        mx.eval(current_token)

        generated_count = 0

        while generated_count < max_tokens:
            # Check for EOS
            if (
                generated_count >= min_tokens
                and (current_token == self.EOS_TOKEN).all()
            ):
                break

            yield current_token
            generated_count += 1

            # Encode current token
            speech_emb = self.encode_speech(current_token)

            # Forward with cache (incremental generation)
            hidden, cache = self.forward(speech_emb, cache)

            # Get logits
            logits = self.decode_to_speech(hidden[:, -1:, :])

            # Sample next token
            current_token = self.sample_next_token(logits, temperature, top_k, top_p)
            mx.eval(current_token)

    def __call__(
        self,
        text_tokens: mx.array,
        speech_tokens: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass for training/evaluation.

        Args:
            text_tokens: Text token IDs (B, T_text)
            speech_tokens: Speech token IDs (B, T_speech) or None

        Returns:
            Speech token logits (B, T_speech, speech_vocab_size)
        """
        # Encode text
        text_emb = self.encode_text(text_tokens)

        if speech_tokens is not None:
            # Encode speech
            speech_emb = self.encode_speech(speech_tokens)
            # Concatenate
            embeddings = mx.concatenate([text_emb, speech_emb], axis=1)
        else:
            embeddings = text_emb

        # Forward
        hidden, _ = self.forward(embeddings)

        # Decode to speech logits
        logits = self.decode_to_speech(hidden)

        return logits

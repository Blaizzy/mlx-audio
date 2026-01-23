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

    def nucleus_sampling(
        self,
        logits: mx.array,
        top_p: float = 0.8,
        top_k: int = 25,
    ) -> mx.array:
        """
        Nucleus (top-p) sampling combined with top-k.

        Matches PyTorch CosyVoice3's nucleus_sampling function exactly:
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True)
            # collect tokens while cum_prob < top_p and count < top_k
            top_ids = indices[prob.multinomial(1, replacement=True)].item()

        Args:
            logits: Token logits (V,) - single position
            top_p: Cumulative probability threshold
            top_k: Maximum number of tokens to consider

        Returns:
            Sampled token ID (scalar)
        """
        # Convert to probabilities (matches PyTorch: weighted_scores.softmax(dim=0))
        probs = mx.softmax(logits, axis=-1)

        # Sort by probability (descending) - matches PyTorch: .sort(descending=True, stable=True)
        sorted_indices = mx.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Evaluate to get actual values for the loop
        mx.eval(sorted_probs, sorted_indices)
        sorted_probs_np = sorted_probs.tolist()
        sorted_indices_np = sorted_indices.tolist()

        # Collect candidates while cum_prob < top_p AND count < top_k
        # Matches PyTorch: if cum_prob < top_p and len(prob) < top_k
        valid_probs = []
        valid_indices = []
        cum_prob = 0.0

        for i in range(len(sorted_probs_np)):
            if cum_prob < top_p and len(valid_probs) < top_k:
                cum_prob += sorted_probs_np[i]
                valid_probs.append(sorted_probs_np[i])
                valid_indices.append(sorted_indices_np[i])
            else:
                break

        if len(valid_probs) == 0:
            # Fallback to argmax
            return mx.argmax(logits)

        # Sample from valid candidates using multinomial
        # PyTorch uses: prob.multinomial(1, replacement=True)
        # mx.random.categorical expects LOGITS, so pass log(probs)
        valid_probs = mx.array(valid_probs)
        valid_indices = mx.array(valid_indices, dtype=mx.int32)
        valid_probs = valid_probs / mx.sum(valid_probs)
        sampled_idx = mx.random.categorical(mx.log(valid_probs)[None, :])[0]
        return valid_indices[sampled_idx]

    def ras_sampling(
        self,
        logits: mx.array,
        decoded_tokens: list,
        top_p: float = 0.8,
        top_k: int = 25,
        win_size: int = 10,
        tau_r: float = 0.1,
    ) -> mx.array:
        """
        Repetition Aware Sampling (RAS).

        Matches PyTorch CosyVoice3's ras_sampling function exactly.
        If a token appears too frequently in recent history, falls back to random sampling.

        Args:
            logits: Token logits (V,) - single position
            decoded_tokens: List of previously decoded token IDs
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            win_size: Window size for repetition detection
            tau_r: Repetition threshold ratio

        Returns:
            Sampled token ID (scalar)
        """
        # First, do nucleus sampling
        top_id = self.nucleus_sampling(logits, top_p=top_p, top_k=top_k)
        mx.eval(top_id)

        # Check for repetition in recent window
        recent_tokens = decoded_tokens[-win_size:] if len(decoded_tokens) > 0 else []
        if len(recent_tokens) > 0:
            top_id_val = int(top_id.item())
            rep_count = sum(1 for t in recent_tokens if t == top_id_val)

            # If token appears too frequently, use random sampling
            # Matches PyTorch: weighted_scores.softmax(dim=0).multinomial(1)
            # mx.random.categorical expects logits, so pass raw logits directly
            if rep_count >= win_size * tau_r:
                top_id = mx.random.categorical(logits[None, :])[0]

        return top_id

    def sample_next_token(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        decoded_tokens: list = None,
        use_ras: bool = True,
    ) -> mx.array:
        """
        Sample next token from logits using RAS (Repetition Aware Sampling).

        Args:
            logits: Token logits (B, 1, V) or (B, V)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            decoded_tokens: List of previously decoded tokens (for RAS)
            use_ras: Whether to use Repetition Aware Sampling

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

        # Use RAS for batch size 1 (most common case)
        if B == 1 and use_ras:
            if decoded_tokens is None:
                decoded_tokens = []
            token = self.ras_sampling(
                logits[0], decoded_tokens, top_p=top_p, top_k=top_k
            )
            return token[None, None] if token.ndim == 0 else token[None, :]

        # Fallback to basic top-k for batch processing
        # mx.random.categorical expects logits (unnormalized log-probs), not probs
        if top_k > 0:
            top_k_indices = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[
                :, :top_k
            ]
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
            sampled_idx = mx.random.categorical(top_k_logits)
            next_token = mx.take_along_axis(
                top_k_indices, sampled_idx[:, None], axis=-1
            )
        else:
            next_token = mx.random.categorical(logits)
            next_token = next_token[:, None]

        return next_token

    def _is_stop_token(self, token_val: int) -> bool:
        """Check if a token is a stop/special token (>= SPEECH_TOKEN_SIZE).

        Matches PyTorch: stop_token_ids = [speech_token_size + i for i in range(200)]
        Any token >= 6561 (the base speech vocab) is a stop token.
        """
        return token_val >= self.SPEECH_TOKEN_SIZE

    def _sample_valid_token(
        self,
        logits: mx.array,
        decoded_tokens: list,
        top_k: int = 25,
        top_p: float = 0.8,
        ignore_eos: bool = True,
        max_trials: int = 100,
    ) -> mx.array:
        """Sample a token, retrying if stop tokens are generated before min_len.

        Matches PyTorch sampling_ids: resamples until a valid speech token
        (< SPEECH_TOKEN_SIZE) is generated when ignore_eos=True.
        """
        for _ in range(max_trials):
            token = self.sample_next_token(
                logits, temperature=1.0, top_k=top_k, top_p=top_p,
                decoded_tokens=decoded_tokens
            )
            mx.eval(token)
            token_val = int(token.item()) if token.size == 1 else int(token[0, 0].item())

            if not ignore_eos or token_val < self.SPEECH_TOKEN_SIZE:
                return token

        # Fallback: return the last sampled token
        return token

    def generate(
        self,
        text_tokens: mx.array,
        prompt_text_tokens: Optional[mx.array] = None,
        prompt_speech_tokens: Optional[mx.array] = None,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        min_tokens: int = 0,
    ) -> Generator[mx.array, None, None]:
        """
        Generate speech tokens autoregressively from text tokens.

        Matches PyTorch CosyVoice3LM.inference + inference_wrapper exactly:
        - Input sequence: [SOS, prompt_text + text, TASK_ID, prompt_speech_tokens]
        - Resamples if stop tokens generated before min_tokens (ignore_eos)
        - Stops on ANY token >= SPEECH_TOKEN_SIZE (not just EOS)

        Args:
            text_tokens: Text token IDs (B, T) - the text to synthesize
            prompt_text_tokens: Optional prompt text tokens (B, T_prompt_text)
            prompt_speech_tokens: Optional prompt speech tokens for voice cloning (B, T_prompt)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (PyTorch doesn't use this; keep 1.0 for match)
            top_k: Top-k sampling (this is the 'sampling' param in PyTorch)
            top_p: Nucleus sampling probability
            min_tokens: Minimum tokens to generate before allowing stop

        Yields:
            Generated speech token IDs (B, 1)
        """
        B = text_tokens.shape[0]

        # Create KV cache for incremental generation
        cache = make_prompt_cache(self.llm)

        # Build input sequence: [SOS, prompt_text + text, TASK_ID, prompt_speech]
        # Following PyTorch: text = torch.concat([prompt_text, text], dim=1)

        # 1. Concatenate prompt_text + text (if prompt_text provided)
        if prompt_text_tokens is not None and prompt_text_tokens.shape[1] > 0:
            combined_text = mx.concatenate([prompt_text_tokens, text_tokens], axis=1)
        else:
            combined_text = text_tokens

        # 2. SOS embedding
        sos_token = mx.full((B, 1), self.SOS_TOKEN, dtype=mx.int32)
        sos_emb = self.encode_speech(sos_token)

        # 3. Text embeddings (combined prompt_text + text)
        text_emb = self.encode_text(combined_text)

        # 4. Task ID embedding
        task_id_token = mx.full((B, 1), self.TASK_ID_TOKEN, dtype=mx.int32)
        task_id_emb = self.encode_speech(task_id_token)

        # 5. Prompt speech embeddings (if provided)
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
        # PyTorch: logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        # We use raw logits and apply softmax in sampling (equivalent)
        logits = self.decode_to_speech(hidden[:, -1:, :])

        # Track decoded tokens for RAS (Repetition Aware Sampling)
        decoded_tokens: List[int] = []

        generated_count = 0

        while generated_count < max_tokens:
            # Sample token with stop-token filtering (matching PyTorch sampling_ids)
            # Before min_tokens: resample if stop token generated (ignore_eos=True)
            # After min_tokens: allow stop tokens (ignore_eos=False)
            ignore_eos = generated_count < min_tokens
            current_token = self._sample_valid_token(
                logits, decoded_tokens, top_k=top_k, top_p=top_p,
                ignore_eos=ignore_eos
            )

            current_token_val = int(current_token.item()) if current_token.size == 1 else int(current_token[0, 0].item())

            # Check for stop token (ANY token >= SPEECH_TOKEN_SIZE)
            # Matches PyTorch: if top_ids in self.stop_token_ids: break
            if self._is_stop_token(current_token_val):
                break

            yield current_token

            # Track token for RAS
            decoded_tokens.append(current_token_val)
            generated_count += 1

            # Encode current token and get next logits
            speech_emb = self.encode_speech(current_token)
            hidden, cache = self.forward(speech_emb, cache)
            logits = self.decode_to_speech(hidden[:, -1:, :])

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

# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.models.qwen2 import ModelArgs as Qwen2Config
from mlx_lm.models.qwen2 import create_attention_mask

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult

from .campplus import CAMPPlus
from .dit import DiT
from .flow import CausalConditionalCFM, CausalMaskedDiffWithDiT, PreLookaheadLayer
from .frontend import CosyVoice3Frontend
from .hift import CausalHiFTGenerator


@dataclass
class ModelConfig(BaseModelArgs):
    """Model configuration for CosyVoice3."""

    model_path: Optional[Path] = None
    sample_rate: int = 24000
    token_frame_rate: int = 25
    token_mel_ratio: int = (
        2  # 25 tokens/s * 2 = 50 mel fps = 24000/480 (HIFT total upsample)
    )
    chunk_size: int = 25
    spk_embed_dim: int = 192

    # LLM config
    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 6761  # Actual from weights
    text_vocab_size: int = 151936

    # Flow config
    flow_input_size: int = 80
    flow_vocab_size: int = 6561
    pre_lookahead_len: int = 3

    # DiT config
    dit_dim: int = 1024
    dit_depth: int = 22
    dit_heads: int = 16
    dit_dim_head: int = 64
    dit_ff_mult: int = 2
    mel_dim: int = 80

    # HIFT config
    hift_base_channels: int = 512
    nb_harmonics: int = 8  # For m_source (l_linear)
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 3])


class LLM(nn.Module):
    """Speech token embedding and decoder projections."""

    def __init__(self, speech_token_size: int, llm_input_size: int, llm_output_size: int):
        super().__init__()
        self.speech_embedding = nn.Embedding(speech_token_size, llm_input_size)
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size, bias=False)


class Model(nn.Module):
    """
    CosyVoice3 for text-to-speech generation.

    This is an LLM-based TTS model with flow matching for mel generation
    and HiFi-GAN vocoder for audio synthesis.

    Pipeline: Text -> LLM -> Speech Tokens -> Flow -> Mel -> HIFT -> Audio
    """

    # Base speech token vocabulary (actual speech codes)
    SPEECH_TOKEN_SIZE = 6561

    # Special tokens (offset from SPEECH_TOKEN_SIZE)
    SOS_TOKEN = 6561  # Start of sequence = speech_token_size + 0
    EOS_TOKEN = 6562  # End of sequence = speech_token_size + 1
    TASK_ID_TOKEN = 6563  # Task identifier = speech_token_size + 2
    FILL_TOKEN = 6564  # Fill token = speech_token_size + 3

    def __init__(self, config: ModelConfig, load_llm: bool = True):
        super().__init__()
        self.config = config
        self.frontend = None  # Initialized in from_pretrained
        self._llm_loaded = load_llm

        # Build CAMPPlus speaker embedding model
        self.campplus = CAMPPlus(
            feat_dim=80, embedding_size=config.spk_embed_dim
        )

        # Build LLM (optional - can be skipped for token-to-wav only)
        if load_llm:
            self.llm = LLM(
                speech_token_size=config.speech_token_size,
                llm_input_size=config.llm_input_size,
                llm_output_size=config.llm_output_size,
            )
            self.qwen2 = Qwen2Model(
                Qwen2Config(
                    model_type="qwen2",
                    hidden_size=config.llm_input_size,
                    intermediate_size=4864,
                    num_hidden_layers=24,
                    num_attention_heads=14,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-6,
                    rope_theta=1000000.0,
                    vocab_size=config.text_vocab_size,
                    tie_word_embeddings=False,
                )
            )

        # Build DiT
        dit = DiT(
            dim=config.dit_dim,
            depth=config.dit_depth,
            heads=config.dit_heads,
            dim_head=config.dit_dim_head,
            ff_mult=config.dit_ff_mult,
            mel_dim=config.mel_dim,
            mu_dim=config.mel_dim,
            spk_dim=config.mel_dim,
            out_channels=config.mel_dim,
            static_chunk_size=config.chunk_size * config.token_mel_ratio,
            num_decoding_left_chunks=-1,
        )

        # Build Flow decoder (CFM)
        cfm = CausalConditionalCFM(
            in_channels=240,
            n_spks=1,
            spk_emb_dim=config.mel_dim,
            sigma_min=1e-6,
            solver="euler",
            t_scheduler="cosine",
            training_cfg_rate=0.2,
            inference_cfg_rate=0.7,
            estimator=dit,
        )

        # Build pre-lookahead layer
        pre_lookahead = PreLookaheadLayer(
            in_channels=config.mel_dim,
            channels=config.dit_dim,
            pre_lookahead_len=config.pre_lookahead_len,
        )

        # Build Flow module
        self.flow = CausalMaskedDiffWithDiT(
            input_size=config.flow_input_size,
            output_size=config.mel_dim,
            spk_embed_dim=config.spk_embed_dim,
            vocab_size=config.flow_vocab_size,
            input_frame_rate=config.token_frame_rate,
            token_mel_ratio=config.token_mel_ratio,
            pre_lookahead_len=config.pre_lookahead_len,
            pre_lookahead_layer=pre_lookahead,
            decoder=cfm,
        )

        # Build HIFT vocoder
        self.hift = CausalHiFTGenerator(
            in_channels=config.mel_dim,
            base_channels=config.hift_base_channels,
            nb_harmonics=config.nb_harmonics,
            sampling_rate=config.sample_rate,
            upsample_rates=config.upsample_rates,
            upsample_kernel_sizes=[16, 11, 7],
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize weights for MLX.

        Remaps PyTorch weight keys to MLX model structure:
        - Parametrizations: combines original0 * original1 -> weight
        - nn.Sequential: .N. -> .layers.N.
        - CausalConv1d wrapper: adds .conv.
        - CausalConvPositionEmbedding: conv1.0. -> conv1.
        - FeedForward nested: ff.0.0. -> ff.layers.0., ff.2. -> ff.layers.3.
        - Linear weights: transpose back if incorrectly transposed
        """
        import re

        # First pass: combine parametrization weights
        combined_weights = {}
        param_pairs = {}

        for key, value in weights.items():
            if ".parametrizations.weight.original" in key:
                # Extract base key (before .parametrizations)
                base_key = key.replace(
                    ".parametrizations.weight.original0", ""
                ).replace(".parametrizations.weight.original1", "")
                if base_key not in param_pairs:
                    param_pairs[base_key] = {}
                if key.endswith("original0"):
                    param_pairs[base_key]["original0"] = value
                else:
                    param_pairs[base_key]["original1"] = value
            else:
                combined_weights[key] = value

        # Combine parametrization pairs using weight normalization formula:
        # weight = g * (v / ||v||) where g=original0, v=original1
        for base_key, pair in param_pairs.items():
            if "original0" in pair and "original1" in pair:
                g = pair["original0"]  # Scale factor (out_ch, 1, 1)
                v = pair["original1"]  # Direction vector (out_ch, in_ch, kernel)
                # Normalize v along all dimensions except output channels
                # For Conv1d: v has shape (out_ch, in_ch, kernel), norm along (1, 2)
                v_norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
                v_normalized = v / v_norm
                combined = g * v_normalized
                # Conv weights need transpose: PyTorch [out, in, k] -> MLX [out, k, in]
                if combined.ndim == 3:
                    combined = mx.transpose(combined, [0, 2, 1])
                combined_weights[f"{base_key}.weight"] = combined

        new_weights = {}

        for key, value in combined_weights.items():
            new_key = key

            # === Flow/DiT mappings ===

            # CausalConvPositionEmbedding: conv1.0. -> conv1., conv2.0. -> conv2.
            new_key = re.sub(
                r"conv_pos_embed\.conv(\d)\.0\.",
                r"conv_pos_embed.conv\1.",
                new_key,
            )

            # TimestepEmbedding: time_mlp.N. -> time_mlp.layers.N.
            new_key = re.sub(
                r"time_mlp\.(\d+)\.",
                r"time_mlp.layers.\1.",
                new_key,
            )

            # Attention to_out: to_out.0. -> to_out.layers.0.
            new_key = re.sub(
                r"to_out\.(\d+)\.",
                r"to_out.layers.\1.",
                new_key,
            )

            # FeedForward nested structure:
            # ff.ff.0.0. -> ff.ff.layers.0. (nested Sequential for first Linear)
            new_key = re.sub(
                r"ff\.ff\.0\.0\.",
                r"ff.ff.layers.0.",
                new_key,
            )
            # ff.ff.2. -> ff.ff.layers.3. (second Linear is at index 3 in MLX)
            new_key = re.sub(
                r"ff\.ff\.2\.",
                r"ff.ff.layers.3.",
                new_key,
            )

            # === HIFT mappings ===

            # conv_pre and conv_post: add .conv for CausalConv1d wrapper
            if "hift.conv_pre." in new_key or "hift.conv_post." in new_key:
                # hift.conv_pre.weight -> hift.conv_pre.conv.weight
                new_key = re.sub(
                    r"hift\.(conv_pre|conv_post)\.(weight|bias)",
                    r"hift.\1.conv.\2",
                    new_key,
                )

            # ups layers: CausalConv1dUpsample has nested .conv
            # hift.ups.N.weight -> hift.ups.N.conv.weight
            new_key = re.sub(
                r"hift\.ups\.(\d+)\.(weight|bias)",
                r"hift.ups.\1.conv.\2",
                new_key,
            )

            # source_downs layers: CausalConv1d/CausalConv1dDownSample have nested .conv
            # hift.source_downs.N.weight -> hift.source_downs.N.conv.weight
            new_key = re.sub(
                r"hift\.source_downs\.(\d+)\.(weight|bias)",
                r"hift.source_downs.\1.conv.\2",
                new_key,
            )

            # f0_predictor.condnet: List of CausalConv1d (each has nested .conv)
            # condnet.N. -> condnet.N.conv. (N is index into list, CausalConv1d has .conv)
            new_key = re.sub(
                r"f0_predictor\.condnet\.(\d+)\.(weight|bias)",
                r"f0_predictor.condnet.\1.conv.\2",
                new_key,
            )

            # ResBlock convs1/convs2: CausalConv1d in list
            # resblocks.X.convs1.Y.weight -> resblocks.X.convs1.Y.conv.weight
            new_key = re.sub(
                r"resblocks\.(\d+)\.(convs1|convs2)\.(\d+)\.(weight|bias)",
                r"resblocks.\1.\2.\3.conv.\4",
                new_key,
            )

            # source_downs: nn.Conv1d (no wrapper)
            # source_resblocks: ResBlock with CausalConv1d
            new_key = re.sub(
                r"source_resblocks\.(\d+)\.(convs1|convs2)\.(\d+)\.(weight|bias)",
                r"source_resblocks.\1.\2.\3.conv.\4",
                new_key,
            )

            # === LLM mappings ===

            # Qwen2 backbone: remap to top-level qwen2 module
            # Raw weights: llm.llm.model.model.X -> qwen2.model.X (inner TextModel)
            if "llm.llm.model.model." in new_key:
                new_key = new_key.replace("llm.llm.model.model.", "qwen2.model.")
            # lm_head lives at Qwen2Model level, not inside .model
            elif new_key.startswith("llm.llm.model.lm_head."):
                new_key = new_key.replace("llm.llm.model.", "qwen2.")
            elif new_key.startswith("llm.llm.lm_head."):
                new_key = new_key.replace("llm.llm.", "qwen2.")
            # Already-converted weights: llm.llm.model.X -> qwen2.model.X
            elif new_key.startswith("llm.llm.model."):
                new_key = new_key.replace("llm.llm.model.", "qwen2.model.")

            new_weights[new_key] = value

        # Handle campplus weights: separate, sanitize with CAMPPlus.sanitize, re-prefix
        campplus_raw = {}
        other_weights = {}
        for key, value in new_weights.items():
            if key.startswith("campplus."):
                campplus_raw[key[len("campplus."):]] = value
            else:
                other_weights[key] = value

        if campplus_raw:
            campplus_sanitized = self.campplus.sanitize(campplus_raw)
            for key, value in campplus_sanitized.items():
                other_weights[f"campplus.{key}"] = value
            return other_weights

        return new_weights


    # --- LLM methods ---

    def encode_text(self, input_ids: mx.array) -> mx.array:
        """Encode text tokens using Qwen2 embeddings."""
        return self.qwen2.model.embed_tokens(input_ids)

    def encode_speech(self, speech_tokens: mx.array) -> mx.array:
        """Embed speech tokens."""
        return self.llm.speech_embedding(speech_tokens)

    def llm_forward(
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

        if cache is None:
            cache = [None] * len(self.qwen2.model.layers)

        mask = create_attention_mask(h, cache[0] if cache else None)

        for layer, c in zip(self.qwen2.model.layers, cache):
            h = layer(h, mask=mask, cache=c)

        h = self.qwen2.model.norm(h)
        return h, cache

    def decode_to_speech(self, hidden_states: mx.array) -> mx.array:
        """Decode hidden states to speech token logits."""
        return self.llm.llm_decoder(hidden_states)

    def nucleus_sampling(
        self,
        logits: mx.array,
        top_p: float = 0.8,
        top_k: int = 25,
    ) -> mx.array:
        """
        Nucleus (top-p) sampling combined with top-k.

        Args:
            logits: Token logits (V,) - single position
            top_p: Cumulative probability threshold
            top_k: Maximum number of tokens to consider

        Returns:
            Sampled token ID (scalar)
        """
        probs = mx.softmax(logits, axis=-1)

        sorted_indices = mx.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        mx.eval(sorted_probs, sorted_indices)
        sorted_probs_np = sorted_probs.tolist()
        sorted_indices_np = sorted_indices.tolist()

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
            return mx.argmax(logits)

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
        top_id = self.nucleus_sampling(logits, top_p=top_p, top_k=top_k)
        mx.eval(top_id)

        recent_tokens = decoded_tokens[-win_size:] if len(decoded_tokens) > 0 else []
        if len(recent_tokens) > 0:
            top_id_val = int(top_id.item())
            rep_count = sum(1 for t in recent_tokens if t == top_id_val)

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
        if logits.ndim == 3:
            logits = logits.squeeze(1)

        B = logits.shape[0]

        if temperature > 0:
            logits = logits / temperature
        else:
            return mx.argmax(logits, axis=-1, keepdims=True)

        if B == 1 and use_ras:
            if decoded_tokens is None:
                decoded_tokens = []
            token = self.ras_sampling(
                logits[0], decoded_tokens, top_p=top_p, top_k=top_k
            )
            return token[None, None] if token.ndim == 0 else token[None, :]

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
        """Check if a token is a stop/special token (>= SPEECH_TOKEN_SIZE)."""
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
        """Sample a token, retrying if stop tokens are generated before min_len."""
        for _ in range(max_trials):
            token = self.sample_next_token(
                logits, temperature=1.0, top_k=top_k, top_p=top_p,
                decoded_tokens=decoded_tokens
            )
            mx.eval(token)
            token_val = int(token.item()) if token.size == 1 else int(token[0, 0].item())

            if not ignore_eos or token_val < self.SPEECH_TOKEN_SIZE:
                return token

        return token

    def generate_speech_tokens(
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

        Input sequence: [SOS, prompt_text + text, TASK_ID, prompt_speech_tokens]

        Args:
            text_tokens: Text token IDs (B, T)
            prompt_text_tokens: Optional prompt text tokens (B, T_prompt_text)
            prompt_speech_tokens: Optional prompt speech tokens for voice cloning (B, T_prompt)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling probability
            min_tokens: Minimum tokens to generate before allowing stop

        Yields:
            Generated speech token IDs (B, 1)
        """
        B = text_tokens.shape[0]

        cache = make_prompt_cache(self.qwen2)

        # Build input: [SOS, prompt_text + text, TASK_ID, prompt_speech]
        if prompt_text_tokens is not None and prompt_text_tokens.shape[1] > 0:
            combined_text = mx.concatenate([prompt_text_tokens, text_tokens], axis=1)
        else:
            combined_text = text_tokens

        sos_token = mx.full((B, 1), self.SOS_TOKEN, dtype=mx.int32)
        sos_emb = self.encode_speech(sos_token)
        text_emb = self.encode_text(combined_text)
        task_id_token = mx.full((B, 1), self.TASK_ID_TOKEN, dtype=mx.int32)
        task_id_emb = self.encode_speech(task_id_token)

        if prompt_speech_tokens is not None and prompt_speech_tokens.shape[1] > 0:
            prompt_emb = self.encode_speech(prompt_speech_tokens)
            embeddings = mx.concatenate(
                [sos_emb, text_emb, task_id_emb, prompt_emb], axis=1
            )
        else:
            embeddings = mx.concatenate([sos_emb, text_emb, task_id_emb], axis=1)

        # Initial forward pass (prompt processing)
        hidden, cache = self.llm_forward(embeddings, cache)
        logits = self.decode_to_speech(hidden[:, -1:, :])

        decoded_tokens: List[int] = []
        generated_count = 0

        while generated_count < max_tokens:
            ignore_eos = generated_count < min_tokens
            current_token = self._sample_valid_token(
                logits, decoded_tokens, top_k=top_k, top_p=top_p,
                ignore_eos=ignore_eos
            )

            current_token_val = int(current_token.item()) if current_token.size == 1 else int(current_token[0, 0].item())

            if self._is_stop_token(current_token_val):
                break

            yield current_token

            decoded_tokens.append(current_token_val)
            generated_count += 1

            speech_emb = self.encode_speech(current_token)
            hidden, cache = self.llm_forward(speech_emb, cache)
            logits = self.decode_to_speech(hidden[:, -1:, :])

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def model_type(self) -> str:
        return "cosyvoice3"

    def token2wav(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: mx.array,
        embedding: mx.array,
        n_timesteps: int = 10,
        streaming: bool = False,
        temperature: float = 1.0,
    ) -> mx.array:
        """
        Convert speech tokens to waveform.

        Args:
            token: Speech tokens (B, T)
            token_len: Token lengths (B,)
            prompt_token: Prompt tokens (B, T_prompt)
            prompt_token_len: Prompt token lengths (B,)
            prompt_feat: Prompt mel features (B, T_prompt * ratio, mel_dim) - PyTorch format
            prompt_feat_len: Prompt feature lengths (B,)
            embedding: Speaker embedding (B, spk_embed_dim)
            n_timesteps: Number of flow ODE steps
            streaming: Whether to use streaming mode
            temperature: Sampling temperature

        Returns:
            Audio waveform (B, T_audio)
        """
        # Generate mel-spectrogram (using inference to get only generated part)
        mel = self.flow.inference(
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            n_timesteps,
            streaming,
            temperature,
        )

        # Generate audio from mel
        audio = self.hift(mel)

        return audio

    def generate(
        self,
        speech_tokens: mx.array,
        speaker_embedding: mx.array,
        prompt_speech_tokens: Optional[mx.array] = None,
        prompt_mel: Optional[mx.array] = None,
        n_timesteps: int = 10,
        streaming: bool = False,
        temperature: float = 1.0,
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate audio from speech tokens.

        Args:
            speech_tokens: Speech token IDs (B, T)
            speaker_embedding: Speaker embedding (B, spk_embed_dim)
            prompt_speech_tokens: Optional prompt tokens (B, T_prompt)
            prompt_mel: Optional prompt mel features (B, T_prompt_mel, mel_dim) - PyTorch format
            n_timesteps: Number of flow ODE steps
            streaming: Whether to use streaming mode
            temperature: Sampling temperature

        Yields:
            GenerationResult with generated audio
        """
        time_start = time.time()

        B = speech_tokens.shape[0]
        token_len = mx.array([speech_tokens.shape[1]] * B)

        # Handle prompts
        if prompt_speech_tokens is None:
            prompt_speech_tokens = mx.zeros((B, 0), dtype=mx.int32)
            prompt_token_len = mx.zeros((B,), dtype=mx.int32)
        else:
            prompt_token_len = mx.array([prompt_speech_tokens.shape[1]] * B)

        # prompt_mel is now (B, T, mel_dim) format (matching PyTorch)
        if prompt_mel is None:
            prompt_mel = mx.zeros((B, 0, self.config.mel_dim))
            prompt_feat_len = mx.zeros((B,), dtype=mx.int32)
        else:
            prompt_feat_len = mx.array([prompt_mel.shape[1]] * B)  # T is at index 1

        # Generate audio
        audio = self.token2wav(
            speech_tokens,
            token_len,
            prompt_speech_tokens,
            prompt_token_len,
            prompt_mel,
            prompt_feat_len,
            speaker_embedding,
            n_timesteps,
            streaming,
            temperature,
        )

        mx.eval(audio)
        time_end = time.time()

        # Calculate metrics
        audio_samples = audio.shape[-1]
        audio_duration_seconds = audio_samples / self.config.sample_rate

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        yield GenerationResult(
            audio=audio.squeeze(),
            sample_rate=self.config.sample_rate,
            samples=audio_samples,
            segment_idx=0,
            token_count=speech_tokens.shape[1],
            audio_samples={
                "samples": audio_samples,
                "samples-per-sec": (
                    round(audio_samples / audio_duration_seconds, 2)
                    if audio_duration_seconds > 0
                    else 0
                ),
            },
            audio_duration=duration_str,
            real_time_factor=(
                audio_duration_seconds / (time_end - time_start)
                if (time_end - time_start) > 0
                else 0
            ),
            prompt={
                "tokens": speech_tokens.shape[1],
                "tokens-per-sec": (
                    round(speech_tokens.shape[1] / (time_end - time_start), 2)
                    if (time_end - time_start) > 0
                    else 0
                ),
            },
            processing_time_seconds=time_end - time_start,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

        mx.clear_cache()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir: Optional[str] = None,
        load_llm: bool = True,
    ) -> "Model":
        """
        Load a pretrained CosyVoice3 model.

        Args:
            model_id: HuggingFace model ID or local path
            local_dir: Optional local directory with weights
            load_llm: Whether to load the LLM (set False for token-to-wav only)

        Returns:
            Loaded Model instance
        """
        import os

        from huggingface_hub import snapshot_download

        # Download or locate model files
        if local_dir and os.path.exists(local_dir):
            model_dir = local_dir
        elif os.path.exists(model_id):
            model_dir = model_id
        else:
            model_dir = snapshot_download(model_id)

        print(f"Loading CosyVoice3 from {model_dir}")

        # Create model
        config = ModelConfig(model_path=Path(model_dir))
        model = cls(config, load_llm=load_llm)

        # Load weights from safetensors
        safetensors_path = os.path.join(model_dir, "model.safetensors")

        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(
                f"model.safetensors not found in {model_dir}.\n"
                "To use CosyVoice3, either:\n"
                "  1. Use the pre-converted model from mlx-community on HuggingFace\n"
                "  2. Convert PyTorch weights first:\n"
                "     python -m mlx_audio.tts.models.cosyvoice3.convert --model-dir /path/to/model"
            )

        weights = mx.load(safetensors_path)
        weights = model.sanitize(dict(weights))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        
        model = cls.post_load_hook(model, Path(model_dir))
        return model

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":


        model.frontend = CosyVoice3Frontend(
            model_path=str(model_path),
            campplus_model=model.campplus,
            sample_rate=model.config.sample_rate,
        )
        return model

    def inference_zero_shot(
        self,
        text: str,
        ref_text: str,
        ref_audio: str,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        top_k: int = 25,
    ) -> Generator[GenerationResult, None, None]:
        """
        Zero-shot voice cloning synthesis.

        Args:
            text: Text to synthesize
            ref_text: Transcript of the reference audio
            ref_audio: Path to the reference audio file
            n_timesteps: Number of flow ODE steps
            temperature: LLM sampling temperature
            top_k: Top-k sampling for LLM

        Yields:
            GenerationResult with synthesized audio
        """
        if not self._llm_loaded:
            raise RuntimeError("LLM not loaded. Use load_llm=True in from_pretrained()")

        if self.frontend is None:
            raise RuntimeError(
                "Frontend not initialized. Use from_pretrained() to load the model."
            )

        time_start = time.time()

        # Process inputs through frontend
        inputs = self.frontend.frontend_zero_shot(text, ref_text, ref_audio)

        # Generate speech tokens from text using LLM
        # With prompt_speech_tokens, the LLM generates continuation tokens (target only).
        # The flow prepends prompt_speech_tokens for the full sequence, then strips
        # the prompt portion from the output mel.
        text_tokens = inputs["text_tokens"]
        prompt_text_tokens = inputs.get("prompt_text_tokens")
        prompt_speech_tokens = inputs.get("prompt_speech_tokens")
        speech_tokens_list = []

        # Calculate min/max tokens for the target text
        # PyTorch uses: min = (text_len - prompt_text_len) * 2, max = ... * 20
        target_text_len = text_tokens.shape[1]
        min_tokens = int(target_text_len * 2)
        max_tokens = int(target_text_len * 20)

        prompt_token_len = (
            prompt_speech_tokens.shape[1] if prompt_speech_tokens is not None else 0
        )

        # Silent/breath tokens (matching PyTorch CosyVoice3Model.silent_tokens)
        silent_tokens = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}
        max_silent_token_num = 5
        cur_silent_token_num = 0

        # Track token values for repetition detection
        token_values = []
        stopped_reason = "eos"
        generated_total = 0

        for token in self.generate_speech_tokens(
            text_tokens,
            prompt_text_tokens=prompt_text_tokens,
            prompt_speech_tokens=prompt_speech_tokens,
            temperature=temperature,
            top_k=top_k,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        ):
            mx.eval(token)
            generated_total += 1
            token_val = int(token.item()) if token.size == 1 else int(token[0, 0].item())

            # Filter excessive consecutive silent tokens (matching PyTorch llm_job)
            if token_val in silent_tokens:
                cur_silent_token_num += 1
                if cur_silent_token_num > max_silent_token_num:
                    continue
            else:
                cur_silent_token_num = 0

            speech_tokens_list.append(token)
            token_values.append(token_val)

            # Repetition detection: check if a window of tokens repeats.
            # Use larger windows (20, 25, 30) to avoid false triggers on
            # sustained sounds or natural speech patterns.
            n = len(token_values)
            if n >= 40:
                for window in (20, 25, 30):
                    if n >= 2 * window:
                        recent = token_values[-window:]
                        prev = token_values[-2 * window : -window]
                        if recent == prev:
                            stopped_reason = f"repetition(w={window})"
                            break
                else:
                    continue
                break  # exit outer for loop

        # Detect if we hit max_tokens
        if stopped_reason == "eos" and generated_total >= max_tokens:
            stopped_reason = "max_tokens"

        if len(speech_tokens_list) == 0:
            raise RuntimeError("LLM generated no speech tokens")

        speech_tokens = mx.concatenate(speech_tokens_list, axis=1)
        mx.eval(speech_tokens)

        gen_len = speech_tokens.shape[1]
        expected_audio_dur = gen_len / self.config.token_frame_rate
        print(
            f"  [zero-shot] target_text_tokens={target_text_len}, "
            f"prompt_speech_tokens={prompt_token_len}, "
            f"generated={gen_len} (min={min_tokens}, max={max_tokens}), "
            f"expected_dur={expected_audio_dur:.1f}s, "
            f"stop={stopped_reason}"
        )

        # Convert speech tokens to audio
        # Pass prompt_speech_tokens to flow so the full token sequence
        # [prompt_tokens, continuation_tokens] provides correct mel generation.
        # The flow strips the prompt mel portion, leaving only target text audio.
        for result in self.generate(
            speech_tokens=speech_tokens,
            speaker_embedding=inputs["speaker_embedding"],
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_mel=inputs.get("prompt_mel"),
            n_timesteps=n_timesteps,
        ):
            # Update timing
            total_time = time.time() - time_start
            result.processing_time_seconds = total_time
            if total_time > 0:
                audio_duration = result.samples / self.config.sample_rate
                result.real_time_factor = audio_duration / total_time
            yield result

    def _generate_from_inputs(
        self,
        inputs: dict,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        top_k: int = 25,
        label: str = "tts",
        max_token_factor: int = 20,
    ) -> Generator[GenerationResult, None, None]:
        """
        Shared LLM generation + Flow synthesis logic.

        Args:
            inputs: Dictionary from frontend (text_tokens, prompt_text_tokens, etc.)
            n_timesteps: Number of flow ODE steps
            temperature: LLM sampling temperature
            top_k: Top-k sampling for LLM
            label: Label for debug output
            max_token_factor: Multiplier for max_tokens (higher = allow longer speech)
        """
        time_start = time.time()

        text_tokens = inputs["text_tokens"]
        prompt_text_tokens = inputs.get("prompt_text_tokens")
        # LLM prompt speech tokens (None for instruct/cross-lingual modes)
        llm_prompt_speech_tokens = inputs.get("prompt_speech_tokens")
        # Flow prompt speech tokens (uses flow-specific key if present, else same as LLM)
        flow_prompt_speech_tokens = inputs.get(
            "flow_prompt_speech_tokens", llm_prompt_speech_tokens
        )
        speech_tokens_list = []

        target_text_len = text_tokens.shape[1]
        min_tokens = int(target_text_len * 2)
        max_tokens = int(target_text_len * max_token_factor)

        llm_prompt_len = (
            llm_prompt_speech_tokens.shape[1]
            if llm_prompt_speech_tokens is not None
            else 0
        )
        flow_prompt_len = (
            flow_prompt_speech_tokens.shape[1]
            if flow_prompt_speech_tokens is not None
            else 0
        )

        silent_tokens = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}
        max_silent_token_num = 5
        cur_silent_token_num = 0

        token_values = []
        stopped_reason = "eos"
        generated_total = 0  # Track total tokens from LLM (including filtered)

        for token in self.generate_speech_tokens(
            text_tokens,
            prompt_text_tokens=prompt_text_tokens,
            prompt_speech_tokens=llm_prompt_speech_tokens,
            temperature=temperature,
            top_k=top_k,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        ):
            mx.eval(token)
            generated_total += 1
            token_val = int(token.item()) if token.size == 1 else int(token[0, 0].item())

            if token_val in silent_tokens:
                cur_silent_token_num += 1
                if cur_silent_token_num > max_silent_token_num:
                    continue
            else:
                cur_silent_token_num = 0

            speech_tokens_list.append(token)
            token_values.append(token_val)

            # Repetition detection: check if a window of tokens repeats.
            # Use larger windows (20, 25, 30) to avoid false triggers on
            # sustained sounds or slow speech patterns.
            n = len(token_values)
            if n >= 40:
                for window in (20, 25, 30):
                    if n >= 2 * window:
                        recent = token_values[-window:]
                        prev = token_values[-2 * window : -window]
                        if recent == prev:
                            stopped_reason = f"repetition(w={window})"
                            break
                else:
                    continue
                break

        # Detect if we hit max_tokens (LLM exhausted without EOS)
        if stopped_reason == "eos" and generated_total >= max_tokens:
            stopped_reason = "max_tokens"

        if len(speech_tokens_list) == 0:
            raise RuntimeError("LLM generated no speech tokens")

        speech_tokens = mx.concatenate(speech_tokens_list, axis=1)
        mx.eval(speech_tokens)

        gen_len = speech_tokens.shape[1]
        expected_audio_dur = gen_len / self.config.token_frame_rate
        print(
            f"  [{label}] target_text_tokens={target_text_len}, "
            f"llm_prompt_tokens={llm_prompt_len}, "
            f"flow_prompt_tokens={flow_prompt_len}, "
            f"generated={gen_len} (min={min_tokens}, max={max_tokens}), "
            f"expected_dur={expected_audio_dur:.1f}s, "
            f"stop={stopped_reason}"
        )

        # Flow uses flow_prompt_speech_tokens for mel conditioning
        for result in self.generate(
            speech_tokens=speech_tokens,
            speaker_embedding=inputs["speaker_embedding"],
            prompt_speech_tokens=flow_prompt_speech_tokens,
            prompt_mel=inputs.get("prompt_mel"),
            n_timesteps=n_timesteps,
        ):
            total_time = time.time() - time_start
            result.processing_time_seconds = total_time
            if total_time > 0:
                audio_duration = result.samples / self.config.sample_rate
                result.real_time_factor = audio_duration / total_time
            yield result

    def inference_instruct(
        self,
        text: str,
        ref_audio: str,
        instruct_text: str,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        top_k: int = 25,
    ) -> Generator[GenerationResult, None, None]:
        """
        Instruct-mode synthesis with style/language control.

        Uses ref_audio for speaker identity and instruct_text to control
        speaking style or language.

        The prompt format: "You are a helpful assistant. {instruct_text}<|endofprompt|>"

        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio file (for speaker identity)
            instruct_text: Instruction (e.g., "Please speak as fast as possible.",
                           "Please express in Cantonese.")
            n_timesteps: Number of flow ODE steps
            temperature: LLM sampling temperature
            top_k: Top-k sampling for LLM

        Yields:
            GenerationResult with synthesized audio
        """
        if not self._llm_loaded:
            raise RuntimeError("LLM not loaded. Use load_llm=True in from_pretrained()")
        if self.frontend is None:
            raise RuntimeError(
                "Frontend not initialized. Use from_pretrained() to load the model."
            )

        inputs = self.frontend.frontend_instruct(text, ref_audio, instruct_text)
        # Use higher max_token_factor for instruct mode since style instructions
        # (e.g., "speak as slow as possible") can require significantly more
        # speech tokens per text token than normal speech.
        yield from self._generate_from_inputs(
            inputs, n_timesteps, temperature, top_k,
            label="instruct", max_token_factor=40,
        )

    def inference_cross_lingual(
        self,
        text: str,
        ref_audio: str,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        top_k: int = 25,
    ) -> Generator[GenerationResult, None, None]:
        """
        Cross-lingual / fine-grained control synthesis.

        Supported control tokens: [breath], [laughter], [noise], etc.
        The system prompt is auto-prepended if not already present.

        Example:
            text="[breath]Hello world,[breath]this is a test."

        Args:
            text: Target text with optional control tokens
            ref_audio: Path to reference audio file (for speaker identity)
            n_timesteps: Number of flow ODE steps
            temperature: LLM sampling temperature
            top_k: Top-k sampling for LLM

        Yields:
            GenerationResult with synthesized audio
        """
        if not self._llm_loaded:
            raise RuntimeError("LLM not loaded. Use load_llm=True in from_pretrained()")
        if self.frontend is None:
            raise RuntimeError(
                "Frontend not initialized. Use from_pretrained() to load the model."
            )

        inputs = self.frontend.frontend_cross_lingual(text, ref_audio)
        yield from self._generate_from_inputs(
            inputs, n_timesteps, temperature, top_k, label="cross-lingual"
        )

    def inference_vc(
        self,
        source_audio: str,
        ref_audio: str,
        n_timesteps: int = 10,
    ) -> Generator[GenerationResult, None, None]:
        """
        Voice conversion: convert source audio to the target speaker's voice.

        Extracts speech tokens from source_audio and uses ref_audio for
        speaker identity. Skips the LLM step entirely.

        Args:
            source_audio: Path to source audio (content to convert)
            ref_audio: Path to reference audio (target voice)
            n_timesteps: Number of flow ODE steps

        Yields:
            GenerationResult with converted audio
        """
        if self.frontend is None:
            raise RuntimeError(
                "Frontend not initialized. Use from_pretrained() to load the model."
            )

        time_start = time.time()

        # Process inputs through frontend (VC mode)
        inputs = self.frontend.frontend_vc(source_audio, ref_audio)

        source_speech_tokens = inputs["source_speech_tokens"]
        prompt_speech_tokens = inputs.get("prompt_speech_tokens")

        gen_len = source_speech_tokens.shape[1]
        prompt_token_len = (
            prompt_speech_tokens.shape[1] if prompt_speech_tokens is not None else 0
        )
        expected_audio_dur = gen_len / self.config.token_frame_rate
        print(
            f"  [vc] source_tokens={gen_len}, "
            f"prompt_speech_tokens={prompt_token_len}, "
            f"expected_dur={expected_audio_dur:.1f}s"
        )

        # Voice conversion: use source tokens directly (no LLM)
        for result in self.generate(
            speech_tokens=source_speech_tokens,
            speaker_embedding=inputs["speaker_embedding"],
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_mel=inputs.get("prompt_mel"),
            n_timesteps=n_timesteps,
        ):
            total_time = time.time() - time_start
            result.processing_time_seconds = total_time
            if total_time > 0:
                audio_duration = result.samples / self.config.sample_rate
                result.real_time_factor = audio_duration / total_time
            yield result

    def text_to_speech(
        self,
        text: str,
        speaker_embedding: Optional[mx.array] = None,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        top_k: int = 25,
    ) -> Generator[GenerationResult, None, None]:
        """
        Simple text-to-speech without voice cloning.

        Args:
            text: Text to synthesize
            speaker_embedding: Optional speaker embedding (random if None)
            n_timesteps: Number of flow ODE steps
            temperature: LLM sampling temperature
            top_k: Top-k sampling

        Yields:
            GenerationResult with synthesized audio
        """
        if not self._llm_loaded:
            raise RuntimeError("LLM not loaded. Use load_llm=True in from_pretrained()")

        if self.frontend is None or self.frontend.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")

        import numpy as np

        # Tokenize text
        text_tokens = self.frontend.tokenize(text)

        # Use random speaker embedding if not provided
        if speaker_embedding is None:
            speaker_embedding = mx.array(
                np.random.randn(1, self.config.spk_embed_dim).astype(np.float32) * 0.1
            )

        # Generate speech tokens
        # Calculate min/max tokens matching PyTorch ratios
        target_text_len = text_tokens.shape[1]
        min_tokens = int(target_text_len * 2)
        max_tokens = int(target_text_len * 20)

        speech_tokens_list = []
        for token in self.generate_speech_tokens(
            text_tokens,
            temperature=temperature,
            top_k=top_k,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        ):
            speech_tokens_list.append(token)

        if len(speech_tokens_list) == 0:
            raise RuntimeError("LLM generated no speech tokens")

        speech_tokens = mx.concatenate(speech_tokens_list, axis=1)
        mx.eval(speech_tokens)

        # Convert to audio
        for result in self.generate(
            speech_tokens=speech_tokens,
            speaker_embedding=speaker_embedding,
            n_timesteps=n_timesteps,
        ):
            yield result

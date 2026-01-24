# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult

from .dit import DiT
from .flow import CausalConditionalCFM, CausalMaskedDiffWithDiT, PreLookaheadLayer
from .frontend import CosyVoice3Frontend
from .hift import CausalHiFTGenerator
from .llm import CosyVoice3LM


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


class Model(nn.Module):
    """
    CosyVoice3 for text-to-speech generation.

    This is an LLM-based TTS model with flow matching for mel generation
    and HiFi-GAN vocoder for audio synthesis.

    Pipeline: Text -> LLM -> Speech Tokens -> Flow -> Mel -> HIFT -> Audio
    """

    def __init__(self, config: ModelConfig, load_llm: bool = True):
        super().__init__()
        self.config = config
        self.frontend = None  # Initialized in from_pretrained

        # Build LLM (optional - can be skipped for token-to-wav only)
        if load_llm:
            self.llm = CosyVoice3LM(
                llm_input_size=config.llm_input_size,
                llm_output_size=config.llm_output_size,
                speech_token_size=config.speech_token_size,
                text_vocab_size=config.text_vocab_size,
            )
        else:
            self.llm = None

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

            # Original has extra wrapper: llm.llm.model.model.X -> llm.llm.model.X
            # (Original: CosyVoiceLLM.llm.model.model, Ours: CosyVoice3LM.llm.model)
            new_key = new_key.replace("llm.llm.model.model.", "llm.llm.model.")

            new_weights[new_key] = value

        return new_weights

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
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())


        # Initialize frontend
        tokenizer_path = os.path.join(model_dir, "CosyVoice-BlankEN")
        campplus_path = os.path.join(model_dir, "campplus.safetensors")
        speech_tokenizer_path = os.path.join(
            model_dir, "speech_tokenizer_v3.safetensors"
        )

        if os.path.exists(tokenizer_path) or os.path.exists(campplus_path):
            print(f"  Initializing frontend with tokenizer: {tokenizer_path}, campplus: {campplus_path}, speech_tokenizer: {speech_tokenizer_path}")
            model.frontend = CosyVoice3Frontend(
                tokenizer_path=(
                    tokenizer_path if os.path.exists(tokenizer_path) else None
                ),
                campplus_path=campplus_path if os.path.exists(campplus_path) else None,
                speech_tokenizer_path=(
                    speech_tokenizer_path
                    if os.path.exists(speech_tokenizer_path)
                    else None
                ),
                sample_rate=config.sample_rate,
            )
            print(f"  Frontend initialized")

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
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Use load_llm=True in from_pretrained()")

        if self.frontend is None:
            raise RuntimeError(
                "Frontend not initialized. Check tokenizer and campplus paths."
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

        for token in self.llm.generate(
            text_tokens,
            prompt_text_tokens=prompt_text_tokens,
            prompt_speech_tokens=prompt_speech_tokens,
            temperature=temperature,
            top_k=top_k,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        ):
            mx.eval(token)
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

            # Repetition detection: check multiple window sizes.
            # If any window of size W repeats consecutively, stop early.
            n = len(token_values)
            if n >= 20:
                for window in (10, 15, 20, 25):
                    if n >= 2 * window:
                        recent = token_values[-window:]
                        prev = token_values[-2 * window : -window]
                        if recent == prev:
                            stopped_reason = f"repetition(w={window})"
                            break
                else:
                    continue
                break  # exit outer for loop

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
        Instruct-mode synthesis with style control.

        Uses ref_audio for speaker identity and instruct_text to control
        speaking style (e.g., "speak slowly", "whisper", "speak with excitement").

        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio file (for speaker identity)
            instruct_text: Style instruction (e.g., "speak slowly")
            n_timesteps: Number of flow ODE steps
            temperature: LLM sampling temperature
            top_k: Top-k sampling for LLM

        Yields:
            GenerationResult with synthesized audio
        """
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Use load_llm=True in from_pretrained()")

        if self.frontend is None:
            raise RuntimeError(
                "Frontend not initialized. Check tokenizer and campplus paths."
            )

        time_start = time.time()

        # Process inputs through frontend (instruct mode)
        inputs = self.frontend.frontend_instruct(text, ref_audio, instruct_text)

        text_tokens = inputs["text_tokens"]
        prompt_text_tokens = inputs.get("prompt_text_tokens")
        prompt_speech_tokens = inputs.get("prompt_speech_tokens")
        speech_tokens_list = []

        target_text_len = text_tokens.shape[1]
        min_tokens = int(target_text_len * 2)
        max_tokens = int(target_text_len * 20)

        prompt_token_len = (
            prompt_speech_tokens.shape[1] if prompt_speech_tokens is not None else 0
        )

        silent_tokens = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}
        max_silent_token_num = 5
        cur_silent_token_num = 0

        token_values = []
        stopped_reason = "eos"

        for token in self.llm.generate(
            text_tokens,
            prompt_text_tokens=prompt_text_tokens,
            prompt_speech_tokens=prompt_speech_tokens,
            temperature=temperature,
            top_k=top_k,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        ):
            mx.eval(token)
            token_val = int(token.item()) if token.size == 1 else int(token[0, 0].item())

            if token_val in silent_tokens:
                cur_silent_token_num += 1
                if cur_silent_token_num > max_silent_token_num:
                    continue
            else:
                cur_silent_token_num = 0

            speech_tokens_list.append(token)
            token_values.append(token_val)

            n = len(token_values)
            if n >= 20:
                for window in (10, 15, 20, 25):
                    if n >= 2 * window:
                        recent = token_values[-window:]
                        prev = token_values[-2 * window : -window]
                        if recent == prev:
                            stopped_reason = f"repetition(w={window})"
                            break
                else:
                    continue
                break

        if len(speech_tokens_list) == 0:
            raise RuntimeError("LLM generated no speech tokens")

        speech_tokens = mx.concatenate(speech_tokens_list, axis=1)
        mx.eval(speech_tokens)

        gen_len = speech_tokens.shape[1]
        expected_audio_dur = gen_len / self.config.token_frame_rate
        print(
            f"  [instruct] target_text_tokens={target_text_len}, "
            f"prompt_speech_tokens={prompt_token_len}, "
            f"generated={gen_len} (min={min_tokens}, max={max_tokens}), "
            f"expected_dur={expected_audio_dur:.1f}s, "
            f"stop={stopped_reason}"
        )

        for result in self.generate(
            speech_tokens=speech_tokens,
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
                "Frontend not initialized. Check tokenizer and campplus paths."
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
        if self.llm is None:
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
        for token in self.llm.generate(
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

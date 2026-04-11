import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import GenerationResult
from .audio_vae import AudioVAE
from .config import LMConfig, ModelArgs
from .dit import UnifiedCFM, VoxCPMLocDiTV2
from .encoder import VoxCPMLocEnc
from .minicpm import MiniCPMModel


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def __call__(self, x):
        x = self.in_proj(x)
        x = mx.tanh(x)
        x = mx.round(x * self.scale) / self.scale
        return self.out_proj(x)


def _trim_audio_silence_vad(
    audio: np.ndarray,
    sample_rate: int,
    max_silence_ms: float = 200.0,
    top_db: float = 35.0,
) -> np.ndarray:
    """Trim leading/trailing silence using energy-based VAD (numpy only)."""
    if audio.size == 0:
        return audio

    y = audio.flatten()
    n = len(y)
    frame_length = 2048
    hop_length = 512

    ref = np.max(np.abs(y))
    if ref <= 0:
        return audio

    threshold = ref * (10.0 ** (-top_db / 20.0))

    # Find speech boundaries using RMS energy
    n_frames = max(0, (n - frame_length) // hop_length + 1)
    first_voice_frame = -1
    last_voice_frame = -1

    for j in range(n_frames):
        idx = j * hop_length
        if idx + frame_length > n:
            break
        rms = np.sqrt(np.mean(y[idx : idx + frame_length] ** 2))
        if rms >= threshold:
            if first_voice_frame < 0:
                first_voice_frame = j
            last_voice_frame = j

    if first_voice_frame < 0:
        return audio

    start = max(0, first_voice_frame * hop_length)
    end = min(n, (last_voice_frame + 1) * hop_length + (frame_length - hop_length))

    max_silence_samples = int(max_silence_ms * sample_rate / 1000.0)
    new_start = max(0, start - max_silence_samples)
    new_end = min(n, end + max_silence_samples)

    return audio[:, new_start:new_end] if audio.ndim == 2 else audio[new_start:new_end]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.feat_dim = args.feat_dim

        # LM Backbone
        self.base_lm = MiniCPMModel(args.lm_config)

        # Residual LM (vocab_size=0, optionally no_rope)
        res_config = LMConfig(**vars(args.lm_config))
        res_config.num_hidden_layers = args.residual_lm_num_layers
        res_config.vocab_size = 0
        res_config.no_rope = args.residual_lm_no_rope
        self.residual_lm = MiniCPMModel(res_config)

        # Encoder
        enc_config = LMConfig(**vars(args.lm_config))
        enc_config.hidden_size = args.encoder_config.hidden_dim
        enc_config.intermediate_size = args.encoder_config.ffn_dim
        enc_config.num_attention_heads = args.encoder_config.num_heads
        enc_config.num_hidden_layers = args.encoder_config.num_layers
        enc_config.kv_channels = args.encoder_config.kv_channels
        enc_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(enc_config, input_dim=args.feat_dim)

        # DiT / CFM
        dit_config = LMConfig(**vars(args.lm_config))
        dit_config.hidden_size = args.dit_config.hidden_dim
        dit_config.intermediate_size = args.dit_config.ffn_dim
        dit_config.num_attention_heads = args.dit_config.num_heads
        dit_config.num_hidden_layers = args.dit_config.num_layers
        dit_config.kv_channels = args.dit_config.kv_channels
        dit_config.vocab_size = 0

        estimator = VoxCPMLocDiTV2(dit_config, in_channels=args.feat_dim)
        self.feat_decoder = UnifiedCFM(
            in_channels=args.feat_dim,
            cfm_params=args.dit_config.cfm_config,
            estimator=estimator,
            mean_mode=args.dit_config.dit_mean_mode,
        )

        # Projections
        self.fsq_layer = ScalarQuantizationLayer(
            args.lm_config.hidden_size,
            args.lm_config.hidden_size,
            args.scalar_quantization_latent_dim,
            args.scalar_quantization_scale,
        )

        self.enc_to_lm_proj = nn.Linear(
            args.encoder_config.hidden_dim, args.lm_config.hidden_size
        )
        self.lm_to_dit_proj = nn.Linear(
            args.lm_config.hidden_size, args.dit_config.hidden_dim
        )
        self.res_to_dit_proj = nn.Linear(
            args.lm_config.hidden_size, args.dit_config.hidden_dim
        )

        # V2: fusion_concat_proj replaces simple addition
        self.fusion_concat_proj = nn.Linear(
            args.lm_config.hidden_size * 2, args.lm_config.hidden_size
        )

        # Stop Predictor
        self.stop_proj = nn.Linear(
            args.lm_config.hidden_size, args.lm_config.hidden_size
        )
        self.stop_head = nn.Linear(args.lm_config.hidden_size, 2, bias=False)

        # Audio VAE V2
        self.audio_vae = AudioVAE(args.audio_vae_config)

        # Special tokens
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        # Placeholder for tokenizer
        self.tokenizer = None
        self._compiled = False

    def compile_model(self):
        """Compile hot paths with mx.compile for faster inference."""
        if self._compiled:
            return
        # DiT estimator (54% of step time) - called 10x per step via CFM
        self.feat_decoder.estimator = mx.compile(self.feat_decoder.estimator)
        # Compile individual LM/encoder layers (preserve module attribute access)
        for layer in self.base_lm.layers:
            layer.self_attn = mx.compile(layer.self_attn)
            layer.mlp = mx.compile(layer.mlp)
        for layer in self.residual_lm.layers:
            layer.self_attn = mx.compile(layer.self_attn)
            layer.mlp = mx.compile(layer.mlp)
        for layer in self.feat_encoder.encoder.layers:
            layer.self_attn = mx.compile(layer.self_attn)
            layer.mlp = mx.compile(layer.mlp)
        self._compiled = True

    @property
    def sample_rate(self):
        return self.args.audio_vae_config.out_sample_rate

    @property
    def _encode_sample_rate(self):
        return self.args.audio_vae_config.sample_rate

    def _tokenize(self, text: str):
        """Tokenize text without BOS token (matching PyTorch behavior)."""
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def _encode_wav(self, audio_input, padding_mode: str = "right") -> mx.array:
        """Load, VAD-trim, pad and VAE-encode audio.

        Args:
            audio_input: file path (str), mx.array, or numpy array.
            padding_mode: "right" or "left".

        Returns:
            audio_feat: (T, P, D) array of latent patches.
        """
        if isinstance(audio_input, str):
            import soundfile as sf

            audio, sr = sf.read(audio_input, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != self._encode_sample_rate:
                try:
                    import librosa

                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=self._encode_sample_rate
                    )
                except ImportError:
                    import scipy.signal

                    num_samples = int(len(audio) * self._encode_sample_rate / sr)
                    audio = scipy.signal.resample(audio, num_samples)
        else:
            # mx.array or numpy array — assume loaded at model.sample_rate,
            # resample to encoder rate if needed
            audio = np.array(audio_input).flatten()
            if self.sample_rate != self._encode_sample_rate:
                import scipy.signal

                num_samples = int(
                    len(audio) * self._encode_sample_rate / self.sample_rate
                )
                audio = scipy.signal.resample(audio, num_samples)

        # VAD trim
        audio_2d = audio[np.newaxis, :]
        audio_2d = _trim_audio_silence_vad(
            audio_2d, self._encode_sample_rate, max_silence_ms=200.0
        )
        audio = audio_2d.flatten()

        # Pad to patch alignment
        patch_len = self.patch_size * self.audio_vae.chunk_size
        if len(audio) % patch_len != 0:
            padding_size = patch_len - len(audio) % patch_len
            if padding_mode == "left":
                audio = np.pad(audio, (padding_size, 0))
            else:
                audio = np.pad(audio, (0, padding_size))

        # VAE encode
        audio_mx = mx.array(audio)[None, None, :]  # (1, 1, T)
        feat = self.audio_vae.encode(audio_mx, self._encode_sample_rate)
        # feat: (1, T', D) in MLX channel-last format
        feat = feat.squeeze(0)  # (T', D)

        # Reshape into patches: (T', D) -> (num_patches, patch_size, D)
        T_prime = feat.shape[0]
        num_patches = T_prime // self.patch_size
        feat = feat[: num_patches * self.patch_size, :]
        feat = feat.reshape(num_patches, self.patch_size, -1)

        return feat

    def _make_ref_prefix(self, ref_feat: mx.array):
        """Build the [ref_start, ref_audio, ref_end] prefix segments.

        Returns:
            tokens, feats, text_mask, audio_mask
        """
        ref_len = ref_feat.shape[0]
        latent_dim = self.audio_vae.latent_dim
        z1 = mx.zeros((1, self.patch_size, latent_dim))

        tokens = mx.concatenate(
            [
                mx.array([self.ref_audio_start_token], dtype=mx.int32),
                mx.zeros(ref_len, dtype=mx.int32),
                mx.array([self.ref_audio_end_token], dtype=mx.int32),
            ]
        )

        feats = mx.concatenate([z1, ref_feat, z1], axis=0)

        t_mask = mx.concatenate(
            [
                mx.array([1], dtype=mx.float32),
                mx.zeros(ref_len, dtype=mx.float32),
                mx.array([1], dtype=mx.float32),
            ]
        )
        a_mask = mx.concatenate(
            [
                mx.array([0], dtype=mx.float32),
                mx.ones(ref_len, dtype=mx.float32),
                mx.array([0], dtype=mx.float32),
            ]
        )

        return tokens, feats, t_mask, a_mask

    def sanitize(self, weights: dict):
        from mlx.utils import tree_flatten

        vae_already_sanitized = False

        # 0. Check if audio_vae weights are present. If not, try to load from pth/safetensors
        has_vae = any(k.startswith("audio_vae.") for k in weights.keys())
        if not has_vae and self.args.model_path:
            model_path = Path(self.args.model_path)
            # Try safetensors first, then pth
            vae_sf = model_path / "audiovae.safetensors"
            vae_pth = model_path / "audiovae.pth"

            vae_weights_raw = None
            if vae_sf.exists():
                vae_weights_raw = {k: v for k, v in mx.load(str(vae_sf)).items()}
            elif vae_pth.exists():
                try:
                    import torch

                    state = torch.load(str(vae_pth), map_location="cpu")
                    if "state_dict" in state:
                        state = state["state_dict"]
                    vae_weights_raw = {}
                    for k, v in state.items():
                        if k.startswith("module."):
                            k = k[7:]
                        vae_weights_raw[k] = mx.array(v.numpy())
                except ImportError:
                    pass

            if vae_weights_raw is not None:
                sanitized_vae = self.audio_vae.sanitize(vae_weights_raw)
                for k, v in sanitized_vae.items():
                    weights[f"audio_vae.{k}"] = v
                vae_already_sanitized = True

        # 1. Sanitize VAE weights if present and not already done
        vae_weights = {k: v for k, v in weights.items() if k.startswith("audio_vae.")}
        vae_weights_stripped = {
            k[len("audio_vae.") :]: v for k, v in vae_weights.items()
        }

        if vae_weights_stripped and not vae_already_sanitized:
            sanitized_vae = self.audio_vae.sanitize(vae_weights_stripped)
            for k in list(vae_weights.keys()):
                del weights[k]
            for k, v in sanitized_vae.items():
                weights[f"audio_vae.{k}"] = v

        # 1b. Extract sr_boundaries buffer (not a parameter)
        sr_key = "audio_vae.decoder._sr_boundaries"
        if sr_key in weights:
            self.audio_vae.decoder._sr_boundaries = weights.pop(sr_key)

        # 2. Shape-fix remaining weights
        new_weights = {}
        curr_shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}

        for k, v in weights.items():
            if k not in curr_shapes:
                new_weights[k] = v
                continue

            target_shape = curr_shapes[k]
            if v.shape == target_shape:
                new_weights[k] = v
            else:
                if len(v.shape) == 2 and v.transpose().shape == target_shape:
                    new_weights[k] = v.transpose()
                else:
                    new_weights[k] = v

        # 3. Add computed buffers (RoPE) if missing
        model_params = dict(tree_flatten(self.parameters()))
        for k, v in model_params.items():
            if k not in new_weights and ("rope" in k):
                new_weights[k] = v

        return new_weights

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path):
        from transformers import AutoTokenizer

        if model.tokenizer is None:
            model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        return model

    def generate(
        self,
        text: str,
        max_tokens: int = 2000,
        min_tokens: int = 2,
        ref_text: Optional[str] = None,
        ref_audio=None,
        prompt_text: Optional[str] = None,
        prompt_audio=None,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming_prefix_len: int = 4,
        warmup_patches: int = 0,
        # CLI compatibility aliases
        cfg_scale: Optional[float] = None,
        ddpm_steps: Optional[int] = None,
        instruct: Optional[str] = None,
        **kwargs,
    ):
        """Generate audio from text with optional voice cloning.

        Modes:
        1. Zero-shot: text only
        2. Voice design: (description)text via instruct param
        3. Continuation: prompt_text + prompt_audio + text
        4. Reference cloning: ref_audio + text
        5. Combined: ref_audio + prompt_text + prompt_audio + text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        # Map CLI aliases — but enforce minimum cfg_value for VoxCPM2
        if cfg_scale is not None:
            cfg_value = max(cfg_scale, 2.0)
        if ddpm_steps is not None:
            inference_timesteps = ddpm_steps

        # Voice design: prepend description as (instruct)text
        if instruct:
            text = f"({instruct}){text}"
            # Voice design gives good context from the start, less warmup needed
            if warmup_patches > 1:
                warmup_patches = 1

        start_time = time.perf_counter()

        scale_emb = (
            self.args.lm_config.scale_emb if self.args.lm_config.use_mup else 1.0
        )
        latent_dim = self.audio_vae.latent_dim

        # Determine mode and build input sequences
        has_ref = ref_audio is not None
        has_prompt = prompt_audio is not None and prompt_text is not None

        if has_ref and has_prompt:
            # Mode 4: Combined reference + continuation
            combined_text = prompt_text + text
            text_ids = self._tokenize(combined_text)
            text_token = mx.array(text_ids + [self.audio_start_token], dtype=mx.int32)
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(ref_audio, padding_mode="right")
            prompt_feat = self._encode_wav(prompt_audio, padding_mode="left")
            prompt_audio_length = prompt_feat.shape[0]

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(
                ref_feat
            )

            text_pad_feat = mx.zeros((text_length, self.patch_size, latent_dim))
            prompt_pad_token = mx.zeros(prompt_audio_length, dtype=mx.int32)

            text_token = mx.concatenate([ref_tokens, text_token, prompt_pad_token])
            audio_feat = mx.concatenate([ref_feats, text_pad_feat, prompt_feat], axis=0)
            text_mask = mx.concatenate(
                [
                    ref_t_mask,
                    mx.ones(text_length, dtype=mx.float32),
                    mx.zeros(prompt_audio_length, dtype=mx.float32),
                ]
            )
            audio_mask = mx.concatenate(
                [
                    ref_a_mask,
                    mx.zeros(text_length, dtype=mx.float32),
                    mx.ones(prompt_audio_length, dtype=mx.float32),
                ]
            )

        elif has_ref:
            # Mode 3: Reference cloning only
            text_ids = self._tokenize(text)
            text_token = mx.array(text_ids + [self.audio_start_token], dtype=mx.int32)
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(ref_audio, padding_mode="right")
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(
                ref_feat
            )

            text_pad_feat = mx.zeros((text_length, self.patch_size, latent_dim))

            text_token = mx.concatenate([ref_tokens, text_token])
            audio_feat = mx.concatenate([ref_feats, text_pad_feat], axis=0)
            text_mask = mx.concatenate(
                [
                    ref_t_mask,
                    mx.ones(text_length, dtype=mx.float32),
                ]
            )
            audio_mask = mx.concatenate(
                [
                    ref_a_mask,
                    mx.zeros(text_length, dtype=mx.float32),
                ]
            )

        elif has_prompt:
            # Mode 2: Continuation only
            combined_text = prompt_text + text
            text_ids = self._tokenize(combined_text)
            text_token = mx.array(text_ids + [self.audio_start_token], dtype=mx.int32)
            text_length = text_token.shape[0]

            prompt_feat = self._encode_wav(prompt_audio, padding_mode="left")
            prompt_audio_length = prompt_feat.shape[0]

            text_pad_feat = mx.zeros((text_length, self.patch_size, latent_dim))
            prompt_pad_token = mx.zeros(prompt_audio_length, dtype=mx.int32)

            text_token = mx.concatenate([text_token, prompt_pad_token])
            audio_feat = mx.concatenate([text_pad_feat, prompt_feat], axis=0)
            text_mask = mx.concatenate(
                [
                    mx.ones(text_length, dtype=mx.float32),
                    mx.zeros(prompt_audio_length, dtype=mx.float32),
                ]
            )
            audio_mask = mx.concatenate(
                [
                    mx.zeros(text_length, dtype=mx.float32),
                    mx.ones(prompt_audio_length, dtype=mx.float32),
                ]
            )

        else:
            # Mode 1: Zero-shot
            text_ids = self._tokenize(text)
            text_token = mx.array(text_ids + [self.audio_start_token], dtype=mx.int32)
            text_length = text_token.shape[0]

            audio_feat = mx.zeros((text_length, self.patch_size, latent_dim))
            text_mask = mx.ones(text_length, dtype=mx.float32)
            audio_mask = mx.zeros(text_length, dtype=mx.float32)

        token_count = len(text_ids)

        # Add batch dimension
        text_token = text_token[None, :]
        audio_feat = audio_feat[None, :, :, :]
        text_mask = text_mask[None, :]
        audio_mask = audio_mask[None, :]

        # Encode audio features
        feat_embed = self.feat_encoder(audio_feat)
        feat_embed = self.enc_to_lm_proj(feat_embed)

        # Text embedding with scale
        text_embed = self.base_lm.embed_tokens(text_token) * scale_emb

        # Combine text and audio embeddings
        combined_embed = (
            text_mask[:, :, None] * text_embed + audio_mask[:, :, None] * feat_embed
        )

        prefix_feat_cond = audio_feat[:, -1, :, :]  # (1, P, D)

        # Initial forward pass
        enc_outputs, lm_cache = self.base_lm(combined_embed)

        # Apply FSQ to audio positions only
        enc_outputs = (
            self.fsq_layer(enc_outputs) * audio_mask[:, :, None]
            + enc_outputs * text_mask[:, :, None]
        )

        lm_hidden = enc_outputs[:, -1, :]

        # V2: fusion_concat_proj for residual input
        residual_input = self.fusion_concat_proj(
            mx.concatenate([enc_outputs, audio_mask[:, :, None] * feat_embed], axis=-1)
        )

        residual_outputs, res_cache = self.residual_lm(residual_input)
        residual_hidden = residual_outputs[:, -1, :]

        # Prepare continuation context for streaming
        has_continuation = audio_mask[0, -1].item() == 1.0
        if has_continuation:
            mask_np = np.array(audio_mask.squeeze(0))
            audio_indices = np.nonzero(mask_np > 0)[0]
            context_len = min(streaming_prefix_len - 1, len(audio_indices))
            last_indices = audio_indices[-context_len:]
            pred_feat_seq = [
                audio_feat[:, int(idx), :, :][:, None, :, :] for idx in last_indices
            ]
        else:
            pred_feat_seq = []

        # In zero-shot/ref modes, warmup patches are generated for conditioning
        # but excluded from decoded audio to avoid onset artifacts.
        warmup_patches = warmup_patches if not has_continuation else 0

        # Generation Loop
        for i in range(max_tokens + warmup_patches):
            # V2: DiT hidden is concatenation (not sum)
            dit_h1 = self.lm_to_dit_proj(lm_hidden)
            dit_h2 = self.res_to_dit_proj(residual_hidden)
            dit_h = mx.concatenate([dit_h1, dit_h2], axis=-1)  # (1, 2*H_dit)

            cond_in = prefix_feat_cond.transpose(0, 2, 1)  # (B, D, P)

            pred_feat = self.feat_decoder.sample(
                mu=dit_h,
                n_timesteps=inference_timesteps,
                patch_size=self.patch_size,
                cond=cond_in,
                cfg_value=cfg_value,
            )

            pred_feat = pred_feat.transpose(0, 2, 1)  # (B, P, D)

            # Only collect patches after warmup
            if i >= warmup_patches:
                pred_feat_seq.append(pred_feat[:, None, :, :])  # (B, 1, P, D)

            curr_embed = self.feat_encoder(pred_feat[:, None, :, :])
            curr_embed = self.enc_to_lm_proj(curr_embed)

            # Stop prediction (only after warmup + min_tokens of real output)
            stop_logits = self.stop_head(nn.silu(self.stop_proj(lm_hidden)))
            stop_flag = mx.argmax(stop_logits, axis=-1).item()
            real_steps = i - warmup_patches
            if real_steps > min_tokens and stop_flag == 1:
                break

            # Autoregressive step
            new_lm_out, lm_cache = self.base_lm(
                inputs_embeds=curr_embed, cache=lm_cache
            )

            lm_hidden = new_lm_out[:, -1, :]
            lm_hidden = self.fsq_layer(lm_hidden)

            # V2: fusion_concat_proj for residual step
            curr_residual_input = self.fusion_concat_proj(
                mx.concatenate([lm_hidden[:, None, :], curr_embed], axis=-1)
            )
            new_res_out, res_cache = self.residual_lm(
                inputs_embeds=curr_residual_input, cache=res_cache
            )
            residual_hidden = new_res_out[:, -1, :]

            prefix_feat_cond = pred_feat

        # Decode
        all_feats = mx.concatenate(pred_feat_seq, axis=1)  # (B, Total, P, D)
        B = all_feats.shape[0]
        all_feats_flat = all_feats.reshape(B, -1, self.feat_dim)  # (B, Total*P, D)

        audio = self.audio_vae.decode(all_feats_flat)
        audio = audio.flatten()

        # Trim continuation prefix if applicable
        if has_continuation:
            patch_len = self.patch_size * self.audio_vae.chunk_size
            trim_samples = patch_len * (streaming_prefix_len - 1)
            # Account for decoder expansion ratio
            decoder_expansion = np.prod(self.args.audio_vae_config.decoder_rates)
            encoder_hop = self.audio_vae.hop_length
            expansion_ratio = decoder_expansion / encoder_hop
            trim_audio_samples = int(trim_samples * expansion_ratio)
            if trim_audio_samples < len(audio):
                audio = audio[trim_audio_samples:]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        samples = audio.shape[0]
        audio_duration_seconds = samples / self.sample_rate

        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
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

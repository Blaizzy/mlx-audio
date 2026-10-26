from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGAN
from mlx_audio.dsp import compute_fbank_kaldi, mel_filters, stft
from mlx_audio.tts.models.base import GenerationResult, adjust_speed
from mlx_audio.utils import load_audio

from .config import ModelConfig
from .emotion import CN_TO_EN, EMOTION_KEYS, QwenEmotion, QwenEmotionConfig
from .s2mel import S2MelConfig, S2MelModel
from .semantic_codec import EnhancedCodec
from .text_frontend import (
    MultilingualTokenizer,
    apply_pronunciation_annotations,
    lang_to_token,
    normalize_case,
    split_text_by_punctuation,
)
from .unifiedvoice import UnifiedVoice, UnifiedVoiceConfig
from .w2vbert import Wav2Vec2BertConfig, Wav2Vec2BertModel
from .w2vbert_features import W2VBertFeatureExtractor, W2VBertFeatureExtractorConfig
from .w2vbert_stats import W2VBertStats


class Model(nn.Module):
    """MLX-native IndexTTS2 (scaffold).

    The full pipeline (w2v-bert -> MaskGCT -> UnifiedVoice -> s2mel diffusion -> BigVGAN)
    is implemented step-by-step. For now this class wires up the public generate()
    interface and emotion-from-text support.
    """

    def __init__(self, config: Union[ModelConfig, dict]):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config

        self.model_type = config.model_type
        self.version = config.version
        self.is_2_5 = config.is_2_5
        self.sample_rate = config.sample_rate

        self._emotion: Optional[QwenEmotion] = None

        self.bigvgan: Optional[BigVGAN] = (
            BigVGAN(config.vocoder) if config.vocoder is not None else None
        )

        self.campplus = None
        if config.campplus is not None:
            # Reuse the existing MLX CAMPPlus implementation.
            from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus

            self.campplus = CAMPPlus(**config.campplus)

        self.semantic_codec = None
        if config.semantic_codec is not None:
            self.semantic_codec = EnhancedCodec(config.semantic_codec)

        # W2V-BERT feature pipeline (semantic encoder to be implemented next)
        self.w2vbert_feature_extractor = W2VBertFeatureExtractor(
            W2VBertFeatureExtractorConfig()
        )
        self.w2vbert_stats = W2VBertStats(dim=1024)
        self.w2vbert = None
        if getattr(config, "w2vbert", None) and isinstance(config.w2vbert, dict):
            # Expect `w2vbert.config` to be the HF config.json dict.
            cfg = config.w2vbert.get("config")
            if isinstance(cfg, dict):
                self.w2vbert = Wav2Vec2BertModel(
                    Wav2Vec2BertConfig(
                        hidden_size=int(cfg["hidden_size"]),
                        num_hidden_layers=int(cfg["num_hidden_layers"]),
                        num_attention_heads=int(cfg["num_attention_heads"]),
                        intermediate_size=int(cfg["intermediate_size"]),
                        feature_projection_input_dim=int(
                            cfg["feature_projection_input_dim"]
                        ),
                        layer_norm_eps=float(cfg.get("layer_norm_eps", 1e-5)),
                        position_embeddings_type=cfg.get(
                            "position_embeddings_type", "relative_key"
                        ),
                        rotary_embedding_base=int(
                            cfg.get("rotary_embedding_base", 10000)
                        ),
                        max_source_positions=int(cfg.get("max_source_positions", 5000)),
                        left_max_position_embeddings=int(
                            cfg.get("left_max_position_embeddings", 64)
                        ),
                        right_max_position_embeddings=int(
                            cfg.get("right_max_position_embeddings", 8)
                        ),
                        conv_depthwise_kernel_size=int(
                            cfg.get("conv_depthwise_kernel_size", 31)
                        ),
                        conformer_conv_dropout=float(
                            cfg.get("conformer_conv_dropout", 0.1)
                        ),
                    )
                )

        self.unifiedvoice = None
        if getattr(config, "unifiedvoice", None) and isinstance(
            config.unifiedvoice, dict
        ):
            tokenizer = None
            bpe_path = None
            if config.model_path is not None:
                if config.tokenizer == "tiktoken":
                    tokenizer = MultilingualTokenizer(config.model_path)
                else:
                    bpe_model = config.unifiedvoice.get("bpe_model", "bpe.model")
                    bpe_path = str((Path(config.model_path) / bpe_model).resolve())
            unify_cfg = dict(config.unifiedvoice)
            unify_cfg["spk_cond_mode"] = config.spk_cond_mode
            self.unifiedvoice = UnifiedVoice(
                UnifiedVoiceConfig.from_dict(unify_cfg),
                bpe_model=bpe_path or config.unifiedvoice.get("bpe_model", "bpe.model"),
                tokenizer=tokenizer,
            )

        self.s2mel = None
        if getattr(config, "s2mel", None) and isinstance(config.s2mel, dict):
            self.s2mel = S2MelModel(
                S2MelConfig.from_dict(
                    config.s2mel, use_gpt_latent=bool(config.use_gpt_latent)
                )
            )

        self.emo_matrix = None
        self.spk_matrix = None
        self.emo_matrix_split = None
        self.spk_matrix_split = None

    @classmethod
    def post_load_hook(cls, model: "Model", model_path) -> "Model":
        """Load optional emotion matrices outside the parameter tree."""
        path = model.config.emotion_matrices
        if path:
            matrices_path = Path(model_path) / path
            if matrices_path.is_file():
                matrices = mx.load(str(matrices_path))
                model.spk_matrix = matrices["spk_matrix"]
                model.emo_matrix = matrices["emo_matrix"]
                bounds = []
                offset = 0
                for size in model.config.emo_num[:-1]:
                    offset += int(size)
                    bounds.append(offset)
                model.emo_matrix_split = list(
                    mx.split(model.emo_matrix, bounds, axis=0)
                )
                model.spk_matrix_split = list(
                    mx.split(model.spk_matrix, bounds, axis=0)
                )
        return model

    def _get_emotion(self) -> QwenEmotion:
        if self._emotion is None:
            model_id = self.config.qwen_emotion_model
            if self.config.model_path:
                local = Path(self.config.model_path) / model_id
                if local.exists():
                    model_id = str(local)
            self._emotion = QwenEmotion(QwenEmotionConfig(model=model_id))
        return self._emotion

    @staticmethod
    def _normalize_emotion_weights(
        values: Optional[list[float] | dict[str, float]],
    ) -> dict[str, float]:
        if values is None:
            return {key: 0.0 for key in EMOTION_KEYS}
        if isinstance(values, dict):
            normalized = {}
            for key, value in values.items():
                key = CN_TO_EN.get(str(key), str(key).lower())
                if key in EMOTION_KEYS:
                    normalized[key] = max(0.0, min(1.2, float(value)))
        else:
            values = list(values)
            if len(values) != len(EMOTION_KEYS):
                raise ValueError("emo_vector must contain exactly 8 values")
            normalized = {
                key: max(0.0, min(1.2, float(value)))
                for key, value in zip(EMOTION_KEYS, values)
            }
        if not any(value > 0 for value in normalized.values()):
            normalized["calm"] = 1.0
        return normalized

    def _compute_emotion_vector(
        self,
        weights: dict[str, float],
        style: mx.array,
        *,
        use_random: bool = False,
    ) -> mx.array:
        """Mix the 2.5 emotion matrix bank using the CAMPPlus style vector."""
        if self.emo_matrix_split is None or self.spk_matrix_split is None:
            raise ValueError(
                "Emotion matrices are not loaded. Convert feat1.pt/feat2.pt with "
                "convert_emotion_matrices.py first."
            )

        style = style.reshape(-1, style.shape[-1]).astype(mx.float32)
        style_norm = mx.sqrt(mx.sum(style * style, axis=-1, keepdims=True))
        selected = []
        for spk_bank, emo_bank, category in zip(
            self.spk_matrix_split, self.emo_matrix_split, EMOTION_KEYS
        ):
            if use_random:
                import random

                index = random.randrange(spk_bank.shape[0])
            else:
                bank = spk_bank.astype(mx.float32)
                bank_norm = mx.sqrt(mx.sum(bank * bank, axis=-1, keepdims=True))
                similarity = (bank @ style[0]) / mx.maximum(
                    bank_norm[:, 0] * style_norm[0, 0], 1e-8
                )
                index = int(mx.argmax(similarity).item())
            selected.append(emo_bank[index])

        matrix = mx.stack(selected, axis=0)
        weight_vector = mx.array(
            [weights.get(category, 0.0) for category in EMOTION_KEYS],
            dtype=mx.float32,
        )
        return (weight_vector[:, None] * matrix).sum(axis=0, keepdims=True)

    @staticmethod
    def _split_weightnorm(key: str, value: mx.array, expected: dict):
        """Rebuild MLX WNConv weight_g/weight_v from stripped torch weights."""
        if not key.endswith(".weight"):
            return None
        base = key[: -len(".weight")]
        g_key, v_key = f"{base}.weight_g", f"{base}.weight_v"
        if g_key not in expected or v_key not in expected:
            return None
        target_shape = expected[g_key].shape
        axes = tuple(index for index, dim in enumerate(target_shape) if dim == 1)
        if not axes:
            axes = tuple(range(1, value.ndim))
        weight_g = mx.sqrt(mx.sum(value * value, axis=axes, keepdims=True))
        # Some converted checkpoints contain fully-zero output channels. A
        # literal 0/0 in WNConv would produce NaN, so keep g=0 but give v a
        # tiny non-zero row, preserving the zero output.
        safe_value = mx.where(weight_g == 0, 1.0, value)
        return {g_key: weight_g, v_key: safe_value}

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Accept both mlx-audio conversion output and existing MLX exports."""
        from mlx.utils import tree_flatten

        expected = dict(tree_flatten(self.parameters()))
        out: dict[str, mx.array] = {}

        for original_key, value in weights.items():
            if original_key in {"emo_matrix", "spk_matrix"}:
                continue
            key = original_key

            # Existing exports keep BigVGAN's upsample ConvTranspose inside a
            # one-element ModuleList and store stripped (weight-norm-free)
            # convolution kernels.
            if key.startswith("bigvgan.ups."):
                parts = key.split(".")
                if len(parts) >= 4 and parts[3] in {"weight", "bias"}:
                    key = ".".join(parts[:3] + ["0"] + parts[3:])

            if key.startswith("s2mel."):
                key = key.replace(".cond_layer.conv.", ".cond_layer.")
                key = key.replace(".t_embedder.linear1.", ".t_embedder.mlp.0.")
                key = key.replace(".t_embedder.linear2.", ".t_embedder.mlp.2.")
                key = key.replace(".t_embedder2.linear1.", ".t_embedder2.mlp.0.")
                key = key.replace(".t_embedder2.linear2.", ".t_embedder2.mlp.2.")
                key = key.replace(".adaLN_modulation.layers.", ".adaLN_modulation.")
                for layer in range(8):
                    key = key.replace(
                        f".wavenet.in_layers.{layer}.conv.",
                        f".wavenet.in_layers.{layer}.",
                    )
                    key = key.replace(
                        f".wavenet.res_skip_layers.{layer}.conv.",
                        f".wavenet.res_skip_layers.{layer}.",
                    )
                key = key.replace("s2mel.gpt_layer.layers.", "s2mel.gpt_layer.")
                if key.startswith("s2mel.gpt_layer.") and key not in expected:
                    continue

            if key.startswith("unifiedvoice.emo_conditioning_encoder."):
                key = key.replace(".embed.conv.", ".embed.conv.0.")
                key = key.replace(".embed.out.", ".embed.out.0.")

            if key.startswith("unifiedvoice.emo_perceiver_encoder."):
                key = key.replace(".norm.weight", ".norm.gamma")
                key = key.replace(".1.w_1.", ".1.0.")
                key = key.replace(".1.w_2.", ".1.2.")
                key = key.replace(".linear_q.", ".to_q.")
                key = key.replace(".linear_out.", ".to_out.")
                if ".linear_k." in key or ".linear_v." in key:
                    prefix = key.rsplit(".", 2)[0]
                    slot = "k" if ".linear_k." in key else "v"
                    out.setdefault("__merge_kv__", {}).setdefault(
                        prefix + ".to_kv.weight", {}
                    )[slot] = value
                    continue

            norm = self._split_weightnorm(key, value, expected)
            if norm is not None:
                out.update(norm)
                continue
            out[key] = value

        merged = out.pop("__merge_kv__", {})
        for key, halves in merged.items():
            if "k" in halves and "v" in halves:
                out[key] = mx.concatenate([halves["k"], halves["v"]], axis=0)

        # Existing exports omit deterministic buffers (anti-alias filters,
        # RoPE caches, position embeddings). Keep the values built by the MLX
        # modules instead of failing a strict load. Do not silently fill real
        # learnable tensors: a missing weight must still fail strict loading.
        derived_suffixes = (".lowpass.filter", ".upsample.filter", ".pos_enc.pe")
        derived_exact = {
            "s2mel.cfm.estimator.final_layer.norm_final.weight",
            "s2mel.cfm.estimator.final_layer.norm_final.bias",
            "s2mel.cfm.estimator.input_pos",
            "s2mel.cfm.estimator.transformer.freqs_cis",
        }
        for key, value in expected.items():
            if key in derived_exact or key.endswith(derived_suffixes):
                out.setdefault(key, value)

        return out

    def _result(self, audio: mx.array, start_time: float) -> GenerationResult:
        samples = int(audio.shape[0])
        audio_duration_seconds = samples / self.sample_rate
        elapsed_time = time.perf_counter() - start_time
        rtf = (audio_duration_seconds / elapsed_time) if elapsed_time > 0 else 0.0

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=0,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={"tokens": 0, "tokens-per-sec": 0},
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _s2mel_ref_mel(self, audio: mx.array) -> mx.array:
        """Match official IndexTTS2 mel_spectrogram settings for s2mel reference."""
        n_fft = 1024
        hop = 256
        n_mels = 80

        pad = int((n_fft - hop) / 2)
        prefix = audio[1 : pad + 1][::-1]
        suffix = audio[-(pad + 1) : -1][::-1]
        y = mx.concatenate([prefix, audio, suffix], axis=0)

        spec = stft(
            y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window="hann",
            center=False,
            pad_mode="reflect",
        ).abs()

        fb = mel_filters(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=0.0,
            f_max=None,
            norm="slaney",
            mel_scale="slaney",
        )
        mel = spec @ fb.T
        mel = mx.log(mx.maximum(mel, 1e-5))
        return mel.T[None, :, :]

    def _align_generated_mel_to_prompt(
        self, mel: mx.array, ref_mel: mx.array
    ) -> mx.array:
        """Match generated mel stats to prompt mel stats.

        This mirrors a common stabilization trick for flow vocoder pipelines when the
        generated mel drifts to an overly low-energy range that leads to near-silent
        waveform output.
        """
        if mel.shape[-1] < 2:
            return mel

        ref_mean = mx.mean(ref_mel, axis=-1, keepdims=True)
        ref_std = mx.std(ref_mel, axis=-1, keepdims=True)
        mel_mean = mx.mean(mel, axis=-1, keepdims=True)
        mel_std = mx.std(mel, axis=-1, keepdims=True)

        eps = 1e-5
        mel_n = (mel - mel_mean) / mx.maximum(mel_std, eps)
        mel_a = mel_n * ref_std + ref_mean

        # Keep values in a reasonable log-mel range for BigVGAN.
        return mx.clip(mel_a, -12.0, 4.0)

    def generate(
        self,
        text: str,
        *,
        ref_audio: Optional[Union[str, mx.array]] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        # Emotion controls
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        emo_vector: Optional[list[float]] = None,
        emo_alpha: float = 1.0,
        use_random: bool = False,
        repetition_penalty: float = 10.0,
        diffusion_steps: int = 40,
        diffusion_cfg_rate: float = 0.7,
        duration_factor: float = 1.0,
        language: Optional[str] = None,
        # Keep signature compatible with mlx_audio.tts.generate
        voice: Optional[str] = None,
        lang_code: str = "en",
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Iterator[GenerationResult]:
        del voice, ref_text, verbose, stream, streaming_interval, kwargs
        lang = (language or lang_code or "en").lower()
        if self.is_2_5 and lang not in {"zh", "en", "ja", "es", "ar"}:
            raise ValueError(
                "IndexTTS-2.5 supports zh, en, ja, es, and ar; "
                f"got language={lang!r}"
            )
        if emo_vector is not None and use_emo_text:
            raise ValueError("emo_vector and use_emo_text are mutually exclusive")

        if ref_audio is None:
            raise ValueError("IndexTTS2 requires ref_audio (speaker prompt audio)")

        if (
            self.unifiedvoice is None
            or self.s2mel is None
            or self.semantic_codec is None
        ):
            raise ValueError(
                "IndexTTS2 is missing required submodules (unifiedvoice/s2mel/semantic_codec). "
                "Make sure you converted all weights into the model folder."
            )

        start_time = time.perf_counter()

        # Load reference audio at 16k (semantic) and 22.05k (mel/vocoder)
        ref_16k = load_audio(ref_audio, sample_rate=16000)
        ref_22k = load_audio(ref_audio, sample_rate=self.sample_rate)

        # W2V-BERT features -> hidden states
        input_features, attn_mask = self.w2vbert_feature_extractor(ref_16k)
        last, hstates = self.w2vbert(
            input_features, attention_mask=attn_mask, output_hidden_states=True
        )
        del last
        hs17 = self.w2vbert_stats(hstates[17])

        # Semantic codec prompt codes + embeddings
        ref_codes, ref_quant = self.semantic_codec.quantize(hs17)
        # ref_mel: (B, 80, T)
        ref_mel = self._s2mel_ref_mel(ref_22k)
        ref_mel_len = mx.array([ref_mel.shape[-1]], dtype=mx.int32)

        # Style from CAMPPlus (Kaldi fbank)
        fb = compute_fbank_kaldi(ref_16k, sample_rate=16000, num_mels=80, dither=0.0)
        fb = fb - mx.mean(fb, axis=0, keepdims=True)
        style = self.campplus(fb[None, :, :])

        prompt_condition, _, _, _, _ = self.s2mel.length_regulator(
            hs17 if self.is_2_5 else ref_quant, ylens=ref_mel_len
        )

        # Emotion vector
        emotion_input = emo_vector
        if use_emo_text:
            text_for_emo = emo_text if emo_text is not None else text
            _, emotion_input = self._get_emotion().infer(text_for_emo)
        emotion_weights = None
        emotion_override = None
        emotion_weight_sum = 0.0
        if emotion_input is not None:
            emotion_weights = self._normalize_emotion_weights(emotion_input)
            scale = max(0.0, min(1.0, float(emo_alpha)))
            if scale != 1.0:
                emotion_weights = {
                    key: value * scale for key, value in emotion_weights.items()
                }
            emotion_override = self._compute_emotion_vector(
                emotion_weights, style, use_random=use_random
            )
            emotion_weight_sum = sum(emotion_weights.values())

        if self.is_2_5:
            text = normalize_case(apply_pronunciation_annotations(text), lang)

        # Text -> semantic codes via UnifiedVoice
        text_tokens = self.unifiedvoice.encode_text(text, lang=lang)

        # Use reference hidden states as speaker/emotion condition.
        # UnifiedVoice expects (B, T, 1024)
        spk_cond = hs17
        emo_cond = hs17

        codes, speech_latent = self.unifiedvoice.inference_speech(
            spk_cond,
            text_tokens,
            emo_cond,
            alpha=float(emo_alpha),
            campplus_embedding=style if self.is_2_5 else None,
            lang_id=lang_to_token(lang) if self.is_2_5 else None,
            emo_vector_override=emotion_override,
            emo_weight_sum=emotion_weight_sum,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            max_generate_length=1500,
            repetition_penalty=float(repetition_penalty),
        )

        # Strip stop token if present
        if (
            codes.shape[1] > 0
            and int(codes[0, -1].item()) == self.unifiedvoice.cfg.stop_mel_token
        ):
            codes = codes[:, :-1]

        if self.is_2_5:
            # IndexTTS-2.5 decodes semantic codes directly; it no longer uses
            # the GPT latent projection from IndexTTS2.
            semantic = self.semantic_codec.decode(codes)
            seconds_scale = float(duration_factor)
            if speed and speed != 1.0:
                seconds_scale = seconds_scale / float(speed)
            target_lengths = mx.array(
                [max(1, int(semantic.shape[1] * 1.72 * seconds_scale))],
                dtype=mx.int32,
            )
            cond, _, _, _, _ = self.s2mel.length_regulator(
                semantic, ylens=target_lengths
            )
        else:
            # IndexTTS2 uses the GPT latent plus semantic codebook embeddings.
            emo_vec_lat = self.unifiedvoice.get_emovec(
                emo_cond, mx.array([emo_cond.shape[1]], dtype=mx.int32)
            )
            mel_lat = self.unifiedvoice.forward_latent(
                speech_latent, text_tokens, codes, emo_vec_lat
            )
            gpt_lat = self.s2mel.project_gpt_latent(mel_lat)

            semantic = self.semantic_codec.vq2emb(codes[None, :, :])
            semantic = semantic + gpt_lat
            code_lens = mx.array([semantic.shape[1]], dtype=mx.int32)
            target_lengths = (code_lens.astype(mx.float32) * 1.72).astype(mx.int32)
            cond, _, _, _, _ = self.s2mel.length_regulator(
                semantic, ylens=target_lengths
            )
        cat_condition = mx.concatenate([prompt_condition, cond], axis=1)

        x_lens = mx.array([cat_condition.shape[1]], dtype=mx.int32)
        mel_all = self.s2mel.cfm.inference(
            cat_condition,
            x_lens,
            ref_mel,
            style,
            None,
            n_timesteps=int(diffusion_steps),
            inference_cfg_rate=float(diffusion_cfg_rate),
        )
        mel = mel_all[:, :, ref_mel.shape[-1] :]

        if mel.shape[-1] == 0:
            raise ValueError(
                "IndexTTS2 generated empty mel sequence (no semantic frames after stop token)."
            )

        # Auto-correct low-energy mel drift before vocoding.
        ref_energy = mx.mean(mx.std(ref_mel, axis=-1))
        mel_energy = mx.mean(mx.std(mel, axis=-1))
        ref_level = mx.mean(mx.mean(ref_mel, axis=-1))
        mel_level = mx.mean(mx.mean(mel, axis=-1))
        if float(mel_energy.item()) < float((ref_energy * 0.75).item()) or float(
            mel_level.item()
        ) < float((ref_level - 1.5).item()):
            mel = self._align_generated_mel_to_prompt(mel, ref_mel)

        audio = self.bigvgan(mel)
        audio = audio.reshape(-1).astype(mx.float32)
        # IndexTTS-2.5 controls tempo in latent space via duration_factor, so
        # do not time-stretch the waveform a second time.
        if not self.is_2_5 and speed and speed != 1.0:
            audio = adjust_speed(audio, speed)

        yield self._result(audio, start_time)

    def vocode(self, mel: mx.array) -> mx.array:
        """Run the BigVGAN vocoder.

        Args:
            mel: (B, n_mels, T) float mel-spectrogram.
        """
        if self.bigvgan is None:
            raise ValueError("BigVGAN vocoder is not configured/loaded")
        audio = self.bigvgan(mel)
        return audio

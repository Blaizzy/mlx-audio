from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGAN
from mlx_audio.tts.indextts2.emotion import QwenEmotion, QwenEmotionConfig
from mlx_audio.tts.models.base import GenerationResult, adjust_speed
from mlx_audio.utils import load_audio
from mlx_audio.dsp import compute_fbank_kaldi, mel_filters, stft

from .config import ModelConfig
from .semantic_codec import RepCodec
from .w2vbert_features import W2VBertFeatureExtractor, W2VBertFeatureExtractorConfig
from .w2vbert_stats import W2VBertStats
from .w2vbert import Wav2Vec2BertConfig, Wav2Vec2BertModel
from .unifiedvoice import UnifiedVoice, UnifiedVoiceConfig
from .s2mel import S2MelConfig, S2MelModel


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
            self.semantic_codec = RepCodec(config.semantic_codec)

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
                        feature_projection_input_dim=int(cfg["feature_projection_input_dim"]),
                        layer_norm_eps=float(cfg.get("layer_norm_eps", 1e-5)),
                        position_embeddings_type=cfg.get(
                            "position_embeddings_type", "relative_key"
                        ),
                        rotary_embedding_base=int(cfg.get("rotary_embedding_base", 10000)),
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
        if getattr(config, "unifiedvoice", None) and isinstance(config.unifiedvoice, dict):
            bpe_model = config.unifiedvoice.get("bpe_model", "bpe.model")
            bpe_path = None
            if config.model_path is not None:
                bpe_path = str((Path(config.model_path) / bpe_model).resolve())
            self.unifiedvoice = UnifiedVoice(
                UnifiedVoiceConfig.from_dict(config.unifiedvoice),
                bpe_model=bpe_path or bpe_model,
            )

        self.s2mel = None
        if getattr(config, "s2mel", None) and isinstance(config.s2mel, dict):
            self.s2mel = S2MelModel(S2MelConfig.from_dict(config.s2mel))

        # TODO: instantiate submodules once implemented and weights are available.

    def _get_emotion(self) -> QwenEmotion:
        if self._emotion is None:
            self._emotion = QwenEmotion(
                QwenEmotionConfig(model=self.config.qwen_emotion_model)
            )
        return self._emotion

    def _result(self, audio: mx.array, start_time: float) -> GenerationResult:
        samples = int(audio.shape[0])
        audio_duration_seconds = samples / self.sample_rate
        elapsed_time = time.perf_counter() - start_time
        rtf = (audio_duration_seconds / elapsed_time) if elapsed_time > 0 else 0.0

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = (
            f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"
        )

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
                "samples-per-sec": (round(samples / elapsed_time, 2) if elapsed_time > 0 else 0),
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

    def _align_generated_mel_to_prompt(self, mel: mx.array, ref_mel: mx.array) -> mx.array:
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
        repetition_penalty: float = 10.0,
        diffusion_steps: int = 40,
        diffusion_cfg_rate: float = 0.7,
        # Keep signature compatible with mlx_audio.tts.generate
        voice: Optional[str] = None,
        lang_code: str = "en",
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Iterator[GenerationResult]:
        del voice, lang_code, ref_text, verbose, stream, streaming_interval, kwargs

        if ref_audio is None:
            raise ValueError("IndexTTS2 requires ref_audio (speaker prompt audio)")

        if self.unifiedvoice is None or self.s2mel is None or self.semantic_codec is None:
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
            ref_quant, ylens=ref_mel_len
        )

        # Emotion vector
        emo_vec = None
        if use_emo_text and emo_vector is None:
            text_for_emo = emo_text if emo_text is not None else text
            _, emo_vector = self._get_emotion().infer(text_for_emo)
        if emo_vector is not None:
            emo_vec = mx.array(emo_vector, dtype=mx.float32)[None, :]

        # Text -> semantic codes via UnifiedVoice
        text_tokens = self.unifiedvoice.encode_text(text)

        # Use reference hidden states as speaker/emotion condition.
        # UnifiedVoice expects (B, T, 1024)
        spk_cond = hs17
        emo_cond = hs17

        codes, speech_latent = self.unifiedvoice.inference_speech(
            spk_cond,
            text_tokens,
            emo_cond,
            alpha=float(emo_alpha),
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            max_generate_length=1500,
            repetition_penalty=float(repetition_penalty),
        )

        # Strip stop token if present
        if codes.shape[1] > 0 and int(codes[0, -1].item()) == self.unifiedvoice.cfg.stop_mel_token:
            codes = codes[:, :-1]

        # Get GPT latent and project to 1024
        emo_vec_lat = self.unifiedvoice.get_emovec(emo_cond, mx.array([emo_cond.shape[1]], dtype=mx.int32))
        mel_lat = self.unifiedvoice.forward_latent(speech_latent, text_tokens, codes, emo_vec_lat)
        gpt_lat = self.s2mel.project_gpt_latent(mel_lat)

        # Semantic embedding of inferred codes
        S_infer = self.semantic_codec.vq2emb(codes[None, :, :])
        S_infer = S_infer + gpt_lat

        code_lens = mx.array([S_infer.shape[1]], dtype=mx.int32)
        target_lengths = (code_lens.astype(mx.float32) * 1.72).astype(mx.int32)

        cond, _, _, _, _ = self.s2mel.length_regulator(S_infer, ylens=target_lengths)
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
        if speed and speed != 1.0:
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

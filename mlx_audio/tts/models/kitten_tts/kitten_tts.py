import math
import time
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelArgs, GenerationResult
from .istftnet import AdainResBlk1d, ConvWeighted, Generator
from .modules import AlbertEmbeddings, AlbertModelArgs, ProsodyPredictor, TextEncoder
from .preprocess import TextPreprocessor
from .quant import maybe_fake_quant


def basic_english_tokenize(text: str) -> List[str]:
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    return re.findall(r"\w+|[^\w\s]", text)


def ensure_punctuation(text: str) -> str:
    """Ensure text ends with punctuation. If not, add a comma."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?,;:":
        text = text + ","
    return text


def chunk_text(text: str, max_len: int = 400) -> List[str]:
    """Split text into chunks for processing long texts."""
    import re

    sentences = re.split(r"[.!?]+", text)
    chunks = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_len:
            chunks.append(ensure_punctuation(sentence))
        else:
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_len:
                    temp_chunk += " " + word if temp_chunk else word
                else:
                    if temp_chunk:
                        chunks.append(ensure_punctuation(temp_chunk.strip()))
                    temp_chunk = word
            if temp_chunk:
                chunks.append(ensure_punctuation(temp_chunk.strip()))

    return chunks


class TextCleaner:
    def __init__(self):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

        self.word_index_dictionary = {s: i for i, s in enumerate(symbols)}

    def __call__(self, text: str) -> List[int]:
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


@dataclass
class ModelConfig(BaseModelArgs):
    # Core dims
    hidden_dim: int
    max_conv_dim: int
    max_dur: int
    n_layer: int
    n_mels: int
    n_token: int
    style_dim: int
    text_encoder_kernel_size: int
    asr_res_dim: int

    # Sub-configs
    plbert: dict
    istftnet: dict

    # Runtime settings
    sample_rate: int = 24000
    decoder_out_dim: Optional[int] = None
    voices_path: str = "voices.npz"
    speed_priors: Optional[dict] = None
    voice_aliases: Optional[dict] = None
    model_path: Optional[str] = None
    activation_quant_modules: Optional[List[str]] = None


class KittenDecoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        max_conv_dim: int,
        decoder_out_dim: int,
        asr_res_dim: int,
        istftnet: dict,
    ):
        super().__init__()
        self.encode = AdainResBlk1d(
            dim_in + 2, max_conv_dim, style_dim, conv_type=mx.conv1d
        )
        self.decode = []
        for _ in range(3):
            self.decode.append(
                AdainResBlk1d(
                    max_conv_dim + 2 + asr_res_dim,
                    max_conv_dim,
                    style_dim,
                    conv_type=mx.conv1d,
                )
            )
        self.decode.append(
            AdainResBlk1d(
                max_conv_dim + 2 + asr_res_dim,
                decoder_out_dim,
                style_dim,
                upsample=True,
                conv_type=mx.conv1d,
            )
        )
        self.F0_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.N_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1, groups=1)
        self.asr_res = [ConvWeighted(dim_in, asr_res_dim, kernel_size=1, padding=0)]
        self.generator = Generator(style_dim=style_dim, **istftnet)

    def __call__(self, asr: mx.array, f0: mx.array, n: mx.array, s: mx.array):
        s = mx.array(s)
        f0_curve = f0
        f0 = self.F0_conv(f0[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        n = self.N_conv(n[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        x = mx.concatenate([asr, f0, n], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        res = True
        for block in self.decode:
            if res:
                x = mx.concatenate([x, asr_res, f0, n], axis=1)
            x = block(x, s)
            if hasattr(block, "upsample_type") and block.upsample_type != "none":
                res = False
        x = self.generator(x, s, f0_curve)
        return x


class KittenAlbertSelfAttention(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.reshape(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, hidden_states, attention_mask=None):
        hidden_states_q = maybe_fake_quant(
            hidden_states, getattr(self, "activation_quant", False)
        )
        mixed_query_layer = self.query(hidden_states_q)
        mixed_key_layer = self.key(hidden_states_q)
        mixed_value_layer = self.value(hidden_states_q)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = mx.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        # ONNX scales QK^T by 1/sqrt(head_dim) (implemented via sqrt on both Q/K).
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = mx.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        context_layer = maybe_fake_quant(
            context_layer, getattr(self, "activation_quant", False)
        )
        context_layer = self.dense(context_layer)
        context_layer = self.LayerNorm(context_layer + hidden_states)
        return context_layer


class KittenAlbertLayer(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.attention = KittenAlbertSelfAttention(config)
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        # Match ONNX tanh-based GELU approximation.
        self._gelu_const = 0.7978846
        self._gelu_const_2 = 0.044715

    def __call__(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ff_chunk(attention_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        return hidden_states

    def ff_chunk(self, attention_output: mx.array) -> mx.array:
        attention_output = maybe_fake_quant(
            attention_output, getattr(self, "activation_quant", False)
        )
        ffn_output = self.ffn(attention_output)
        x = ffn_output
        ffn_output = (
            0.5
            * x
            * (1.0 + mx.tanh(self._gelu_const * (x + self._gelu_const_2 * (x**3))))
        )
        ffn_output = maybe_fake_quant(
            ffn_output, getattr(self, "activation_quant", False)
        )
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class KittenAlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.albert_layers = [
            KittenAlbertLayer(config) for _ in range(config.inner_group_num)
        ]

    def __call__(self, hidden_states, attention_mask=None):
        for layer_module in self.albert_layers:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class KittenAlbertEncoder(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size, config.hidden_size
        )
        self.albert_layer_groups = [
            KittenAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)
        ]

    def __call__(self, hidden_states, attention_mask=None):
        hidden_states = maybe_fake_quant(
            hidden_states, getattr(self, "activation_quant", False)
        )
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        for i in range(self.config.num_hidden_layers):
            group_idx = int(
                i / (self.config.num_hidden_layers / self.config.num_hidden_groups)
            )
            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states, attention_mask
            )
            hidden_states = layer_group_output
        return hidden_states


class KittenAlbert(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = KittenAlbertEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = nn.tanh(self.pooler(sequence_output[:, 0]))

        return sequence_output, pooled_output


class KittenProsodyPredictor(ProsodyPredictor):
    def __call__(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        en = mx.matmul(mx.transpose(d), alignment)
        return mx.squeeze(duration, axis=-1), en


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.speed_priors = config.speed_priors or {}
        self.voice_aliases = config.voice_aliases or {}

        self.bert = KittenAlbert(
            AlbertModelArgs(vocab_size=config.n_token, **config.plbert)
        )
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config.hidden_dim)
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = KittenProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_dim,
            nlayers=config.n_layer,
            max_dur=config.max_dur,
            dropout=0.0,
        )
        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )
        self.decoder = KittenDecoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            max_conv_dim=config.max_conv_dim,
            decoder_out_dim=config.decoder_out_dim or config.max_conv_dim,
            asr_res_dim=config.asr_res_dim,
            istftnet=config.istftnet,
        )

        self._enable_activation_quant(config.activation_quant_modules or [])

        self._text_cleaner = TextCleaner()
        self._preprocessor = TextPreprocessor()
        self._phonemizer = None
        self.voices: Dict[str, np.ndarray] = {}

    def _enable_activation_quant(self, module_names: List[str]):
        if not module_names:
            return
        module_set = set(module_names)
        for name, module in self.named_modules():
            if not name:
                continue
            if any(q == name or q.startswith(f"{name}.") for q in module_set):
                setattr(module, "activation_quant", True)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Compatibility: older exports used dot-form Snake alpha names (alpha1.0 / alpha2.0).
        if any(".alpha1." in k or ".alpha2." in k for k in weights.keys()) and not any(
            "alpha1_" in k or "alpha2_" in k for k in weights.keys()
        ):
            remapped = {}
            for k, v in weights.items():
                nk = k.replace(".alpha1.", ".alpha1_").replace(".alpha2.", ".alpha2_")
                remapped[nk] = v
            return remapped
        return weights

    @classmethod
    def post_load_hook(cls, model, model_path: Path):
        voices_path = Path(model_path) / model.config.voices_path
        if voices_path.exists():
            model._load_voices(voices_path)
        return model

    def _get_phonemizer(self):
        if self._phonemizer is None:
            try:
                from phonemizer.backend import EspeakBackend
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "phonemizer is required for KittenTTS. Install with `pip install phonemizer`."
                ) from exc

            self._phonemizer = EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True
            )
        return self._phonemizer

    def _load_voices(self, path: Path):
        voices = np.load(path)
        self.voices = {k: voices[k].astype(np.float32) for k in voices.files}

    def _prepare_inputs(
        self, text: str, voice: str, speed: float, clean_text: bool
    ) -> tuple[mx.array, mx.array, float]:
        if voice in self.voice_aliases:
            voice = self.voice_aliases[voice]

        if voice not in self.voices:
            raise ValueError(
                f"Voice '{voice}' not available. Choose from: {sorted(self.voices.keys())}"
            )

        if voice in self.speed_priors:
            speed = speed * self.speed_priors[voice]

        if clean_text:
            text = self._preprocessor(text)

        phonemizer = self._get_phonemizer()
        phonemes_list = phonemizer.phonemize([text])
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = " ".join(phonemes)
        tokens = self._text_cleaner(phonemes)
        tokens.insert(0, 0)
        tokens.append(0)

        input_ids = mx.array([tokens], dtype=mx.int32)
        ref_id = min(len(text), self.voices[voice].shape[0] - 1)
        ref_s = mx.array(self.voices[voice][ref_id : ref_id + 1])

        return input_ids, ref_s, speed

    @dataclass
    class Output:
        audio: mx.array
        pred_dur: Optional[mx.array] = None

    def __call__(
        self,
        input_ids: mx.array,
        ref_s: mx.array,
        speed: Number = 1.0,
        return_output: bool = False,
    ) -> Union["Model.Output", mx.array]:
        input_lengths = mx.array([input_ids.shape[-1]])
        text_mask = mx.arange(int(input_lengths.max()))[None, ...]
        text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(
            input_lengths.dtype
        )
        text_mask = text_mask + 1 > input_lengths[:, None]
        bert_out, _ = self.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))
        bert_out = maybe_fake_quant(
            bert_out, getattr(self.bert_encoder, "activation_quant", False)
        )
        d_en = self.bert_encoder(bert_out).transpose(0, 2, 1)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = mx.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=None).astype(mx.int32)[0]
        indices = mx.concatenate(
            [mx.repeat(mx.array(i), int(n)) for i, n in enumerate(pred_dur)]
        )
        pred_aln_trg = mx.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, mx.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg[None, :]
        en = d.transpose(0, 2, 1) @ pred_aln_trg
        f0_pred, n_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        audio = self.decoder(asr, f0_pred, n_pred, ref_s[:, :128])[0]
        mx.eval(audio, pred_dur)
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def generate(
        self,
        text: str,
        voice: str = "expr-voice-5-m",
        speed: float = 1.0,
        clean_text: bool = True,
        chunk_size: int = 400,
        crossfade_ms: int = 20,
        fade_out_ms: int = 200,
        tail_silence_ms: int = 200,
        **kwargs,
    ):
        if not self.voices:
            raise RuntimeError("Voices are not loaded. Ensure voices.npz is present.")

        text = text.strip()
        if not text:
            return

        # Avoid chunking for short inputs to prevent boundary artifacts.
        if len(text) <= chunk_size:
            chunks = [ensure_punctuation(text)]
        else:
            chunks = chunk_text(text, max_len=chunk_size)

        crossfade_samples = int(self.sample_rate * max(crossfade_ms, 0) / 1000)
        fade_out_samples = int(self.sample_rate * max(fade_out_ms, 0) / 1000)
        tail_silence_samples = int(self.sample_rate * max(tail_silence_ms, 0) / 1000)

        def _apply_tail(audio: mx.array):
            if audio is None:
                return audio
            # Trim spurious trailing spurts after a brief silence (common end artifact).
            try:
                hop = max(1, int(self.sample_rate * 0.01))  # 10 ms
                tail_len = min(audio.shape[0], int(self.sample_rate * 1.0))  # 1s
                if tail_len > hop * 3:
                    tail_np = np.array(audio[-tail_len:])
                    n_frames = tail_np.shape[0] // hop
                    if n_frames > 3:
                        tail_np = tail_np[-n_frames * hop :].reshape(n_frames, hop)
                        rms = np.sqrt(np.mean(tail_np * tail_np, axis=1))
                        if rms.max() > 1e-6:
                            rms_norm = rms / (rms.max() + 1e-9)
                            silence_rel = 0.1
                            resume_rel = 0.2
                            min_silence_frames = max(3, int(0.03 / 0.01))  # 30 ms

                            run = 0
                            for i in range(len(rms_norm) - 1, -1, -1):
                                if rms_norm[i] < silence_rel:
                                    run += 1
                                else:
                                    if run >= min_silence_frames:
                                        low_end = i + run
                                        # If audio resumes after the low-energy span, trim it.
                                        if np.any(rms_norm[low_end + 1 :] > resume_rel):
                                            cut_frame = low_end + 1
                                            cut_idx = (
                                                audio.shape[0]
                                                - tail_len
                                                + cut_frame * hop
                                            )
                                            audio = audio[:cut_idx]
                                        break
                                    run = 0
            except Exception:
                pass

            if fade_out_samples > 0:
                # Dynamic fade: find last energetic frame near the end and fade from there.
                # This helps avoid audible truncation if the model cuts mid-phoneme.
                hop = max(1, int(self.sample_rate * 0.01))  # 10 ms
                tail_len = min(
                    audio.shape[0],
                    int(self.sample_rate * max(fade_out_ms, 400) / 1000),
                )
                fade_start = max(0, audio.shape[0] - fade_out_samples)
                if tail_len > hop:
                    tail = np.array(audio[-tail_len:])
                    n_frames = tail.shape[0] // hop
                    if n_frames > 0:
                        tail = tail[-n_frames * hop :].reshape(n_frames, hop)
                        rms = np.sqrt(np.mean(tail * tail, axis=1))
                        if rms.size > 0:
                            thr = max(rms.max() * 0.05, 1e-4)
                            idxs = np.where(rms > thr)[0]
                            if len(idxs):
                                last_idx = idxs[-1]
                                fade_start = audio.shape[0] - tail_len + last_idx * hop
                fade_len = audio.shape[0] - fade_start
                if fade_len < fade_out_samples:
                    fade_start = max(0, audio.shape[0] - fade_out_samples)
                    fade_len = audio.shape[0] - fade_start
                if fade_len > 0:
                    fade_start_i = int(fade_start)
                    fade_len_i = int(fade_len)
                    t = mx.arange(fade_len_i, dtype=audio.dtype) / fade_len_i
                    fade_curve = 1.0 - t
                    audio = mx.concatenate(
                        [audio[:fade_start_i], audio[fade_start_i:] * fade_curve],
                        axis=0,
                    )
            if tail_silence_samples > 0:
                audio = mx.concatenate(
                    [audio, mx.zeros((tail_silence_samples,), dtype=audio.dtype)],
                    axis=0,
                )
            return audio

        def _crossfade(prev_audio: mx.array, next_audio: mx.array):
            if crossfade_samples <= 0:
                return prev_audio, next_audio
            fade = min(crossfade_samples, prev_audio.shape[0], next_audio.shape[0])
            if fade <= 0:
                return prev_audio, next_audio
            t = mx.arange(fade, dtype=prev_audio.dtype) / fade
            fade_out = 1.0 - t
            fade_in = t
            blended = prev_audio[-fade:] * fade_out + next_audio[:fade] * fade_in
            out = mx.concatenate([prev_audio[:-fade], blended], axis=0)
            remainder = next_audio[fade:]
            return out, remainder

        start_time = time.time()
        pending_audio = None
        pending_tokens = 0
        emit_idx = 0

        for text_chunk in chunks:
            input_ids, ref_s, speed = self._prepare_inputs(
                text_chunk, voice, speed, clean_text
            )
            audio = self(input_ids, ref_s, speed).reshape(-1)

            if pending_audio is None:
                pending_audio = audio
                pending_tokens = input_ids.shape[-1]
                continue

            out_audio, pending_audio = _crossfade(pending_audio, audio)
            token_count = pending_tokens
            pending_tokens = input_ids.shape[-1]

            now = time.time()
            segment_time = now - start_time
            start_time = now

            samples = out_audio.shape[0] if out_audio is not None else 0
            assert samples > 0, "No audio generated"

            sample_rate = self.sample_rate
            audio_duration_seconds = samples / sample_rate

            rtf = (
                segment_time / audio_duration_seconds
                if audio_duration_seconds > 0
                else 0
            )
            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=out_audio,
                samples=samples,
                sample_rate=sample_rate,
                segment_idx=emit_idx,
                token_count=token_count,
                audio_duration=duration_str,
                real_time_factor=round(rtf, 2),
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": (
                        round(token_count / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                processing_time_seconds=segment_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            emit_idx += 1
            mx.clear_cache()

        if pending_audio is not None:
            pending_audio = _apply_tail(pending_audio)
            now = time.time()
            segment_time = now - start_time

            samples = pending_audio.shape[0] if pending_audio is not None else 0
            assert samples > 0, "No audio generated"

            sample_rate = self.sample_rate
            audio_duration_seconds = samples / sample_rate
            rtf = (
                segment_time / audio_duration_seconds
                if audio_duration_seconds > 0
                else 0
            )
            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=pending_audio,
                samples=samples,
                sample_rate=sample_rate,
                segment_idx=emit_idx,
                token_count=pending_tokens,
                audio_duration=duration_str,
                real_time_factor=round(rtf, 2),
                prompt={
                    "tokens": pending_tokens,
                    "tokens-per-sec": (
                        round(pending_tokens / segment_time, 2)
                        if segment_time > 0
                        else 0
                    ),
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": (
                        round(samples / segment_time, 2) if segment_time > 0 else 0
                    ),
                },
                processing_time_seconds=segment_time,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            mx.clear_cache()

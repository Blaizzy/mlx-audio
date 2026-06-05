"""Nemotron 3.5 ASR (FastConformer prompted RNN-T) — MLX, offline path.

Pipeline: log-mel -> FastConformer encoder -> language-prompt fusion -> RNN-T
greedy. Streaming (cache-aware) is added in a later phase.
"""

from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.nemotron_asr.audio import PreprocessArgs, log_mel_spectrogram
from mlx_audio.stt.models.nemotron_asr.config import (
    ModelConfig,
    parse_encoder,
    parse_preprocess,
)
from mlx_audio.stt.models.nemotron_asr.decoder import (
    JointArgs,
    JointNetwork,
    JointNetworkArgs,
    PredictArgs,
    PredictNetwork,
    PredictNetworkArgs,
)
from mlx_audio.stt.models.nemotron_asr.encoder import Conformer, EncoderArgs
from mlx_audio.stt.models.nemotron_asr.tokenizer import VocabTokenizer, strip_lang_tag
from mlx_audio.stt.utils import load_audio


class _Featurizer(nn.Module):
    """Holds the .nemo mel filterbank + window so they load as weights
    (keys preprocessor.featurizer.{fb,window}) and stay bit-identical."""

    def __init__(self, n_mels: int, n_fft: int, win_length: int):
        super().__init__()
        self.fb = mx.zeros((1, n_mels, n_fft // 2 + 1))
        self.window = mx.zeros((win_length,))


class _Preprocessor(nn.Module):
    def __init__(self, n_mels: int, n_fft: int, win_length: int):
        super().__init__()
        self.featurizer = _Featurizer(n_mels, n_fft, win_length)


class STTOutput:
    def __init__(self, text: str, language: Optional[str] = None, tokens=None):
        self.text = text
        self.language = language
        self.tokens = tokens or []

    def __repr__(self):
        return f"STTOutput(text={self.text!r}, language={self.language!r})"


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, ModelConfig):
            config = config._config

        enc = config.get("encoder", {})
        joint = config.get("joint", {})
        vocab = config.get("vocabulary") or []
        vocab_size = int(joint.get("num_classes", len(vocab) or 13087))
        md = config.get("model_defaults", {})
        decoding = config.get("decoding", {}) or {}
        greedy = decoding.get("greedy", {}) if isinstance(decoding, dict) else {}
        pred_hidden = int(
            config.get("decoder", {}).get("prednet", {}).get("pred_hidden", 640)
        )
        joint_hidden = int(joint.get("jointnet", {}).get("joint_hidden", 640))

        self.preprocess_args = parse_preprocess(config)
        self.encoder_args = parse_encoder(config)
        d = self.encoder_args.d_model
        self.vocab_size = vocab_size
        self.blank_id = vocab_size  # blank index (joint output dim = vocab+1)
        self.num_prompts = int(md.get("num_prompts", config.get("num_prompts", 128)))
        self.prompt_dictionary = dict(md.get("prompt_dictionary", {}) or {})
        self.max_symbols = int(greedy.get("max_symbols", 10)) if greedy else 10
        self.tokenizer = VocabTokenizer(vocab)

        self.preprocessor = _Preprocessor(
            self.preprocess_args.features,
            self.preprocess_args.n_fft,
            self.preprocess_args.win_length,
        )
        self.encoder = Conformer(self.encoder_args)
        # prompt_kernel list -> weight keys prompt_kernel.{0,2}.*
        self.prompt_kernel = [
            nn.Linear(d + self.num_prompts, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        ]
        self.decoder = PredictNetwork(
            PredictArgs(
                blank_as_pad=True,
                vocab_size=vocab_size,
                prednet=PredictNetworkArgs(pred_hidden=pred_hidden, pred_rnn_layers=2),
            )
        )
        self.joint = JointNetwork(
            JointArgs(
                num_classes=vocab_size,
                vocabulary=vocab,
                jointnet=JointNetworkArgs(
                    joint_hidden=joint_hidden,
                    activation="relu",
                    encoder_hidden=d,
                    pred_hidden=pred_hidden,
                ),
            )
        )
        self.train(False)  # eval mode

    @classmethod
    def from_config(cls, config) -> "Model":
        return cls(config)

    def sanitize(self, weights: dict) -> dict:
        # converter already matches MLX layout/keys; nothing to remap.
        return weights

    # ---- prompt fusion ----
    def _fuse(self, encoded: mx.array, prompt_id: int) -> mx.array:
        B, T, _ = encoded.shape
        onehot = mx.zeros((B, T, self.num_prompts), dtype=encoded.dtype)
        onehot[:, :, prompt_id] = 1.0
        x = mx.concatenate([encoded, onehot], axis=-1)
        for layer in self.prompt_kernel:
            x = layer(x)
        return x

    def _resolve_prompt_id(self, target_lang: str) -> int:
        if target_lang in self.prompt_dictionary:
            return int(self.prompt_dictionary[target_lang])
        return int(self.prompt_dictionary.get("auto", 0))

    # ---- greedy RNN-T ----
    def _greedy(self, enc_post: mx.array) -> list[int]:
        T = enc_post.shape[1]
        last_token = self.blank_id
        hidden = None
        hyp: list[int] = []
        t = 0
        new_sym = 0
        while t < T:
            feat = enc_post[:, t : t + 1]
            cur = (
                mx.array([[last_token]], dtype=mx.int32)
                if last_token != self.blank_id
                else None
            )
            dec_out, (h, c) = self.decoder(cur, hidden)
            dec_out = dec_out.astype(feat.dtype)
            logits = self.joint(feat, dec_out)
            pred = int(mx.argmax(logits))
            if pred != self.blank_id:
                last_token = pred
                hidden = (h.astype(feat.dtype), c.astype(feat.dtype))
                hyp.append(pred)
                new_sym += 1
                if self.max_symbols is not None and new_sym >= self.max_symbols:
                    t += 1
                    new_sym = 0
            else:
                t += 1
                new_sym = 0
        return hyp

    # ---- public API ----
    def encode(self, mel: mx.array, target_lang: str = "en-US") -> mx.array:
        enc = self.encoder(mel)
        return self._fuse(enc, self._resolve_prompt_id(target_lang))

    def generate(
        self,
        audio: Union[str, Path, mx.array],
        *,
        target_lang: str = "en-US",
        strip_lang_tags: bool = True,
        dtype: mx.Dtype = mx.float32,
        **kwargs,
    ) -> STTOutput:
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, self.preprocess_args.sample_rate, dtype=dtype)
        else:
            audio = audio.astype(dtype)

        mel = log_mel_spectrogram(
            audio,
            self.preprocessor.featurizer.fb,
            self.preprocessor.featurizer.window,
            self.preprocess_args,
        ).astype(dtype)
        enc_post = self.encode(mel, target_lang)
        ids = self._greedy(enc_post)
        text = self.tokenizer.decode(ids)
        lang = None
        if strip_lang_tags:
            text, lang = strip_lang_tag(text)
        return STTOutput(text=text, language=lang or target_lang, tokens=ids)

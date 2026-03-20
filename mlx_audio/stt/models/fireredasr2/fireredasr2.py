import json
import math
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput

from .config import ModelConfig


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)
        self.subsampling = 4
        self.context = 7  # left=3 + current=1 + right=3

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, T, D) -> (N, T, D, 1) for Conv2d with NHWC layout
        x = mx.expand_dims(x, axis=-1)  # (N, T, D, 1)
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        # MLX output is (N, T, D, C) in NHWC
        # PyTorch does transpose(1,2).view(N, T, C*D) on NCHW -> (N, T, C, D) -> (N, T, C*D)
        # so we need (N, T, C, D) -> (N, T, C*D) to match
        N, T, D, C = x.shape
        x = x.transpose(0, 1, 3, 2)  # (N, T, C, D)
        x = x.reshape(N, T, C * D)
        x = self.out(x)
        return x


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe_positive = np.zeros((max_len, d_model), dtype=np.float32)
        pe_negative = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )
        pe_positive[:, 0::2] = np.sin(position * div_term)
        pe_positive[:, 1::2] = np.cos(position * div_term)
        pe_negative[:, 0::2] = np.sin(-position * div_term)
        pe_negative[:, 1::2] = np.cos(-position * div_term)

        pe_positive = pe_positive[::-1].copy()
        pe_negative = pe_negative[1:]
        pe = np.concatenate([pe_positive, pe_negative], axis=0)[None, :, :]
        self.pe = mx.array(pe)

    def __call__(self, x: mx.array) -> mx.array:
        T_max = self.pe.shape[1]
        T = x.shape[1]
        start = T_max // 2 - T + 1
        end = T_max // 2 + T
        return self.pe[:, start:end, :]


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net_0 = nn.LayerNorm(d_model)  # pre layer norm
        self.net_1 = nn.Linear(d_model, d_model * 4)  # expand
        self.net_4 = nn.Linear(d_model * 4, d_model)  # project

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = self.net_0(x)
        out = self.net_1(out)
        out = out * mx.sigmoid(out)  # swish
        out = self.net_4(out)
        return out + residual


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 4, kernel_size=1, bias=False
        )
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
            groups=d_model * 2,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.pointwise_conv2 = nn.Conv1d(
            d_model * 2, d_model, kernel_size=1, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = self.pre_layer_norm(x)
        # mlx Conv1d expects (N, T, C) - no transpose needed
        out = self.pointwise_conv1(out)
        # GLU: split in half along last dim, gate = sigmoid(second half)
        a, b = mx.split(out, 2, axis=-1)
        out = a * mx.sigmoid(b)
        out = self.depthwise_conv(out)
        out = self.batch_norm(out)
        out = out * mx.sigmoid(out)  # swish
        out = self.pointwise_conv2(out)
        return out + residual


class RelPosMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * self.d_k, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.scale = 1.0 / (self.d_k**0.5)

        self.pos_bias_u = mx.zeros((n_head, self.d_k))
        self.pos_bias_v = mx.zeros((n_head, self.d_k))

    def _rel_shift(self, x: mx.array) -> mx.array:
        N, H, T1, T2 = x.shape
        zero_pad = mx.zeros((N, H, T1, 1))
        x_padded = mx.concatenate([zero_pad, x], axis=-1)
        x_padded = x_padded.reshape(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:, :].reshape(N, H, T1, T2)
        x = x[:, :, :, : T2 // 2 + 1]
        return x

    def __call__(
        self, q: mx.array, k: mx.array, v: mx.array, pos_emb: mx.array
    ) -> mx.array:
        N, T = q.shape[0], q.shape[1]
        residual = q

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).reshape(N, T, self.n_head, self.d_k)
        k = self.w_ks(k).reshape(N, -1, self.n_head, self.d_k)
        v = self.w_vs(v).reshape(N, -1, self.n_head, self.d_k)

        # q shape: (N, T, n_head, d_k) - need (N, n_head, T, d_k) for matmul
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        p = self.linear_pos(pos_emb).reshape(
            pos_emb.shape[0], -1, self.n_head, self.d_k
        )
        p = p.transpose(0, 2, 1, 3)

        # relative position attention (Transformer-XL style)
        # q needs to go back to (N, T, n_head, d_k) for adding bias, then back
        q_t = q.transpose(0, 2, 1, 3)  # (N, T, n_head, d_k)
        q_with_bias_u = (q_t + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_with_bias_v = (q_t + self.pos_bias_v).transpose(0, 2, 1, 3)

        matrix_ac = q_with_bias_u @ k.transpose(0, 1, 3, 2)
        matrix_bd = q_with_bias_v @ p.transpose(0, 1, 3, 2)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = (matrix_ac + matrix_bd) * self.scale
        attn = mx.softmax(attn_scores, axis=-1)

        output = attn @ v
        output = output.transpose(0, 2, 1, 3).reshape(N, T, -1)
        output = self.fc(output)
        return output + residual


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, kernel_size: int = 33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model)
        self.conv = ConformerConvolution(d_model, kernel_size)
        self.ffn2 = ConformerFeedForward(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, pos_emb: mx.array) -> mx.array:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb)
        out = self.conv(out)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class ConformerEncoder(nn.Module):
    def __init__(
        self, idim: int, n_layers: int, n_head: int, d_model: int, kernel_size: int = 33
    ):
        super().__init__()
        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_stack = [
            ConformerBlock(d_model, n_head, kernel_size) for _ in range(n_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # pad right by context-1=6 frames
        pad_right = mx.zeros(
            (x.shape[0], self.input_preprocessor.context - 1, x.shape[2])
        )
        x = mx.concatenate([x, pad_right], axis=1)
        x = self.input_preprocessor(x)
        pos_emb = self.positional_encoding(x)
        for layer in self.layer_stack:
            x = layer(x, pos_emb)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = mx.array(pe[None, :, :])

    def __call__(self, length: int) -> mx.array:
        return self.pe[:, :length, :]


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)  # has bias
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)  # has bias
        self.fc = nn.Linear(n_head * self.d_k, d_model)  # has bias
        self.scale = 1.0 / (self.d_k**0.5)

    def __call__(
        self, q: mx.array, k: mx.array, v: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        N = q.shape[0]
        q = self.w_qs(q).reshape(N, -1, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        k = self.w_ks(k).reshape(N, -1, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        v = self.w_vs(v).reshape(N, -1, self.n_head, self.d_k).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = mx.where(mask, attn, mx.array(-1e9))
        attn = mx.softmax(attn, axis=-1)
        if mask is not None:
            attn = mx.where(mask, attn, mx.array(0.0))
        output = attn @ v
        output = output.transpose(0, 2, 1, 3).reshape(N, -1, self.n_head * self.d_k)
        return self.fc(output)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = DecoderMultiHeadAttention(d_model, n_head)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = DecoderMultiHeadAttention(d_model, n_head)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_model * 4)

    def __call__(
        self,
        x: mx.array,
        enc_output: mx.array,
        self_attn_mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x
        x_norm = self.self_attn_norm(x)
        if cache is not None:
            xq = x_norm[:, -1:, :]
            residual = residual[:, -1:, :]
            if self_attn_mask is not None:
                self_attn_mask = self_attn_mask[:, -1:, :]
        else:
            xq = x_norm
        x = residual + self.self_attn(xq, x_norm, x_norm, self_attn_mask)

        residual = x
        x = residual + self.cross_attn(self.cross_attn_norm(x), enc_output, enc_output)

        residual = x
        x = residual + self.mlp(self.mlp_norm(x))

        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(nn.gelu(self.w_1(x)))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        odim: int,
        n_layers: int,
        n_head: int,
        d_model: int,
        pe_maxlen: int = 5000,
    ):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.n_layers = n_layers
        self.scale = d_model**0.5

        self.tgt_word_emb = nn.Embedding(odim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.layer_stack = [DecoderLayer(d_model, n_head) for _ in range(n_layers)]
        self.layer_norm_out = nn.LayerNorm(d_model)
        self.tgt_word_prj = nn.Linear(d_model, odim, bias=False)

    def _topk(self, x: mx.array, k: int) -> Tuple[mx.array, mx.array]:
        if k >= x.shape[-1]:
            idx = mx.argsort(-x, axis=-1)
            return mx.take_along_axis(x, idx, axis=-1), idx
        neg = -x
        idx = mx.argpartition(neg, kth=k, axis=-1)[..., :k]
        vals = mx.take_along_axis(x, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        return mx.take_along_axis(vals, order, axis=-1), mx.take_along_axis(
            idx, order, axis=-1
        )

    def beam_search(
        self,
        enc_output: mx.array,
        beam_size: int = 3,
        max_len: int = 0,
        softmax_smoothing: float = 1.25,
        length_penalty: float = 0.6,
        eos_penalty: float = 1.0,
    ) -> Tuple[mx.array, float]:
        B = beam_size
        T_enc = enc_output.shape[1]
        max_decode = max_len if max_len > 0 else T_enc
        INF = 1e10

        enc_expanded = mx.repeat(enc_output, B, axis=0)  # (B, T_enc, H)

        ys = mx.full((B, 1), self.sos_id, dtype=mx.int32)
        scores = mx.array([0.0] + [-INF] * (B - 1)).reshape(B, 1)
        is_finished = mx.zeros((B, 1))
        caches = [None] * self.n_layers
        confidences = mx.zeros((B, 1))

        for t in range(max_decode):
            seq_len = ys.shape[1]
            causal = mx.expand_dims(mx.tril(mx.ones((seq_len, seq_len))), axis=0)

            emb = self.tgt_word_emb(ys) * self.scale + self.positional_encoding(seq_len)

            dec_output = emb
            new_caches = []
            for i, layer in enumerate(self.layer_stack):
                dec_output = layer(dec_output, enc_expanded, causal, cache=caches[i])
                new_caches.append(dec_output)
            caches = new_caches

            dec_output = self.layer_norm_out(dec_output)
            logits = self.tgt_word_prj(dec_output[:, -1, :])
            t_scores = mx.log(mx.softmax(logits / softmax_smoothing, axis=-1) + 1e-10)

            if eos_penalty != 1.0:
                eos_col = t_scores[:, self.eos_id : self.eos_id + 1] * eos_penalty
                t_scores = mx.concatenate(
                    [
                        t_scores[:, : self.eos_id],
                        eos_col,
                        t_scores[:, self.eos_id + 1 :],
                    ],
                    axis=-1,
                )

            t_topB_scores, t_topB_ys = self._topk(t_scores, B)

            # mask finished beams
            mask_score = mx.array([0.0] + [-INF] * (B - 1)).reshape(1, B)
            mask_score = mx.repeat(mask_score, B, axis=0)
            is_fin = is_finished.astype(mx.float32)
            t_topB_scores = t_topB_scores * (1 - is_fin) + mask_score * is_fin
            t_topB_ys = t_topB_ys * (1 - is_finished.astype(mx.int32)) + (
                self.eos_id * is_finished.astype(mx.int32)
            )

            all_scores = (scores + t_topB_scores).reshape(1, B * B)
            best_scores, best_idx = self._topk(all_scores, B)
            scores = best_scores.reshape(B, 1)

            beam_idx = (best_idx.reshape(B) // B).astype(mx.int32)

            ys = ys[beam_idx]
            new_tokens = t_topB_ys.reshape(B * B)[best_idx.reshape(B)].reshape(B, 1)
            ys = mx.concatenate([ys, new_tokens], axis=1)

            confidences = confidences[beam_idx]
            new_conf = t_topB_scores.reshape(B * B)[best_idx.reshape(B)].reshape(B, 1)
            confidences = mx.concatenate([confidences, mx.exp(new_conf)], axis=1)

            caches = [c[beam_idx] if c is not None else None for c in caches]

            is_finished = (new_tokens == self.eos_id).astype(mx.int32)
            mx.eval(is_finished)
            if is_finished.sum().item() == B:
                break

        # GNMT length penalty
        ys_lengths = mx.sum(ys != self.eos_id, axis=-1, keepdims=True).astype(
            mx.float32
        )
        if length_penalty > 0.0:
            penalty = mx.power((5.0 + ys_lengths) / 6.0, length_penalty)
            final_scores = scores / penalty
        else:
            final_scores = scores

        best = mx.argmax(final_scores.reshape(-1)).item()
        best_seq = ys[best, 1:]  # remove SOS
        best_conf = confidences[best, 1:]
        return best_seq, best_conf


class FireRedASR2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        enc = config.encoder
        dec = config.decoder

        self.encoder = ConformerEncoder(
            config.idim, enc.n_layers, enc.n_head, enc.d_model, enc.kernel_size
        )
        self.decoder = TransformerDecoder(
            config.sos_id,
            config.eos_id,
            config.pad_id,
            config.odim,
            dec.n_layers,
            dec.n_head,
            dec.d_model,
            dec.pe_maxlen,
        )

        self._tokenizer = None
        self._cmvn = None

    def _load_cmvn(self, model_path: str):
        cmvn_path = Path(model_path) / "cmvn.json"
        if cmvn_path.exists():
            with open(cmvn_path) as f:
                data = json.load(f)
            self._cmvn = (
                mx.array(data["means"], dtype=mx.float32),
                mx.array(data["istd"], dtype=mx.float32),
            )

    def _load_tokenizer(self, model_path: str):
        dict_path = Path(model_path) / "dict.txt"
        spm_path = Path(model_path) / "train_bpe1000.model"
        if dict_path.exists():
            id2word = []
            with open(dict_path, encoding="utf8") as f:
                for i, line in enumerate(f):
                    tokens = line.strip().split()
                    if len(tokens) >= 2:
                        word = tokens[0]
                    elif len(tokens) == 1:
                        word = tokens[0]
                    else:
                        word = " "
                    if word == "<space>":
                        word = " "
                    id2word.append(word)
            self._tokenizer = id2word

            if spm_path.exists():
                import sentencepiece as spm

                self._sp = spm.SentencePieceProcessor()
                self._sp.Load(str(spm_path))
            else:
                self._sp = None

    def _detokenize(self, ids) -> str:
        if self._tokenizer is None:
            return ""
        SPM_SPACE = "\u2581"
        tokens = [
            self._tokenizer[int(i)] for i in ids if 0 <= int(i) < len(self._tokenizer)
        ]
        text = "".join(tokens)
        text = text.replace(SPM_SPACE, " ").strip()
        text = re.sub(r"(<blank>)|(<sil>)", "", text)
        return text.lower()

    def _extract_fbank(self, audio: mx.array) -> mx.array:
        from mlx_audio.dsp import compute_fbank_kaldi

        waveform = audio.flatten()
        # scale to int16 range since compute_fbank_kaldi expects raw waveform amplitudes
        if mx.abs(waveform).max().item() <= 1.0:
            waveform = waveform * 32768.0
        # 16kHz: 25ms frame = 400 samples, 10ms shift = 160 samples
        features = compute_fbank_kaldi(
            waveform,
            sample_rate=16000,
            win_len=400,
            win_inc=160,
            num_mels=80,
            snip_edges=True,
            dither=0.0,
        )
        return features

    def generate(self, audio_or_path, beam_size: int = 3, **kwargs) -> STTOutput:
        start_time = time.time()

        if isinstance(audio_or_path, (str, Path)):
            from mlx_audio.stt.utils import load_audio

            audio = load_audio(str(audio_or_path), sr=16000)
        else:
            audio = audio_or_path

        features = self._extract_fbank(audio)

        if self._cmvn is not None:
            means, istd = self._cmvn
            features = (features - means) * istd

        features = mx.expand_dims(features, axis=0)  # (1, T, 80)
        mx.eval(features)

        enc_output = self.encoder(features)
        mx.eval(enc_output)

        softmax_smoothing = kwargs.get("softmax_smoothing", 1.25)
        length_penalty = kwargs.get("length_penalty", 0.6)
        eos_penalty = kwargs.get("eos_penalty", 1.0)
        max_len = kwargs.get("max_len", 0)

        best_seq, best_conf = self.decoder.beam_search(
            enc_output,
            beam_size=beam_size,
            max_len=max_len,
            softmax_smoothing=softmax_smoothing,
            length_penalty=length_penalty,
            eos_penalty=eos_penalty,
        )
        mx.eval(best_seq, best_conf)

        # trim at EOS
        seq_list = best_seq.tolist()
        try:
            eos_pos = seq_list.index(self.config.eos_id)
            seq_list = seq_list[:eos_pos]
        except ValueError:
            pass

        text = self._detokenize(seq_list)
        confidence = (
            float(mx.mean(best_conf[: len(seq_list)]).item()) if seq_list else 0.0
        )

        total_time = time.time() - start_time

        return STTOutput(
            text=text,
            segments=[{"text": text, "confidence": round(confidence, 3)}],
            total_time=round(total_time, 3),
            generation_tokens=len(seq_list),
        )

    def sanitize(self, weights: dict) -> dict:
        new_weights = {}
        for k, v in weights.items():
            new_k = k

            # rename Conv2d subsampling: conv.0 -> conv1, conv.2 -> conv2
            new_k = new_k.replace(
                "input_preprocessor.conv.0.", "input_preprocessor.conv1."
            )
            new_k = new_k.replace(
                "input_preprocessor.conv.2.", "input_preprocessor.conv2."
            )

            # rename FFN sequential indices: net.0 -> net_0, net.1 -> net_1, net.4 -> net_4
            new_k = re.sub(r"\.net\.(\d+)\.", r".net_\1.", new_k)

            # transpose Conv1d weights: PyTorch (out, in, K) -> MLX (out, K, in)
            if "pointwise_conv1.weight" in new_k or "pointwise_conv2.weight" in new_k:
                v = mx.transpose(v, axes=(0, 2, 1))
            elif "depthwise_conv.weight" in new_k:
                v = mx.transpose(v, axes=(0, 2, 1))
            # transpose Conv2d weights: PyTorch (out, in, H, W) -> MLX (out, H, W, in)
            elif (
                "input_preprocessor.conv" in new_k and "weight" in new_k and v.ndim == 4
            ):
                v = mx.transpose(v, axes=(0, 2, 3, 1))

            new_weights[new_k] = v

        # weight tying: copy embedding to output projection if missing
        if (
            "decoder.tgt_word_prj.weight" not in new_weights
            and "decoder.tgt_word_emb.weight" in new_weights
        ):
            new_weights["decoder.tgt_word_prj.weight"] = new_weights[
                "decoder.tgt_word_emb.weight"
            ]

        return new_weights

    @staticmethod
    def post_load_hook(model, model_path):
        model._load_cmvn(str(model_path))
        model._load_tokenizer(str(model_path))
        return model


Model = FireRedASR2

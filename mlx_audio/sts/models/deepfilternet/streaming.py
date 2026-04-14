"""True streaming helpers for DeepFilterNet enhancement."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Iterable, Iterator, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .network import DfNet
from .network_df1 import DfNetV1

if TYPE_CHECKING:
    from .model import DeepFilterNetModel


@dataclass
class DeepFilterNetStreamingConfig:
    """Configuration for real-time DeepFilterNet streaming."""

    pad_end_frames: int = 3
    compensate_delay: bool = True


class DeepFilterNetStreamer:
    """Stateful DeepFilterNet streamer with per-hop processing.

    This implements a true streaming state machine:
    - persistent analysis/synthesis overlap state
    - persistent feature normalization EMA state
    - frame-wise model execution with bounded temporal buffers
    """

    def __init__(
        self,
        model: "DeepFilterNetModel",
        config: Optional[DeepFilterNetStreamingConfig] = None,
    ):
        self.model = model
        self.p = model.config
        self.config = config or DeepFilterNetStreamingConfig()

        if isinstance(model.model, DfNetV1):
            raise NotImplementedError(
                "True stateful streaming is currently implemented for DeepFilterNet2/3."
            )
        if not isinstance(model.model, DfNet):
            raise TypeError(
                f"Unsupported model type for streaming: {type(model.model)}"
            )

        self.net: DfNet = model.model
        self.reset()

    def reset(self) -> None:
        p = self.p
        self._sample_in = np.zeros((0,), dtype=np.float32)
        self._sample_out = np.zeros((0,), dtype=np.float32)

        # libDF-style streaming STFT/ISTFT memories.
        self._analysis_mem = np.zeros((p.fft_size - p.hop_size,), dtype=np.float32)
        self._synth_mem = np.zeros((p.fft_size - p.hop_size,), dtype=np.float32)

        # Feature normalization EMA states.
        self._alpha = np.float32(self.model._norm_alpha())
        self._one_minus_alpha = np.float32(1.0 - self._alpha)
        self._erb_state = np.linspace(-60.0, -90.0, p.nb_erb, dtype=np.float32)
        self._df_state = np.linspace(0.001, 0.0001, p.nb_df, dtype=np.float32)

        # Lookahead alignment queues (raw frame -> shifted frame).
        self._spec_q: Deque[np.ndarray] = deque()
        self._frame_count = 0

        # Model temporal buffers (causal context) as fixed-size rolling tensors.
        self._enc_erb0_hist = mx.zeros((1, 1, 3, p.nb_erb), dtype=mx.float32)
        self._enc_df0_hist = mx.zeros((1, 2, 3, p.nb_df), dtype=mx.float32)
        self._df_convp_hist = mx.zeros(
            (1, p.conv_ch, self.p.df_pathway_kernel_size_t, p.nb_df), dtype=mx.float32
        )
        self._spec_past: Deque[np.ndarray] = deque(maxlen=max(1, self.p.df_order))
        self._erb_fb_np = np.asarray(self.model.erb_fb, dtype=np.float32)
        self._has_erb_fb = (
            self._erb_fb_np.ndim == 2
            and self._erb_fb_np.shape[0] == self.p.freq_bins
            and self._erb_fb_np.shape[1] == self.p.nb_erb
        )

        # GRU hidden states for streaming.
        self._enc_emb_state: Optional[List[mx.array]] = None
        self._erb_dec_state: Optional[List[mx.array]] = None
        self._df_dec_state: Optional[List[mx.array]] = None

        self._emitted_samples = 0
        self._delay_samples = self.p.fft_size - self.p.hop_size
        self._delay_dropped = 0

    @property
    def hop_size(self) -> int:
        return self.p.hop_size

    def process_chunk(self, chunk: np.ndarray, is_last: bool = False) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if x.size:
            self._sample_in = np.concatenate([self._sample_in, x], axis=0)

        out_frames: List[np.ndarray] = []
        while self._sample_in.shape[0] >= self.p.hop_size:
            frame = self._sample_in[: self.p.hop_size]
            self._sample_in = self._sample_in[self.p.hop_size :]
            y = self._process_hop(frame)
            if y is not None:
                out_frames.append(y)

        if is_last:
            pad = np.zeros(
                (self.config.pad_end_frames * self.p.hop_size,), dtype=np.float32
            )
            if pad.size:
                self._sample_in = np.concatenate([self._sample_in, pad], axis=0)
            while self._sample_in.shape[0] >= self.p.hop_size:
                frame = self._sample_in[: self.p.hop_size]
                self._sample_in = self._sample_in[self.p.hop_size :]
                y = self._process_hop(frame)
                if y is not None:
                    out_frames.append(y)

        if not out_frames:
            return np.zeros((0,), dtype=np.float32)
        y = np.concatenate(out_frames, axis=0)

        if self.config.compensate_delay:
            # Match offline/official delay compensation by dropping initial d samples.
            if self._delay_dropped < self._delay_samples:
                need = self._delay_samples - self._delay_dropped
                drop = min(need, y.shape[0])
                y = y[drop:]
                self._delay_dropped += drop
        self._emitted_samples += y.shape[0]
        return y

    def flush(self) -> np.ndarray:
        return self.process_chunk(np.zeros((0,), dtype=np.float32), is_last=True)

    def process_iterable(self, chunks: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        for chunk in chunks:
            out = self.process_chunk(chunk, is_last=False)
            if out.size:
                yield out
        tail = self.flush()
        if tail.size:
            yield tail

    def _process_hop(self, hop_td: np.ndarray) -> Optional[np.ndarray]:
        spec = self._analysis_frame(hop_td)
        feat_erb, feat_df = self._features_frame(spec)
        self._spec_q.append(spec)
        self._frame_count += 1

        la = self.p.conv_lookahead
        if self._frame_count <= la:
            return None

        feat_erb_t = feat_erb
        feat_df_t = feat_df
        spec_t = self._spec_q.popleft()

        spec_e = self._infer_frame_v23(spec_t, feat_erb_t, feat_df_t)
        y = self._synthesis_frame(spec_e)
        return y.astype(np.float32, copy=False)

    def _analysis_frame(self, hop_td: np.ndarray) -> np.ndarray:
        p = self.p
        frame_td = np.concatenate([self._analysis_mem, hop_td], axis=0)
        frame_win = frame_td * np.asarray(self.model._vorbis, dtype=np.float32)
        spec = np.fft.rfft(frame_win, n=p.fft_size).astype(np.complex64) * np.float32(
            self.model.wnorm
        )

        if self._analysis_mem.size:
            split = self._analysis_mem.size - p.hop_size
            if split > 0:
                self._analysis_mem[:split] = self._analysis_mem[p.hop_size :]
            self._analysis_mem[split:] = hop_td
        return spec

    def _synthesis_frame(self, spec_norm: np.ndarray) -> np.ndarray:
        p = self.p
        # Rust realfft inverse is unnormalized. NumPy irfft is normalized, so multiply by N.
        td = np.fft.irfft(spec_norm, n=p.fft_size).astype(np.float32) * np.float32(
            p.fft_size
        )
        td *= np.asarray(self.model._vorbis, dtype=np.float32)

        out = td[: p.hop_size] + self._synth_mem[: p.hop_size]
        split = self._synth_mem.size - p.hop_size
        if split > 0:
            self._synth_mem[:split] = self._synth_mem[p.hop_size :]
            self._synth_mem[split:] = td[p.hop_size : p.hop_size + p.hop_size]
        else:
            self._synth_mem[:] = td[p.hop_size :]
        return out

    def _features_frame(self, spec: np.ndarray):
        p = self.p
        mag_sq = np.square(spec.real, dtype=np.float32) + np.square(
            spec.imag, dtype=np.float32
        )

        if self._has_erb_fb:
            erb_e = mag_sq @ self._erb_fb_np
        else:
            if self.model.erb_widths is None:
                raise ValueError("Missing both ERB filterbank and ERB band widths.")
            bands = []
            start = 0
            for w in self.model.erb_widths:
                stop = start + int(w)
                bands.append(np.mean(mag_sq[start:stop], dtype=np.float32))
                start = stop
            erb_e = np.asarray(bands, dtype=np.float32)

        erb_db = np.float32(10.0) * np.log10(erb_e + np.float32(1e-10))
        self._erb_state = erb_db * self._one_minus_alpha + self._erb_state * self._alpha
        feat_erb = (erb_db - self._erb_state) / np.float32(40.0)

        df = spec[: p.nb_df]
        mag = np.abs(df).astype(np.float32)
        self._df_state = mag * self._one_minus_alpha + self._df_state * self._alpha
        denom = np.sqrt(self._df_state)
        feat_df = np.stack([df.real / denom, df.imag / denom], axis=-1).astype(
            np.float32
        )

        return feat_erb.astype(np.float32), feat_df

    def _infer_frame_v23(
        self,
        spec_t: np.ndarray,
        feat_erb_t: np.ndarray,
        feat_df_t: np.ndarray,
    ) -> np.ndarray:
        p = self.p
        b = 1

        spec_mx = mx.array(
            np.stack([spec_t.real, spec_t.imag], axis=-1)[
                None, None, None, :, :
            ].astype(np.float32)
        )
        feat_erb_mx = mx.array(feat_erb_t[None, None, None, :].astype(np.float32))
        feat_df_mx = mx.array(
            feat_df_t[None, None, :, :].astype(np.float32)
        )  # [B,T,F,2]
        feat_df_mx = mx.transpose(feat_df_mx, (0, 3, 1, 2))  # [B,2,T,F]

        self._enc_erb0_hist = self._append_history(self._enc_erb0_hist, feat_erb_mx)
        self._enc_df0_hist = self._append_history(self._enc_df0_hist, feat_df_mx)

        e0 = self._apply_conv_last(self.net.enc.erb_conv0, self._enc_erb0_hist)
        e1 = self._apply_conv_last(self.net.enc.erb_conv1, e0)
        e2 = self._apply_conv_last(self.net.enc.erb_conv2, e1)
        e3 = self._apply_conv_last(self.net.enc.erb_conv3, e2)

        c0 = self._apply_conv_last(self.net.enc.df_conv0, self._enc_df0_hist)
        c1 = self._apply_conv_last(self.net.enc.df_conv1, c0)

        cemb = mx.transpose(c1, (0, 2, 3, 1)).reshape(b, 1, -1)
        cemb = self.net.enc.df_fc_emb(cemb)
        emb = mx.transpose(e3, (0, 2, 3, 1)).reshape(b, 1, -1)
        emb = mx.concatenate([emb, cemb], axis=-1) if p.enc_concat else emb + cemb

        emb, self._enc_emb_state = self._squeezed_gru_step(
            self.net.enc.emb_gru, emb, self._enc_emb_state
        )
        lsnr = (
            self.net.enc.lsnr_fc(emb) * self.net.enc.lsnr_scale
            + self.net.enc.lsnr_offset
        )

        m = self._erb_decoder_step(emb, e3, e2, e1, e0)
        spec_m = self.net.mask(spec_mx, m)

        df_coefs = self._df_decoder_step(emb, c0)
        df_coefs = df_coefs.reshape(b, 1, self.net.nb_df, self.net.df_order, 2)
        df_coefs = mx.transpose(df_coefs, (0, 3, 1, 2, 4))  # [B,O,T,F,2]

        spec_e = self._df_assign_step(spec_mx, spec_m, df_coefs, spec_t)
        spec_e_np = np.array(
            spec_e[0, 0, 0, :, 0] + 1j * spec_e[0, 0, 0, :, 1], dtype=np.complex64
        )
        return spec_e_np

    def _append_history(self, history: mx.array, frame: mx.array) -> mx.array:
        return mx.concatenate([history[:, :, 1:, :], frame], axis=2)

    def _apply_conv_last(self, layer: dict, x: mx.array) -> mx.array:
        x = layer["1"](x)
        if "3" in layer:
            x = layer["2"](x)
            x = layer["3"].norm(x)
        else:
            x = layer["2"].norm(x)
        x = nn.relu(x)
        return x[:, :, -1:, :]

    def _squeezed_gru_step(
        self,
        gru_mod,
        x: mx.array,
        state: Optional[List[mx.array]],
    ):
        x_in = gru_mod.linear_in(x)
        new_state: List[mx.array] = []
        cur = x_in
        for i, gru in enumerate(gru_mod.gru_layers):
            h0 = (
                state[i]
                if state is not None and i < len(state)
                else mx.zeros((cur.shape[0], gru_mod.hidden_size), dtype=cur.dtype)
            )
            cur = gru(cur, h0)
            new_state.append(cur[:, -1, :])
        if gru_mod.linear_out is not None:
            cur = gru_mod.linear_out(cur)
        return cur, new_state

    def _erb_decoder_step(
        self, emb: mx.array, e3: mx.array, e2: mx.array, e1: mx.array, e0: mx.array
    ) -> mx.array:
        b, t = emb.shape[:2]
        f8 = e3.shape[3]
        emb_d, self._erb_dec_state = self._squeezed_gru_step(
            self.net.erb_dec.emb_gru, emb, self._erb_dec_state
        )
        emb_d = emb_d.reshape(b, t, f8, -1)
        emb_d = mx.transpose(emb_d, (0, 3, 1, 2))

        d3 = self.net.erb_dec._apply_pathway(self.net.erb_dec.conv3p, e3) + emb_d
        d3 = nn.relu(self.net.erb_dec._apply_transpose(self.net.erb_dec.convt3, d3))
        d2 = self.net.erb_dec._apply_pathway(self.net.erb_dec.conv2p, e2) + d3
        d2 = nn.relu(self.net.erb_dec._apply_transpose(self.net.erb_dec.convt2, d2))
        d1 = self.net.erb_dec._apply_pathway(self.net.erb_dec.conv1p, e1) + d2
        d1 = nn.relu(self.net.erb_dec._apply_transpose(self.net.erb_dec.convt1, d1))
        d0 = self.net.erb_dec._apply_pathway(self.net.erb_dec.conv0p, e0) + d1
        m = mx.sigmoid(self.net.erb_dec._apply_output(self.net.erb_dec.conv0_out, d0))
        return m

    def _df_decoder_step(self, emb: mx.array, c0: mx.array) -> mx.array:
        c, self._df_dec_state = self._squeezed_gru_step(
            self.net.df_dec.df_gru, emb, self._df_dec_state
        )
        if self.net.df_dec.df_skip is not None:
            c = c + self.net.df_dec.df_skip(emb)

        self._df_convp_hist = self._append_history(self._df_convp_hist, c0)
        c0p = self.net.df_dec._apply_convp(self._df_convp_hist)[:, :, -1:, :]
        c0p = mx.transpose(c0p, (0, 2, 3, 1))  # [B,1,F,O*2]

        c_out = self.net.df_dec.df_out(c)
        c_out = (
            c_out.reshape(
                c.shape[0],
                c.shape[1],
                self.net.df_dec.df_bins,
                self.net.df_dec.df_out_ch,
            )
            + c0p
        )
        return c_out

    def _df_assign_step(
        self,
        spec: mx.array,
        spec_m: mx.array,
        df_coefs: mx.array,
        spec_t: np.ndarray,
    ) -> mx.array:
        p = self.p
        self._spec_past.append(spec_t.astype(np.complex64))
        left = p.df_order - p.df_lookahead - 1

        past: List[np.ndarray] = list(self._spec_past)[:-1]
        need_past = max(0, left - len(past))
        spec_window: List[np.ndarray] = [
            np.zeros((p.freq_bins,), dtype=np.complex64) for _ in range(need_past)
        ]
        if left:
            spec_window.extend(past[-left:])
        spec_window.append(spec_t.astype(np.complex64))

        fut_avail = min(p.df_lookahead, len(self._spec_q))
        for i in range(fut_avail):
            spec_window.append(self._spec_q[i].astype(np.complex64))
        for _ in range(p.df_lookahead - fut_avail):
            spec_window.append(np.zeros((p.freq_bins,), dtype=np.complex64))

        sw = np.stack(spec_window, axis=0)[:, : self.net.nb_df]  # [O,F]
        co = np.array(df_coefs[0, :, 0, :, :], dtype=np.float32)  # [O,F,2]
        sr = sw.real.astype(np.float32)
        si = sw.imag.astype(np.float32)
        cr = co[..., 0]
        ci = co[..., 1]
        out_r = np.sum(sr * cr - si * ci, axis=0)
        out_i = np.sum(sr * ci + si * cr, axis=0)

        low = mx.array(np.stack([out_r, out_i], axis=-1)[None, None, None, :, :])
        if p.enc_concat:
            return mx.concatenate([low, spec_m[:, :, :, self.net.nb_df :, :]], axis=3)

        spec_df = mx.concatenate([low, spec[:, :, :, self.net.nb_df :, :]], axis=3)
        return mx.concatenate(
            [
                spec_df[:, :, :, : self.net.nb_df, :],
                spec_m[:, :, :, self.net.nb_df :, :],
            ],
            axis=3,
        )

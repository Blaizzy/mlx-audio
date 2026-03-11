"""DeepFilterNet v1 architecture in pure MLX."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import DeepFilterNetConfig
from .network import BatchNorm, ConvBlock, ConvTransposeBlock, DeepFilterOp, Mask


class GroupedLinear(nn.Module):
    """Grouped linear layer matching df.modules.GroupedLinear."""

    def __init__(
        self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = True
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.shuffle = shuffle if groups > 1 else False
        self.layers = [
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        ys = []
        for i, layer in enumerate(self.layers):
            xs = x[..., i * self.input_size : (i + 1) * self.input_size]
            ys.append(layer(xs))
        y = mx.concatenate(ys, axis=-1)
        if self.shuffle and y.ndim == 3:
            b, t, _ = y.shape
            y = y.reshape(b, t, self.hidden_size, self.groups)
            y = mx.transpose(y, (0, 1, 3, 2)).reshape(b, t, -1)
        return y


class PyTorchGRUCell(nn.Module):
    """Single-layer GRU with PyTorch-compatible parameter names and equations."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih_l0 = mx.zeros((3 * hidden_size, input_size))
        self.weight_hh_l0 = mx.zeros((3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_l0 = mx.zeros((3 * hidden_size,))
            self.bias_hh_l0 = mx.zeros((3 * hidden_size,))
        else:
            self.bias_ih_l0 = None
            self.bias_hh_l0 = None

    def __call__(
        self, x: mx.array, h: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        # x: [T, B, I], h: [B, H]
        h_t = h
        if h_t is None:
            h_t = mx.zeros((x.shape[1], self.hidden_size), dtype=x.dtype)

        outputs = []
        H = self.hidden_size
        for t in range(x.shape[0]):
            xt = x[t]
            gi = xt @ mx.transpose(self.weight_ih_l0)
            gh = h_t @ mx.transpose(self.weight_hh_l0)
            if self.bias_ih_l0 is not None:
                gi = gi + self.bias_ih_l0
                gh = gh + self.bias_hh_l0

            i_r, i_z, i_n = gi[:, :H], gi[:, H : 2 * H], gi[:, 2 * H :]
            h_r, h_z, h_n = gh[:, :H], gh[:, H : 2 * H], gh[:, 2 * H :]

            r = mx.sigmoid(i_r + h_r)
            z = mx.sigmoid(i_z + h_z)
            n = mx.tanh(i_n + r * h_n)
            h_t = n + z * (h_t - n)
            outputs.append(h_t)

        out = mx.stack(outputs, axis=0)
        return out, mx.expand_dims(h_t, axis=0)


class GroupedGRULayer(nn.Module):
    """One grouped-GRU layer (splits channels across groups)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        batch_first: bool = True,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.batch_first = batch_first
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.layers = [
            PyTorchGRUCell(self.input_size, self.hidden_size) for _ in range(groups)
        ]

    def __call__(
        self, x: mx.array, h0: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        # x: [B, T, I] if batch_first else [T, B, I]
        if self.batch_first:
            x_tbi = mx.transpose(x, (1, 0, 2))
        else:
            x_tbi = x

        bsz = x_tbi.shape[1]
        if h0 is None:
            h0 = mx.zeros((self.groups, bsz, self.hidden_size), dtype=x_tbi.dtype)

        ys = []
        hs = []
        for i, layer in enumerate(self.layers):
            xg = x_tbi[..., i * self.input_size : (i + 1) * self.input_size]
            yg, hg = layer(xg, h0[i])
            ys.append(yg)
            hs.append(hg)

        y = mx.concatenate(ys, axis=-1)
        h = mx.concatenate(hs, axis=0)

        if self.batch_first:
            y = mx.transpose(y, (1, 0, 2))
        return y, h


class GroupedGRU(nn.Module):
    """GroupedGRU matching DeepFilterNet v1 implementation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        batch_first: bool = True,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.input_size = input_size
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size // groups
        self.shuffle = shuffle if groups > 1 else False
        self.add_outputs = add_outputs

        self.grus = [
            GroupedGRULayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                groups=groups,
                batch_first=batch_first,
            )
            for i in range(num_layers)
        ]

    def __call__(
        self, x: mx.array, state: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        dim0, dim1, _ = x.shape
        b = dim0 if self.batch_first else dim1
        if state is None:
            state = mx.zeros(
                (self.num_layers * self.groups, b, self.hidden_size), dtype=x.dtype
            )

        out = mx.zeros((dim0, dim1, self.hidden_size * self.groups), dtype=x.dtype)
        outstates = []
        cur = x
        h = self.groups
        for i, gru in enumerate(self.grus):
            cur, s = gru(cur, state[i * h : (i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                cur = cur.reshape(cur.shape[0], cur.shape[1], -1, self.groups)
                cur = mx.transpose(cur, (0, 1, 3, 2)).reshape(
                    cur.shape[0], cur.shape[1], -1
                )
            out = out + cur if self.add_outputs else cur

        return out, mx.concatenate(outstates, axis=0)


class ConvKxF(nn.Module):
    """convkxf-style block with v1 naming (sconv/sconvt/norm)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        f: int = 3,
        fstride: int = 2,
        lookahead: int = 0,
        batch_norm: bool = True,
        mode: str = "normal",
        depthwise: bool = True,
        complex_in: bool = False,
        act: str = "relu",
    ):
        super().__init__()
        stride_f = 1 if f == 1 else fstride
        groups = min(in_ch, out_ch) if depthwise else 1
        if in_ch % groups != 0 or out_ch % groups != 0:
            groups = 1
        if complex_in and groups % 2 == 0:
            groups //= 2

        kernel = (k, f)
        self.mode = mode
        self.act = act
        if mode == "normal":
            self.sconv = ConvBlock(
                in_ch,
                out_ch,
                kernel,
                groups,
                stride_f,
                lookahead=lookahead,
                use_bias=(not batch_norm),
            )
        elif mode == "transposed":
            self.sconvt = ConvTransposeBlock(in_ch, out_ch, kernel, groups, stride_f)
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        self.has_pw = groups > 1
        self.pwconv = ConvBlock(out_ch, out_ch, (1, 1), 1, 1) if self.has_pw else None
        self.norm = BatchNorm(out_ch) if batch_norm else None

    def __call__(self, x: mx.array) -> mx.array:
        if self.mode == "normal":
            y = self.sconv(x)
        else:
            y = self.sconvt(x)

        if self.pwconv is not None:
            y = self.pwconv(y)
        if self.norm is not None:
            y = self.norm.norm(y)

        if self.act == "relu":
            return nn.relu(y)
        if self.act == "sigmoid":
            return mx.sigmoid(y)
        return y


class EncoderV1(nn.Module):
    """V1 encoder with grouped convolutions and linear layers."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        p = config
        layer_width = p.conv_ch
        wf = p.conv_width_factor

        k = p.conv_k_enc
        k0 = 1 if k == 1 and p.conv_lookahead == 0 else max(2, k)

        cl = 1 if p.conv_lookahead > 0 else 0
        self.erb_conv0 = ConvKxF(
            1,
            layer_width,
            k=k0,
            fstride=1,
            lookahead=cl,
            batch_norm=True,
            depthwise=p.conv_depthwise,
        )
        cl = 1 if p.conv_lookahead > 1 else 0
        self.erb_conv1 = ConvKxF(
            layer_width * wf**0,
            layer_width * wf**1,
            k=k,
            lookahead=cl,
            batch_norm=True,
            depthwise=p.conv_depthwise,
        )
        cl = 1 if p.conv_lookahead > 2 else 0
        self.erb_conv2 = ConvKxF(
            layer_width * wf**1,
            layer_width * wf**2,
            k=k,
            lookahead=cl,
            batch_norm=True,
            depthwise=p.conv_depthwise,
        )
        self.erb_conv3 = ConvKxF(
            layer_width * wf**2,
            layer_width * wf**2,
            k=k,
            fstride=1,
            batch_norm=True,
            depthwise=p.conv_depthwise,
        )

        self.clc_conv0 = ConvKxF(
            2,
            layer_width,
            k=k0,
            fstride=1,
            lookahead=p.conv_lookahead,
            batch_norm=True,
            depthwise=p.conv_depthwise,
        )
        self.clc_conv1 = ConvKxF(
            layer_width,
            layer_width * wf**1,
            k=k,
            batch_norm=True,
            depthwise=p.conv_depthwise,
        )

        self.emb_dim = layer_width * p.nb_erb // 4 * wf**2
        self.clc_fc_emb = GroupedLinear(
            layer_width * p.nb_df // 2,
            self.emb_dim,
            groups=p.linear_groups,
            shuffle=p.group_shuffle,
        )

        self.emb_gru = GroupedGRU(
            self.emb_dim,
            p.emb_hidden_dim,
            num_layers=p.emb_num_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(p.emb_hidden_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def __call__(self, feat_erb: mx.array, feat_spec: mx.array):
        # feat_erb: [B,1,T,Fe], feat_spec: [B,2,T,F]
        b = feat_erb.shape[0]
        t = feat_erb.shape[2]

        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)

        c0 = self.clc_conv0(feat_spec)
        c1 = self.clc_conv1(c0)

        cemb = mx.transpose(c1, (2, 0, 1, 3)).reshape(t, b, -1)
        cemb = self.clc_fc_emb(cemb)
        emb = mx.transpose(e3, (2, 0, 1, 3)).reshape(t, b, -1)
        emb = emb + cemb
        emb, _ = self.emb_gru(emb)

        emb_bt = mx.transpose(emb, (1, 0, 2))
        lsnr = self.lsnr_fc(emb_bt) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb_bt, c0, lsnr


class ErbDecoderV1(nn.Module):
    """V1 ERB-band decoder with transposed convolutions and skip connections."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        p = config
        layer_width = p.conv_ch
        wf = p.conv_width_factor

        self.emb_width = layer_width * wf**2
        self.emb_dim = self.emb_width * (p.nb_erb // 4)
        self.fc_emb = nn.Sequential(
            GroupedLinear(
                p.emb_hidden_dim,
                self.emb_dim,
                groups=p.linear_groups,
                shuffle=p.group_shuffle,
            ),
            nn.ReLU(),
        )

        k = p.conv_k_dec
        self.conv3p = ConvKxF(
            layer_width * wf**2, self.emb_width, k=1, f=1, fstride=1, batch_norm=True
        )
        self.convt3 = ConvKxF(
            self.emb_width,
            layer_width * wf**2,
            k=k,
            fstride=1,
            batch_norm=True,
            depthwise=p.conv_depthwise,
        )
        self.conv2p = ConvKxF(
            layer_width * wf**2,
            layer_width * wf**2,
            k=1,
            f=1,
            fstride=1,
            batch_norm=True,
        )
        self.convt2 = ConvKxF(
            layer_width * wf**2,
            layer_width * wf**1,
            k=k,
            batch_norm=True,
            depthwise=p.convt_depthwise,
            mode=p.conv_dec_mode,
        )
        self.conv1p = ConvKxF(
            layer_width * wf**1,
            layer_width * wf**1,
            k=1,
            f=1,
            fstride=1,
            batch_norm=True,
        )
        self.convt1 = ConvKxF(
            layer_width * wf**1,
            layer_width * wf**0,
            k=k,
            batch_norm=True,
            depthwise=p.convt_depthwise,
            mode=p.conv_dec_mode,
        )
        self.conv0p = ConvKxF(
            layer_width, layer_width, k=1, f=1, fstride=1, batch_norm=True
        )
        self.conv0_out = ConvKxF(
            layer_width, 1, k=k, fstride=1, batch_norm=False, act="sigmoid"
        )

    def __call__(
        self, emb: mx.array, e3: mx.array, e2: mx.array, e1: mx.array, e0: mx.array
    ) -> mx.array:
        b, _, t, f8 = e3.shape
        emb = self.fc_emb(emb)
        emb = emb.reshape(b, t, -1, f8)
        emb = mx.transpose(emb, (0, 2, 1, 3))

        p3, emb = self._align(self.conv3p(e3), emb)
        e3 = self.convt3(p3 + emb)
        p2, e3 = self._align(self.conv2p(e2), e3)
        e2 = self.convt2(p2 + e3)
        p1, e2 = self._align(self.conv1p(e1), e2)
        e1 = self.convt1(p1 + e2)
        p0, e1 = self._align(self.conv0p(e0), e1)
        m = self.conv0_out(p0 + e1)
        return m

    @staticmethod
    def _align(a: mx.array, b: mx.array) -> Tuple[mx.array, mx.array]:
        t = min(a.shape[2], b.shape[2])
        f = min(a.shape[3], b.shape[3])
        return a[:, :, :t, :f], b[:, :, :t, :f]


class DfDecoderV1(nn.Module):
    """V1 deep-filtering decoder producing complex DF coefficients."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        p = config
        layer_width = p.conv_ch

        self.df_order = p.df_order
        self.df_bins = p.nb_df

        self.clc_convp = ConvKxF(
            layer_width,
            self.df_order * 2,
            k=1,
            f=1,
            fstride=1,
            batch_norm=True,
            complex_in=True,
        )
        self.clc_gru = GroupedGRU(
            p.emb_hidden_dim,
            p.df_hidden_dim,
            num_layers=p.df_num_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.clc_fc_out = nn.Sequential(
            nn.Linear(p.df_hidden_dim, self.df_bins * self.df_order * 2), nn.Tanh()
        )
        self.clc_fc_a = nn.Sequential(nn.Linear(p.df_hidden_dim, 1), nn.Sigmoid())

    def __call__(self, emb: mx.array, c0: mx.array):
        b, t, _ = emb.shape
        c, _ = self.clc_gru(mx.transpose(emb, (1, 0, 2)))  # [T, B, H]
        c0p = mx.transpose(self.clc_convp(c0), (0, 2, 1, 3))  # [B, T, O*2, F]

        c = mx.transpose(c, (1, 0, 2))
        alpha = self.clc_fc_a(c)
        coefs = self.clc_fc_out(c)
        coefs = coefs.reshape(b, t, self.df_order * 2, self.df_bins)
        coefs = (coefs + c0p).reshape(b, t, self.df_order, 2, self.df_bins)
        coefs = mx.transpose(coefs, (0, 1, 2, 4, 3))  # [B, T, O, F, 2]
        return coefs, alpha


class DfNetV1(nn.Module):
    """DeepFilterNet v1 network."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        self.config = config
        self.freq_bins = config.fft_size // 2 + 1
        self.erb_fb = mx.zeros((self.freq_bins, config.nb_erb))
        self.enc = EncoderV1(config)
        self.erb_dec = ErbDecoderV1(config)
        self.mask = Mask()
        self.clc_dec = DfDecoderV1(config)
        self.df_op = DeepFilterOp(config.nb_df, config.df_order, config.df_lookahead)

    def __call__(self, spec: mx.array, feat_erb: mx.array, feat_spec: mx.array):
        # feat_spec: [B, 1, T, F, 2] -> [B, 2, T, F]
        feat_spec = feat_spec.squeeze(1)
        feat_spec = mx.transpose(feat_spec, (0, 3, 1, 2))

        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)
        m = self._align_time(m, spec.shape[2], fill_value=1.0)
        spec_m = self.mask(spec, m)

        df_coefs_bt = self.clc_dec(emb, c0)
        df_coefs, df_alpha = df_coefs_bt
        df_coefs = mx.transpose(df_coefs, (0, 2, 1, 3, 4))  # [B, O, T, F, 2]
        df_coefs = self._align_time(
            df_coefs, spec.shape[2], fill_value=0.0, time_axis=2
        )
        df_alpha = self._align_time(
            df_alpha, spec.shape[2], fill_value=0.0, time_axis=1
        )
        spec_e = self.df_op(spec_m, df_coefs, alpha=df_alpha)

        return spec_e, m, lsnr, df_coefs

    @staticmethod
    def _align_time(
        x: mx.array,
        target_t: int,
        fill_value: float = 0.0,
        time_axis: int = 2,
    ) -> mx.array:
        t = x.shape[time_axis]
        if t == target_t:
            return x
        if t > target_t:
            slices = [slice(None)] * x.ndim
            slices[time_axis] = slice(0, target_t)
            return x[tuple(slices)]

        pad_shape = list(x.shape)
        pad_shape[time_axis] = target_t - t
        pad = mx.full(tuple(pad_shape), fill_value, dtype=x.dtype)
        return mx.concatenate([x, pad], axis=time_axis)

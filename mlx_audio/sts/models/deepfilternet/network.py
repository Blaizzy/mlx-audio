"""
DeepFilterNet neural network architecture for MLX.

Exact match of PyTorch implementation for direct weight loading.

Based on the DeepFilterNet architecture by Hendrik Schröter et al.
https://github.com/Rikorose/DeepFilterNet
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import DeepFilterNetConfig


class GroupedLinearEinsum(nn.Module):
    """Grouped linear layer. Weight shape: (groups, ws, hs)"""

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        self.groups = groups
        self.ws = input_size // groups
        self.hs = hidden_size // groups
        self.weight = mx.zeros((groups, self.ws, self.hs))

    def __call__(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        x = x.reshape(b, t, self.groups, self.ws)
        x = mx.einsum("btgi,gih->btgh", x, self.weight)
        return x.reshape(b, t, self.groups * self.hs)


class PyTorchGRU(nn.Module):
    """GRU implementation matching PyTorch's behavior exactly.

    PyTorch applies bias_hh even when h=None (uses h=zeros internally).
    This implementation matches PyTorch's GRU equations:

    r = σ(W_ir @ x + b_ir + W_hr @ h + b_hr)
    z = σ(W_iz @ x + b_iz + W_hz @ h + b_hz)
    n = tanh(W_in @ x + b_in + r ⊙ (W_hn @ h + b_hn))
    h' = (1 - z) ⊙ n + z ⊙ h

    Key difference from MLX's GRU: PyTorch applies bias_hh to ALL gates (r, z, n),
    while MLX's default only applies it to the n gate.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        scale = 1.0 / math.sqrt(hidden_size)

        # Weights: Wx = weight_ih, Wh = weight_hh (PyTorch naming)
        # Shape: [3*H, input_size] and [3*H, hidden_size]
        self.Wx = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, input_size)
        )
        self.Wh = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, hidden_size)
        )

        if bias:
            # b = bias_ih [3*H], bhn = bias_hh [3*H] - FULL bias for all gates!
            self.b = mx.random.uniform(low=-scale, high=scale, shape=(3 * hidden_size,))
            self.bhn = mx.random.uniform(
                low=-scale, high=scale, shape=(3 * hidden_size,)
            )
        else:
            self.b = None
            self.bhn = None

    def __call__(self, x: mx.array, hidden: Optional[mx.array] = None) -> mx.array:
        """Forward pass. Input shape: [T, B, H]. Returns [T, B, H]."""
        H = self.hidden_size

        # Input projection with bias_ih
        if self.b is not None:
            gates_x = mx.addmm(self.b, x, self.Wx.T)  # [T, B, 3*H]
        else:
            gates_x = x @ self.Wx.T

        # Split into [r, z, n] components
        gates_x_r = gates_x[..., :H]  # [T, B, H]
        gates_x_z = gates_x[..., H : 2 * H]  # [T, B, H]
        gates_x_n = gates_x[..., 2 * H :]  # [T, B, H]

        # Split bias_hh into [b_hr, b_hz, b_hn]
        if self.bhn is not None:
            bhn_r = self.bhn[:H]
            bhn_z = self.bhn[H : 2 * H]
            bhn_n = self.bhn[2 * H :]
        else:
            bhn_r = bhn_z = bhn_n = None

        all_hidden = []
        h = hidden

        for t_idx in range(x.shape[0]):
            # Get input gates for this timestep
            r_x = gates_x_r[t_idx, ...]  # [B, H]
            z_x = gates_x_z[t_idx, ...]  # [B, H]
            n_x = gates_x_n[t_idx, ...]  # [B, H]

            if h is not None:
                # Hidden projection: h @ Wh.T
                gates_h = h @ self.Wh.T  # [B, 3*H]
                r_h = gates_h[..., :H]  # [B, H]
                z_h = gates_h[..., H : 2 * H]  # [B, H]
                n_h = gates_h[..., 2 * H :]  # [B, H]

                # Add bias_hh to each gate's hidden contribution
                if bhn_r is not None:
                    r_h = r_h + bhn_r
                    z_h = z_h + bhn_z
                    n_h = n_h + bhn_n

                # Compute gates
                r = mx.sigmoid(r_x + r_h)
                z = mx.sigmoid(z_x + z_h)
            else:
                # PyTorch uses h=zeros when None, so still applies bias_hh!
                # r = σ(r_x + 0 @ Wh_r.T + bhr) = σ(r_x + bhr)
                # z = σ(z_x + 0 @ Wh_z.T + bhz) = σ(z_x + bhz)
                # n_h = bhn (used with r below)
                if bhn_r is not None:
                    r = mx.sigmoid(r_x + bhn_r)
                    z = mx.sigmoid(z_x + bhn_z)
                    n_h = mx.broadcast_to(bhn_n, r_x.shape)  # [B, H]
                else:
                    r = mx.sigmoid(r_x)
                    z = mx.sigmoid(z_x)
                    n_h = mx.zeros_like(n_x)

            # n gate: n = tanh(n_x + r * n_h)
            n = mx.tanh(n_x + r * n_h)

            # New hidden: h' = (1 - z) * n + z * h
            if h is not None:
                h_new = (1 - z) * n + z * h
            else:
                h_new = (1 - z) * n

            all_hidden.append(h_new)
            h = h_new

        return mx.stack(all_hidden, axis=0)


class SqueezedGRU(nn.Module):
    """GRU with input/output projections matching PyTorch SqueezedGRU_S."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups),
            nn.ReLU(),
        )
        # Use MLX native GRU kernels (batch-first) for better runtime.
        self.gru_layers = [nn.GRU(hidden_size, hidden_size) for _ in range(num_layers)]
        self.linear_out = (
            nn.Sequential(
                GroupedLinearEinsum(
                    hidden_size, output_size or hidden_size, linear_groups
                ),
                nn.ReLU(),
            )
            if output_size
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_in(x)
        for gru in self.gru_layers:
            h0 = mx.zeros((x.shape[0], self.hidden_size), dtype=x.dtype)
            x = gru(x, h0)
        if self.linear_out:
            x = self.linear_out(x)
        return x


class Encoder(nn.Module):
    """Encoder matching PyTorch structure exactly."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        p = config
        self.erb_bins = p.nb_erb
        # DeepFilterNet2 uses a different embedding pathway shape than DeepFilterNet3.
        self._is_v2_style = p.enc_concat
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = (
            p.emb_hidden_dim if self._is_v2_style else (p.conv_ch * p.nb_erb // 4)
        )
        self.enc_concat = p.enc_concat
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

        # Create conv layers as dict to match PyTorch Sequential indices
        # erb_conv0: separable=False (gcd(1,64)=1), indices: 0=pad, 1=conv, 2=bn, 3=relu
        self.erb_conv0 = self._make_conv(1, p.conv_ch, p.conv_kernel_inp, False)
        # erb_conv1/2/3: separable=True, indices: 0=pad, 1=conv, 2=pointwise, 3=bn, 4=relu
        self.erb_conv1 = self._make_conv(
            p.conv_ch, p.conv_ch, p.conv_kernel, True, fstride=2
        )
        self.erb_conv2 = self._make_conv(
            p.conv_ch, p.conv_ch, p.conv_kernel, True, fstride=2
        )
        self.erb_conv3 = self._make_conv(
            p.conv_ch, p.conv_ch, p.conv_kernel, True, fstride=1
        )

        # df_conv0: separable=True (gcd(2,64)=2), indices: 0=pad, 1=conv, 2=pointwise, 3=bn, 4=relu
        self.df_conv0 = self._make_conv(2, p.conv_ch, p.conv_kernel_inp, True)
        self.df_conv1 = self._make_conv(
            p.conv_ch, p.conv_ch, p.conv_kernel, True, fstride=2
        )

        # Linear layers
        self.df_fc_emb = nn.Sequential(
            GroupedLinearEinsum(
                p.conv_ch * p.nb_df // 2, self.emb_in_dim, p.enc_linear_groups
            ),
            nn.ReLU(),
        )

        emb_gru_in_dim = self.emb_in_dim * 2 if self.enc_concat else self.emb_in_dim
        emb_gru_out_size = None if self._is_v2_style else self.emb_out_dim
        self.emb_gru = SqueezedGRU(
            emb_gru_in_dim, self.emb_dim, emb_gru_out_size, 1, p.linear_groups
        )

        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())

    def _make_conv(
        self,
        in_ch: int,
        out_ch: int,
        kernel: List[int],
        separable: bool,
        fstride: int = 1,
    ):
        """Create conv layer dict matching PyTorch Sequential indices."""
        # PyTorch order: calculate groups, then check separable conditions
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel) == 1:
            separable = False  # No pointwise after 1x1 conv, but keep groups

        layer = {}
        # Index 0: padding (no weights)
        # Index 1: conv
        layer["1"] = ConvBlock(in_ch, out_ch, tuple(kernel), groups, fstride)

        if groups > 1:
            # Index 2: pointwise conv
            layer["2"] = ConvBlock(out_ch, out_ch, (1, 1), 1, 1)
            # Index 3: bn
            layer["3"] = BatchNorm(out_ch)
        else:
            # Index 2: bn
            layer["2"] = BatchNorm(out_ch)

        return layer

    def __call__(self, feat_erb: mx.array, feat_spec: mx.array):
        # ERB encoding
        e0 = self._apply_conv(self.erb_conv0, feat_erb)
        e1 = self._apply_conv(self.erb_conv1, e0)
        e2 = self._apply_conv(self.erb_conv2, e1)
        e3 = self._apply_conv(self.erb_conv3, e2)

        # DF encoding
        c0 = self._apply_conv(self.df_conv0, feat_spec)
        c1 = self._apply_conv(self.df_conv1, c0)

        # Combine
        cemb = mx.transpose(c1, (0, 2, 3, 1)).reshape(c1.shape[0], c1.shape[2], -1)
        cemb = self.df_fc_emb(cemb)
        emb = mx.transpose(e3, (0, 2, 3, 1)).reshape(e3.shape[0], e3.shape[2], -1)
        if self.enc_concat:
            emb = mx.concatenate([emb, cemb], axis=-1)
        else:
            emb = emb + cemb

        emb = self.emb_gru(emb)
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr

    def _apply_conv(self, layer: dict, x: mx.array) -> mx.array:
        """Apply conv layer from dict."""
        x = layer["1"](x)  # Main conv

        # Check if separable (has pointwise conv at index '2')
        if "3" in layer:
            # Separable: '2' is pointwise, '3' is BatchNorm
            x = layer["2"](x)
            x = layer["3"].norm(x)
        else:
            # Non-separable: '2' is BatchNorm
            x = layer["2"].norm(x)

        return nn.relu(x)


class ErbDecoder(nn.Module):
    """ERB decoder matching PyTorch structure."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        p = config
        self.emb_in_dim = (
            p.emb_hidden_dim if p.enc_concat else (p.conv_ch * p.nb_erb // 4)
        )
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4

        self.emb_gru = SqueezedGRU(
            self.emb_in_dim,
            self.emb_dim,
            self.emb_out_dim,
            max(1, p.emb_num_layers - 1),
            p.linear_groups,
        )

        # Pathway convs (1x1, no activation)
        self.conv3p = self._make_pathway_conv(p.conv_ch)
        self.conv2p = self._make_pathway_conv(p.conv_ch)
        self.conv1p = self._make_pathway_conv(p.conv_ch)
        self.conv0p = self._make_pathway_conv(p.conv_ch)

        # Transposed convs
        # Matches PyTorch: convt3 is a regular conv block (not transposed).
        self.convt3 = self._make_regular_conv(p.conv_ch, p.convt_kernel)
        self.convt2 = self._make_transpose_conv(p.conv_ch, p.convt_kernel, fstride=2)
        self.convt1 = self._make_transpose_conv(p.conv_ch, p.convt_kernel, fstride=2)

        # Output conv
        self.conv0_out = self._make_output_conv(p.conv_ch, p.convt_kernel)

    def _make_pathway_conv(self, ch: int):
        """1x1 conv + bn, no activation. Uses depthwise groups for 1x1."""
        groups = ch  # depthwise for 1x1 conv with same in/out channels
        return {"0": ConvBlock(ch, ch, (1, 1), groups, 1), "1": BatchNorm(ch)}

    def _make_regular_conv(self, ch: int, kernel: List[int]):
        """Regular decoder conv block."""
        return {
            "0": ConvBlock(ch, ch, tuple(kernel), ch, 1),
            "1": ConvBlock(ch, ch, (1, 1), 1, 1),
            "2": BatchNorm(ch),
        }

    def _make_transpose_conv(self, ch: int, kernel: List[int], fstride: int = 1):
        """Transposed conv with upsample + conv."""
        return {
            "0": ConvTransposeBlock(ch, ch, tuple(kernel), ch, fstride),
            "1": ConvBlock(ch, ch, (1, 1), 1, 1),
            "2": BatchNorm(ch),
        }

    def _make_output_conv(self, in_ch: int, kernel: List[int]):
        """Output conv with sigmoid activation."""
        return {"0": ConvBlock(in_ch, 1, kernel, 1, 1), "1": BatchNorm(1)}

    def __call__(
        self, emb: mx.array, e3: mx.array, e2: mx.array, e1: mx.array, e0: mx.array
    ):
        b, t = emb.shape[:2]
        f8 = e3.shape[3]

        emb = self.emb_gru(emb)
        emb = emb.reshape(b, t, f8, -1)
        emb = mx.transpose(emb, (0, 3, 1, 2))

        d3 = self._apply_pathway(self.conv3p, e3) + emb
        d3 = nn.relu(self._apply_transpose(self.convt3, d3))
        d2 = self._apply_pathway(self.conv2p, e2) + d3
        d2 = nn.relu(self._apply_transpose(self.convt2, d2))
        d1 = self._apply_pathway(self.conv1p, e1) + d2
        d1 = nn.relu(self._apply_transpose(self.convt1, d1))
        d0 = self._apply_pathway(self.conv0p, e0) + d1
        m = mx.sigmoid(self._apply_output(self.conv0_out, d0))

        return m

    def _apply_pathway(self, layer: dict, x: mx.array) -> mx.array:
        x = layer["0"](x)
        x = layer["1"].norm(x)
        return nn.relu(x)

    def _apply_transpose(self, layer: dict, x: mx.array) -> mx.array:
        x = layer["0"](x)
        x = layer["1"](x)
        x = layer["2"].norm(x)
        return x

    def _apply_output(self, layer: dict, x: mx.array) -> mx.array:
        x = layer["0"](x)
        x = layer["1"].norm(x)
        return x


class DfDecoder(nn.Module):
    """Deep filtering decoder."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        p = config
        self.emb_in_dim = (
            p.emb_hidden_dim if p.enc_concat else (p.conv_ch * p.nb_erb // 4)
        )
        self.emb_dim = p.df_hidden_dim
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2

        self.df_convp = {
            "1": ConvBlock(
                p.conv_ch,
                self.df_out_ch,
                (p.df_pathway_kernel_size_t, 1),
                math.gcd(p.conv_ch, self.df_out_ch),
                1,
            ),
            "2": ConvBlock(self.df_out_ch, self.df_out_ch, (1, 1), 1, 1),
            "3": BatchNorm(self.df_out_ch),
        }

        # DeepFilterNet2/3 checkpoints both use 8 groups for df_gru.linear_in.
        self.df_gru = SqueezedGRU(
            self.emb_in_dim, self.emb_dim, None, p.df_num_layers, 8
        )
        self.df_skip = (
            GroupedLinearEinsum(self.emb_in_dim, self.emb_dim, p.linear_groups)
            if p.df_gru_skip == "groupedlinear"
            else None
        )
        self.df_out = nn.Sequential(
            GroupedLinearEinsum(
                self.emb_dim, self.df_bins * self.df_out_ch, p.linear_groups
            ),
            nn.Tanh(),
        )
        self.df_fc_a = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

    def __call__(self, emb: mx.array, c0: mx.array):
        b, t = emb.shape[:2]

        c = self.df_gru(emb)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        c0 = self._apply_convp(c0)
        c0 = mx.transpose(c0, (0, 2, 3, 1))  # [B, T, F, O*2]

        c_out = self.df_out(c)
        c_out = c_out.reshape(b, t, self.df_bins, self.df_out_ch) + c0

        return c_out

    def _apply_convp(self, x: mx.array) -> mx.array:
        x = self.df_convp["1"](x)
        x = self.df_convp["2"](x)
        x = self.df_convp["3"].norm(x)
        return nn.relu(x)


class ConvBlock(nn.Module):
    """2D convolution block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: tuple,
        groups: int,
        fstride: int,
        upsample: int = 1,
        lookahead: int = 0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.kernel = kernel
        self.groups = groups
        self.fstride = fstride
        self.upsample = upsample
        raw_left = kernel[0] - 1 - lookahead
        self.time_crop = max(0, -raw_left)
        left = max(0, raw_left)
        right = max(0, lookahead)
        self.time_pad = (left, right)
        self.freq_pad = kernel[1] // 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias

        # Weight: stored in PyTorch format [out_ch, in_ch // groups, kH, kW]
        # Will be transposed for MLX conv2d which expects [out_ch, kH, kW, in_ch // groups]
        self.weight = mx.zeros((out_ch, in_ch // groups, kernel[0], kernel[1]))
        self.bias = mx.zeros((out_ch,)) if use_bias else None
        self._weight_ohwi: Optional[mx.array] = None
        self._weight_token: Optional[int] = None

    def _weight_cache(self) -> mx.array:
        token = id(self.weight)
        if self._weight_ohwi is None or self._weight_token != token:
            self._weight_ohwi = mx.transpose(self.weight, (0, 2, 3, 1))
            self._weight_token = token
        return self._weight_ohwi

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T, F]
        x = mx.transpose(x, (0, 2, 3, 1))  # [B, T, F, C]

        if self.upsample > 1:
            x = mx.repeat(x, self.upsample, axis=2)

        if self.time_crop > 0:
            x = x[:, self.time_crop :, :, :]

        x = mx.pad(
            x,
            [
                (0, 0),
                (self.time_pad[0], self.time_pad[1]),
                (self.freq_pad, self.freq_pad),
                (0, 0),
            ],
        )

        # Use native grouped conv2d instead of Python per-group loops.
        x = mx.conv2d(
            x, self._weight_cache(), stride=(1, self.fstride), groups=self.groups
        )
        if self.bias is not None:
            x = x + self.bias.reshape((1, 1, 1, -1))

        x = mx.transpose(x, (0, 3, 1, 2))  # [B, C, T, F]
        return x


class ConvTransposeBlock(nn.Module):
    """2D transposed convolution block matching PyTorch ConvTranspose2d semantics."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: tuple, groups: int, fstride: int
    ):
        super().__init__()
        self.kernel = kernel
        self.groups = groups
        self.fstride = fstride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.padding = (kernel[0] - 1, kernel[1] // 2)
        self.output_padding = (0, kernel[1] // 2)
        # PyTorch ConvTranspose2d layout: [in_ch, out_ch // groups, kH, kW]
        self.weight = mx.zeros((in_ch, out_ch // groups, kernel[0], kernel[1]))
        self._weight_token: Optional[int] = None
        self._dense_weight: Optional[mx.array] = None
        self._group_weights: Optional[List[mx.array]] = None

    def _refresh_weight_cache(self) -> None:
        token = id(self.weight)
        if self._weight_token == token:
            return
        self._weight_token = token
        self._dense_weight = None
        self._group_weights = None

        if self.groups <= 1:
            self._dense_weight = mx.transpose(self.weight, (1, 2, 3, 0))
            return

        in_pg = self.in_ch // self.groups
        out_pg = self.out_ch // self.groups
        total_in = in_pg * self.groups
        if in_pg == 0 or out_pg == 0:
            return

        blocks = []
        grouped = []
        for g in range(self.groups):
            w_g = self.weight[
                g * in_pg : (g + 1) * in_pg, :, :, :
            ]  # [I_pg, O_pg, kT, kF]
            w_t = mx.transpose(w_g, (1, 2, 3, 0))  # [O_pg, kT, kF, I_pg]
            grouped.append(w_t)
            left = mx.zeros(
                (out_pg, w_t.shape[1], w_t.shape[2], g * in_pg), dtype=w_t.dtype
            )
            right = mx.zeros(
                (out_pg, w_t.shape[1], w_t.shape[2], total_in - (g + 1) * in_pg),
                dtype=w_t.dtype,
            )
            blocks.append(mx.concatenate([left, w_t, right], axis=3))
        self._group_weights = grouped
        self._dense_weight = mx.concatenate(blocks, axis=0)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T, F] -> [B, T, F, C]
        x = mx.transpose(x, (0, 2, 3, 1))
        self._refresh_weight_cache()

        if self._dense_weight is not None:
            x = mx.conv_transpose2d(
                x,
                self._dense_weight,
                stride=(1, self.fstride),
                padding=self.padding,
                output_padding=self.output_padding,
            )
        else:
            # MLX currently supports only groups=1 for conv_transpose2d, so run per-group.
            in_pg = self.in_ch // self.groups
            outs = []
            weights = self._group_weights or []
            for g in range(self.groups):
                x_g = x[:, :, :, g * in_pg : (g + 1) * in_pg]
                w_g = weights[g]
                y_g = mx.conv_transpose2d(
                    x_g,
                    w_g,
                    stride=(1, self.fstride),
                    padding=self.padding,
                    output_padding=self.output_padding,
                )
                outs.append(y_g)
            x = mx.concatenate(outs, axis=-1)

        # [B, T, F, C] -> [B, C, T, F]
        x = mx.transpose(x, (0, 3, 1, 2))
        return x


class BatchNorm(nn.Module):
    """BatchNorm for inference."""

    def __init__(self, num_features: int):
        super().__init__()
        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)
        self.running_mean = mx.zeros(num_features)
        self.running_var = mx.ones(num_features)

    def norm(self, x: mx.array, eps: float = 1e-5) -> mx.array:
        # x: [B, C, T, F]
        x = mx.transpose(x, (0, 2, 3, 1))  # [B, T, F, C]
        x = (x - self.running_mean) / mx.sqrt(self.running_var + eps)
        x = x * self.weight + self.bias
        x = mx.transpose(x, (0, 3, 1, 2))  # [B, C, T, F]
        return x


class Mask(nn.Module):
    """ERB mask application."""

    def __init__(self):
        super().__init__()
        self.erb_inv_fb = mx.zeros((32, 481))  # Will be loaded

    def __call__(self, spec: mx.array, mask: mx.array) -> mx.array:
        mask = mask @ self.erb_inv_fb
        mask = mx.expand_dims(mask, axis=-1)
        return spec * mask


class DeepFilterOp(nn.Module):
    """Deep filtering operation matching PyTorch's MF.DF exactly."""

    def __init__(self, df_bins: int, df_order: int, lookahead: int = 0):
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        self.lookahead = lookahead

    def __call__(
        self, spec: mx.array, coefs: mx.array, alpha: Optional[mx.array] = None
    ) -> mx.array:
        # spec: [B, 1, T, F, 2], coefs: [B, O, T, F_df, 2]
        b, _, t, _, _ = spec.shape

        # Padding: (df_order - 1 - lookahead) on left, lookahead on right
        # This matches PyTorch's ConstantPad2d((0, 0, frame_size - 1 - lookahead, lookahead), 0.0)
        pad_left = self.df_order - 1 - self.lookahead
        pad_right = self.lookahead

        # Remove channel dimension for processing
        spec_df = spec[:, 0, :, : self.df_bins, :]  # [B, T, F_df, 2]

        # Pad along time dimension: [(0,0), (pad_left, pad_right), (0,0), (0,0)]
        spec_padded = mx.pad(spec_df, [(0, 0), (pad_left, pad_right), (0, 0), (0, 0)])
        # spec_padded shape: [B, T + pad_left + pad_right, F_df, 2]

        # Loop over df_order (small: 5), not over time (large), to avoid
        # building an O(T) stack of sliding windows.
        out_r = mx.zeros((b, t, self.df_bins), dtype=spec.dtype)
        out_i = mx.zeros((b, t, self.df_bins), dtype=spec.dtype)
        for k in range(self.df_order):
            window = spec_padded[:, k : (k + t), :, :]  # [B, T, F_df, 2]
            sr = window[..., 0]
            si = window[..., 1]
            cr = coefs[:, k, :, :, 0]
            ci = coefs[:, k, :, :, 1]
            out_r = out_r + (sr * cr - si * ci)
            out_i = out_i + (sr * ci + si * cr)

        spec_f = mx.expand_dims(
            mx.stack([out_r, out_i], axis=-1), axis=1
        )  # [B, 1, T, F, 2]

        # Match libDF assign_df: only replace/blend first df_bins bins; keep others unchanged.
        out = spec
        if alpha is not None:
            a = mx.reshape(alpha, (b, 1, t, 1, 1))
            low = spec_f * a + spec[:, :, :, : self.df_bins, :] * (1 - a)
        else:
            low = spec_f
        out = mx.concatenate([low, spec[:, :, :, self.df_bins :, :]], axis=3)
        return out


class DfNet(nn.Module):
    """Main DeepFilterNet architecture."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        p = config
        self.config = config

        self.conv_lookahead = p.conv_lookahead
        self.df_lookahead = p.df_lookahead
        self.erb_fb = mx.zeros((p.fft_size // 2 + 1, p.nb_erb))
        self.enc = Encoder(p)
        self.erb_dec = ErbDecoder(p)
        self.mask = Mask()
        self.df_dec = DfDecoder(p)
        self.df_order = p.df_order
        self.df_op = DeepFilterOp(p.nb_df, p.df_order, p.df_lookahead)
        self.nb_df = p.nb_df
        self.freq_bins = p.fft_size // 2 + 1

    @staticmethod
    def _apply_lookahead(x: mx.array, lookahead: int, time_axis: int = 2) -> mx.array:
        """Match PyTorch ConstantPad2d((0,0,-lookahead,lookahead),0) behavior."""
        if lookahead <= 0:
            return x
        if x.shape[time_axis] <= lookahead:
            return x

        slices = [slice(None)] * x.ndim
        slices[time_axis] = slice(lookahead, None)
        shifted = x[tuple(slices)]
        pad_shape = list(x.shape)
        pad_shape[time_axis] = lookahead
        pad = mx.zeros(tuple(pad_shape), dtype=x.dtype)
        return mx.concatenate([shifted, pad], axis=time_axis)

    def __call__(self, spec: mx.array, feat_erb: mx.array, feat_spec: mx.array):
        # feat_spec comes in as [B, 1, T, F', 2] like PyTorch, needs reshape to [B, 2, T, F']
        b = feat_spec.shape[0]
        feat_spec = feat_spec.squeeze(1)  # [B, T, F', 2]
        feat_spec = mx.transpose(feat_spec, (0, 3, 1, 2))  # [B, 2, T, F']

        feat_erb = self._apply_lookahead(feat_erb, self.conv_lookahead, time_axis=2)
        feat_spec = self._apply_lookahead(feat_spec, self.conv_lookahead, time_axis=2)

        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)

        m = self.erb_dec(emb, e3, e2, e1, e0)
        spec_m = self.mask(spec, m)

        df_coefs = self.df_dec(emb, c0)
        b, t = df_coefs.shape[:2]
        df_coefs = df_coefs.reshape(b, t, self.nb_df, self.df_order, 2)
        df_coefs = mx.transpose(df_coefs, (0, 3, 1, 2, 4))

        # DeepFilterNet2 and DeepFilterNet3 behave slightly differently here in this MLX port:
        # - DF2 path matches better when DF sees masked spectrum directly.
        # - DF3 path matches better with legacy low-bin DF + masked high-bin fusion.
        if self.config.enc_concat:
            spec_e = self.df_op(spec_m, df_coefs)
        else:
            spec_df = self.df_op(spec, df_coefs)
            spec_e = mx.concatenate(
                [spec_df[:, :, :, : self.nb_df, :], spec_m[:, :, :, self.nb_df :, :]],
                axis=3,
            )

        return spec_e, m, lsnr, df_coefs

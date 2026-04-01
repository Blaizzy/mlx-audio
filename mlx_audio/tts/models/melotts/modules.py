"""Core flow and encoder modules for MeloTTS (VITS2-based)."""

import mlx.core as mx
import mlx.nn as nn

from .attentions import Encoder
from .hifigan import Conv1dPT
from .transforms import piecewise_rational_quadratic_transform


def sequence_mask(lengths, max_len=None):
    """Create (B, 1, T) float mask from sequence lengths."""
    if max_len is None:
        max_len = int(lengths.max().item())
    mask = mx.arange(max_len)[None, :] < lengths[:, None]
    return mask[:, None, :].astype(mx.float32)


class WN(nn.Module):
    """WaveNet module with dilated convolutions and gated activations."""

    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = []
        self.res_skip_layers = []

        if gin_channels > 0:
            self.cond_layer = Conv1dPT(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                Conv1dPT(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            self.res_skip_layers.append(
                Conv1dPT(hidden_channels, res_skip_channels, 1)
            )

    def __call__(self, x, x_mask, g=None):
        output = mx.zeros_like(x)

        if g is not None and self.gin_channels > 0:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            t_act = mx.tanh(x_in[:, : self.hidden_channels, :])
            s_act = mx.sigmoid(x_in[:, self.hidden_channels :, :])
            acts = t_act * s_act

            res_skip = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                res = res_skip[:, : self.hidden_channels, :]
                skip = res_skip[:, self.hidden_channels :, :]
                x = (x + res) * x_mask
                output = output + skip
            else:
                output = output + res_skip

        return output * x_mask


class ResidualCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows."""

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = Conv1dPT(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
            p_dropout=p_dropout,
        )
        post_out = self.half_channels * (1 if mean_only else 2)
        self.post = Conv1dPT(hidden_channels, post_out, 1)
        self.post.conv.weight = mx.zeros_like(self.post.conv.weight)

    def __call__(self, x, x_mask, g=None, reverse=False):
        x0 = x[:, : self.half_channels, :]
        x1 = x[:, self.half_channels :, :]

        h = self.pre(x0)
        h = self.enc(h, x_mask, g=g)
        h = self.post(h)

        if not self.mean_only:
            m = h[:, : self.half_channels, :]
            logs = h[:, self.half_channels :, :]
        else:
            m = h
            logs = mx.zeros_like(m)

        if not reverse:
            x1 = m + x1 * mx.exp(logs) * x_mask
            x = mx.concatenate([x0, x1], axis=1)
            logdet = mx.sum(logs * x_mask)
            return x, logdet
        else:
            x1 = (x1 - m) * mx.exp(-logs) * x_mask
            x = mx.concatenate([x0, x1], axis=1)
            return x


class TransformerCouplingLayer(nn.Module):
    """Transformer-based coupling layer for normalizing flows."""

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        n_heads,
        p_dropout=0,
        filter_channels=0,
        mean_only=False,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = Conv1dPT(self.half_channels, hidden_channels, 1)
        self.enc = Encoder(
            hidden_channels,
            filter_channels or hidden_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
        )
        post_out = self.half_channels * (1 if mean_only else 2)
        self.post = Conv1dPT(hidden_channels, post_out, 1)
        self.post.conv.weight = mx.zeros_like(self.post.conv.weight)

    def __call__(self, x, x_mask, g=None, reverse=False):
        x0 = x[:, : self.half_channels, :]
        x1 = x[:, self.half_channels :, :]

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        h = self.post(h) * x_mask

        if not self.mean_only:
            m = h[:, : self.half_channels, :]
            logs = h[:, self.half_channels :, :]
        else:
            m = h
            logs = mx.zeros_like(m)

        if not reverse:
            x1 = m + x1 * mx.exp(logs) * x_mask
            x = mx.concatenate([x0, x1], axis=1)
            logdet = mx.sum(logs * x_mask)
            return x, logdet
        else:
            x1 = (x1 - m) * mx.exp(-logs) * x_mask
            x = mx.concatenate([x0, x1], axis=1)
            return x


class PosteriorEncoder(nn.Module):
    """Encodes mel spectrograms to latent space."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.pre = Conv1dPT(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = Conv1dPT(hidden_channels, out_channels * 2, 1)

    def __call__(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, x.shape[2])

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m = stats[:, : self.out_channels, :]
        logs = stats[:, self.out_channels :, :]

        z = (m + mx.random.normal(m.shape) * mx.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Log(nn.Module):
    """Log transform for normalizing flows."""

    def __call__(self, x, x_mask, reverse=False):
        if not reverse:
            y = mx.log(mx.clip(x, a_min=1e-5, a_max=None)) * x_mask
            logdet = mx.sum(-y * x_mask)
            return y, logdet
        else:
            y = mx.exp(x) * x_mask
            return y


class Flip(nn.Module):
    """Channel flip for normalizing flows."""

    def __call__(self, x, *args, reverse=False, **kwargs):
        x = x[:, ::-1, :]
        if not reverse:
            logdet = mx.array(0.0)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    """Learnable element-wise scale and shift for normalizing flows."""

    def __init__(self, channels):
        super().__init__()
        self.m = mx.zeros((channels, 1))
        self.logs = mx.zeros((channels, 1))

    def __call__(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + mx.exp(self.logs) * x
            y = y * x_mask
            logdet = mx.sum(self.logs * x_mask)
            return y, logdet
        else:
            y = (x - self.m) * mx.exp(-self.logs) * x_mask
            return y


class DDSConv(nn.Module):
    """Dilated depth-separable convolutions."""

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = []
        self.convs_1x1 = []
        self.norms_1 = []
        self.norms_2 = []

        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                Conv1dPT(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                    groups=channels,
                )
            )
            self.convs_1x1.append(Conv1dPT(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def __call__(self, x, x_mask, g=None):
        if g is not None:
            x = x + g

        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = y.transpose(0, 2, 1)
            y = self.norms_1[i](y)
            y = y.transpose(0, 2, 1)
            y = nn.gelu(y)

            y = self.convs_1x1[i](y)
            y = y.transpose(0, 2, 1)
            y = self.norms_2[i](y)
            y = y.transpose(0, 2, 1)
            y = nn.gelu(y)

            y = self.drop(y)
            x = x + y

        return x * x_mask


class ConvFlow(nn.Module):
    """Piecewise rational quadratic coupling for stochastic duration predictor."""

    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = Conv1dPT(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = Conv1dPT(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        self.proj.conv.weight = mx.zeros_like(self.proj.conv.weight)

    def __call__(self, x, x_mask, g=None, reverse=False):
        x0 = x[:, : self.half_channels, :]
        x1 = x[:, self.half_channels :, :]

        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        B, C, T = x0.shape
        h = h.reshape(B, self.half_channels, -1, T).transpose(0, 1, 3, 2)

        unnormalized_widths = h[..., : self.num_bins] / mx.sqrt(
            mx.array(float(self.filter_channels))
        )
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / mx.sqrt(
            mx.array(float(self.filter_channels))
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = mx.concatenate([x0, x1], axis=1) * x_mask
        if not reverse:
            logdet = mx.sum(logdet * x_mask)
            return x, logdet
        else:
            return x


class StochasticDurationPredictor(nn.Module):
    """Predicts phoneme durations using a normalizing flow."""

    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()

        self.flows = [ElementwiseAffine(2)]
        for _ in range(n_flows):
            self.flows.append(
                ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(Flip())

        self.post_pre = Conv1dPT(1, filter_channels, 1)
        self.post_proj = Conv1dPT(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(
            filter_channels, kernel_size, 3, p_dropout=p_dropout
        )

        self.post_flows = [ElementwiseAffine(2)]
        for _ in range(n_flows):
            self.post_flows.append(
                ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(Flip())

        self.pre = Conv1dPT(in_channels, filter_channels, 1)
        self.proj = Conv1dPT(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, 3, p_dropout=p_dropout)

        if gin_channels > 0:
            self.cond = Conv1dPT(gin_channels, filter_channels, 1)

    def __call__(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask

            e_q = mx.random.normal(w.shape) * x_mask
            z_q = e_q

            logdet_tot_q = mx.array(0.0)
            for flow in self.post_flows:
                result = flow(z_q, x_mask, g=x)
                if isinstance(result, tuple):
                    z_q, logdet_q = result
                    logdet_tot_q = logdet_tot_q + logdet_q
                else:
                    z_q = result

            z_u, z1 = mx.split(z_q, 2, axis=1)
            u = mx.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask

            logdet_tot_q = logdet_tot_q + mx.sum(
                (nn.log_sigmoid(z_u) + nn.log_sigmoid(-z_u)) * x_mask
            )

            logq = (
                mx.sum(-0.5 * (mx.log(mx.array(2 * 3.141592653589793)) + e_q**2) * x_mask)
                - logdet_tot_q
            )

            logdet_tot = mx.array(0.0)
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot = logdet_tot + logdet
            z = mx.concatenate([z0, z1], axis=1)

            for flow in self.flows:
                result = flow(z, x_mask, g=x)
                if isinstance(result, tuple):
                    z, logdet = result
                    logdet_tot = logdet_tot + logdet
                else:
                    z = result

            nll = (
                mx.sum(0.5 * (mx.log(mx.array(2 * 3.141592653589793)) + z**2) * x_mask)
                - logdet_tot
            )

            return nll + logq
        else:
            z = mx.random.normal((x.shape[0], 2, x.shape[2])) * noise_scale
            for flow in reversed(self.flows):
                z = flow(z, x_mask, g=x, reverse=True)
                if isinstance(z, tuple):
                    z = z[0]

            z0, z1 = mx.split(z, 2, axis=1)
            w = self.log_flow(z0, x_mask, reverse=True)
            logw = mx.log(mx.clip(w, a_min=1e-5, a_max=None)) * x_mask
            return logw


class DurationPredictor(nn.Module):
    """Deterministic duration predictor (simpler alternative to SDP)."""

    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        padding = (kernel_size - 1) // 2

        self.conv_1 = Conv1dPT(in_channels, filter_channels, kernel_size, padding=padding)
        self.norm_1 = nn.LayerNorm(filter_channels)
        self.conv_2 = Conv1dPT(
            filter_channels, filter_channels, kernel_size, padding=padding
        )
        self.norm_2 = nn.LayerNorm(filter_channels)
        self.proj = Conv1dPT(filter_channels, 1, 1)

        self.drop = nn.Dropout(p_dropout)

        if gin_channels > 0:
            self.cond = Conv1dPT(gin_channels, in_channels, 1)

    def __call__(self, x, x_mask, g=None):
        if g is not None:
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = nn.relu(x)
        x = x.transpose(0, 2, 1)
        x = self.norm_1(x)
        x = x.transpose(0, 2, 1)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = nn.relu(x)
        x = x.transpose(0, 2, 1)
        x = self.norm_2(x)
        x = x.transpose(0, 2, 1)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    """Text encoder combining phone, tone, language, and BERT embeddings."""

    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.tone_emb = nn.Embedding(16, hidden_channels)  # num_tones
        self.language_emb = nn.Embedding(10, hidden_channels)  # num_languages
        self.bert_proj = Conv1dPT(1024, hidden_channels, 1)
        self.ja_bert_proj = Conv1dPT(768, hidden_channels, 1)

        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout,
            gin_channels=gin_channels,
        )
        self.proj = Conv1dPT(hidden_channels, out_channels * 2, 1)

    def __call__(self, x, x_lengths, tone, language, bert, ja_bert=None, g=None):
        x_emb = self.emb(x).transpose(0, 2, 1)
        tone_emb = self.tone_emb(tone).transpose(0, 2, 1)
        lang_emb = self.language_emb(language).transpose(0, 2, 1)
        bert_emb = self.bert_proj(bert)
        if ja_bert is not None:
            bert_emb = bert_emb + self.ja_bert_proj(ja_bert)

        x = x_emb + tone_emb + lang_emb + bert_emb

        x_mask = sequence_mask(x_lengths, x.shape[2])

        x = self.encoder(x * x_mask, x_mask, g=g)

        stats = self.proj(x) * x_mask
        m = stats[:, : self.out_channels, :]
        logs = stats[:, self.out_channels :, :]

        return x, m, logs, x_mask

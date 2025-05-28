import mlx.core as mx
import mlx.nn as nn


class Snake(nn.Module):
    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = False
    ):
        super().__init__()

        self.alpha_logscale = alpha_logscale

        self.alpha = (
            mx.zeros(in_features) if alpha_logscale else mx.ones(in_features)
        ) * alpha

    def __call__(self, x: mx.array):
        alpha = self.alpha[None, :, None]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)

        x += (1.0 / (alpha + 1e-9)) * mx.power(mx.sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = False
    ):
        super().__init__()

        self.alpha_logscale = alpha_logscale

        self.alpha = (
            mx.zeros(in_features) if alpha_logscale else mx.ones(in_features)
        ) * alpha
        self.beta = (
            mx.zeros(in_features) if alpha_logscale else mx.ones(in_features)
        ) * alpha

    def __call__(self, x: mx.array):
        alpha = self.alpha[None, None, :]
        beta = self.beta[None, None, :]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)

        x += (1.0 / (beta + 1e-9)) * mx.power(mx.sin(x * alpha), 2)

        return x

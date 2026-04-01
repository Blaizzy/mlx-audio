"""Piecewise rational quadratic spline transforms for normalizing flows."""

import mlx.core as mx
import mlx.nn as nn

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    """Find bin indices for inputs given sorted bin boundaries."""
    inputs = inputs[..., None]
    return mx.sum(inputs >= bin_locations, axis=-1) - 1


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs,
    )
    return outputs, logabsdet


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = mx.zeros_like(inputs)
    logabsdet = mx.zeros_like(inputs)

    unnormalized_derivatives = mx.concatenate(
        [
            mx.ones(unnormalized_derivatives.shape[:-1] + (1,)),
            unnormalized_derivatives,
            mx.ones(unnormalized_derivatives.shape[:-1] + (1,)),
        ],
        axis=-1,
    )

    outputs = mx.where(outside_interval_mask, inputs, outputs)
    logabsdet = mx.where(outside_interval_mask, 0.0, logabsdet)

    outputs_inside, logabsdet_inside = rational_quadratic_spline(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs = mx.where(inside_interval_mask, outputs_inside, outputs)
    logabsdet = mx.where(inside_interval_mask, logabsdet_inside, logabsdet)

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    widths = mx.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = mx.cumsum(widths, axis=-1)
    cumwidths = mx.pad(cumwidths, [(0, 0)] * (cumwidths.ndim - 1) + [(1, 0)])
    cumwidths = (right - left) * cumwidths + left
    cumwidths = cumwidths[..., 0:1] * 0 + cumwidths  # force contiguous
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = mx.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = mx.cumsum(heights, axis=-1)
    cumheights = mx.pad(cumheights, [(0, 0)] * (cumheights.ndim - 1) + [(1, 0)])
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = min_derivative + nn.softplus(unnormalized_derivatives)

    if inverse:
        bin_idx = searchsorted(cumheights + 1e-6, inputs)
    else:
        bin_idx = searchsorted(cumwidths + 1e-6, inputs)

    bin_idx = mx.clip(bin_idx, 0, num_bins - 1)

    # Gather parameters for the relevant bins
    input_cumwidths = _gather(cumwidths, bin_idx)
    input_bin_widths = _gather(widths, bin_idx)
    input_cumheights = _gather(cumheights, bin_idx)
    input_heights = _gather(heights, bin_idx)
    input_delta = input_heights / input_bin_widths
    input_derivatives = _gather(derivatives, bin_idx)
    input_derivatives_plus_one = _gather(derivatives[..., 1:], bin_idx)

    if inverse:
        a = (input_heights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        b = (input_heights) * (input_delta - input_derivatives)
        c = -input_delta * (inputs - input_cumheights)
        a = a + 1e-8

        discriminant = b * b - 4 * a * c
        discriminant = mx.maximum(discriminant, 0)

        root = (2 * c) / (-b - mx.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = (
            input_delta
            * input_delta
            * (
                input_derivatives_plus_one * root * root
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root) * (1 - root)
            )
        )
        logabsdet = mx.log(derivative_numerator + 1e-8) - 2 * mx.log(
            mx.abs(denominator) + 1e-8
        )
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta * theta + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = (
            input_delta
            * input_delta
            * (
                input_derivatives_plus_one * theta * theta
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta) * (1 - theta)
            )
        )
        logabsdet = mx.log(derivative_numerator + 1e-8) - 2 * mx.log(
            mx.abs(denominator) + 1e-8
        )
        return outputs, logabsdet


def _gather(params, indices):
    """Gather elements along the last dimension using indices."""
    idx = mx.clip(indices, 0, params.shape[-1] - 1)
    return mx.take_along_axis(params, mx.expand_dims(idx, -1), axis=-1).squeeze(-1)

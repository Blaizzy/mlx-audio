# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
    alpha_transform_type: str = "cosine",
) -> mx.array:
    """Create a beta schedule based on alpha_bar function."""

    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))

    return mx.array(betas, dtype=mx.float32)


@dataclass
class SchedulerOutput:
    """Output from scheduler step."""

    prev_sample: mx.array
    x0_pred: Optional[mx.array] = None


class DPMSolverMultistepScheduler:
    """DPM-Solver multistep scheduler for fast diffusion sampling.

    Implements DPM-Solver++ algorithm for efficient sampling with multi-order updates.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "v_prediction",
        solver_order: int = 2,
        lower_order_final: bool = True,
        final_sigmas_type: str = "zero",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.solver_order = solver_order
        self.lower_order_final = lower_order_final
        self.final_sigmas_type = final_sigmas_type

        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = mx.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule in ["scaled_linear", "squaredcos_cap_v2"]:
            self.betas = betas_for_alpha_bar(
                num_train_timesteps, alpha_transform_type="cosine"
            )
        elif beta_schedule == "cosine":
            self.betas = betas_for_alpha_bar(
                num_train_timesteps, alpha_transform_type="cosine"
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)

        # DPM-Solver parameters
        self.alpha_t = mx.sqrt(self.alphas_cumprod)
        self.sigma_t = mx.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = mx.log(self.alpha_t) - mx.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        self.init_noise_sigma = 1.0

        # State
        self.num_inference_steps = None
        self.timesteps = None
        self.model_outputs: List[Optional[mx.array]] = [None] * self.solver_order
        self.lower_order_nums = 0
        self._step_index = None

        # Precomputed values for inference
        self._cached_alpha_t: List[float] = []
        self._cached_sigma_t: List[float] = []
        self._cached_lambda: List[float] = []

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    def set_timesteps(self, num_inference_steps: int):
        """Set the number of inference steps."""
        self.num_inference_steps = num_inference_steps

        # Create timesteps - linspace from max to 0
        timestep_values = []
        for i in range(num_inference_steps):
            t = (self.num_train_timesteps - 1) * (1.0 - i / num_inference_steps)
            timestep_values.append(int(round(t)))

        self.timesteps = mx.array(timestep_values, dtype=mx.int32)

        # Precompute values for each inference timestep
        self._cached_alpha_t = []
        self._cached_sigma_t = []
        self._cached_lambda = []

        alpha_t_np = np.array(self.alpha_t.tolist())
        sigma_t_np = np.array(self.sigma_t.tolist())

        for t in timestep_values:
            sigma = np.sqrt((1 - alpha_t_np[t] ** 2) / (alpha_t_np[t] ** 2))
            alpha_t_val = 1.0 / np.sqrt(sigma**2 + 1.0)
            sigma_t_val = sigma * alpha_t_val
            lambda_val = np.log(alpha_t_val) - np.log(sigma_t_val)

            self._cached_alpha_t.append(float(alpha_t_val))
            self._cached_sigma_t.append(float(sigma_t_val))
            self._cached_lambda.append(float(lambda_val))

        # Add final step values
        self._cached_alpha_t.append(1.0)
        self._cached_sigma_t.append(0.0)
        self._cached_lambda.append(float("inf"))

        # Reset state
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self._step_index = None

    def _convert_model_output(
        self,
        model_output: mx.array,
        sample: mx.array,
        step_idx: int,
    ) -> mx.array:
        """Convert model output to x0 prediction based on prediction type."""
        alpha_t = self._cached_alpha_t[step_idx]
        sigma_t = self._cached_sigma_t[step_idx]

        if self.prediction_type == "epsilon":
            # model predicts noise
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == "v_prediction":
            # model predicts v = alpha_t * noise - sigma_t * x0
            x0_pred = alpha_t * sample - sigma_t * model_output
        elif self.prediction_type == "sample":
            x0_pred = model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return x0_pred

    def _dpm_solver_first_order_update(
        self,
        x0_pred: mx.array,
        sample: mx.array,
        step_idx: int,
    ) -> mx.array:
        """First order DPM-Solver++ update."""
        alpha_t = self._cached_alpha_t[step_idx + 1]
        sigma_t_next = self._cached_sigma_t[step_idx + 1]
        sigma_t_curr = self._cached_sigma_t[step_idx]

        lambda_t = self._cached_lambda[step_idx + 1]
        lambda_s = self._cached_lambda[step_idx]
        h = lambda_t - lambda_s

        sigma_ratio = sigma_t_next / sigma_t_curr if sigma_t_curr > 0 else 0.0
        exp_neg_h = math.exp(-h)

        prev_sample = sigma_ratio * sample - alpha_t * (exp_neg_h - 1.0) * x0_pred

        return prev_sample

    def _dpm_solver_second_order_update(
        self,
        x0_pred: mx.array,
        prev_x0: mx.array,
        sample: mx.array,
        step_idx: int,
    ) -> mx.array:
        """Second order DPM-Solver++ update."""
        alpha_t = self._cached_alpha_t[step_idx + 1]
        sigma_t_next = self._cached_sigma_t[step_idx + 1]
        sigma_t_curr = self._cached_sigma_t[step_idx]

        lambda_t = self._cached_lambda[step_idx + 1]
        lambda_s0 = self._cached_lambda[step_idx]
        lambda_s1 = self._cached_lambda[step_idx - 1] if step_idx > 0 else lambda_s0

        h = lambda_t - lambda_s0
        h0 = lambda_s0 - lambda_s1
        r0 = h0 / h if h != 0 else 1.0

        D0 = x0_pred
        D1 = (1.0 / r0) * (x0_pred - prev_x0) if r0 != 0 else mx.zeros_like(x0_pred)

        sigma_ratio = sigma_t_next / sigma_t_curr if sigma_t_curr > 0 else 0.0
        exp_neg_h = math.exp(-h)

        prev_sample = (
            sigma_ratio * sample
            - alpha_t * (exp_neg_h - 1.0) * D0
            - 0.5 * alpha_t * (exp_neg_h - 1.0) * D1
        )

        return prev_sample

    def step(
        self,
        model_output: mx.array,
        timestep: Union[int, mx.array],
        sample: mx.array,
        prev_x0: Optional[mx.array] = None,
    ) -> SchedulerOutput:
        """Perform one step of the DPM-Solver.

        Args:
            model_output: Direct output from the model
            timestep: Current timestep
            sample: Current sample
            prev_x0: Previous x0 prediction for multi-order updates

        Returns:
            SchedulerOutput with prev_sample and x0_pred
        """
        if self._step_index is None:
            self._step_index = 0

        step_idx = self._step_index

        # Convert model output to x0 prediction
        x0_pred = self._convert_model_output(model_output, sample, step_idx)

        # Store model output for multi-order updates
        for i in range(self.solver_order - 1, 0, -1):
            self.model_outputs[i] = self.model_outputs[i - 1]
        self.model_outputs[0] = x0_pred

        # Determine order for this step
        lower_order_final_flag = (step_idx == self.num_inference_steps - 1) and (
            (self.lower_order_final and self.num_inference_steps < 15)
            or self.final_sigmas_type == "zero"
        )

        if self.lower_order_nums < 1 or lower_order_final_flag:
            order = 1
        elif self.solver_order == 2 or self.lower_order_nums < 2:
            order = 2
        else:
            order = self.solver_order

        # Compute previous sample based on order
        if order == 1:
            prev_sample = self._dpm_solver_first_order_update(x0_pred, sample, step_idx)
        elif order == 2:
            if prev_x0 is not None:
                prev_sample = self._dpm_solver_second_order_update(
                    x0_pred, prev_x0, sample, step_idx
                )
            elif self.model_outputs[1] is not None:
                prev_sample = self._dpm_solver_second_order_update(
                    x0_pred, self.model_outputs[1], sample, step_idx
                )
            else:
                # Fall back to first order
                prev_sample = self._dpm_solver_first_order_update(
                    x0_pred, sample, step_idx
                )
        else:
            # Higher orders not implemented, fall back to second order
            if prev_x0 is not None:
                prev_sample = self._dpm_solver_second_order_update(
                    x0_pred, prev_x0, sample, step_idx
                )
            else:
                prev_sample = self._dpm_solver_first_order_update(
                    x0_pred, sample, step_idx
                )

        # Update lower order count
        if self.lower_order_nums < self.solver_order - 1:
            self.lower_order_nums += 1

        # Update step index
        self._step_index += 1

        return SchedulerOutput(prev_sample=prev_sample, x0_pred=x0_pred)

    def reset(self):
        """Reset scheduler state for new generation."""
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self._step_index = None

    def scale_model_input(
        self, sample: mx.array, timestep: Optional[int] = None
    ) -> mx.array:
        """Scale model input (identity for DPM-Solver)."""
        return sample

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Add noise to samples."""
        if timesteps.ndim == 0:
            timesteps = mx.expand_dims(timesteps, 0)

        alpha_t = self.alpha_t[timesteps]
        sigma_t = self.sigma_t[timesteps]

        # Reshape for broadcasting
        while alpha_t.ndim < original_samples.ndim:
            alpha_t = mx.expand_dims(alpha_t, -1)
            sigma_t = mx.expand_dims(sigma_t, -1)

        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

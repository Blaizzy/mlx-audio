# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Optional, Union

import mlx.core as mx

from ..vibevoice.scheduler import DPMSolverMultistepScheduler as BaseDPMSolver
from ..vibevoice.scheduler import SchedulerOutput


class SDEDPMSolverMultistepScheduler(BaseDPMSolver):
    """Extends the VibeVoice DPM-Solver with SDE-DPM-Solver++ (stochastic variant).

    The stochastic variant injects noise at each solver step, producing higher-quality
    speech for KugelAudio's diffusion architecture.
    """

    def _dpm_solver_first_order_update(
        self,
        x0_pred: mx.array,
        sample: mx.array,
        step_idx: int,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        alpha_s = self._cached_alpha_t[step_idx + 1]
        sigma_s = self._cached_sigma_t[step_idx + 1]
        sigma_t = self._cached_sigma_t[step_idx]

        lambda_s = self._cached_lambda[step_idx + 1]
        lambda_t = self._cached_lambda[step_idx]
        h = lambda_s - lambda_t

        sigma_ratio = sigma_s / sigma_t if sigma_t > 0 else 0.0
        exp_neg_h = math.exp(-h)
        exp_neg_2h = math.exp(-2.0 * h)

        return (
            (sigma_ratio * exp_neg_h) * sample
            + (alpha_s * (1 - exp_neg_2h)) * x0_pred
            + sigma_s * math.sqrt(1.0 - exp_neg_2h) * noise
        )

    def _dpm_solver_second_order_update(
        self,
        x0_pred: mx.array,
        prev_x0: mx.array,
        sample: mx.array,
        step_idx: int,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        alpha_s = self._cached_alpha_t[step_idx + 1]
        sigma_s = self._cached_sigma_t[step_idx + 1]
        sigma_t = self._cached_sigma_t[step_idx]

        lambda_s = self._cached_lambda[step_idx + 1]
        lambda_s0 = self._cached_lambda[step_idx]
        lambda_s1 = self._cached_lambda[step_idx - 1] if step_idx > 0 else lambda_s0

        h = lambda_s - lambda_s0
        h0 = lambda_s0 - lambda_s1
        r0 = h0 / h if h != 0 else 1.0

        D0 = x0_pred
        D1 = (1.0 / r0) * (x0_pred - prev_x0) if r0 != 0 else mx.zeros_like(x0_pred)

        sigma_ratio = sigma_s / sigma_t if sigma_t > 0 else 0.0
        exp_neg_h = math.exp(-h)
        exp_neg_2h = math.exp(-2.0 * h)

        return (
            (sigma_ratio * exp_neg_h) * sample
            + (alpha_s * (1 - exp_neg_2h)) * D0
            + 0.5 * (alpha_s * (1 - exp_neg_2h)) * D1
            + sigma_s * math.sqrt(1.0 - exp_neg_2h) * noise
        )

    def step(
        self,
        model_output: mx.array,
        timestep: Union[int, mx.array],  # pylint: disable=unused-argument
        sample: mx.array,
        prev_x0: Optional[mx.array] = None,
        noise: Optional[mx.array] = None,
    ) -> SchedulerOutput:
        if self._step_index is None:
            self._step_index = 0

        step_idx = self._step_index

        if noise is None:
            noise = mx.random.normal(sample.shape)

        x0_pred = self._convert_model_output(model_output, sample, step_idx)

        for i in range(self.solver_order - 1, 0, -1):
            self.model_outputs[i] = self.model_outputs[i - 1]
        self.model_outputs[0] = x0_pred

        lower_order_final_flag = (step_idx == self.num_inference_steps - 1) and (
            (self.lower_order_final and self.num_inference_steps < 15)
            or self.final_sigmas_type == "zero"
        )

        if self.lower_order_nums < 1 or lower_order_final_flag:
            prev_sample = self._dpm_solver_first_order_update(
                x0_pred, sample, step_idx, noise=noise
            )
        else:
            use_prev = prev_x0 if prev_x0 is not None else self.model_outputs[1]
            if use_prev is not None:
                prev_sample = self._dpm_solver_second_order_update(
                    x0_pred, use_prev, sample, step_idx, noise=noise
                )
            else:
                prev_sample = self._dpm_solver_first_order_update(
                    x0_pred, sample, step_idx, noise=noise
                )

        if self.lower_order_nums < self.solver_order - 1:
            self.lower_order_nums += 1

        self._step_index += 1

        return SchedulerOutput(prev_sample=prev_sample, x0_pred=x0_pred)

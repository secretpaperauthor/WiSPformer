# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class TemporalDecayLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, loss_mode: str = "L1"):
        super().__init__()

        valid_modes = {"L1", "L2", "L1L2"}
        if loss_mode not in valid_modes:
            raise ValueError(f"loss_mode must be one of {valid_modes}, got {loss_mode!r}")

        self.alpha = alpha
        self.loss_mode = loss_mode

        self.register_buffer("_cached_weights", t.empty(0), persistent=False)
        self._cached_seq_len = None

    def _get_weights(self, pred: t.Tensor) -> t.Tensor:
        seq_len = pred.size(1)

        cache_invalid = (
            self._cached_weights.numel() == 0
            or self._cached_seq_len != seq_len
            or self._cached_weights.device != pred.device
            or self._cached_weights.dtype != pred.dtype
        )

        if cache_invalid:
            self._cached_weights = t.arange(
                1,
                seq_len + 1,
                device=pred.device,
                dtype=pred.dtype,
            ).pow(-self.alpha).view(1, seq_len, 1)

            self._cached_seq_len = seq_len

        return self._cached_weights

    def forward(self, pred: t.Tensor, gt: t.Tensor) -> t.Tensor:
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

        weights = self._get_weights(pred)

        if self.loss_mode == "L1":
            loss = F.l1_loss(pred, gt, reduction="none")
        elif self.loss_mode == "L2":
            loss = F.mse_loss(pred, gt, reduction="none")
        else:
            loss = F.l1_loss(pred, gt, reduction="none") + F.mse_loss(
                pred, gt, reduction="none"
            )

        return (loss * weights).mean()

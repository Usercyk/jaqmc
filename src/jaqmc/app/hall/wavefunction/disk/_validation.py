# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Coupling domains of rational fitted disk envelopes."""

import math


def validate_fitted_coupling(
    kappa: float,
    *,
    quartic_slope: float = 0.0,
    denominator_slopes: tuple[float, ...] = (),
) -> None:
    """Validate a decaying quartic envelope and its rational denominators.

    Args:
        kappa: Nonnegative interaction strength.
        quartic_slope: Slope in the denominator of the negative quartic term.
        denominator_slopes: Slopes in other denominators ``1 + slope * kappa``.

    Raises:
        ValueError: If the coupling is invalid, a denominator vanishes, or the
            quartic envelope no longer confines electrons on the plane.
    """
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")
    if 1 + quartic_slope * kappa <= 0:
        raise ValueError("kappa must preserve a decaying quartic envelope.")
    if any(1 + slope * kappa == 0 for slope in denominator_slopes):
        raise ValueError("kappa lies at a pole of the fitted envelope.")

# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Pair-correlation estimator for the disk geometry.

For a fixed point :math:`\mathbf r=(r, 0)`, this estimator computes the
dimensionless pair-correlation function

.. math::

    g(\mathbf r, \mathbf r') =
    \frac{\rho^{(2)}(\mathbf r, \mathbf r')}
    {\rho^{(1)}(\mathbf r)\rho^{(1)}(\mathbf r')},

where

.. math::

    \rho^{(2)}(\mathbf r, \mathbf r') =
    \left\langle \sum_{i\ne j}
    \delta^{(2)}(\mathbf r-\mathbf r_i)
    \delta^{(2)}(\mathbf r'-\mathbf r_j) \right\rangle.

The scalar ``r`` supplied in the configuration fixes the first electron's
radial coordinate.  Rotational symmetry of the disk is used to put that
electron on the positive x axis.  The second position :math:`\mathbf r'` is
returned on a two-dimensional Cartesian grid.

Both delta functions are approximated by finite histogram cells.  The first
uses an annulus of width ``reference_bin_width`` (the output bin width by
default); every selected configuration is rotated before the second electron
is histogrammed.  The one-particle densities used in the denominator are
accumulated from the same Monte Carlo samples.  The unnormalized ordered-pair
density is also returned as ``pair_correlation:rho2``.
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from jax import numpy as jnp
from jax.typing import ArrayLike

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class DiskPairCorrelation(Estimator):
    r"""Pair correlation :math:`g((r,0),\mathbf r')` on a disk.

    Args:
        bins: Number of bins along each Cartesian axis of :math:`\mathbf r'`.
        max_r: The :math:`x'` and :math:`y'` histogram range is
            ``[-max_r, max_r]``.
        r: Fixed radial coordinate of the first electron.  Rotational
            symmetry places the reference point at ``(r, 0)``.
        reference_bin_width: Radial width used to approximate the delta
            function at the fixed point.  If omitted, use the Cartesian
            output-bin width ``2 * max_r / bins``.
    """

    bins: int = 200
    max_r: float = 20.0
    r: float = 0.0
    reference_bin_width: float | None = None

    def __post_init__(self) -> None:
        if self.bins <= 0:
            raise ValueError("bins must be positive")
        if self.max_r <= 0:
            raise ValueError("max_r must be positive")
        if self.r < 0:
            raise ValueError("r must be non-negative")
        if self.r > self.max_r:
            raise ValueError("r must not exceed max_r")
        if self.reference_bin_width is not None and self.reference_bin_width <= 0:
            raise ValueError("reference_bin_width must be positive")

    @property
    def _reference_bin_width(self) -> float:
        if self.reference_bin_width is not None:
            return self.reference_bin_width
        return 2 * self.max_r / self.bins

    @property
    def _reference_area(self) -> float:
        half_width = self._reference_bin_width / 2
        inner_r = max(0.0, self.r - half_width)
        outer_r = self.r + half_width
        return math.pi * (outer_r**2 - inner_r**2)

    def init(self, data: Data, rngs: PRNGKey) -> dict[str, jnp.ndarray]:
        del data, rngs
        n_devices = 1
        histogram_shape = (n_devices, self.bins, self.bins)
        scalar_shape = (n_devices,)
        return {
            "pair_counts": jnp.zeros(histogram_shape),
            "density_counts": jnp.zeros(histogram_shape),
            "reference_counts": jnp.zeros(scalar_shape),
            "sample_counts": jnp.zeros(scalar_shape),
            "electron_counts": jnp.zeros(scalar_shape),
        }

    def evaluate_batch_walkers(
        self,
        params: Any,
        batched_data: BatchedData,
        prev_walker_stats: Mapping[str, Any],
        state: dict[str, jnp.ndarray],
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        del params, prev_walker_stats, rngs
        electrons = batched_data.data.electrons[..., :2]
        batch_size, nelec, _ = electrons.shape

        x = electrons[..., 0]
        y = electrons[..., 1]
        radii = jnp.hypot(x, y)
        half_width = self._reference_bin_width / 2
        inner_r = max(0.0, self.r - half_width)
        outer_r = self.r + half_width
        reference_mask = (radii >= inner_r) & (radii < outer_r)

        # The first electron labels the rotation.  After rotating by minus
        # its polar angle, it lies on the positive x axis and every possible
        # second electron supplies one sample of r'.  At r == 0 the reference
        # direction is undefined, so retain the laboratory frame; rotational
        # sampling makes the resulting rho(0, r') isotropic.
        if self.r == 0:
            x_prime = jnp.broadcast_to(x[:, None, :], (batch_size, nelec, nelec))
            y_prime = jnp.broadcast_to(y[:, None, :], (batch_size, nelec, nelec))
        else:
            reference_theta = jnp.arctan2(y, x)
            cos_theta = jnp.cos(reference_theta)[..., :, None]
            sin_theta = jnp.sin(reference_theta)[..., :, None]
            target_x = x[:, None, :]
            target_y = y[:, None, :]
            x_prime = target_x * cos_theta + target_y * sin_theta
            y_prime = -target_x * sin_theta + target_y * cos_theta

        not_self = ~jnp.eye(nelec, dtype=bool)[None, :, :]
        weights = reference_mask[..., :, None] & not_self
        pair_counts, _, _ = jnp.histogram2d(
            x_prime.reshape(-1),
            y_prime.reshape(-1),
            bins=self.bins,
            range=((-self.max_r, self.max_r), (-self.max_r, self.max_r)),
            weights=weights.reshape(-1),
        )
        density_counts, _, _ = jnp.histogram2d(
            x.reshape(-1),
            y.reshape(-1),
            bins=self.bins,
            range=((-self.max_r, self.max_r), (-self.max_r, self.max_r)),
        )

        # Accumulate raw counts and normalize only after summing across all
        # devices.  This avoids an extra device-count factor that would arise
        # from summing independently normalized device-local histograms.
        return {}, {
            "pair_counts": state["pair_counts"] + parallel_jax.psum(pair_counts),
            "density_counts": state["density_counts"]
            + parallel_jax.psum(density_counts),
            "reference_counts": (
                state["reference_counts"] + parallel_jax.psum(jnp.sum(reference_mask))
            ),
            "sample_counts": state["sample_counts"] + parallel_jax.psum(batch_size),
            "electron_counts": (
                state["electron_counts"] + parallel_jax.psum(batch_size * nelec)
            ),
        }

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def finalize_state(
        self, state: Mapping[str, ArrayLike], *, n_steps: int
    ) -> dict[str, Any]:
        r"""Return :math:`g`, its density factors, and grid metadata.

        Raises:
            ValueError: If no steps or samples have completed, or counts are invalid.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        bin_width = 2 * self.max_r / self.bins
        bin_area = bin_width**2
        # Finalization runs on the host. Undefined bins deliberately contain NaN.
        counts = {
            key: np.asarray(state[key], dtype=np.float64)
            for key in (
                "pair_counts",
                "density_counts",
                "reference_counts",
                "sample_counts",
                "electron_counts",
            )
        }
        if any(
            not np.isfinite(value).all() or (value < 0).any()
            for value in counts.values()
        ):
            raise ValueError("pair-correlation counts must be finite and nonnegative")
        sample_count = float(np.sum(counts["sample_counts"]))
        if sample_count <= 0:
            raise ValueError("state must contain at least one sampled configuration")

        pair_counts = np.sum(counts["pair_counts"], axis=0)
        density_counts = np.sum(counts["density_counts"], axis=0)
        reference_count = float(np.sum(counts["reference_counts"]))
        electron_count = float(np.sum(counts["electron_counts"]))

        rho2 = pair_counts / (sample_count * self._reference_area * bin_area)
        rho1_target = density_counts / (sample_count * bin_area)
        rho1_reference = reference_count / (sample_count * self._reference_area)
        denominator = rho1_reference * rho1_target
        valid = denominator > 0
        g = np.full_like(rho2, np.nan)
        np.divide(rho2, denominator, out=g, where=valid)

        # In a fixed-N finite system the standard reduced correlation above
        # approaches (N-1)/N for independent particles.  Also expose the
        # conventional finite-N corrected version whose independent-particle
        # baseline is exactly one.
        electron_number = electron_count / sample_count
        g_finite_n = np.full_like(g, np.nan)
        if electron_number > 1:
            g_finite_n = electron_number / (electron_number - 1) * g

        centers = np.linspace(
            -self.max_r + bin_width / 2,
            self.max_r - bin_width / 2,
            self.bins,
        )
        return {
            "pair_correlation": g,
            "pair_correlation:g_finite_n": g_finite_n,
            "pair_correlation:rho2": rho2,
            "pair_correlation:rho1_target": rho1_target,
            "pair_correlation:rho1_reference": rho1_reference,
            "pair_correlation:x": centers,
            "pair_correlation:y": centers,
            "pair_correlation:bin_area": bin_area,
            "pair_correlation:reference_r": self.r,
            "pair_correlation:reference_x": self.r,
            "pair_correlation:reference_y": 0.0,
            "pair_correlation:reference_bin_width": self._reference_bin_width,
            "pair_correlation:reference_area": self._reference_area,
            "pair_correlation:sample_count": sample_count,
            "pair_correlation:electron_number": electron_number,
            "pair_correlation:n_steps": n_steps,
        }

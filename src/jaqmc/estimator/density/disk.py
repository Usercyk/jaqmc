# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Disk density estimator for the disk geometry."""

from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.histogram import HistogramEstimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class DiskDensity(HistogramEstimator):
    r"""Electron density on the disk.

    Accumulates both a 2-D histogram of electron positions in Cartesian
    coordinates (x, y) and a 1-D histogram of their radii on the disk.

    Args:
        bins: Number of bins for both x and y dimensions.
        range_max: The range of the histogram will be
            [-range_max, range_max] for both dimensions.
        radial_bin_width_pixels: Width of a radial bin measured in Cartesian
            histogram pixels.  The default of two suppresses lattice artefacts
            when plotting an azimuthally averaged density profile.
    """

    bins: int = 100
    range_max: float = 10.0
    radial_bin_width_pixels: float = 2.0

    @property
    def radial_bins(self) -> int:
        """Number of bins spanning ``[0, range_max]``.

        A Cartesian pixel has width ``2 * range_max / bins``.  Therefore a
        radial bin that is ``w`` pixels wide requires ``bins / (2 * w)`` bins
        between zero and ``range_max``.

        Raises:
            ValueError: If ``radial_bin_width_pixels`` is not positive.
        """
        if self.radial_bin_width_pixels <= 0:
            raise ValueError("radial_bin_width_pixels must be positive")
        return max(1, round(self.bins / (2 * self.radial_bin_width_pixels)))

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        return (self.bins, self.bins), [
            (-self.range_max, self.range_max),
            (-self.range_max, self.range_max),
        ]

    def extract(self, data: Data) -> jnp.ndarray:
        return data["electrons"][..., :2]

    def init(self, data: Data, rngs) -> dict[str, jnp.ndarray]:
        state = super().init(data, rngs)
        shape = (1, self.radial_bins)
        state["radial_histogram"] = jnp.zeros(shape)
        state["radial_compensation"] = jnp.zeros(shape)
        return state

    def evaluate_batch_walkers(
        self, params, batched_data, prev_walker_stats, state, rngs
    ) -> tuple[dict, dict[str, jnp.ndarray]]:
        stats, new_state = super().evaluate_batch_walkers(
            params, batched_data, prev_walker_stats, state, rngs
        )

        xy = batched_data.data["electrons"][..., :2]
        radii = jnp.linalg.norm(xy, axis=-1).reshape(-1)
        radial_counts = jnp.histogram(
            radii,
            bins=self.radial_bins,
            range=(0.0, self.range_max),
        )[0]

        adjusted = parallel_jax.psum(radial_counts) - state["radial_compensation"]
        radial_sum = state["radial_histogram"] + adjusted
        radial_compensation = (radial_sum - state["radial_histogram"]) - adjusted
        new_state["radial_histogram"] = radial_sum
        new_state["radial_compensation"] = radial_compensation
        return stats, new_state

    def finalize_state(self, state: dict[str, jnp.ndarray], *, n_steps: int) -> dict:
        result = super().finalize_state(state, n_steps=n_steps)
        result["density_radial"] = jnp.sum(state["radial_histogram"], axis=0)
        result["density_radial:n_steps"] = n_steps
        return result

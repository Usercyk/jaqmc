# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Two-dimensional pair-correlation estimator for electrons on a torus."""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

__all__ = ["TorusPairCorrelation"]


@configurable_dataclass
class TorusPairCorrelation(Estimator):
    r"""Pair correlation :math:`g(\Delta u, \Delta v)` on a torus.

    For every unordered electron pair, accumulates both directed fractional
    displacements :math:`\mathbf r_i-\mathbf r_j` and
    :math:`\mathbf r_j-\mathbf r_i`, wrapped into the fundamental domain
    :math:`[-1/2,1/2)\times[-1/2,1/2)`. The histogram is normalized by its
    bin area and by the number of electrons and walkers. With the same
    :math:`N^2` convention as the spherical estimator, independent uniformly
    distributed electrons approach :math:`(N-1)/N`.

    The accumulated correlation is normalized over walkers and devices on
    every evaluation step. :meth:`finalize_state` averages it over steps and
    returns the 2-D ``pair_correlation`` array and its bin-center axes
    ``pair_correlation:u`` and ``pair_correlation:v``.

    Args:
        bins_u: Number of bins for the first fractional displacement.
        bins_v: Number of bins for the second fractional displacement.
        data_field: Name of the field holding fractional torus coordinates.
        name: Prefix used for evaluation-digest keys.
    """

    bins_u: int = 50
    bins_v: int = 50
    data_field: str = runtime_dep(default="electrons")
    name: str = "pair_correlation"

    def __post_init__(self) -> None:
        for name, bins in (("bins_u", self.bins_u), ("bins_v", self.bins_v)):
            if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
                raise ValueError(f"{name} must be a positive integer. Got {bins!r}.")

    def init(self, data: Data, rngs: PRNGKey) -> dict[str, jnp.ndarray]:
        del data, rngs
        shape = (jax.device_count(), self.bins_u, self.bins_v)
        return {
            "histogram": jnp.zeros(shape),
            "compensation": jnp.zeros(shape),
        }

    def _pair_displacements(self, electrons: jnp.ndarray) -> jnp.ndarray:
        """Compute directed fractional displacements for all distinct pairs.

        Returns:
            Wrapped displacements with shape
            ``(batch * n_electrons * (n_electrons - 1), 2)``.
        """
        nelec = electrons.shape[-2]
        pair_indices = jnp.triu_indices(nelec, 1)
        displacements = (electrons[..., :, None, :] - electrons[..., None, :, :])[
            :, *pair_indices
        ]
        directed = jnp.concatenate([displacements, -displacements], axis=1)
        return ((directed + 0.5) % 1.0 - 0.5).reshape(-1, 2)

    def evaluate_batch_walkers(
        self,
        params: Any,
        batched_data: BatchedData,
        prev_walker_stats: Mapping[str, Any],
        state: dict[str, jnp.ndarray],
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        del params, prev_walker_stats, rngs
        electrons = batched_data.data[self.data_field]
        if electrons.ndim != 3 or electrons.shape[-1] != 2:
            raise ValueError(
                "Batched torus electron coordinates must have shape "
                "(batch, n_electrons, 2), with the last axis ordered as "
                f"(u, v). Got {electrons.shape}."
            )

        batch_size, nelec, _ = electrons.shape
        pair_displacements = self._pair_displacements(electrons)
        counts = jnp.histogramdd(
            pair_displacements,
            bins=(self.bins_u, self.bins_v),
            range=((-0.5, 0.5), (-0.5, 0.5)),
        )[0]

        # Both pair directions are already present. Dividing by the fractional
        # bin area multiplies the raw counts by bins_u * bins_v.
        global_batch_size = batch_size * jax.device_count()
        increment = counts * self.bins_u * self.bins_v / (global_batch_size * nelec**2)
        adjusted = increment - state["compensation"]
        new_sum = state["histogram"] + adjusted
        new_compensation = (new_sum - state["histogram"]) - adjusted
        return {}, {"histogram": new_sum, "compensation": new_compensation}

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        del walker_stats
        return {}

    def finalize_stats(
        self, mean_stats: Mapping[str, Any], state: dict[str, jnp.ndarray]
    ) -> dict[str, Any]:
        del mean_stats, state
        return {}

    def finalize_state(
        self, state: dict[str, jnp.ndarray], *, n_steps: int
    ) -> dict[str, Any]:
        correlation = jnp.sum(state["histogram"], axis=0) / n_steps
        u_centers = (jnp.arange(self.bins_u) + 0.5) / self.bins_u - 0.5
        v_centers = (jnp.arange(self.bins_v) + 0.5) / self.bins_v - 0.5
        return {
            self.name: correlation,
            f"{self.name}:u": u_centers,
            f"{self.name}:v": v_centers,
            f"{self.name}:n_steps": n_steps,
        }

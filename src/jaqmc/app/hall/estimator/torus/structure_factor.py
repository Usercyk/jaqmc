# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physical density structure factors at reciprocal torus momenta."""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.torus import get_torus_lattice
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class TorusStructureFactor(PerWalkerEstimator):
    """Connected physical structure factor without angular averaging.

    Args:
        flux: Number of magnetic flux quanta, fixing the physical area.
        tau: Modular parameter in the upper half-plane (runtime dependency).
        indices: Integer reciprocal labels, including zero if desired.
        data_field: Field containing fractional electron coordinates.

    Notes:
        Connected subtraction happens after averaging all steps and walkers.
        The output is not a projected guiding-center structure factor.
    """

    flux: int = 1
    tau: complex = runtime_dep(default=1j)
    indices: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1))
    data_field: str = runtime_dep(default="electrons")

    def __post_init__(self) -> None:
        get_torus_lattice(self.flux, self.tau)
        if not self.indices or any(
            len(pair) != 2
            or any(not isinstance(n, int) or isinstance(n, bool) for n in pair)
            for pair in self.indices
        ):
            raise ValueError("indices must contain integer reciprocal pairs.")

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, prev_walker_stats, rngs
        electrons = data[self.data_field]
        if electrons.ndim != 2 or electrons.shape[-1] != 2 or len(electrons) == 0:
            raise ValueError("Torus coordinates must have shape (N, 2) with N > 0.")
        phase = 2 * jnp.pi * jnp.asarray(self.indices) @ electrons.T
        rho = jnp.sum(jnp.exp(1j * phase), axis=-1)
        return {
            "structure_factor:rho": rho,
            "structure_factor:raw": jnp.abs(rho) ** 2 / len(electrons),
            "structure_factor:n_electrons": jnp.asarray(len(electrons)),
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: None
    ) -> dict[str, Any]:
        del state
        stats = jax.tree.map(lambda x: jnp.mean(x, axis=0), batched_stats)
        rho = stats["structure_factor:rho"]
        nelec = stats["structure_factor:n_electrons"]
        l1, l2, _ = get_torus_lattice(self.flux, self.tau)
        lattice = jnp.array([[l1, 0.0], [l2.real, l2.imag]])
        stats["structure_factor"] = (
            stats["structure_factor:raw"] - jnp.abs(rho) ** 2 / nelec
        )
        stats["structure_factor:indices"] = jnp.asarray(self.indices)
        stats["structure_factor:k"] = (
            2 * jnp.pi * jnp.asarray(self.indices) @ jnp.linalg.inv(lattice).T
        )
        return stats

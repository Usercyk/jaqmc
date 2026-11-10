# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
r"""Edge Green's function estimator for the disk geometry.

This module estimates a normalized edge Green's function by keeping the
first electron on the edge of the disk and evaluating wave-function ratios
after rotating that electron along a fixed-radius arc.
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class DiskEdgeGreenDirect(PerWalkerEstimator):
    r"""Direct normalized edge Green's function estimator on the disk.

    The sampler is expected to provide configurations where
    ``electrons[0]`` is fixed at ``(edge_radius, 0)``. For each sampled
    configuration, this estimator moves only that electron to points
    ``edge_radius * (cos(theta), sin(theta))`` and computes

    .. math::

        G(\theta) / G(0) = \Psi(r_\theta, r_1, \ldots)
        / \Psi(r_0, r_1, \ldots).

    Args:
        edge_radius: Radius of the disk edge used for the displaced electron.
        n_theta: Number of angular points sampled on the edge arc.
        theta_min: Smallest angle included in the arc.
        theta_max: Largest angle included in the arc.
        f_log_psi: Complex log-psi function (runtime dependency).
        data_field: Name of the electron-coordinate field in ``Data``.
    """

    edge_radius: float = 6.0
    n_theta: int = 96
    theta_min: float = 0.05
    theta_max: float = np.pi
    f_log_psi: Any = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, jnp.ndarray], Any]:
        """Evaluate wave-function ratios for one walker.

        ``displaced[i]`` is a full electron configuration whose first electron
        has been moved to the corresponding edge target coordinate.
        ``jax.vmap`` evaluates ``f_log_psi`` over all such displaced
        configurations at once.

        Returns:
            Tuple[dict[str, jnp.ndarray], Any]: A pair where the first
            element is a dictionary of local statistics containing the
            key ``"edge_green_normalized"`` mapped to an array of
            normalized Green's function values with shape ``(n_theta,)``,
            and the second element is the (unchanged) estimator state.
        """
        del prev_walker_stats, rngs
        electrons = data[self.data_field]
        theta = jnp.linspace(self.theta_min, self.theta_max, self.n_theta)
        target = self.edge_radius * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

        # 当前构型的 electrons[0] 已由 sampler 固定为 (R, 0)
        logpsi_0 = self.f_log_psi(params, data)

        displaced = jnp.broadcast_to(electrons, (self.n_theta, *electrons.shape))
        displaced = displaced.at[:, 0, :].set(target)

        def evaluate(electrons_new: jnp.ndarray) -> jnp.ndarray:
            """Compute log-psi after replacing the electron coordinates.

            Args:
                electrons_new: Electron coordinates for which to evaluate
                    the complex log wavefunction. Shape matches the
                    sampler's electron array for a single configuration.

            Returns:
                Complex log-psi evaluated at the provided coordinates.
            """
            return self.f_log_psi(params, data.merge({self.data_field: electrons_new}))

        logpsi_theta = jax.vmap(evaluate)(displaced)
        green_normalized = jnp.exp(logpsi_theta - logpsi_0)

        return {"edge_green_normalized": green_normalized}, state

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(local_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        del state
        green = jnp.nanmean(batched_stats["edge_green_normalized"], axis=0)
        return {
            "edge_green_normalized": green,
            "edge_green:abs": jnp.abs(green),
            "edge_green:theta": jnp.linspace(
                self.theta_min, self.theta_max, self.n_theta
            ),
        }

# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Metropolis-Hastings sampling with one electron pinned on the disk edge."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from jaqmc.array_types import PRNGKey, PyTree
from jaqmc.geometry.disk import disk_proposal
from jaqmc.sampler.base import BatchLogProb
from jaqmc.sampler.mcmc import MCMCSampler, MCMCState, SamplingProposal
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

__all__ = ["PinnedDiskMCMCSampler", "pinned_disk_proposal"]


def pinned_disk_proposal(
    rngs: PRNGKey,
    x: Mapping[str, jnp.ndarray],
    stddev: float | jnp.ndarray,
) -> Mapping[str, jnp.ndarray]:
    """Propose disk moves while leaving the first electron fixed.

    Args:
        rngs: Random key for the proposal.
        x: Batched electron coordinates under the ``"electrons"`` key.
        stddev: Gaussian proposal width.

    Returns:
        A proposed data mapping with ``electrons[:, 0, :]`` unchanged.
    """
    proposed = disk_proposal(rngs, x, stddev)
    return {
        **proposed,
        "electrons": proposed["electrons"].at[:, 0, :].set(x["electrons"][:, 0, :]),
    }


@configurable_dataclass
class PinnedDiskMCMCSampler(MCMCSampler):
    """MCMC sampler for the conditional edge Green-function distribution.

    The first electron is pinned at ``(edge_radius, 0)`` before each sampler
    step, while the remaining electrons are updated by Metropolis-Hastings.

    Args:
        edge_radius: Radius of the pinned edge electron.
        sampling_proposal: Proposal function which preserves the pinned
            electron.
    """

    edge_radius: float = 6.0
    sampling_proposal: SamplingProposal = runtime_dep(default=pinned_disk_proposal)

    def _pin_electron(self, data: Mapping[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """Return data with the first electron fixed at ``(edge_radius, 0)``."""
        pinned_position = jnp.array([self.edge_radius, 0.0])
        return {
            **data,
            "electrons": data["electrons"].at[:, 0, :].set(pinned_position),
        }

    def step(
        self,
        batch_log_prob: BatchLogProb,
        data: PyTree,
        state: MCMCState,
        rngs: PRNGKey,
    ) -> tuple[PyTree, dict[str, Any], MCMCState]:
        """Run adaptive MH updates with the first electron fixed at the edge.

        Args:
            batch_log_prob: Log-probability function over a batch.
            data: Current batched electron configurations.
            state: Adaptive MCMC state.
            rngs: Random state.

        Returns:
            Tuple of ``(data, stats, new_state)`` after one sampler step.
        """
        return super().step(batch_log_prob, self._pin_electron(data), state, rngs)

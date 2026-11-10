# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Static structure-factor estimator for the disk geometry.

For :math:`\rho_{\mathbf q}=\sum_j\exp(-i\mathbf q\cdot\mathbf r_j)`, the
ordinary equal-time static structure factor is

.. math::

    S(\mathbf q) = \frac{1}{N}
    \left\langle\rho_{\mathbf q}\rho_{-\mathbf q}\right\rangle.

Unlike a periodic cell, the disk has continuous wave vectors.  This estimator
therefore evaluates a configurable one-dimensional grid of magnitudes and
averages over equally spaced directions of :math:`\mathbf q`.  It also returns
the connected structure factor

.. math::

    S_c(\mathbf q) = \frac{1}{N}\left(
    \left\langle |\rho_{\mathbf q}|^2\right\rangle
    - \left|\left\langle\rho_{\mathbf q}\right\rangle\right|^2
    \right),

which removes the disconnected contribution from the inhomogeneous density
profile of a finite droplet.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass


def _kahan_add(
    total: jnp.ndarray,
    compensation: jnp.ndarray,
    value: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add ``value`` to an accumulated array using Kahan summation.

    Returns:
        Updated total and compensation arrays.
    """
    adjusted = value - compensation
    new_total = total + adjusted
    new_compensation = (new_total - total) - adjusted
    return new_total, new_compensation


@configurable_dataclass
class DiskStructureFactor(Estimator):
    r"""Angle-averaged static structure factor :math:`S(q)` on a disk.

    Args:
        q_min: Smallest wave-vector magnitude.  Including zero is useful
            because the ordinary structure factor must obey ``S(0) = N``.
        q_max: Largest wave-vector magnitude, in inverse coordinate units.
        num_q: Number of linearly spaced magnitudes, including both endpoints.
        num_angles: Number of equally spaced directions used for the angular
            average.  One direction is sufficient in expectation for a
            rotationally invariant state; additional directions reduce Monte
            Carlo noise.

    Notes:
        ``structure_factor`` is the ordinary structure factor and
        ``structure_factor:connected`` is its connected counterpart.  These
        are particle-density structure factors; no lowest-Landau-level form
        factor or guiding-centre conversion is applied.
    """

    q_min: float = 0.0
    q_max: float = 5.0
    num_q: int = 101
    num_angles: int = 8

    def __post_init__(self) -> None:
        if self.q_min < 0:
            raise ValueError("q_min must be non-negative")
        if self.q_max < self.q_min:
            raise ValueError("q_max must not be smaller than q_min")
        if self.num_q <= 0:
            raise ValueError("num_q must be positive")
        if self.num_angles <= 0:
            raise ValueError("num_angles must be positive")

    @property
    def q_values(self) -> jnp.ndarray:
        """Linearly spaced wave-vector magnitudes."""
        return jnp.linspace(self.q_min, self.q_max, self.num_q)

    @property
    def q_angles(self) -> jnp.ndarray:
        """Inversion-inequivalent directions used in the angular average."""
        # S(q) = S(-q), so [0, pi) contains every independent direction.
        return jnp.pi * jnp.arange(self.num_angles) / self.num_angles

    @property
    def q_vectors(self) -> jnp.ndarray:
        """Wave vectors with shape ``(num_q, num_angles, 2)``."""
        directions = jnp.stack(
            [jnp.cos(self.q_angles), jnp.sin(self.q_angles)], axis=-1
        )
        return self.q_values[:, None, None] * directions[None, :, :]

    def init(self, data: Data, rngs: PRNGKey) -> dict[str, jnp.ndarray]:
        del data, rngs
        vector_shape = (1, self.num_q, self.num_angles)
        scalar_shape = (1,)
        # Each state leaf must own a distinct buffer. Sampling stages donate
        # estimator state to JIT-compiled steps; reusing one array object for
        # several leaves would attempt to donate that buffer more than once.
        return {
            "rho_real_sum": jnp.zeros(vector_shape),
            "rho_real_compensation": jnp.zeros(vector_shape),
            "rho_imag_sum": jnp.zeros(vector_shape),
            "rho_imag_compensation": jnp.zeros(vector_shape),
            "rho_abs2_sum": jnp.zeros(vector_shape),
            "rho_abs2_compensation": jnp.zeros(vector_shape),
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

        # Map over directions so that the largest temporary array has shape
        # (batch, nelec, num_q), rather than additionally carrying num_angles.
        directions = jnp.stack(
            [jnp.cos(self.q_angles), jnp.sin(self.q_angles)], axis=-1
        )

        def direction_sums(direction: jnp.ndarray):
            q_vectors = self.q_values[:, None] * direction[None, :]
            phase = jnp.einsum("bnd,qd->bnq", electrons, q_vectors)
            rho_real = jnp.sum(jnp.cos(phase), axis=1)
            rho_imag = -jnp.sum(jnp.sin(phase), axis=1)
            return (
                jnp.sum(rho_real, axis=0),
                jnp.sum(rho_imag, axis=0),
                jnp.sum(rho_real**2 + rho_imag**2, axis=0),
            )

        rho_real, rho_imag, rho_abs2 = jax.lax.map(direction_sums, directions)
        # lax.map returns (num_angles, num_q); state uses (num_q, num_angles).
        additions = {
            "rho_real": rho_real.T,
            "rho_imag": rho_imag.T,
            "rho_abs2": rho_abs2.T,
        }

        new_state = dict(state)
        for name, value in additions.items():
            new_sum, new_compensation = _kahan_add(
                state[f"{name}_sum"],
                state[f"{name}_compensation"],
                parallel_jax.psum(value),
            )
            new_state[f"{name}_sum"] = new_sum
            new_state[f"{name}_compensation"] = new_compensation
        new_state["sample_counts"] = state["sample_counts"] + parallel_jax.psum(
            batch_size
        )
        new_state["electron_counts"] = state["electron_counts"] + parallel_jax.psum(
            batch_size * nelec
        )
        return {}, new_state

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def finalize_state(
        self, state: Mapping[str, jnp.ndarray], *, n_steps: int
    ) -> dict[str, Any]:
        r"""Normalize accumulated density modes and return :math:`S(q)`.

        Returns:
            Ordinary, connected, and disconnected structure factors together
            with their wave-vector and sampling metadata.

        Raises:
            ValueError: If no evaluation steps or samples have been completed.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        sample_count = jnp.sum(state["sample_counts"])
        if float(sample_count) <= 0:
            raise ValueError("state must contain at least one sampled configuration")

        electron_count = jnp.sum(state["electron_counts"])
        electron_number = electron_count / sample_count
        rho_real_mean = jnp.sum(state["rho_real_sum"], axis=0) / sample_count
        rho_imag_mean = jnp.sum(state["rho_imag_sum"], axis=0) / sample_count
        rho_abs2_mean = jnp.sum(state["rho_abs2_sum"], axis=0) / sample_count

        raw_by_angle = rho_abs2_mean / electron_number
        disconnected_by_angle = (rho_real_mean**2 + rho_imag_mean**2) / electron_number
        # Roundoff at q=0 can make this tiny and negative in float32.
        connected_by_angle = jnp.maximum(raw_by_angle - disconnected_by_angle, 0.0)

        return {
            "structure_factor": jnp.mean(raw_by_angle, axis=-1),
            "structure_factor:connected": jnp.mean(connected_by_angle, axis=-1),
            "structure_factor:disconnected": jnp.mean(disconnected_by_angle, axis=-1),
            "structure_factor:q": self.q_values,
            "structure_factor:q_angles": self.q_angles,
            "structure_factor:q_vectors": self.q_vectors,
            "structure_factor:rho_mean_real": rho_real_mean,
            "structure_factor:rho_mean_imag": rho_imag_mean,
            "structure_factor:sample_count": sample_count,
            "structure_factor:electron_number": electron_number,
            "structure_factor:n_steps": n_steps,
        }

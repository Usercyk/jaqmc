# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Torus geometry utilities for MCMC sampling on the torus."""

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey


def _torus_move(
    rngs: PRNGKey,
    x: jnp.ndarray,
    stddev: float | jnp.ndarray,
    tau: complex,
):
    # The last axis contains the two fractional torus coordinates (u, v),
    # both with unit period. Sample an isotropic displacement in Cartesian
    # coordinates and transform it back through z / L1 = u + tau * v.
    cartesian_displacement = jax.random.normal(rngs, shape=x.shape) * stddev

    displacement_v = cartesian_displacement[..., 1] / tau.imag
    displacement_u = cartesian_displacement[..., 0] - tau.real * displacement_v
    displacement = jnp.stack((displacement_u, displacement_v), axis=-1)
    # Wrap the Gaussian proposal back into the fundamental domain [0, 1).
    return (x + displacement) % 1.0


def torus_proposal(
    rngs: PRNGKey,
    x,
    stddev: float | jnp.ndarray,
    tau: complex,
):
    r"""Propose an isotropic Gaussian move on a torus.

    Coordinates are fractional, with physical positions given by
    :math:`z/L_1=u+\tau v`. ``stddev`` is the standard deviation of each
    Cartesian component in units of :math:`L_1`.

    Args:
        rngs: Random number generator key.
        x: Fractional torus coordinates with final axis ``(u, v)``.
        stddev: Cartesian proposal standard deviation in units of ``L1``.
        tau: Torus modular parameter in the upper half-plane.

    Returns:
        Proposed coordinates, wrapped into the fractional unit cell.
    """
    return jax.tree.map(lambda a: _torus_move(rngs, a, stddev, tau), x)

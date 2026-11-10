# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Disk geometry utilities for MCMC sampling on the disk."""

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey


def _disk_move(rngs: PRNGKey, x: jnp.ndarray, stddev: float | jnp.ndarray):
    key_dx, key_dy = jax.random.split(rngs)
    dx = jax.random.normal(key_dx, x[..., 0].shape) * stddev
    dy = jax.random.normal(key_dy, x[..., 1].shape) * stddev
    return jnp.stack([x[..., 0] + dx, x[..., 1] + dy], axis=-1)


def disk_proposal(rngs: PRNGKey, x, stddev: float | jnp.ndarray):
    """Propose MCMC moves on the disk, operating on PyTree leaves.

    Args:
        rngs: Random key.
        x: Current configuration (array or PyTree of arrays).
        stddev: Gaussian width of the move.

    Returns:
        New configuration with the same structure as ``x``.
    """
    return jax.tree.map(lambda a: _disk_move(rngs, a, stddev), x)

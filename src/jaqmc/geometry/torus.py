# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Torus geometry utilities for MCMC sampling on the torus."""

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey


def _torus_move(rngs: PRNGKey, x: jnp.ndarray, stddev: float | jnp.ndarray):
    # The last axis contains the two fractional torus coordinates (u, v),
    # both with unit period.
    displacement = jax.random.normal(rngs, shape=x.shape) * stddev
    # Wrap the Gaussian proposal back into the fundamental domain [0, 1).
    return (x + displacement) % 1.0


def torus_proposal(rngs: PRNGKey, x, stddev: float | jnp.ndarray):
    return jax.tree.map(lambda a: _torus_move(rngs, a, stddev), x)

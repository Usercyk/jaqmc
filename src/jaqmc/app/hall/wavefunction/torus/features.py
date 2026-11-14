# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Periodic input features for torus wavefunctions."""

from jax import numpy as jnp

__all__ = ["torus_features"]


def torus_features(electrons: jnp.ndarray) -> jnp.ndarray:
    r"""Build periodic one-electron features.

    Args:
        electrons: Fractional coordinates with shape ``(..., n_electrons, 2)``.

    Returns:
        Four features per electron: cosine/sine pairs for its two fractional
        coordinates. The output shape is ``(..., n_electrons, 4)``.
    """
    u, v = electrons[..., 0], electrons[..., 1]
    return jnp.stack(
        [
            jnp.cos(2.0 * jnp.pi * u),
            jnp.sin(2.0 * jnp.pi * u),
            jnp.cos(2.0 * jnp.pi * v),
            jnp.sin(2.0 * jnp.pi * v),
        ],
        axis=-1,
    )

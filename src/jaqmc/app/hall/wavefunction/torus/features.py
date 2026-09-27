# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Periodic input features for torus wavefunctions."""

from jax import numpy as jnp

__all__ = ["center_of_mass_features", "torus_features"]


def center_of_mass_features(electrons: jnp.ndarray) -> jnp.ndarray:
    r"""Return periodic center-of-mass features for every electron.

    ``electrons`` contains fractional lattice coordinates ``(u, v)``. The
    center of mass is represented by the phases associated with
    ``sum_i u_i`` and ``sum_i v_i``. Using the sum rather than the arithmetic
    mean makes the representation periodic when any electron is translated by
    a primitive lattice vector.

    Args:
        electrons: Fractional coordinates with shape ``(..., n_electrons, 2)``.

    Returns:
        The four features ``(cos(2 pi sum u), sin(2 pi sum u),
        cos(2 pi sum v), sin(2 pi sum v))``, broadcast to shape
        ``(..., n_electrons, 4)``.
    """
    center = jnp.sum(electrons, axis=-2)
    phase = 2.0 * jnp.pi * center
    features = jnp.stack(
        [
            jnp.cos(phase[..., 0]),
            jnp.sin(phase[..., 0]),
            jnp.cos(phase[..., 1]),
            jnp.sin(phase[..., 1]),
        ],
        axis=-1,
    )
    return jnp.broadcast_to(features[..., None, :], (*electrons.shape[:-1], 4))


def torus_features(electrons: jnp.ndarray) -> jnp.ndarray:
    r"""Build periodic one-electron and center-of-mass features.

    Args:
        electrons: Fractional coordinates with shape ``(..., n_electrons, 2)``.

    Returns:
        Eight features per electron: cosine/sine pairs for its two fractional
        coordinates, followed by four center-of-mass features shared by all
        electrons. The output shape is ``(..., n_electrons, 8)``.
    """
    u, v = electrons[..., 0], electrons[..., 1]
    one_electron_features = jnp.stack(
        [
            jnp.cos(2.0 * jnp.pi * u),
            jnp.sin(2.0 * jnp.pi * u),
            jnp.cos(2.0 * jnp.pi * v),
            jnp.sin(2.0 * jnp.pi * v),
        ],
        axis=-1,
    )
    return jnp.concatenate(
        [one_electron_features, center_of_mass_features(electrons)], axis=-1
    )

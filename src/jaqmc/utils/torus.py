# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from jax import numpy as jnp


def get_torus_lattice(
    flux: int, tau: complex
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Return the torus lattice vectors :math:`\mathbf{L}_1` and :math:`\mathbf{L}_2`.

    Args:
        flux: The number of magnetic flux quanta :math:`N_\phi`.
        tau: The complex torus modular parameter :math:`\tau`.

    Returns:
        A tuple of two numbers representing the torus lattice vectors.
    """
    l1 = jnp.sqrt(2 * jnp.pi * flux / tau.imag)
    l2 = l1 * tau
    area = l1 * l2.imag
    return l1, l2, area

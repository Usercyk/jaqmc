# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math

from jax import numpy as jnp


def validate_shear(shear: tuple[float, float]) -> None:
    """Check the two real components of a constant traceless shear.

    Raises:
        ValueError: If the shear does not contain two finite real numbers.
    """
    if (
        not isinstance(shear, (tuple, list))
        or len(shear) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in shear
        )
    ):
        raise ValueError("shear must contain two finite real numbers.")


def shear_matrix(shear: jnp.ndarray | tuple[float, float]) -> jnp.ndarray:
    r"""Return :math:`\exp[[s_1,s_2],[s_2,-s_1]]`, including at zero shear.

    The result is symmetric, positive definite and has unit determinant.
    This function supports JAX differentiation with respect to both components.

    Args:
        shear: Two real traceless-shear components.

    Returns:
        A ``(2, 2)`` unimodular matrix.
    """
    s = jnp.asarray(shear)
    s = s.astype(jnp.result_type(s.dtype, 1.0))
    squared = jnp.sum(s**2)
    # h^2 = |s|^2 I. Series in |s|^2 avoid singular derivatives of |s| at zero.
    # Keep the unused square-root branch finite as well for higher-order AD.
    radius = jnp.sqrt(jnp.maximum(squared, 1e-4))
    cosh = jnp.where(
        squared < 1e-4,
        1 + squared * (0.5 + squared * (1 / 24 + squared / 720)),
        jnp.cosh(radius),
    )
    sinhc = jnp.where(
        squared < 1e-4,
        1 + squared * (1 / 6 + squared * (1 / 120 + squared / 5040)),
        jnp.sinh(radius) / radius,
    )
    return cosh * jnp.eye(2) + sinhc * jnp.array([[s[0], s[1]], [s[1], -s[0]]])


def get_torus_lattice(
    flux: int, tau: complex
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Return the torus lattice vectors :math:`\mathbf{L}_1` and :math:`\mathbf{L}_2`.

    Args:
        flux: The number of magnetic flux quanta :math:`N_\phi`.
        tau: The complex torus modular parameter :math:`\tau`.

    Returns:
        A tuple of two numbers representing the torus lattice vectors.

    Raises:
        ValueError: If flux is not a positive integer or tau is not a finite
            modular parameter in the upper half-plane.
    """
    if not isinstance(flux, int) or isinstance(flux, bool) or flux <= 0:
        raise ValueError(f"flux must be a positive integer. Got {flux!r}.")

    tau = complex(tau)
    if not math.isfinite(tau.real) or not math.isfinite(tau.imag):
        raise ValueError(f"tau must be finite. Got {tau!r}.")
    if tau.imag <= 0:
        raise ValueError(
            "tau must lie in the upper half-plane, so its imaginary "
            f"part must be positive. Got {tau!r}."
        )

    l1 = jnp.sqrt(2 * jnp.pi * flux / tau.imag)
    l2 = l1 * tau
    area = l1 * l2.imag
    return l1, l2, area

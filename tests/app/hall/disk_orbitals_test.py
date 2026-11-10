# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Disk orbital derivatives agree with finite differences at the origin."""

from typing import cast

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.wavefunction.disk.mhpo import DiskOrbitals, EnvelopeType


@pytest.mark.x64_modes
@pytest.mark.parametrize("radius", [0.0, 1.0], ids=["origin", "scale_boundary"])
@pytest.mark.parametrize(
    "mode",
    [
        EnvelopeType.m_learnable,
        EnvelopeType.m_learnable2,
        EnvelopeType.m_learnable2_old,
    ],
)
def test_orbital_derivatives_at_origin_and_scale_boundary(
    x64_mode: bool, mode: EnvelopeType, radius: float
) -> None:
    dtype = jnp.float64 if x64_mode else jnp.float32
    h_one = jnp.ones((2, 3), dtype=dtype)
    point = jnp.array([radius, 0.0], dtype=dtype)
    orbitals = DiskOrbitals(Q=4, nspins=(2, 0), ndets=1, envelope_type=mode)
    x, y = jnp.array([0.0, 1.0], dtype=dtype), jnp.array([0.0, 0.5], dtype=dtype)
    params = orbitals.init(jax.random.key(0), h_one, x, y)

    def evaluate(position: jax.Array) -> jax.Array:
        return cast(
            jax.Array,
            orbitals.apply(
                params, h_one, x.at[0].set(position[0]), y.at[0].set(position[1])
            ),
        )

    evaluate = jax.jit(evaluate)
    jacobian = jax.jit(jax.jacfwd(evaluate))(point)
    hessian = jax.jit(jax.jacfwd(jax.jacfwd(evaluate)))(point)
    step = 1e-3 if x64_mode else 2e-2
    directions = jnp.eye(2, dtype=dtype) * step
    first = jnp.stack(
        [(evaluate(point + v) - evaluate(point - v)) / (2 * step) for v in directions],
        axis=-1,
    )
    second = jnp.stack(
        [
            jnp.stack(
                [
                    (
                        evaluate(point + v + w)
                        - evaluate(point + v - w)
                        - evaluate(point - v + w)
                        + evaluate(point - v - w)
                    )
                    / (4 * step**2)
                    for w in directions
                ],
                axis=-1,
            )
            for v in directions
        ],
        axis=-2,
    )
    atol = 1e-5 if x64_mode else 5e-3
    assert np.isfinite(jacobian).all()
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(jacobian, first, rtol=atol, atol=atol)
    np.testing.assert_allclose(hessian, second, rtol=atol, atol=atol)


def test_high_angular_momenta_preserve_finite_nonzero_orbitals() -> None:
    orbitals = DiskOrbitals(
        Q=200, nspins=(1, 0), ndets=1, envelope_type=EnvelopeType.m_learnable2
    )
    h_one = jnp.ones((1, 3))
    x, y = jnp.array([20.0]), jnp.array([0.0])
    params = orbitals.init(jax.random.key(0), h_one, x, y)
    value = jax.jit(orbitals.apply)(params, h_one, x, y)
    assert np.isfinite(value).all()
    assert np.all(np.abs(value) > 1e-6)

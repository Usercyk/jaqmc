# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.geometry.torus import torus_proposal


def test_proposal_is_isotropic_in_cartesian_coordinates():
    key = jax.random.PRNGKey(42)
    x = jnp.full((128, 2), 0.5)
    stddev = 1e-3
    tau = 0.5 + 0.5j * float(jnp.sqrt(3.0))

    proposed = torus_proposal(key, x, stddev, tau)
    fractional_displacement = proposed - x
    cartesian_displacement = jnp.stack(
        (
            fractional_displacement[..., 0]
            + tau.real * fractional_displacement[..., 1],
            tau.imag * fractional_displacement[..., 1],
        ),
        axis=-1,
    )

    expected = jax.random.normal(key, shape=x.shape) * stddev
    np.testing.assert_allclose(cartesian_displacement, expected, atol=1e-7)


def test_square_torus_matches_fractional_isotropic_proposal():
    key = jax.random.PRNGKey(7)
    x = jnp.array([[0.01, 0.99], [0.4, 0.6]])
    stddev = 0.1

    proposed = torus_proposal(key, x, stddev, 1j)
    expected = (x + jax.random.normal(key, shape=x.shape) * stddev) % 1.0

    np.testing.assert_allclose(proposed, expected)
    assert jnp.all((proposed >= 0.0) & (proposed < 1.0))

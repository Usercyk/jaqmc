# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for torus magnetic orbitals and the MHPO wavefunction."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.wavefunction.torus.mhpo import TorusMHPO, torus_envelope
from jaqmc.utils.torus import get_torus_lattice
from jaqmc.utils.wiring import wire

_FLUX = 3
_TAU = 0.31 + 1.07j
_THETA_TERMS = 48


def _sample_positions() -> jnp.ndarray:
    l1, l2, _ = get_torus_lattice(_FLUX, _TAU)
    u = jnp.array([0.13, 0.57])
    v = jnp.array([0.29, 0.71])
    return u * l1 + v * l2


class TestTorusEnvelope:
    def test_shape_dtype_and_derivatives(self):
        z = _sample_positions()
        values = jax.jit(
            lambda positions: torus_envelope(positions, _FLUX, _TAU, _THETA_TERMS)
        )(z)

        assert values.shape == (z.shape[0], 3 * _FLUX)
        assert jnp.iscomplexobj(values)
        assert jnp.all(jnp.isfinite(values))

        def evaluate(real_coordinates):
            position = real_coordinates[0] + 1j * real_coordinates[1]
            return torus_envelope(position, _FLUX, _TAU, _THETA_TERMS)

        point = jnp.array([0.37, 0.41])
        jacobian = jax.jacfwd(evaluate)(point)
        hessian = jax.jacfwd(jax.jacfwd(evaluate))(point)
        assert jnp.all(jnp.isfinite(jacobian))
        assert jnp.all(jnp.isfinite(hessian))

    def test_magnetic_boundary_conditions(self):
        z = _sample_positions()
        values = torus_envelope(z, _FLUX, _TAU, _THETA_TERMS)
        l1, l2, _ = get_torus_lattice(_FLUX, _TAU)

        for period in (l1, l2):
            phase = (-1) ** _FLUX * jnp.exp(
                (jnp.conj(period) * z - period * jnp.conj(z)) / 4
            )
            translated = torus_envelope(z + period, _FLUX, _TAU, _THETA_TERMS)
            np.testing.assert_allclose(
                translated,
                phase[:, None] * values,
                rtol=1e-5,
                atol=1e-5,
            )

    @pytest.mark.requires_x64
    def test_first_three_landau_levels_are_orthonormal(self):
        grid_size = 48
        grid = (jnp.arange(grid_size) + 0.5) / grid_size
        u, v = jnp.meshgrid(grid, grid, indexing="ij")
        l1, l2, area = get_torus_lattice(_FLUX, _TAU)
        z = (u * l1 + v * l2).reshape(-1)

        values = torus_envelope(z, _FLUX, _TAU, _THETA_TERMS)
        overlap = area / grid_size**2 * (jnp.conj(values).T @ values)

        np.testing.assert_allclose(
            overlap,
            jnp.eye(3 * _FLUX),
            rtol=2e-12,
            atol=2e-12,
        )

    @pytest.mark.parametrize(
        ("flux", "tau"),
        [(0, _TAU), (_FLUX, 0j), (_FLUX, complex(float("inf"), 1))],
    )
    def test_rejects_invalid_geometry(self, flux, tau):
        with pytest.raises(ValueError):
            torus_envelope(jnp.array([0.1 + 0.2j]), flux, tau, _THETA_TERMS)


def test_torus_mhpo_is_antisymmetric():
    wavefunction = TorusMHPO(ndets=2, num_heads=2, heads_dim=8, num_layers=1)
    wire(wavefunction, nspins=(2, 0), flux=_FLUX, tau=_TAU)

    electrons = jnp.array([[0.17, 0.23], [0.61, 0.74]])
    data = HallData(electrons=electrons)
    params = wavefunction.init_params(data, jax.random.PRNGKey(0))

    logpsi = wavefunction.evaluate(params, data)["logpsi"]
    swapped_data = HallData(electrons=electrons[::-1])
    swapped_logpsi = wavefunction.evaluate(params, swapped_data)["logpsi"]
    exchange_ratio = jnp.exp(swapped_logpsi - logpsi)

    np.testing.assert_allclose(exchange_ratio, -1, rtol=1e-5, atol=1e-5)

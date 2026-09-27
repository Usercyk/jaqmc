# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for sigma-form torus orbitals and input features."""

import warnings

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.wavefunction.torus.features import (
    center_of_mass_features,
    torus_features,
)
from jaqmc.app.hall.wavefunction.torus.sigma_mhpo import (
    TorusSigmaMHPO,
    torus_sigma_envelope,
)
from jaqmc.utils.torus import get_torus_lattice
from jaqmc.utils.wiring import wire


def test_torus_features_include_periodic_center_of_mass_by_default():
    electrons = jnp.array([[0.13, 0.27], [0.41, 0.59]])

    features = torus_features(electrons)

    assert features.shape == (2, 8)
    np.testing.assert_allclose(
        features[:, 4:],
        center_of_mass_features(electrons),
        rtol=1e-6,
        atol=1e-6,
    )

    for electron in range(electrons.shape[0]):
        shifted = electrons.at[electron].add(jnp.array([1.0, -1.0]))
        np.testing.assert_allclose(
            torus_features(shifted),
            features,
            rtol=1e-6,
            atol=1e-6,
        )


def test_sigma_envelope_uses_the_standard_symmetric_gauge_origin():
    flux = 3
    tau = 1.3j
    l1, l2, _ = get_torus_lattice(flux, tau)
    z = jnp.array([0.13 * l1 + 0.29 * l2, 0.57 * l1 + 0.71 * l2])
    values = torus_sigma_envelope(z, flux, tau)

    for period in (l1, l2):
        phase = (-1) ** flux * jnp.exp(
            (jnp.conj(period) * z - period * jnp.conj(z)) / 4
        )
        translated = torus_sigma_envelope(z + period, flux, tau)
        np.testing.assert_allclose(
            translated,
            phase[:, None] * values,
            rtol=2e-5,
            atol=1e-5,
        )


def test_sigma_mhpo_uses_the_standard_symmetric_gauge_origin():
    flux = 3
    tau = 1j
    wavefunction = TorusSigmaMHPO(
        ndets=1,
        num_heads=1,
        heads_dim=4,
        num_layers=1,
        theta_terms=12,
        eisen_terms=24,
    )
    wire(wavefunction, nspins=(1, 0), flux=flux, tau=tau)
    electrons = jnp.array([[0.17, 0.23]])
    data = HallData(electrons=electrons)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        params = wavefunction.init_params(data, jax.random.PRNGKey(0))
        logpsi = wavefunction.evaluate(params, data)["logpsi"]

        l1, l2, _ = get_torus_lattice(flux, tau)
        z = electrons[0, 0] * l1 + electrons[0, 1] * l2
        for shift, period in ((jnp.array([1.0, 0.0]), l1), (jnp.array([0.0, 1.0]), l2)):
            translated = wavefunction.evaluate(
                params,
                HallData(electrons=electrons.at[0].add(shift)),
            )["logpsi"]
            ratio = jnp.exp(translated - logpsi)
            expected = (-1) ** flux * jnp.exp(
                (jnp.conj(period) * z - period * jnp.conj(z)) / 4
            )
            np.testing.assert_allclose(ratio, expected, rtol=2e-5, atol=1e-5)

# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU analytic checks of torus metrics, stresses, and product-state actions."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.config import HallTorusConfig
from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.torus import TorusGraviton
from jaqmc.app.hall.hamiltonian.torus import TorusPotential
from jaqmc.app.hall.wavefunction.torus.mhpo import torus_envelope
from jaqmc.estimator.ewald import EwaldSum2D
from jaqmc.estimator.kinetic.torus import TorusKinetic
from jaqmc.utils.torus import get_torus_lattice, shear_matrix


@pytest.mark.parametrize("shear", [(0.0, 0.0), (0.3, -0.2)])
def test_metric_and_cartesian_kinetic_match_analytic_gaussian(shear):
    flux, tau = 6, 0.3 + 1.2j
    l1, l2, _ = get_torus_lattice(flux, tau)
    lattice = jnp.array([[l1, 0], [l2.real, l2.imag]])
    data = HallData(electrons=jnp.array([[0.13, 0.21], [0.37, 0.42]]))
    alpha = 0.7 + 0.2j

    def logpsi(params, d):
        del params
        return -alpha * jnp.sum((d.electrons @ lattice) ** 2) / 2

    kinetic = TorusKinetic(flux=flux, tau=tau, mass_shear=shear, f_log_psi=logpsi)
    stats, _ = kinetic.evaluate_single_walker({}, data, {}, None, jax.random.key(0))
    metric = shear_matrix(shear)
    np.testing.assert_allclose(jnp.linalg.det(metric), 1.0, atol=2e-7)
    cartesian = data.electrons @ lattice
    covariant = -alpha * cartesian + 0.5j * jnp.stack(
        [cartesian[:, 1], -cartesian[:, 0]], axis=-1
    )
    expected = len(cartesian) * alpha * jnp.trace(metric) / 2
    expected -= jnp.einsum("ab,ia,ib->", metric, covariant, covariant) / 2
    np.testing.assert_allclose(stats["energy:kinetic"], expected, rtol=3e-6)
    _, tensor = kinetic.covariant_derivatives({}, data)
    steps = jnp.eye(2) * 1e-3
    finite_difference = jnp.array(
        [
            -0.5
            * jnp.einsum(
                "ab,iab->",
                (
                    kinetic.fractional_metric(jnp.array(shear) + step)
                    - kinetic.fractional_metric(jnp.array(shear) - step)
                )
                / 2e-3,
                tensor,
            )
            for step in steps
        ]
    )
    np.testing.assert_allclose(kinetic.stress(tensor), finite_difference, rtol=1e-3)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_isotropic_kinetic_has_exact_landau_energies(level):
    flux, tau = 4, 0.25 + 1.1j
    l1, _, _ = get_torus_lattice(flux, tau)

    def logpsi(params, data):
        del params
        u, v = data.electrons[0]
        orbitals = torus_envelope(l1 * (u + tau * v), flux, tau, 6)
        return jnp.log(orbitals[level * flux])

    kinetic = TorusKinetic(flux=flux, tau=tau, f_log_psi=logpsi)
    data = HallData(electrons=jnp.array([[0.13, 0.27]]))
    stats, _ = jax.jit(kinetic.evaluate_single_walker)(
        {}, data, {}, None, jax.random.key(0)
    )
    np.testing.assert_allclose(stats["energy:kinetic"], level + 0.5, atol=3e-5)


@pytest.mark.parametrize("shear", [(0.0, 0.0), (0.35, -0.21)])
def test_deformed_ewald_and_shear_derivatives(shear):
    data = HallData(electrons=jnp.array([[0.12, 0.23], [0.45, 0.61]]))
    potential = TorusPotential(
        flux=6, tau=0.3 + 1.1j, interaction_shear=shear, ewald_gmax=16, nlatvec=3
    )
    potential.init(data, jax.random.key(0))
    transform = shear_matrix(jnp.array(shear) / 2)
    lattice = potential.lattice @ transform.T
    independent = EwaldSum2D(lattice, ewald_gmax=16, nlatvec=3)
    expected = independent.energy(data.electrons @ lattice, -jnp.ones(2))
    np.testing.assert_allclose(potential.potential(data.electrons), expected, atol=2e-6)
    derivative = potential.stress(data.electrons)
    steps = jnp.eye(2) * 1e-3
    finite_difference = jnp.array(
        [
            (
                potential.potential(data.electrons, jnp.array(shear) + step)
                - potential.potential(data.electrons, jnp.array(shear) - step)
            )
            / 2e-3
            for step in steps
        ]
    )
    np.testing.assert_allclose(derivative, finite_difference, atol=1e-4, rtol=2e-3)
    np.testing.assert_allclose(
        potential.stress(data.electrons.at[0].add(jnp.array([1, -1]))),
        derivative,
        atol=2e-6,
    )
    hessian = jax.jit(jax.jacfwd(jax.jacfwd(potential.stress)))(data.electrons)
    assert np.all(np.isfinite(hessian))


def test_product_subspace_action_matches_direct_wavefunction_derivatives():
    data = HallData(electrons=jnp.array([[0.12, 0.23], [0.45, 0.61]]))

    def logpsi(params, d):
        del params
        return -(0.7 + 0.2j) * jnp.sum(d.electrons**2)

    estimator = TorusGraviton(
        flux=6,
        tau=0.3 + 1.1j,
        mass_shear=(0.2, -0.1),
        interaction_shear=(0.1, 0.15),
        ewald_gmax=6,
        f_log_psi=logpsi,
    )
    estimator.init(data, jax.random.key(0))
    stats, _ = jax.jit(estimator.evaluate_single_walker)(
        {}, data, {}, None, jax.random.key(0)
    )

    def product_wavefunctions(x):
        factors = estimator.trial_factors(x)
        return jnp.exp(logpsi({}, data.merge({"electrons": x}))) * factors

    values = product_wavefunctions(data.electrons)
    gradient = jax.jacfwd(product_wavefunctions)(data.electrons)
    hessian = jax.jacfwd(jax.jacfwd(product_wavefunctions))(data.electrons)
    connection = (
        jnp.pi
        * estimator.flux
        * jnp.stack([data.electrons[:, 1], -data.electrons[:, 0]], axis=-1)
    )
    tensor = jnp.einsum("aibic->aibc", hessian)
    tensor += 1j * jnp.einsum("ib,aic->aibc", connection, gradient)
    tensor += 1j * jnp.einsum("ic,aib->aibc", connection, gradient)
    tensor -= jnp.einsum("ib,ic,a->aibc", connection, connection, values)
    action = -0.5 * jnp.einsum(
        "bc,aibc->a", estimator.kinetic.fractional_metric(), tensor
    )
    action += estimator.potential.potential(data.electrons) * values
    action /= jnp.exp(logpsi({}, data))
    np.testing.assert_allclose(stats["graviton:hamiltonian"][0], action, rtol=2e-5)
    np.testing.assert_allclose(
        stats["graviton:hamiltonian_square"],
        jnp.outer(action.conj(), action),
        rtol=5e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        stats["graviton:unprojected_norm"][0], stats["graviton:unprojected_norm"][1]
    )
    np.testing.assert_allclose(
        stats["graviton:gram"][0],
        estimator.trial_factors(data.electrons),
        rtol=1e-5,
        atol=2e-7,
    )


def test_coulomb_trial_factors_vanish_at_linear_pair_nodes_and_are_periodic():
    data = HallData(electrons=jnp.array([[0.2, 0.3], [0.4, 0.6]]))
    estimator = TorusGraviton(
        flux=6,
        ewald_gmax=6,
        trial_radius=0.5,
        f_log_psi=lambda p, d: -jnp.sum(d.electrons**2),
    )
    estimator.init(data, jax.random.key(0))
    l1, _, _ = get_torus_lattice(6, 1j)
    large = data.electrons.at[1].set(data.electrons[0] + jnp.array([0.01 / l1, 0.0]))
    small = data.electrons.at[1].set(data.electrons[0] + jnp.array([0.005 / l1, 0.0]))
    ratio = estimator.trial_factors(small)[1] / estimator.trial_factors(large)[1]
    np.testing.assert_allclose(ratio, 0.5, rtol=2e-3)
    # Bare stress instead grows as 1/r. The Hamiltonian and transition source
    # retain this physical stress while only the variational basis is bounded.
    np.testing.assert_allclose(
        estimator.potential.stress(small)[0] / estimator.potential.stress(large)[0],
        2.0,
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        estimator.trial_factors(data.electrons.at[0].add(jnp.array([1, -1]))),
        estimator.trial_factors(data.electrons),
        atol=2e-6,
    )


@pytest.mark.parametrize("field", ["mass_shear", "interaction_shear"])
@pytest.mark.parametrize("shear", [(float("nan"), 0.0), (0.0,), (1j, 0.0)])
def test_rejects_invalid_metric_parameters(field, shear):
    with pytest.raises(ValueError, match="shear"):
        HallTorusConfig(**{field: shear})

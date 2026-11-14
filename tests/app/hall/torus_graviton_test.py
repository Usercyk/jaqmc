# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact finite-dimensional checks of projection and quadrupole spectra."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.torus import TorusGraviton
from jaqmc.app.hall.graviton import solve_graviton_subspace


def test_nonorthogonal_ground_manifold_recovers_exact_chiral_spectrum():
    basis = np.array(
        [[1, 1, 0.3, -0.2j], [0, 0.8, 0.1, 0.6], [0, 0, 1, 0.5], [0, 0, 0.4j, 1]],
        dtype=complex,
    )
    energy = np.array([-1.0, -1.0, 0.2, 1.4])
    sources = np.array([[0.5, 0.2j], [0.3, 0.7], [2.0, 0.4], [0.5j, 3.0]])
    gram = basis.conj().T @ basis
    hamiltonian = basis.conj().T @ np.diag(energy) @ basis
    transitions = basis.conj().T @ sources
    spectrum = solve_graviton_subspace(
        gram,
        hamiltonian,
        transitions,
        n_electrons=2,
        ground_indices=(0, 1),
        ground_energy=-1.0,
        hamiltonian_square=basis.conj().T @ np.diag(energy**2) @ basis,
    )

    np.testing.assert_allclose(spectrum.excitation_energies, [1.2, 2.4], atol=1e-12)
    np.testing.assert_allclose(spectrum.weights, np.abs(sources[2:]) ** 2 / 2)
    assert spectrum.energy_variances is not None
    assert spectrum.residual_norms is not None
    np.testing.assert_allclose(spectrum.energy_variances, 0, atol=1e-12)
    np.testing.assert_allclose(spectrum.residual_norms, 0, atol=1e-7)
    np.testing.assert_allclose(
        spectrum.coefficients.conj().T @ gram @ spectrum.coefficients,
        np.eye(2),
        atol=1e-12,
    )
    np.testing.assert_allclose((basis @ spectrum.coefficients)[:2], 0, atol=1e-12)
    np.testing.assert_allclose(spectrum.low_energy_weights(1.5), [2.0, 0.08])
    assert spectrum.chirality(1.5) == pytest.approx((0.08 - 2) / 2.08)
    assert np.isnan(spectrum.chirality(0.5))
    grid = np.array([1.2, 2.4])
    lorentzian = 0.1 / (np.pi * ((grid[:, None] - np.array([1.2, 2.4])) ** 2 + 0.01))
    np.testing.assert_allclose(
        spectrum.broaden(grid, 0.1), lorentzian @ spectrum.weights
    )

    # A single reference leaves the other zero-energy partner in the spectrum.
    incomplete = solve_graviton_subspace(
        gram, hamiltonian, transitions, n_electrons=2, ground_indices=(0,)
    )
    np.testing.assert_allclose(
        incomplete.excitation_energies, [0, 1.2, 2.4], atol=1e-12
    )


def test_gram_truncation_and_unprojected_hamiltonian_residual():
    # The trial mixes with an approximate reference. Keeping all four H blocks
    # is required when the reference is not an exact eigenstate.
    basis = np.array([[1, 0.5, 0.5], [0, 1, 1]])
    gram = basis.T @ basis
    operator = np.array([[0.0, 0.3j], [-0.3j, 2.0]])
    hamiltonian = basis.T @ operator @ basis
    sources = basis.T @ np.array([[0.2, -0.1j], [1, 2j]])
    spectrum = solve_graviton_subspace(
        gram,
        hamiltonian,
        sources,
        n_electrons=1,
        hamiltonian_square=basis.T @ operator @ operator @ basis,
    )
    np.testing.assert_allclose(spectrum.excitation_energies, [2.0])
    np.testing.assert_allclose(spectrum.weights, [[1, 4]])
    assert len(spectrum.retained_gram_eigenvalues) == 1
    assert abs(spectrum.gram_eigenvalues[0]) < 1e-12
    assert spectrum.energy_variances is not None
    assert spectrum.residual_norms is not None
    np.testing.assert_allclose(spectrum.energy_variances, [0.09], atol=1e-12)
    np.testing.assert_allclose(spectrum.residual_norms, [0.3], atol=1e-12)


@pytest.mark.parametrize("energy_offset", [0.0, 100.0])
def test_full_residual_includes_reference_and_out_of_basis_components(
    energy_offset: float,
) -> None:
    basis = np.array(
        [[1, 0.3, -0.2j], [0, 2, 0.4], [0, 0, 1], [0, 0, 0]], dtype=complex
    )
    operator = np.array(
        [
            [0.4, 0.2j, 0, 0],
            [-0.2j, 2, 0, 0.7 + 0.2j],
            [0, 0, 3, 0],
            [0, 0.7 - 0.2j, 0, 4],
        ]
    ) + energy_offset * np.eye(4)
    actions = operator @ basis
    spectrum = solve_graviton_subspace(
        basis.conj().T @ basis,
        basis.conj().T @ actions,
        np.zeros((3, 2)),
        n_electrons=1,
        hamiltonian_square=actions.conj().T @ actions,
    )
    states = basis @ spectrum.coefficients
    energies = spectrum.excitation_energies + spectrum.ground_energy
    residuals = operator @ states - states * energies
    assert spectrum.energy_variances is not None
    assert spectrum.residual_norms is not None
    np.testing.assert_allclose(spectrum.excitation_energies, [1.6, 2.6], atol=1e-12)
    np.testing.assert_allclose(
        spectrum.energy_variances, np.sum(np.abs(residuals) ** 2, axis=0), atol=1e-10
    )
    np.testing.assert_allclose(spectrum.energy_variances, [0.57, 0], atol=1e-10)
    np.testing.assert_allclose(
        spectrum.residual_norms, np.linalg.norm(residuals, axis=0), atol=1e-5
    )
    # The projected Ritz equation is exact despite the nonzero full residual.
    np.testing.assert_allclose(states.conj().T @ residuals, 0, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_energy_variance_clips_only_roundoff_sized_negative_moments(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    second_moment = np.nextafter(dtype(1), dtype(0))
    spectrum = solve_graviton_subspace(
        np.eye(2, dtype=dtype),
        np.diag(np.array([0, 1], dtype=dtype)),
        np.zeros((2, 2)),
        n_electrons=1,
        hamiltonian_square=np.diag(np.array([0, second_moment], dtype=dtype)),
    )
    assert spectrum.energy_variances is not None
    assert spectrum.residual_norms is not None
    np.testing.assert_array_equal(spectrum.energy_variances, [0])
    np.testing.assert_array_equal(spectrum.residual_norms, [0])


def test_zero_excited_space_returns_no_peaks():
    spectrum = solve_graviton_subspace(
        np.diag([1.0, 0.0]),
        np.diag([0.5, 0.0]),
        np.zeros((2, 2)),
        n_electrons=1,
        hamiltonian_square=np.diag([0.25, 0.0]),
    )
    assert spectrum.weights.shape == (0, 2)
    np.testing.assert_array_equal(spectrum.broaden(np.linspace(0, 1, 4), 0.1), 0)
    assert np.isnan(spectrum.chirality(1.0))
    assert spectrum.energy_variances is not None
    assert spectrum.residual_norms is not None
    assert spectrum.energy_variances.shape == (0,)
    assert spectrum.residual_norms.shape == (0,)


def test_float32_constant_operators_do_not_generate_spurious_modes():
    factors = np.array([1.0, 0.2, -0.3], dtype=np.float32)
    gram = np.outer(factors, factors)
    spectrum = solve_graviton_subspace(
        gram, gram * 0.5, np.zeros((3, 2)), n_electrons=1
    )
    assert spectrum.excitation_energies.size == 0
    assert spectrum.energy_variances is None
    assert spectrum.residual_norms is None


def test_evaluation_digest_reports_per_electron_peaks_and_subspace_moments():
    estimator = TorusGraviton(
        ewald_gmax=1, f_log_psi=lambda p, d: -jnp.sum(d.electrons**2)
    )
    estimator.init(
        HallData(electrons=jnp.array([[0.1, 0.2], [0.4, 0.6]])), jax.random.key(0)
    )
    matrices = {
        "graviton:gram": np.diag([1.0, 4.0, 9.0]),
        "graviton:hamiltonian": np.diag([1.0, 8.0, 36.0]),
        "graviton:hamiltonian_square": np.diag([1.0, 17.0, 180.0]),
        "graviton:transitions": np.array([[0, 0], [2, 4j], [3j, 3]]),
    }
    digest = estimator.finalize_stats(
        {name: np.stack([matrix, matrix]) for name, matrix in matrices.items()}, None
    )
    np.testing.assert_allclose(digest["graviton:excitation_energies"], [1, 3])
    np.testing.assert_allclose(digest["graviton:weights"], [[0.5, 2], [0.5, 0.5]])
    np.testing.assert_allclose(digest["graviton:subspace_weight"], [1, 2.5])
    np.testing.assert_allclose(digest["graviton:subspace_first_moment"], [2, 3.5])
    np.testing.assert_allclose(digest["graviton:subspace_centroid"], [2, 1.4])
    np.testing.assert_allclose(digest["graviton:energy_variances"], [0.25, 4])
    np.testing.assert_allclose(digest["graviton:residual_norms"], [0.5, 2])
    assert digest["graviton:n_ground_references"] == 1


@pytest.mark.parametrize(
    "gram,kwargs,match",
    [
        (np.diag([1.0, -0.1]), {}, "negative eigenvalue"),
        (np.eye(2), {"ground_indices": (0, 1)}, "ground_energy is required"),
        (np.eye(2), {"rcond": 0.0}, "rcond"),
        (np.eye(2), {"ground_indices": (2,)}, "ground_indices"),
        (np.array([[1, np.nan], [np.nan, 1]]), {}, "finite"),
        (np.eye(2), {"hamiltonian_square": np.eye(3)}, "basis shape"),
        (np.eye(2), {"hamiltonian_square": np.full((2, 2), np.nan)}, "finite"),
        (np.eye(2), {"hamiltonian_square": np.zeros((2, 2))}, "negative Ritz"),
    ],
)
def test_rejects_invalid_spectral_inputs(gram, kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve_graviton_subspace(
            gram, np.eye(2), np.zeros((2, 2)), n_electrons=1, **kwargs
        )

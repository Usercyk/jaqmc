# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Ground-space projection and finite-subspace quadrupole spectra.

These host-side routines consume averaged matrix elements, including matrices
from independently constructed multi-reference bases. They do not assume the
reference wavefunction is an exact eigenstate or perform any optimization.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GravitonSpectrum:
    """Ritz excitations and per-electron weights in ``(plus, minus)`` order.

    Attributes:
        excitation_energies: Ritz energies relative to the reference energy.
        weights: Nonnegative peak weights with shape ``(n_modes, 2)``.
        coefficients: Eigenvectors in the original input basis.
        gram_eigenvalues: All projected Gram eigenvalues before truncation,
            after normalizing trial columns by their unprojected norms.
        retained_gram_eigenvalues: Projected Gram eigenvalues kept by the solver.
        reference_gram_eigenvalues: Eigenvalues of the reference overlap block.
        hermiticity_error: Relative anti-Hermitian norms of the input Gram and
            Hamiltonian matrices, before symmetrizing, in that order.
        ground_energy: Energy subtracted from the Ritz eigenvalues.
        energy_variances: Full-Hamiltonian energy variances per Ritz state,
            or ``None`` when Hamiltonian-action overlaps were not supplied.
    """

    excitation_energies: np.ndarray
    weights: np.ndarray
    coefficients: np.ndarray
    gram_eigenvalues: np.ndarray
    retained_gram_eigenvalues: np.ndarray
    reference_gram_eigenvalues: np.ndarray
    hermiticity_error: np.ndarray
    ground_energy: float
    energy_variances: np.ndarray | None = None

    @property
    def residual_norms(self) -> np.ndarray | None:
        """Full-Hamiltonian residual norms, or ``None`` without second moments."""
        if self.energy_variances is None:
            return None
        return np.sqrt(self.energy_variances)

    def broaden(self, energies: np.ndarray, eta: float) -> np.ndarray:
        """Evaluate Lorentzian spectra, with an artificial display width.

        Args:
            energies: One-dimensional energy grid in the Hamiltonian units.
            eta: Positive Lorentzian half-width in the same energy units.

        Returns:
            Spectra with shape ``(len(energies), 2)`` in plus/minus order.

        Raises:
            ValueError: If the width or energy grid is invalid.
        """
        energies = np.asarray(energies, dtype=float)
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive.")
        if energies.ndim != 1 or not np.all(np.isfinite(energies)):
            raise ValueError("energies must be a finite one-dimensional grid.")
        differences = energies[:, None] - self.excitation_energies[None, :]
        return (eta / (np.pi * (differences**2 + eta**2))) @ self.weights

    def low_energy_weights(self, cutoff: float) -> np.ndarray:
        """Sum unbroadened peak weights in ``0 < excitation_energy <= cutoff``.

        Returns:
            Two weights in plus/minus order.

        Raises:
            ValueError: If the cutoff is not finite and positive.
        """
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive.")
        selected = (self.excitation_energies > 0) & (self.excitation_energies <= cutoff)
        return self.weights[selected].sum(axis=0)

    def chirality(self, cutoff: float) -> float:
        """Return ``(W_minus - W_plus) / (W_minus + W_plus)`` below a cutoff.

        Returns:
            Low-energy chirality, or NaN when the window has zero weight.
        """
        plus, minus = self.low_energy_weights(cutoff)
        return float((minus - plus) / (minus + plus)) if minus + plus > 0 else np.nan


def _whitener(
    matrix: np.ndarray, rcond: float, *, scale_floor: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Gram whitener, all eigenvalues, and retained eigenvalues.

    Raises:
        ValueError: If the overlap has a significant negative eigenvalue.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(eigenvalues.max(initial=0)), scale_floor)
    tolerance = rcond * scale
    if np.any(eigenvalues < -tolerance):
        raise ValueError("Gram matrix has a significant negative eigenvalue.")
    keep = eigenvalues > tolerance
    return (
        eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]),
        eigenvalues,
        eigenvalues[keep],
    )


def _validate_subspace_inputs(
    gram: np.ndarray,
    hamiltonian: np.ndarray,
    transitions: np.ndarray,
    n_electrons: int,
    ground_indices: tuple[int, ...],
    rcond: float,
) -> None:
    """Validate matrix shapes, finite entries, and solver settings.

    Raises:
        ValueError: If matrix inputs or solver settings are inconsistent.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix.")
    if hamiltonian.shape != gram.shape or transitions.shape != (len(gram), 2):
        raise ValueError("Hamiltonian and transition shapes must match the basis.")
    if not all(np.all(np.isfinite(x)) for x in (gram, hamiltonian, transitions)):
        raise ValueError("All matrix elements must be finite.")
    if (
        not isinstance(n_electrons, int)
        or isinstance(n_electrons, bool)
        or n_electrons <= 0
    ):
        raise ValueError("n_electrons must be a positive integer.")
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must lie strictly between zero and one.")
    if (
        not ground_indices
        or len(set(ground_indices)) != len(ground_indices)
        or any(index < 0 or index >= len(gram) for index in ground_indices)
    ):
        raise ValueError("ground_indices must contain distinct valid basis indices.")


def solve_graviton_subspace(
    gram: np.ndarray,
    hamiltonian: np.ndarray,
    transitions: np.ndarray,
    *,
    n_electrons: int,
    ground_indices: tuple[int, ...] = (0,),
    ground_energy: float | None = None,
    hamiltonian_square: np.ndarray | None = None,
    rcond: float = 1e-6,
) -> GravitonSpectrum:
    r"""Project reference states out and solve :math:`Hc=EGc` on the complement.

    Args:
        gram: Averaged overlaps of the full nonorthogonal basis, ``(B, B)``.
        hamiltonian: Averaged ``<phi_a|H|phi_b>`` matrix, ``(B, B)``.
        transitions: Averaged ``<phi_a|O_+/-|0>`` matrix, ``(B, 2)``. The
            source state must be normalized in the same convention as the basis.
        n_electrons: Electron count for per-electron peak normalization.
        ground_indices: Basis indices spanning the supplied ground space. No
            assumption is made that one state covers the full torus manifold.
        ground_energy: Reference energy. Defaults to ``H[0,0]/G[0,0]`` for a
            single reference, using its actual index. Required for multiple
            references, including split or approximate ground manifolds.
        hamiltonian_square: Optional full-space action overlaps
            ``<H phi_a|H phi_b>`` with shape ``(B, B)``. These provide energy
            variances and residual norms, including components along the
            removed reference space and outside the full input basis.
        rcond: Relative Gram eigenvalue cutoff for both projection and solving.
            Trial columns are equilibrated by their unprojected norms. The
            equilibrated unprojected trial-block norm also bounds the cutoff
            scale to avoid resolving cancellation noise in constant operators.

    Returns:
        Excitation energies, channel weights, and numerical diagnostics. Empty
        excited spaces return empty peak arrays. Negative excitation energies
        are retained as diagnostics of the input basis and reference quality.

    Raises:
        ValueError: If inputs are inconsistent, nonfinite, have an indefinite
            overlap, yield significantly negative energy variances, or do not
            contain a nonzero reference state.
    """
    moment_precision = max(
        (
            np.finfo(np.asarray(matrix).real.dtype).eps
            for matrix in (gram, hamiltonian, hamiltonian_square)
            if matrix is not None
            and np.issubdtype(np.asarray(matrix).dtype, np.inexact)
        ),
        default=np.finfo(float).eps,
    )
    gram = np.asarray(gram, dtype=np.complex128)
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    transitions = np.asarray(transitions, dtype=np.complex128)
    _validate_subspace_inputs(
        gram, hamiltonian, transitions, n_electrons, ground_indices, rcond
    )
    if hamiltonian_square is not None:
        hamiltonian_square = np.asarray(hamiltonian_square, dtype=np.complex128)
        if hamiltonian_square.shape != gram.shape:
            raise ValueError("hamiltonian_square must match the basis shape.")
        if not np.all(np.isfinite(hamiltonian_square)):
            raise ValueError("hamiltonian_square must contain finite entries.")
    size = len(gram)
    hermiticity_error = np.array(
        [
            float(np.linalg.norm(x - x.conj().T)) / max(float(np.linalg.norm(x)), 1e-30)
            for x in (gram, hamiltonian)
        ]
    )
    gram = (gram + gram.conj().T) / 2
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2
    reference = np.asarray(ground_indices)
    trial = np.array([i for i in range(size) if i not in ground_indices], dtype=int)
    reference_gram = gram[np.ix_(reference, reference)]
    reference_whitener, reference_eigenvalues, _ = _whitener(reference_gram, rcond)
    if reference_whitener.shape[1] == 0:
        raise ValueError("The reference space has zero norm.")
    if ground_energy is None:
        if len(reference) != 1:
            raise ValueError("ground_energy is required for multiple references.")
        index = reference[0]
        ground_energy = float((hamiltonian[index, index] / gram[index, index]).real)
    if not np.isfinite(ground_energy):
        raise ValueError("ground_energy must be finite.")

    projection = np.zeros((size, len(trial)), dtype=np.complex128)
    projection[trial, np.arange(len(trial))] = 1
    projection[reference] = -(
        reference_whitener
        @ reference_whitener.conj().T
        @ gram[np.ix_(reference, trial)]
    )
    trial_gram = gram[np.ix_(trial, trial)]
    trial_norms = np.sqrt(np.maximum(trial_gram.diagonal().real, 0.0))
    trial_norms = np.where(trial_norms > 0, trial_norms, 1.0)
    projection /= trial_norms[None, :]
    equilibrated_gram = trial_gram / trial_norms[:, None] / trial_norms[None, :]
    projected_gram = projection.conj().T @ gram @ projection
    projected_gram = (projected_gram + projected_gram.conj().T) / 2
    whitener, eigenvalues, retained = _whitener(
        projected_gram,
        rcond,
        scale_floor=float(np.linalg.norm(equilibrated_gram)),
    )
    basis = projection @ whitener
    effective_hamiltonian = basis.conj().T @ hamiltonian @ basis
    energies, vectors = np.linalg.eigh(
        (effective_hamiltonian + effective_hamiltonian.conj().T) / 2
    )
    coefficients = basis @ vectors
    amplitudes = coefficients.conj().T @ transitions
    energy_variances = None
    if hamiltonian_square is not None:
        norms = np.einsum("an,ab,bn->n", coefficients.conj(), gram, coefficients).real
        second_moments = (
            np.einsum(
                "an,ab,bn->n", coefficients.conj(), hamiltonian_square, coefficients
            ).real
            / norms
        )
        energy_variances = second_moments - energies**2
        # Bound roundoff before cancellation in the nonorthogonal contractions,
        # including zero-energy states whose second moment also cancels to zero.
        absolute_coefficients = np.abs(coefficients)
        moment_scale = (
            np.einsum(
                "an,ab,bn->n",
                absolute_coefficients,
                np.abs(hamiltonian_square),
                absolute_coefficients,
            )
            / norms
        )
        norm_scale = (
            np.einsum(
                "an,ab,bn->n",
                absolute_coefficients,
                np.abs(gram),
                absolute_coefficients,
            )
            / norms
        )
        tolerance = 32 * moment_precision * (moment_scale + energies**2 * norm_scale)
        if np.any(energy_variances < -tolerance):
            raise ValueError("Second moments yield negative Ritz energy variances.")
        energy_variances = np.maximum(energy_variances, 0.0)
    return GravitonSpectrum(
        excitation_energies=energies - ground_energy,
        weights=np.abs(amplitudes) ** 2 / n_electrons,
        coefficients=coefficients,
        gram_eigenvalues=eigenvalues,
        retained_gram_eigenvalues=retained,
        reference_gram_eigenvalues=reference_eigenvalues,
        hermiticity_error=hermiticity_error,
        ground_energy=ground_energy,
        energy_variances=energy_variances,
    )

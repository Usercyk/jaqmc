# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Torus shear stresses and ground-state-generated quadrupole subspaces."""

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.graviton import solve_graviton_subspace
from jaqmc.app.hall.hamiltonian.torus import TorusPotential
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.estimator.kinetic.torus import TorusKinetic
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


@configurable_dataclass
class TorusStress(PerWalkerEstimator):
    """Local derivatives of the complete real-space torus Hamiltonian.

    Args:
        flux: Positive number of flux quanta.
        tau: Physical torus shape, independently of both shear metrics.
        mass_shear: Cartesian inverse-mass shear components.
        interaction_shear: Covariant real-space interaction shear components.
        interaction_strength: Coulomb strength in cyclotron energy units.
        ewald_gmax: Reciprocal index cutoff for the reference Ewald sum.
        nlatvec: Real-space image cutoff for the reference Ewald sum.
        f_log_psi: Complex log-wavefunction evaluator (runtime dependency).
        data_field: Field holding fractional electron coordinates.

    Notes:
        ``stress:mass`` and ``stress:interaction`` contain the two derivatives
        at their respective configured shears. They represent different
        perturbations and are not a strict LLL-projected Raman operator.
    """

    flux: int = 1
    tau: complex = runtime_dep(default=1j)
    mass_shear: tuple[float, float] = (0.0, 0.0)
    interaction_shear: tuple[float, float] = (0.0, 0.0)
    interaction_strength: float = 1.0
    ewald_gmax: int = 200
    nlatvec: int = 1
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def init(self, data: Data, rngs: PRNGKey) -> None:
        self.kinetic = TorusKinetic(
            flux=self.flux,
            tau=self.tau,
            mass_shear=self.mass_shear,
            f_log_psi=self.f_log_psi,
            data_field=self.data_field,
        )
        self.potential = TorusPotential(
            flux=self.flux,
            tau=self.tau,
            interaction_shear=self.interaction_shear,
            interaction_strength=self.interaction_strength,
            ewald_gmax=self.ewald_gmax,
            nlatvec=self.nlatvec,
        )
        self.potential.init(data, rngs)
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        _, tensor = self.kinetic.covariant_derivatives(params, data)
        return {
            "stress:mass": self.kinetic.stress(tensor),
            "stress:interaction": self.potential.stress(data[self.data_field]),
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(walker_stats, include_variance=False)


@configurable_dataclass
class TorusGraviton(TorusStress):
    r"""Ritz quadrupole spectrum in a regularized interaction-stress basis.

    Args:
        rcond: Relative Gram cutoff used by host-side digest diagonalization.
        trial_radius: Positive pair-suppression radius in magnetic lengths.
            Trial factors are ``(1, R*T1, R*T2)`` with a smooth periodic
            factor ``R`` vanishing quadratically at pair coincidences. This
            avoids the divergent kinetic form of bare Coulomb-stress trial
            states with linear fermionic zeros. The physical Hamiltonian and
            transition operators are unchanged.

    Notes:
        This estimator projects out the sampled reference state only. Other
        torus ground partners require additional matrix elements and the
        multi-reference ``solve_graviton_subspace`` interface. The two trial
        states yield at most two Ritz peaks. Reported moments describe that
        retained subspace, not a converged full spectrum.
    """

    rcond: float = 1e-6
    trial_radius: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.rcond) or not 0 < self.rcond < 1:
            raise ValueError("rcond must lie strictly between zero and one.")
        if not np.isfinite(self.trial_radius) or self.trial_radius <= 0:
            raise ValueError("trial_radius must be finite and positive.")

    def init(self, data: Data, rngs: PRNGKey) -> None:
        super().init(data, rngs)
        self.n_electrons = len(data[self.data_field])
        if self.n_electrons <= 0:
            raise ValueError("Graviton evaluation requires at least one electron.")
        return None

    def trial_factors(self, electrons: jnp.ndarray) -> jnp.ndarray:
        r"""Return :math:`(1,R T_1^V,R T_2^V)` with periodic pair suppression.

        The embedding ``(exp(2*pi*i*delta)-1) @ lattice / (2*pi)`` has squared
        norm equal to the Cartesian pair distance squared near coincidence,
        while remaining smooth and periodic over the full torus. Multiplying
        ``1-exp(-distance_squared/(2*trial_radius**2))`` over pairs supplies
        the quadratic zeros. No suppression is applied to transition sources.

        Args:
            electrons: Fractional electron coordinates, shape ``(N, 2)``.

        Returns:
            Three real factors multiplying the reference wavefunction.
        """
        i, j = jnp.triu_indices(len(electrons), k=1)
        delta = electrons[i] - electrons[j]
        embedding = (
            jnp.expm1(2j * jnp.pi * delta) @ self.potential.lattice / (2 * jnp.pi)
        )
        squared = jnp.sum(embedding.real**2 + embedding.imag**2, axis=-1)
        suppression = jnp.prod(-jnp.expm1(-squared / (2 * self.trial_radius**2)))
        stress = self.potential.stress(electrons)
        return jnp.concatenate([jnp.ones(1, dtype=stress.dtype), suppression * stress])

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        electrons = data[self.data_field]
        covariant_grad, tensor = self.kinetic.covariant_derivatives(params, data)
        metric = self.kinetic.fractional_metric()
        local_energy = -0.5 * jnp.einsum("ab,iab->", metric, tensor)
        local_energy += self.potential.potential(electrons)

        stress = self.potential.stress(electrons)
        factors = self.trial_factors(electrons)
        factor_grad = jax.jacfwd(self.trial_factors)(electrons)
        factor_hessian = jax.jacfwd(jax.jacfwd(self.trial_factors))(electrons)
        diagonal_hessian = jnp.einsum("aibic->aibc", factor_hessian)
        correction = -0.5 * jnp.einsum("bc,aibc->a", metric, diagonal_hessian)
        correction -= jnp.einsum("bc,ib,aic->a", metric, covariant_grad, factor_grad)
        h_factors = factors * local_energy + correction
        chiral = jnp.array([stress[0] + 1j * stress[1], stress[0] - 1j * stress[1]])
        return {
            "stress:mass": self.kinetic.stress(tensor),
            "stress:interaction": stress,
            "graviton:gram": jnp.outer(factors.conj(), factors),
            "graviton:hamiltonian": jnp.outer(factors.conj(), h_factors),
            "graviton:transitions": jnp.outer(factors.conj(), chiral),
            "graviton:unprojected_norm": jnp.abs(chiral) ** 2 / self.n_electrons,
        }, state

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: None
    ) -> dict[str, Any]:
        del state
        stats = jax.tree.map(lambda x: jnp.mean(x, axis=0), batched_stats)
        spectrum = solve_graviton_subspace(
            np.asarray(stats["graviton:gram"]),
            np.asarray(stats["graviton:hamiltonian"]),
            np.asarray(stats["graviton:transitions"]),
            n_electrons=self.n_electrons,
            rcond=self.rcond,
        )
        weights = spectrum.weights.sum(axis=0)
        first_moment = spectrum.excitation_energies @ spectrum.weights
        centroid = np.full(2, np.nan)
        np.divide(first_moment, weights, out=centroid, where=weights > 0)
        retained = spectrum.retained_gram_eigenvalues
        stats.update(
            {
                "graviton:excitation_energies": spectrum.excitation_energies,
                "graviton:weights": spectrum.weights,
                "graviton:coefficients": spectrum.coefficients,
                "graviton:gram_eigenvalues": spectrum.gram_eigenvalues,
                "graviton:retained_gram_eigenvalues": retained,
                "graviton:hermiticity_error": spectrum.hermiticity_error,
                "graviton:ground_energy": spectrum.ground_energy,
                "graviton:subspace_weight": weights,
                "graviton:subspace_first_moment": first_moment,
                "graviton:subspace_centroid": centroid,
                "graviton:n_ground_references": 1,
            }
        )
        return stats

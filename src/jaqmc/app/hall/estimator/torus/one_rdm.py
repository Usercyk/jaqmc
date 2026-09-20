# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""One-body reduced density matrix estimator for electrons on a torus.

The matrix is represented in the n=0,1,2 torus Landau-level orbital basis
used by the MHPO wavefunction. A uniformly sampled auxiliary point replaces
each electron in turn, giving

.. math::

    \rho_{ij} = A \left\langle
        \sum_a \frac{\Psi(R_a')}{\Psi(R)}
        \varphi_i(r_a)\,\varphi_j^*(r')
    \right\rangle,

where :math:`A=2\pi N_\phi` is the torus area in magnetic-length units.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.app.hall.wavefunction.torus.mhpo import torus_envelope
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.torus import get_torus_lattice
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

__all__ = ["TorusOneRDM"]


@configurable_dataclass
class TorusOneRDM(PerWalkerEstimator):
    r"""One-body RDM in the first three torus Landau levels.

    The basis is ordered by Landau-level index and then guiding-centre index:
    :math:`(0,0),\ldots,(0,N_\phi-1),(1,0),\ldots,(2,N_\phi-1)`.
    The result is a complex :math:`3N_\phi\times3N_\phi` matrix. Its
    diagonal contains orbital occupations, while the traces of the
    ``one_rdm:n0``, ``one_rdm:n1``, and ``one_rdm:n2`` blocks give
    the occupation of each Landau level. The total trace gives the particle
    number represented in this three-level basis.

    Args:
        flux: Positive number of magnetic flux quanta :math:`N_\phi`.
        tau: Torus modular parameter in the upper half-plane.
        theta_terms: Half-width of the theta-series truncation used by the
            magnetic orbitals.
        f_log_psi: Complex log-wavefunction evaluator.
        data_field: Field containing fractional coordinates ``(u, v)``.
    """

    flux: int = 2
    tau: complex = runtime_dep(default=1j)
    theta_terms: int = 48
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def __post_init__(self) -> None:
        if (
            not isinstance(self.flux, int)
            or isinstance(self.flux, bool)
            or self.flux <= 0
        ):
            raise ValueError(f"flux must be a positive integer. Got {self.flux!r}.")

        tau = complex(self.tau)
        if not jnp.isfinite(tau.real) or not jnp.isfinite(tau.imag):
            raise ValueError(f"tau must be finite. Got {tau!r}.")
        if tau.imag <= 0:
            raise ValueError(
                "tau must lie in the upper half-plane, so its imaginary "
                f"part must be positive. Got {tau!r}."
            )
        self.tau = tau

        if (
            not isinstance(self.theta_terms, int)
            or isinstance(self.theta_terms, bool)
            or self.theta_terms <= 0
        ):
            raise ValueError(
                f"theta_terms must be a positive integer. Got {self.theta_terms!r}."
            )

    def init(self, data: Data, rngs: PRNGKey) -> None:
        del data, rngs
        self._l1, self._l2, self._area = get_torus_lattice(self.flux, self.tau)
        return None

    def _orbitals(self, electrons: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the normalized n=0,1,2 magnetic orbitals.

        Returns:
            Complex orbital values with a final axis of length ``3 * flux``.
        """
        z = electrons[..., 0] * self._l1 + electrons[..., 1] * self._l2
        return torus_envelope(z, self.flux, self.tau, self.theta_terms)

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        del prev_walker_stats
        electrons = data[self.data_field]
        if electrons.ndim != 2 or electrons.shape[-1] != 2:
            raise ValueError(
                "Torus electron coordinates must have shape (n_electrons, 2), "
                f"with the last axis ordered as (u, v). Got {electrons.shape}."
            )
        nelec = electrons.shape[0]

        r_prime = jax.random.uniform(rngs, (2,), minval=0.0, maxval=1.0)
        displaced = jnp.repeat(electrons[None], nelec, axis=0)
        indices = jnp.arange(nelec)
        displaced = displaced.at[indices, indices].set(r_prime)

        logpsi = self.f_log_psi(params, data)

        def eval_displaced(displaced_electrons: jnp.ndarray) -> jnp.ndarray:
            return self.f_log_psi(
                params, data.merge({self.data_field: displaced_electrons})
            )

        logpsi_prime = jax.vmap(eval_displaced)(displaced)
        wf_ratio = jnp.exp(logpsi_prime - logpsi)

        orbitals = self._orbitals(electrons)
        orbitals_prime = self._orbitals(r_prime)
        one_rdm = self._area * jnp.sum(
            wf_ratio[:, None, None]
            * orbitals[:, :, None]
            * jnp.conj(orbitals_prime)[None, None, :],
            axis=0,
        )
        return {"one_rdm": one_rdm}, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        del state
        one_rdm = jnp.nanmean(batched_stats["one_rdm"], axis=0)
        return {
            "one_rdm": one_rdm,
            "one_rdm:diagonal": jnp.diagonal(one_rdm),
            "one_rdm:trace": jnp.trace(one_rdm),
            "one_rdm:n0": one_rdm[: self.flux, : self.flux],
            "one_rdm:n1": one_rdm[self.flux : 2 * self.flux, self.flux : 2 * self.flux],
            "one_rdm:n2": one_rdm[
                2 * self.flux : 3 * self.flux, 2 * self.flux : 3 * self.flux
            ],
        }

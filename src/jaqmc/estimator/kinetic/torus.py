# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Kinetic-energy estimator on a torus in symmetric gauge.

Electron positions are fractional lattice coordinates ``(u, v)`` with

.. math::

    z = L_1(u + \tau v), \qquad
    L_1^2\operatorname{Im}\tau = 2\pi N_\phi \ell_B^2.

The reported energy is in units of :math:`\hbar\omega_c`.
"""

from collections.abc import Mapping
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import grad_maybe_complex, hessian_maybe_complex
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

__all__ = ["TorusKinetic"]


@configurable_dataclass
class TorusKinetic(PerWalkerEstimator):
    r"""Local kinetic-energy estimator on a general torus.

    The estimator uses symmetric-gauge covariant derivatives

    .. math::

        D_u=\partial_u+i\pi N_\phi v, \qquad
        D_v=\partial_v-i\pi N_\phi u,

    and evaluates

    .. math::

        \frac{H}{\hbar\omega_c}
        =-\frac{|\tau|^2D_u^2
          -\tau_1(D_uD_v+D_vD_u)+D_v^2}
        {4\pi N_\phi\tau_2}.

    Derivatives are taken with respect to the stored fractional coordinates
    ``(u, v)``. For :math:`f=\log\psi`, second derivatives of the wavefunction
    are reconstructed using
    :math:`\partial_a\partial_b\psi/\psi=f_{ab}+f_af_b`.

    Args:
        flux: Positive number of magnetic flux quanta :math:`N_\phi`.
        tau: Torus modular parameter in the upper half-plane.
        f_log_psi: Complex log-wavefunction evaluate function (runtime dep).
        data_field: Electron-coordinate field name (runtime dep).
    """

    flux: int = 1
    tau: complex = runtime_dep(default=1j)
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
        if electrons.ndim != 2 or electrons.shape[-1] != 2:
            raise ValueError(
                "Torus electron coordinates must have shape (n_electrons, 2), "
                f"with the last axis ordered as (u, v). Got {electrons.shape}."
            )

        def log_psi(p: Params, x: jnp.ndarray) -> jnp.ndarray:
            return self.f_log_psi(p, data.merge({self.data_field: x}))

        grad_log_psi = grad_maybe_complex(log_psi, argnums=1)(params, electrons)
        hessian_log_psi = hessian_maybe_complex(log_psi, argnums=1)(params, electrons)

        grad_u = grad_log_psi[:, 0]
        grad_v = grad_log_psi[:, 1]

        # Keep only the diagonal electron blocks: the one-body Hamiltonian has
        # no derivatives coupling coordinates of different electrons.
        particle_hessian = jnp.einsum("iaib->iab", hessian_log_psi)
        hessian_uu = particle_hessian[:, 0, 0]
        hessian_uv = particle_hessian[:, 0, 1]
        hessian_vv = particle_hessian[:, 1, 1]

        tau_1 = self.tau.real
        tau_2 = self.tau.imag
        tau_abs_squared = tau_1**2 + tau_2**2

        metric_laplacian = jnp.sum(
            tau_abs_squared * hessian_uu - 2 * tau_1 * hessian_uv + hessian_vv
        )
        metric_grad_squared = jnp.sum(
            tau_abs_squared * grad_u**2 - 2 * tau_1 * grad_u * grad_v + grad_v**2
        )
        second_order = -(metric_laplacian + metric_grad_squared) / (
            4 * jnp.pi * self.flux * tau_2
        )

        u, v = electrons[:, 0], electrons[:, 1]
        first_order = (
            -0.5j
            / tau_2
            * jnp.sum(
                (tau_abs_squared * v + tau_1 * u) * grad_u - (u + tau_1 * v) * grad_v
            )
        )
        vector_potential_squared = (
            jnp.pi
            * self.flux
            / (4 * tau_2)
            * jnp.sum(u**2 + 2 * tau_1 * u * v + tau_abs_squared * v**2)
        )

        kinetic_energy = second_order + first_order + vector_potential_squared
        return {"energy:kinetic": kinetic_energy}, state

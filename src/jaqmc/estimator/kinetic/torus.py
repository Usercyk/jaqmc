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

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import grad_maybe_complex, hessian_maybe_complex
from jaqmc.utils.torus import get_torus_lattice, shear_matrix, validate_shear
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

    and contracts their symmetric second derivatives with
    :math:`M=L^{-1}g_m^{-1}L^{-T}`, where :math:`L` maps fractional to
    Cartesian magnetic-length coordinates. At zero mass shear this reduces to

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
        mass_shear: Cartesian inverse-mass shear components. The metric is
            ``exp([[s1, s2], [s2, -s1]])`` independently of the cell shape.
        f_log_psi: Complex log-wavefunction evaluate function (runtime dep).
        data_field: Electron-coordinate field name (runtime dep).
    """

    flux: int = 1
    tau: complex = runtime_dep(default=1j)
    mass_shear: tuple[float, float] = (0.0, 0.0)
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def __post_init__(self) -> None:
        validate_shear(self.mass_shear)
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

    def fractional_metric(self, shear: jnp.ndarray | None = None) -> jnp.ndarray:
        """Return the inverse mass tensor in fractional coordinates.

        Args:
            shear: Optional inverse-mass shear overriding ``mass_shear``.

        Returns:
            The matrix ``L^-1 @ g^-1 @ L^-T`` in magnetic-length units.
        """
        l1, l2, _ = get_torus_lattice(self.flux, self.tau)
        lattice = jnp.array([[l1, l2.real], [0.0, l2.imag]])
        inverse = jnp.linalg.inv(lattice)
        metric = shear_matrix(self.mass_shear if shear is None else shear)
        return inverse @ metric @ inverse.T

    def covariant_derivatives(
        self, params: Params, data: Data
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Return :math:`D\Psi/\Psi` and its symmetric second-order tensor.

        Returns:
            Covariant gradients ``(N, 2)`` and symmetric covariant second
            derivatives divided by the wavefunction, with shape ``(N, 2, 2)``.

        Raises:
            ValueError: If coordinates are not shaped ``(N, 2)``.
        """
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

        # Keep only the diagonal electron blocks: the one-body Hamiltonian has
        # no derivatives coupling coordinates of different electrons.
        particle_hessian = jnp.einsum("iaib->iab", hessian_log_psi)
        connection = (
            jnp.pi * self.flux * jnp.stack([electrons[:, 1], -electrons[:, 0]], axis=-1)
        )
        covariant_grad = grad_log_psi + 1j * connection
        tensor = particle_hessian + jnp.einsum(
            "ia,ib->iab", covariant_grad, covariant_grad
        )
        return covariant_grad, tensor

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        _, tensor = self.covariant_derivatives(params, data)
        kinetic_energy = -0.5 * jnp.einsum("ab,iab->", self.fractional_metric(), tensor)
        return {"energy:kinetic": kinetic_energy}, state

    def stress(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Return the two local mass-stress components from covariant derivatives.

        Args:
            tensor: Symmetric covariant second derivatives divided by the
                wavefunction, as returned by ``covariant_derivatives``.

        Returns:
            The two derivatives of the local kinetic energy at ``mass_shear``.
        """
        derivatives = jax.jacfwd(self.fractional_metric)(
            jnp.asarray(self.mass_shear, dtype=float)
        )
        return -0.5 * jnp.einsum("abc,iab->c", derivatives, tensor)

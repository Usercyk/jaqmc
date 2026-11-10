# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Potential-energy estimator on a torus with a neutralizing background."""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.app.hall.config import InteractionType
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.estimator.ewald import EwaldSum2D
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.torus import get_torus_lattice, shear_matrix, validate_shear
from jaqmc.utils.wiring import runtime_dep

__all__ = ["TorusPotential"]


@configurable_dataclass
class TorusPotential(PerWalkerEstimator):
    r"""Periodic Coulomb energy on a general torus.

    Electron positions are fractional coordinates `(u, v)` satisfying

    .. math::

        z=L_1(u+\tau v), \qquad
        L_1^2\operatorname{Im}\tau=2\pi N_\phi\ell_B^2.

    The two-dimensional Ewald sum evaluates the three-dimensional Coulomb
    interaction :math:`1/r`, including a uniform neutralizing background and
    the Madelung interaction with periodic self-images. Coordinates are
    converted to Cartesian magnetic-length units before evaluation. The
    interaction metric deforms positions and lattice vectors together,
    independently of the wavefunction's physical torus boundary conditions.

    Args:
        interaction_type: Interaction potential form. Only Coulomb is supported.
        flux: Positive number of magnetic flux quanta :math:`N_\phi`.
        tau: Torus modular parameter in the upper half-plane.
        interaction_strength: Ratio
            :math:`(e^2/\epsilon\ell_B)/(\hbar\omega_c)` multiplying the
            dimensionless Ewald energy.
        interaction_shear: Covariant real-space interaction-metric shear.
        ewald_gmax: Reciprocal-lattice index cutoff used by the Ewald sum.
        nlatvec: Real-space periodic-image cutoff used by the Ewald sum.
    """

    interaction_type: InteractionType = InteractionType.coulomb
    flux: int = 1
    tau: complex = runtime_dep(default=1j)
    interaction_strength: float = 1.0
    interaction_shear: tuple[float, float] = (0.0, 0.0)
    ewald_gmax: int = 200
    nlatvec: int = 1

    def __post_init__(self) -> None:
        validate_shear(self.interaction_shear)
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
            not isinstance(self.ewald_gmax, int)
            or isinstance(self.ewald_gmax, bool)
            or self.ewald_gmax <= 0
        ):
            raise ValueError(
                f"ewald_gmax must be a positive integer. Got {self.ewald_gmax!r}."
            )
        if (
            not isinstance(self.nlatvec, int)
            or isinstance(self.nlatvec, bool)
            or self.nlatvec < 0
        ):
            raise ValueError(
                f"nlatvec must be a non-negative integer. Got {self.nlatvec!r}."
            )

    def init(self, data: Data, rngs: PRNGKey) -> None:
        del data, rngs
        l1, l2, _ = get_torus_lattice(self.flux, self.tau)
        zero = jnp.zeros_like(l1)
        self.lattice = jnp.stack([jnp.stack([l1, zero]), jnp.stack([l2.real, l2.imag])])
        # Select reciprocal vectors near the configured interaction geometry.
        # Subsequent AD uses a relative deformation around this reference.
        self.reference_transform = shear_matrix(jnp.asarray(self.interaction_shear) / 2)
        self.ewald = EwaldSum2D(
            self.lattice @ self.reference_transform.T,
            ewald_gmax=self.ewald_gmax,
            nlatvec=self.nlatvec,
        )
        return None

    def potential(
        self, electrons: jnp.ndarray, shear: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Evaluate the complete periodic potential at a fixed shear.

        Args:
            electrons: Fractional coordinates with shape ``(N, 2)``.
            shear: Optional covariant interaction shear overriding the config.

        Returns:
            Ewald potential including background and periodic self-images.

        Raises:
            ValueError: If the interaction or coordinate shape is unsupported.
        """
        if self.interaction_type != InteractionType.coulomb:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")
        if electrons.ndim != 2 or electrons.shape[-1] != 2:
            raise ValueError("Torus electron coordinates must have shape (N, 2).")
        s = jnp.asarray(self.interaction_shear) if shear is None else shear
        deformation = shear_matrix(s / 2) @ jnp.linalg.inv(self.reference_transform)
        charges = -jnp.ones(electrons.shape[0], dtype=electrons.dtype)
        return self.interaction_strength * self.ewald.energy(
            electrons @ self.lattice @ self.reference_transform.T,
            charges,
            deformation=deformation,
        )

    def stress(self, electrons: jnp.ndarray) -> jnp.ndarray:
        """Return the two potential-energy shear derivatives at the config shear.

        Returns:
            Real multiplication operators ``(T1, T2)`` in cyclotron units.
        """
        return jax.jacfwd(self.potential, argnums=1)(
            electrons, jnp.asarray(self.interaction_shear, dtype=float)
        )

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, prev_walker_stats, rngs
        electrons = data["electrons"]
        if electrons.ndim != 2 or electrons.shape[-1] != 2:
            raise ValueError(
                "Torus electron coordinates must have shape (n_electrons, 2), "
                f"with the last axis ordered as (u, v). Got {electrons.shape}."
            )

        return {"energy:potential": self.potential(electrons)}, state

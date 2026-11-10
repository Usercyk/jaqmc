# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""One-body reduced density matrix (1-RDM) estimator for the disk.

Computes the 1-RDM in the disk basis via Monte Carlo sampling.  In the
symmetric gauge, with magnetic length :math:`\ell_B=1` and
:math:`z=x-iy`, the normalized orbitals used here are

.. math::
    \phi_{nm}(z,\bar z) =
    (-1)^n\sqrt{\frac{n!}{2\pi 2^m(n+m)!}}\,
    z^m L_n^m(|z|^2/2)e^{-|z|^2/4},

where :math:`n` is the Landau-level index and the angular-momentum label
:math:`m` takes the values :math:`-n,-n+1,\ldots`.  Equivalently, the
non-negative guiding-centre index is :math:`q=m+n`.  The first three
generalized Laguerre polynomials give

.. math::
    \phi_{0m} &= \frac{z^m}{\sqrt{2\pi 2^m m!}}e^{-|z|^2/4}, \\
    \phi_{1m} &= -\frac{z^m(m+1-|z|^2/2)}
      {\sqrt{2\pi 2^m(m+1)!}}e^{-|z|^2/4}, \\
    \phi_{2m} &= \sqrt{\frac{2}{2\pi 2^m(m+2)!}}\frac{z^m}{2}
      \left[(|z|^2/2)^2-2(m+2)|z|^2/2+(m+1)(m+2)\right]
      e^{-|z|^2/4}.

The implementation expands :math:`z^mL_n^m(z\bar z/2)` into non-negative
powers of :math:`z` and :math:`\bar z`.  This avoids a removable numerical
singularity at :math:`z=0` for the legitimate negative-:math:`m` orbitals in
higher Landau levels.
"""

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


def make_disk_basis(m: int, landau_level: int = 0):
    r"""Build a disk Landau-level basis function.

    Args:
        m: Angular-momentum label.  The :math:`L_z` eigenvalue is
            :math:`-m\hbar` with the convention :math:`z=x-iy`.
        landau_level: Landau-level index :math:`n` (zero based).

    Returns:
        A callable ``phi_m(electrons)``.

    Raises:
        ValueError: If ``landau_level`` is negative or ``m < -landau_level``.
    """
    if landau_level < 0:
        raise ValueError("landau_level must be non-negative")
    if m < -landau_level:
        raise ValueError("m must satisfy m >= -landau_level")

    # Eq. (3.41) of Jain, Composite Fermions.  gammaln keeps the
    # normalization stable when many angular-momentum orbitals are retained.
    log_norm = 0.5 * (
        ss.gammaln(landau_level + 1)
        - np.log(2 * np.pi)
        - m * np.log(2.0)
        - ss.gammaln(landau_level + m + 1)
    )
    norm_factor = (-1) ** landau_level * np.exp(log_norm)

    # L_n^m(t) = sum_k (-1)^k C(n+m, n-k) t^k/k!, t=z*zbar/2.
    # For m < 0, all terms with k < -m vanish.  Dropping them explicitly
    # prevents evaluating a spurious negative power of z at the origin.
    laguerre_powers = np.arange(max(0, -m), landau_level + 1)
    polynomial_coeffs = jnp.asarray(
        [
            (-1) ** k
            * ss.comb(landau_level + m, landau_level - k)
            / (ss.factorial(k) * 2**k)
            for k in laguerre_powers
        ]
    )
    z_powers = jnp.asarray(m + laguerre_powers)
    zbar_powers = jnp.asarray(laguerre_powers)

    def phi_m(electrons: jnp.ndarray) -> jnp.ndarray:
        x, y = electrons[..., 0], electrons[..., 1]
        z = x - 1j * y
        r_sq = x**2 + y**2
        radial_polynomial = jnp.sum(
            polynomial_coeffs
            * z[..., None] ** z_powers
            * jnp.conj(z[..., None]) ** zbar_powers,
            axis=-1,
        )
        return norm_factor * radial_polynomial * jnp.exp(-r_sq / 4)

    return phi_m


def _uniform_sample_disk(key: PRNGKey, radius_max: float = 10.0) -> jnp.ndarray:
    """Sample a single point uniformly on a disk up to radius_max.

    Args:
        key: JAX random key.
        radius_max: Maximum radius to sample.

    Returns:
        Array of shape ``(2,)`` containing ``(x, y)``.
    """
    key1, key2 = jax.random.split(key)
    r = radius_max * jnp.sqrt(jax.random.uniform(key1, ()))
    theta = jax.random.uniform(key2, (), minval=0.0, maxval=2 * jnp.pi)
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)])


@configurable_dataclass
class DiskOneRDM(PerWalkerEstimator):
    r"""One-body reduced density matrix on the disk.

    Args:
        num_orbitals: Number of guiding-centre orbitals in each Landau level.
            They are labelled by :math:`q=0,\ldots,M-1`; their physical
            angular-momentum labels are :math:`m=q-n` in Landau level ``n``.
        num_landau_levels: Number of Landau levels, starting at :math:`n=0`.
        flux: Magnetic flux.
        f_log_psi: Complex log-psi function (runtime dep).
        data_field: Name of the data field (runtime dep).
        radius_max: Maximum radius for uniform sampling.
    """

    num_orbitals: int | None = None
    num_landau_levels: int = 3
    flux: int = 2
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")
    radius_max: float = 15.0

    def init(self, data: Data, rngs: PRNGKey) -> None:
        """Build the list of disk basis closures.

        Raises:
            ValueError: If the number of orbitals or Landau levels is not positive.
        """
        norbs_per_level = self.flux + 1

        if self.num_orbitals is not None:
            norbs_per_level = self.num_orbitals
        if norbs_per_level < 1:
            raise ValueError("num_orbitals must be positive")
        if self.num_landau_levels < 1:
            raise ValueError("num_landau_levels must be positive")

        # Flatten the (n, q) basis in Landau-level-major order.  The physical
        # angular-momentum label in Jain's convention is m=q-n, so the first
        # two higher LLs include m=-1 and m=-2,-1, respectively.
        orbital_labels = [
            (n, q, q - n)
            for n in range(self.num_landau_levels)
            for q in range(norbs_per_level)
        ]
        self._orbitals = [
            make_disk_basis(m, landau_level=n) for n, _, m in orbital_labels
        ]
        self._orbital_landau_levels = jnp.asarray(
            [n for n, _, _ in orbital_labels], dtype=jnp.int32
        )
        self._orbital_guiding_centres = jnp.asarray(
            [q for _, q, _ in orbital_labels], dtype=jnp.int32
        )
        self._orbital_angular_momenta = jnp.asarray(
            [m for _, _, m in orbital_labels], dtype=jnp.int32
        )
        return None

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
        nelec = electrons.shape[0]

        # Sample r' uniformly on the disk
        r_prime = _uniform_sample_disk(rngs, self.radius_max)
        area = jnp.pi * self.radius_max**2

        # Build displaced configs
        data_prime_electrons = jnp.repeat(electrons[None], nelec, axis=0)
        idx = jnp.arange(nelec)
        data_prime_electrons = data_prime_electrons.at[idx, idx].set(r_prime)

        logpsi = self.f_log_psi(params, data)

        def eval_displaced(displaced_electrons: jnp.ndarray) -> jnp.ndarray:
            return self.f_log_psi(
                params, data.merge({self.data_field: displaced_electrons})
            )

        logpsi_prime = jax.vmap(eval_displaced)(data_prime_electrons)
        wf_ratio = jnp.exp(logpsi_prime - logpsi)

        # Orbitals at original positions: shape (nelec, norbs)
        varphi = jnp.stack([orb(electrons) for orb in self._orbitals], axis=-1)
        # Orbitals at r': shape (norbs,)
        varphi_prime = jnp.stack([orb(r_prime[None]) for orb in self._orbitals])[..., 0]

        one_rdm = area * jnp.sum(
            wf_ratio[:, None, None]
            * varphi[:, :, None]
            * jnp.conj(varphi_prime)[None, None, :],
            axis=0,
        )

        return {"one_rdm": one_rdm}, state

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(local_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        one_rdm = jnp.nanmean(batched_stats["one_rdm"], axis=0)
        # The exact 1-RDM is Hermitian.  Use its Hermitian part for nonlinear
        # derived observables while retaining the raw matrix so that the
        # anti-Hermitian Monte Carlo error remains available as a diagnostic.
        one_rdm_h = (one_rdm + jnp.conj(one_rdm.T)) / 2
        occupations = jnp.real(jnp.diagonal(one_rdm_h))
        trace = jnp.real(jnp.trace(one_rdm_h))
        purity = jnp.real(jnp.trace(one_rdm_h @ one_rdm_h))
        natural_occupations = jnp.linalg.eigvalsh(one_rdm_h)[::-1]
        landau_level_occupations = (
            jnp.zeros((self.num_landau_levels,), dtype=occupations.dtype)
            .at[self._orbital_landau_levels]
            .add(occupations)
        )
        angular_momentum_z = -jnp.sum(self._orbital_angular_momenta * occupations)
        cyclotron_energy = jnp.sum((self._orbital_landau_levels + 0.5) * occupations)
        matrix_norm = jnp.linalg.norm(one_rdm)
        hermiticity_error = jnp.where(
            matrix_norm > 0,
            jnp.linalg.norm(one_rdm - jnp.conj(one_rdm.T)) / matrix_norm,
            0.0,
        )
        effective_orbitals = jnp.where(purity > 0, trace**2 / purity, jnp.nan)

        return {
            "one_rdm": one_rdm,
            "one_rdm:diagonal": occupations,
            "one_rdm:trace": trace,
            "one_rdm:purity": purity,
            "one_rdm:effective_orbitals": effective_orbitals,
            "one_rdm:natural_occupations": natural_occupations,
            "one_rdm:landau_level_occupations": landau_level_occupations,
            "one_rdm:angular_momentum_z": angular_momentum_z,
            "one_rdm:cyclotron_energy": cyclotron_energy,
            "one_rdm:hermiticity_error": hermiticity_error,
            "one_rdm:basis_landau_level": self._orbital_landau_levels,
            "one_rdm:basis_guiding_centre": self._orbital_guiding_centres,
            "one_rdm:basis_angular_momentum": self._orbital_angular_momenta,
        }

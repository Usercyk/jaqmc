# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
r"""Sigma-function MHPO ansatz on a rectangular torus.

The sigma representation implemented here is restricted to rectangular tori.
It accepts the workflow's modular parameter ``tau``, but requires its real
part to vanish.
"""

import math
import warnings

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.wavefunction.torus.jastrow import TorusJastrow
from jaqmc.array_types import Params
from jaqmc.utils.torus import get_torus_lattice
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.backbone.psiformer import PsiformerBackbone
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction
from jaqmc.wavefunction.output.orbital import SplitChannelDense

from .mhpo import _lowest_landau_level_kernel_init

__all__ = ["TorusSigmaMHPO"]


def _validate_geometry(
    flux: int,
    tau: complex,
    theta_terms: int,
    eisen_terms: int,
) -> None:
    get_torus_lattice(flux, tau)
    tau = complex(tau)
    if not math.isclose(tau.real, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "TorusSigmaMHPO only supports rectangular lattices, which require "
            f"Re(tau) = 0. Got tau={tau!r}."
        )
    if (
        not isinstance(theta_terms, int)
        or isinstance(theta_terms, bool)
        or theta_terms <= 0
    ):
        raise ValueError(
            f"theta_terms must be a positive integer. Got {theta_terms!r}."
        )
    if (
        not isinstance(eisen_terms, int)
        or isinstance(eisen_terms, bool)
        or eisen_terms <= 0
    ):
        raise ValueError(
            f"eisen_terms must be a positive integer. Got {eisen_terms!r}."
        )


def _theta1_series(
    u: jnp.ndarray, q: jnp.ndarray, *, n_terms: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate theta1 and its first two derivatives with respect to ``u``.

    Returns:
        ``theta1``, ``theta1'``, and ``theta1''`` evaluated at ``u``.
    """
    n = jnp.arange(n_terms)
    a = n + 0.5
    k = 2 * n + 1
    sign = (-1.0) ** n

    # Writing sine and cosine in exponential form avoids first constructing
    # trigonometric values with unbounded imaginary arguments.
    exponent = (a**2) * jnp.log(q)
    exp_plus = jnp.exp(exponent + 1j * k * u[..., None])
    exp_minus = jnp.exp(exponent - 1j * k * u[..., None])
    sin_terms = sign * (exp_plus - exp_minus) / 1j
    cos_terms = sign * (exp_plus + exp_minus)

    theta1 = jnp.sum(sin_terms, axis=-1)
    theta1_p = jnp.sum(k * cos_terms, axis=-1)
    theta1_pp = jnp.sum(-(k**2) * sin_terms, axis=-1)
    return theta1, theta1_p, theta1_pp


def _theta1_prime0(q: jnp.ndarray, *, n_terms: int) -> jnp.ndarray:
    n = jnp.arange(n_terms)
    a = n + 0.5
    k = 2 * n + 1
    return jnp.sum(2.0 * ((-1.0) ** n) * (q ** (a * a)) * k)


def _eisenstein_e2(q: jnp.ndarray, *, n_terms: int) -> jnp.ndarray:
    n = jnp.arange(1, n_terms + 1)
    qn = (q * q) ** n
    return 1.0 - 24.0 * jnp.sum(n * qn / (1.0 - qn))


def _lattice_parameters(
    *, flux: int, tau: complex, eisen_terms: int
) -> tuple[jnp.ndarray, ...]:
    x_len, l2, area = get_torus_lattice(flux, tau)
    y_len = l2.imag
    omega1 = x_len / 2.0
    q = jnp.exp(-jnp.pi * y_len / x_len)
    e2 = _eisenstein_e2(q, n_terms=eisen_terms)
    eta1 = jnp.pi**2 * e2 / (12.0 * omega1)
    return x_len, y_len, area, omega1, q, eta1


def _weierstrass_log_sigma(
    z: jnp.ndarray,
    *,
    omega1: jnp.ndarray,
    q: jnp.ndarray,
    eta1: jnp.ndarray,
    theta_terms: int,
) -> jnp.ndarray:
    u = jnp.pi * z / (2.0 * omega1)
    theta1, _, _ = _theta1_series(u, q, n_terms=theta_terms)
    theta1p0 = _theta1_prime0(q, n_terms=theta_terms)
    constant = jnp.log(2.0 * omega1 / jnp.pi) - jnp.log(theta1p0)
    quadratic = eta1 * z**2 / (2.0 * omega1)
    return constant + quadratic + jnp.log(theta1)


def _weierstrass_zeta(
    z: jnp.ndarray,
    *,
    omega1: jnp.ndarray,
    q: jnp.ndarray,
    eta1: jnp.ndarray,
    theta_terms: int,
) -> jnp.ndarray:
    u = jnp.pi * z / (2.0 * omega1)
    theta1, theta1_p, _ = _theta1_series(u, q, n_terms=theta_terms)
    prefactor = jnp.pi / (2.0 * omega1)
    return eta1 * z / omega1 + prefactor * theta1_p / theta1


def _weierstrass_wp(
    z: jnp.ndarray,
    *,
    omega1: jnp.ndarray,
    q: jnp.ndarray,
    eta1: jnp.ndarray,
    theta_terms: int,
) -> jnp.ndarray:
    u = jnp.pi * z / (2.0 * omega1)
    theta1, theta1_p, theta1_pp = _theta1_series(u, q, n_terms=theta_terms)
    prefactor = jnp.pi / (2.0 * omega1)
    logarithmic_second_derivative = theta1_pp / theta1 - (theta1_p / theta1) ** 2
    return -eta1 / omega1 - prefactor**2 * logarithmic_second_derivative


def _sigma_zeros_and_momenta(
    flux: int,
    x_len: jnp.ndarray,
    y_len: jnp.ndarray,
    eta1: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return zeros and holomorphic corrections for every degenerate orbital."""
    guiding_centres = jnp.arange(flux)
    zero_indices = jnp.arange(flux)
    zero_x = x_len / flux * (zero_indices - (flux - 1) / 2)
    zero_y = y_len / flux * (guiding_centres - flux / 2)
    zeros = zero_x[None, :] + 1j * zero_y[:, None]

    # A product of Weierstrass sigma functions contains the holomorphic
    # quadratic exp(N * eta1 * z**2 / Lx). Correct it to the symmetric-gauge
    # factor exp(z**2 / 4). The corresponding linear term makes the modulus
    # periodic along the imaginary period. The legacy expressions coincide
    # with these corrections only for a square torus.
    quadratic = 0.25 - flux * eta1 / x_len
    momenta = 1j * (flux - 2 * guiding_centres) * (eta1 * y_len - jnp.pi) / x_len
    return zeros, momenta, quadratic


def _sigma_lll_log_normalizers(
    *,
    flux: int,
    x_len: jnp.ndarray,
    y_len: jnp.ndarray,
    omega1: jnp.ndarray,
    q: jnp.ndarray,
    eta1: jnp.ndarray,
    theta_terms: int,
) -> jnp.ndarray:
    r"""Return the log norm of each unnormalized sigma-product LLL orbital.

    A sigma product and a theta-function orbital with matching zeros are the
    same holomorphic section up to a constant.  We evaluate that constant at
    one nonzero point per guiding centre and compare against the analytically
    normalized theta orbital.  This avoids a costly two-dimensional numerical
    normalization on every envelope evaluation.
    """
    zeros, momenta, quadratic = _sigma_zeros_and_momenta(flux, x_len, y_len, eta1)
    cell_x = x_len / flux
    cell_y = y_len / flux
    reference_z = zeros[:, 0] + 0.371 * cell_x + 0.419j * cell_y

    log_sigma = _weierstrass_log_sigma(
        reference_z[:, None] - zeros,
        omega1=omega1,
        q=q,
        eta1=eta1,
        theta_terms=theta_terms,
    )
    sigma_log_abs = jnp.real(
        jnp.sum(log_sigma, axis=-1)
        + jnp.conj(momenta) * reference_z
        + quadratic * reference_z**2
        - jnp.abs(reference_z) ** 2 / 4.0
    )

    # The zeros select the characteristic a=-n/N and b=N/2.  Combine the
    # symmetric-gauge Gaussian with each theta-series term, as in the regular
    # torus envelope, so the reference value remains numerically bounded.
    guiding_centres = jnp.arange(flux)
    characteristics = -guiding_centres / flux
    rho = reference_z.imag / y_len
    series_centres = jnp.rint(-rho - characteristics)
    offsets = jnp.arange(-theta_terms, theta_terms + 1)
    series_indices = (
        series_centres[:, None] + offsets[None, :] + characteristics[:, None]
    )
    aspect_ratio = y_len / x_len
    real_exponent = -jnp.pi * flux * aspect_ratio * (series_indices + rho[:, None]) ** 2
    imaginary_exponent = (reference_z.real * reference_z.imag / 2.0)[
        :, None
    ] + 2.0 * jnp.pi * series_indices * (
        flux * reference_z.real[:, None] / x_len + flux / 2
    )
    theta_values = jnp.sum(jnp.exp(real_exponent + 1j * imaginary_exponent), axis=-1)
    theta_log_abs = (
        -0.25 * jnp.log(jnp.pi) - 0.5 * jnp.log(x_len) + jnp.log(jnp.abs(theta_values))
    )
    return sigma_log_abs - theta_log_abs


def torus_sigma_envelope(
    z: jnp.ndarray,
    flux: int,
    tau: complex = 1j,
    theta_terms: int = 24,
    eisen_terms: int = 48,
) -> jnp.ndarray:
    r"""Evaluate normalized sigma-form orbitals on a rectangular torus.

    The input ``z`` is a Cartesian complex coordinate in magnetic-length
    units.  The returned basis is ordered first by level and then by guiding
    centre: ``(psi0_0, ..., psi0_N-1, psi1_0, ..., psi2_N-1)``.

    The sigma product gives an unnormalized lowest-level orbital.  Its norm is
    fixed by comparison with the equivalent unit-normalized theta section.
    The first- and second-raised expressions each have norm ``1/sqrt(2)``
    relative to that sigma product, so both receive the corresponding
    ``sqrt(2)`` ladder normalization.

    Args:
        z: Complex Cartesian positions in magnetic-length units.
        flux: Positive number of magnetic flux quanta.
        tau: Purely imaginary modular parameter. ``1j`` is the square lattice.
        theta_terms: Number of terms retained in the theta1 series.
        eisen_terms: Number of terms retained in the Eisenstein series.

    Returns:
        Complex basis values with shape ``z.shape + (3 * flux,)``.
    """
    _validate_geometry(flux, tau, theta_terms, eisen_terms)
    x_len, y_len, _, omega1, q, eta1 = _lattice_parameters(
        flux=flux,
        tau=tau,
        eisen_terms=eisen_terms,
    )
    zeros, momenta, quadratic = _sigma_zeros_and_momenta(flux, x_len, y_len, eta1)

    dz = z[..., None, None] - zeros
    log_sigma = _weierstrass_log_sigma(
        dz,
        omega1=omega1,
        q=q,
        eta1=eta1,
        theta_terms=theta_terms,
    )
    log_psi0 = (
        jnp.sum(log_sigma, axis=-1)
        + jnp.conj(momenta) * z[..., None]
        + quadratic * z[..., None] ** 2
        - jnp.abs(z[..., None]) ** 2 / 4.0
    )
    log_normalizers = _sigma_lll_log_normalizers(
        flux=flux,
        x_len=x_len,
        y_len=y_len,
        omega1=omega1,
        q=q,
        eta1=eta1,
        theta_terms=theta_terms,
    )
    psi0 = jnp.exp(log_psi0 - log_normalizers)

    zeta = _weierstrass_zeta(
        dz,
        omega1=omega1,
        q=q,
        eta1=eta1,
        theta_terms=theta_terms,
    )
    summed_zeta = (
        jnp.sum(zeta, axis=-1)
        + jnp.conj(momenta)
        + 2.0 * quadratic * z[..., None]
        - 0.5 * jnp.conj(z[..., None])
    )
    psi1 = jnp.sqrt(2.0) * summed_zeta * psi0

    wp = _weierstrass_wp(
        dz,
        omega1=omega1,
        q=q,
        eta1=eta1,
        theta_terms=theta_terms,
    )
    psi2 = (
        jnp.sqrt(2.0) * (summed_zeta**2 - jnp.sum(wp, axis=-1) + 2.0 * quadratic) * psi0
    )
    return jnp.concatenate([psi0, psi1, psi2], axis=-1)


class _TorusSigmaOrbitals(nn.Module):
    """Mix learned coefficients with normalized sigma-form orbitals."""

    nspins: tuple[int, int]
    flux: int
    ndets: int
    tau: complex
    theta_terms: int
    eisen_terms: int

    def setup(self) -> None:
        num_basis = 3 * self.flux
        features = [num_basis, sum(self.nspins), self.ndets]
        kernel_init = _lowest_landau_level_kernel_init(self.flux)
        self.orbitals_real = SplitChannelDense(
            channels=self.nspins,
            features=features,
            kernel_init=kernel_init,
        )
        self.orbitals_imag = SplitChannelDense(
            channels=self.nspins,
            features=features,
            kernel_init=kernel_init,
        )

    def __call__(self, h_one: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        coefficients = self.orbitals_real(h_one) + 1j * self.orbitals_imag(h_one)
        envelope = torus_sigma_envelope(
            z,
            flux=self.flux,
            tau=self.tau,
            theta_terms=self.theta_terms,
            eisen_terms=self.eisen_terms,
        )
        orbitals = jnp.sum(coefficients * envelope[..., None, None], axis=1)
        return jnp.moveaxis(orbitals, -1, 0)


class TorusSigmaMHPO(Wavefunction[HallData, ComplexWFOutput]):
    r"""Sigma-function MHPO ansatz restricted to rectangular tori.

    ``HallData.electrons`` stores fractional coordinates ``(u, v)``.  They are
    mapped to the centered Cartesian cell before the sigma orbitals are
    evaluated. Only purely imaginary ``tau`` values are accepted. The square
    lattice ``tau=1j`` is the only validated geometry for this deprecated ansatz.
    """

    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()
    tau: complex = runtime_dep()

    ndets: int = 1
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 4
    theta_terms: int = 24
    eisen_terms: int = 48

    def setup(self) -> None:
        _validate_geometry(
            self.flux,
            self.tau,
            self.theta_terms,
            self.eisen_terms,
        )
        warnings.warn(
            "TorusSigmaMHPO is deprecated and should only be used on the "
            "square lattice (tau=1j). Other rectangular lattices are accepted "
            "for compatibility but have not been validated.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.backbone_layer = PsiformerBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
        )
        self.orbital_layer = _TorusSigmaOrbitals(
            nspins=self.nspins,
            flux=self.flux,
            ndets=self.ndets,
            tau=self.tau,
            theta_terms=self.theta_terms,
            eisen_terms=self.eisen_terms,
        )

        x_len, y_len, *_ = _lattice_parameters(
            flux=self.flux,
            tau=self.tau,
            eisen_terms=self.eisen_terms,
        )
        self.jastrow_layer = TorusJastrow(
            nspins=self.nspins,
            l1=x_len,
            l2=1j * y_len,
        )

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        u, v = electrons[..., 0], electrons[..., 1]
        x_len, y_len, *_ = _lattice_parameters(
            flux=self.flux,
            tau=self.tau,
            eisen_terms=self.eisen_terms,
        )

        h_one = jnp.stack(
            [
                jnp.cos(2.0 * jnp.pi * u),
                jnp.sin(2.0 * jnp.pi * u),
                jnp.cos(2.0 * jnp.pi * v),
                jnp.sin(2.0 * jnp.pi * v),
            ],
            axis=-1,
        )
        h_one = self.backbone_layer(h_one)

        z = (u - 0.5) * x_len + 1j * (v - 0.5) * y_len
        orbitals = self.orbital_layer(h_one, z)

        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        logpsi = jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax
        logpsi += self.jastrow_layer(electrons)
        return ComplexWFOutput(logpsi=logpsi)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]

# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
r"""Magnetic-harmonic product-orbital (MHPO) ansatz on a torus.

Coordinates are fractional lattice coordinates (u, v). In magnetic-length
units, the physical position and flux-quantization condition are

.. math::

    z = L_1(u + \tau v), \qquad
    L_1^2\operatorname{Im}\tau = 2\pi N_\phi.
"""

import jax
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

__all__ = ["TorusMHPO"]


def torus_envelope(
    z: jnp.ndarray, flux: int, tau: complex, theta_terms: int
) -> jnp.ndarray:
    r"""Evaluate the first three torus Landau levels in symmetric gauge.

    All lengths are expressed in magnetic-length units, so l_B = 1. Basis
    functions are ordered first by Landau-level index and then by guiding-centre
    index: (0, 0), ..., (0, N_phi - 1), ..., (2, N_phi - 1).

    The Gaussian prefactor and each theta-series term are combined before
    exponentiation. The resulting real exponent is
    -pi N_phi Im(tau) (q + rho)^2, avoiding cancellation between a large theta
    value and a small Gaussian prefactor.

    Args:
        z: Complex electron positions in magnetic-length units.
        flux: Positive number of magnetic flux quanta N_phi.
        tau: Torus modular parameter with positive imaginary part.
        theta_terms:
            Half-width of the theta-series truncation.
            A total of 2 * theta_terms + 1 terms are retained around the dominant term.

    Returns:
        Complex basis values with shape z.shape + (3 * flux,).
    """
    l1, _, _ = get_torus_lattice(flux, tau)
    alpha = flux / l1

    guiding_centres = jnp.arange(flux)
    characteristics = 0.5 + guiding_centres / flux

    # rho is the fractional coordinate along L_2. Centre each finite series
    # on its largest-magnitude term before adding a fixed number of neighbours.
    rho = z.imag / (l1 * tau.imag)
    series_centres = jax.lax.stop_gradient(jnp.rint(-rho[..., None] - characteristics))

    offsets = jnp.arange(-theta_terms, theta_terms + 1)
    q = series_centres[..., None] + offsets + characteristics[:, None]

    # Algebraically combine P(z, z*) with E_(m,j)(Z). The real part is
    # manifestly non-positive, preventing overflow in the theta series.
    real_exponent = -jnp.pi * flux * tau.imag * (q + rho[..., None, None]) ** 2
    real_z_scaled = flux * z.real / l1
    imag_exponent = (
        z.real[..., None, None] * z.imag[..., None, None] / 2
        + jnp.pi * flux * tau.real * q**2
        + 2 * jnp.pi * q * (real_z_scaled[..., None, None] + flux / 2)
    )
    terms = jnp.exp(real_exponent + 1j * imag_exponent)

    derivative_factor = 2j * jnp.pi * q
    p_t0 = jnp.sum(terms, axis=-1)
    p_t1 = jnp.sum(derivative_factor * terms, axis=-1)
    p_t2 = jnp.sum(derivative_factor**2 * terms, axis=-1)

    normalization = 1 / (jnp.pi**0.25 * jnp.sqrt(l1))
    c = (z - jnp.conj(z))[..., None] / 2
    psi0 = normalization * p_t0
    psi1 = -jnp.sqrt(2.0) * normalization * (alpha * p_t1 + c * p_t0)
    psi2 = (
        jnp.sqrt(2.0)
        * normalization
        * (alpha**2 * p_t2 + 2 * alpha * c * p_t1 + (c**2 + 0.5) * p_t0)
    )
    return jnp.concatenate([psi0, psi1, psi2], axis=-1)


class TorusOrbitals(nn.Module):
    r"""Mix learned coefficients with a torus Landau-level basis.

    Args:
        nspins: (n_up, n_down) electron counts.
        flux: Number of magnetic flux quanta N_phi.
        tau: Torus modular parameter in the upper half-plane.
        ndets: Number of determinants.
        theta_terms: Half-width of the theta-series truncation. A total of
            2 * theta_terms + 1 terms are retained around the dominant term.
    """

    nspins: tuple[int, int]
    flux: int
    tau: complex
    ndets: int
    theta_terms: int = 48

    def setup(self) -> None:
        # Every retained Landau level has N_phi independent guiding-centre
        # orbitals. Here n = 0, 1, 2, hence 3 * N_phi basis functions.
        num_basis = 3 * self.flux
        features = [num_basis, sum(self.nspins), self.ndets]
        self.orbitals_real = SplitChannelDense(
            channels=self.nspins,
            features=features,
        )
        self.orbitals_imag = SplitChannelDense(
            channels=self.nspins,
            features=features,
        )

    def __call__(self, h_one: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Construct orbital matrices.

        Returns:
            Complex orbital matrices with shape (ndets, nelec, nelec).
        """
        coefficients = self.orbitals_real(h_one) + 1j * self.orbitals_imag(h_one)

        envelope = torus_envelope(z, self.flux, self.tau, self.theta_terms)

        orbitals = jnp.sum(coefficients * envelope[..., None, None], axis=1)
        return jnp.moveaxis(orbitals, -1, 0)


class TorusMHPO(Wavefunction[HallData, ComplexWFOutput]):
    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()
    tau: complex = runtime_dep()

    ndets: int = 1
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 4

    def setup(self) -> None:
        self.backbone_layer = PsiformerBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
        )
        self.orbital_layer = TorusOrbitals(
            nspins=self.nspins, flux=self.flux, tau=self.tau, ndets=self.ndets
        )

        l1, l2, _ = get_torus_lattice(self.flux, self.tau)
        self.jastrow_layer = TorusJastrow(nspins=self.nspins, l1=l1, l2=l2)

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons

        # Fractional coordinates: r = u L_1 + v L_2, with u, v periodic
        # modulo one. z is measured in magnetic-length units.
        u, v = electrons[..., 0], electrons[..., 1]
        l1, l2, _ = get_torus_lattice(self.flux, self.tau)

        # Embed each circle with sine/cosine pairs so the learned coefficients
        # are exactly periodic under either primitive-lattice translation.
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

        z = u * l1 + v * l2
        orbitals = self.orbital_layer(h_one, z)

        # Stable complex sum of determinants. For complex matrices, signs
        # are unit phases and logdets are logarithms of determinant moduli.
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        logpsi = jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

        # Cusp condition
        jastrow = self.jastrow_layer(electrons)
        logpsi += jastrow

        return ComplexWFOutput(logpsi=logpsi)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]

# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Product orbital wavefunction on the disk.

The input features are Cartesian coordinates on the disk:
``[x, y]``.
These are fed into a PsiformerBackbone (which appends spin encoding),
followed by disk orbital projection and a disk Jastrow factor.
"""

import enum

import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.backbone.psiformer import PsiformerBackbone
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction
from jaqmc.wavefunction.output.orbital import SplitChannelDense

from .jastrow import DiskJastrow

__all__ = ["DiskMHPO", "EnvelopeType"]


class EnvelopeType(enum.StrEnum):
    """Types of envelope functions for disk orbitals."""

    no_m_learnable = "no_m_learnable"
    m_learnable = "m_learnable"
    m_learnable2 = "m_learnable2"
    m_learnable2_old = "m_learnable2_old"


class DiskOrbitals(nn.Module):
    r"""Disk orbital construction.

    Combines learned features with disk basis functions
    :math:`z^m \exp(-r^2 \sigma)` where :math:`z = x - iy`.

    Args:
        Q: Maximum angular momentum (number of flux quanta).
        nspins: ``(n_up, n_down)`` electron counts.
        ndets: Number of determinants.
        envelope_type: The type of envelope to apply to the orbitals.
        num_orbits: Orbital count override. Defaults to ``Q``.
    """

    Q: float
    nspins: tuple[int, int]
    ndets: int
    envelope_type: EnvelopeType = EnvelopeType.no_m_learnable
    num_orbits: int | None = None

    def setup(self) -> None:
        if self.num_orbits is None:
            self.num_orbitals = int(self.Q)
        else:
            self.num_orbitals = self.num_orbits
        features = [self.num_orbitals, sum(self.nspins), self.ndets]
        self.orbitals_real = SplitChannelDense(channels=self.nspins, features=features)
        self.orbitals_imag = SplitChannelDense(channels=self.nspins, features=features)
        num_electrons = sum(self.nspins)

        if self.envelope_type == EnvelopeType.m_learnable2_old:
            self.sigma = self.param(
                "sigma",
                nn.initializers.constant(0.25),
                (self.num_orbitals,),
            )
        else:
            self.sigma = self.param(
                "sigma",
                nn.initializers.constant(0.25),
                (num_electrons, self.num_orbitals),
            )
            self.amplitude = self.param(
                "amplitude",
                nn.initializers.ones,
                (num_electrons, self.num_orbitals),
            )
            self.amplitude_imag = self.param(
                "amplitude_imag",
                nn.initializers.zeros,
                (num_electrons, self.num_orbitals),
            )

    def __call__(
        self, h_one: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        orbitals = self.orbitals_real(h_one) + 1j * self.orbitals_imag(h_one)

        if self.envelope_type == EnvelopeType.m_learnable2_old:
            r_square = (x**2 + y**2)[..., None]
            sigma = self.sigma

            m = jnp.arange(self.num_orbitals)
            z = (x - 1j * y)[..., None]
            safe_z = jnp.where(z == 0, 1.0, z)

            log_norm = jnp.asarray(
                -0.5 * ss.gammaln(np.arange(self.num_orbitals) + 1.0)
                - 0.5 * np.arange(self.num_orbitals) * np.log(2.0)
            )
            envelope_log = m * jnp.log(safe_z) + log_norm - r_square * jnp.abs(sigma)
            envelope = jnp.where(
                z == 0,
                jnp.where(
                    m == 0,
                    jnp.exp(log_norm - r_square * jnp.abs(sigma)),
                    0.0,
                ),
                jnp.exp(envelope_log),
            )

            orbitals = jnp.sum(
                orbitals * envelope[..., None, None],
                axis=1,
            )
            return jnp.moveaxis(orbitals, -1, 0)

        r_square = (x**2 + y**2)[..., None, None]

        sigma_mi = jnp.swapaxes(jnp.abs(self.sigma), 0, 1)
        amplitude_mi = jnp.swapaxes(self.amplitude + 1j * self.amplitude_imag, 0, 1)

        sigma_mi = sigma_mi[None, ...]
        amplitude_mi = amplitude_mi[None, ...]

        if self.envelope_type == EnvelopeType.no_m_learnable:
            envelope = amplitude_mi * jnp.exp(-r_square * sigma_mi)

        elif self.envelope_type == EnvelopeType.m_learnable:
            m = jnp.arange(self.num_orbitals)
            z = (x - 1j * y)[..., None]

            z_power = z**m

            log_norm = jnp.asarray(
                -0.5 * ss.gammaln(np.arange(self.num_orbitals) + 1.0)
            )
            envelope = (
                amplitude_mi
                * z_power[..., None]
                * jnp.exp(log_norm[None, :, None] - r_square * sigma_mi)
            )

        elif self.envelope_type == EnvelopeType.m_learnable2:
            m = jnp.arange(self.num_orbitals)
            r_square = x**2 + y**2
            theta = jnp.arctan2(y, x)
            r = jnp.sqrt(r_square)
            safe_r = jnp.where(r > 0, r, 1.0)
            log_r = jnp.log(safe_r)
            log_norm = jnp.asarray(
                -0.5 * ss.gammaln(np.arange(self.num_orbitals) + 1.0)
                - 0.5 * np.arange(self.num_orbitals) * np.log(2.0)
            )

            log_magnitude = (
                log_norm[None, :, None]
                + log_r[:, None, None] * m[None, :, None]
                - r_square[:, None, None] * sigma_mi
            )

            phase = jnp.exp(-1j * theta[:, None, None] * m[None, :, None])
            basis = jnp.exp(log_magnitude) * phase
            zero_at_positive_m = (r == 0)[:, None, None] & (m > 0)[None, :, None]
            basis = jnp.where(
                zero_at_positive_m,
                0.0,
                basis,
            )

            envelope = amplitude_mi * basis

        else:
            raise ValueError(f"Unknown envelope type: {self.envelope_type}")

        orbitals = jnp.sum(
            orbitals * envelope[..., None],
            axis=-3,
        )

        return jnp.moveaxis(orbitals, -1, 0)  # (ndets, nelec, nelec)


class DiskMHPO(Wavefunction[HallData, ComplexWFOutput]):
    r"""Product orbital ansatz on the disk.

    Architecture:
        1. Input: ``(x, y)`` Cartesian features
        2. Backbone: :class:`~jaqmc.wavefunction.backbone.psiformer.PsiformerBackbone`
        3. Orbitals: Disk orbital projection
        4. Jastrow: Disk Jastrow factor
        5. Output: ``slogdet`` of orbital matrix to complex ``log psi``

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
        monopole_strength: :math:`Q`.
        flux: Magnetic flux :math:`Q`.
        ndets: Number of determinants.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        num_layers: Number of Psiformer layers.
        flux_per_elec: Reserved compatibility field. Disk orbitals use ``flux``
            directly and do not apply flux attachment.
        envelope_type: The type of envelope to apply to the orbitals.
        num_orbitals: Orbital count override. Defaults to ``flux``.
    """

    nspins: tuple[int, int] = runtime_dep()
    monopole_strength: float = runtime_dep()
    flux: int = runtime_dep()
    ndets: int = 1
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
    flux_per_elec: int = 0
    envelope_type: EnvelopeType = EnvelopeType.m_learnable2
    num_orbitals: int | None = None

    def setup(self) -> None:
        reduced_flux = self.flux
        self.backbone_layer = PsiformerBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
        )
        self.orbital_layer = DiskOrbitals(
            Q=reduced_flux,
            nspins=self.nspins,
            ndets=self.ndets,
            envelope_type=self.envelope_type,
            num_orbits=self.num_orbitals,
        )
        self.jastrow_layer = DiskJastrow(nspins=self.nspins)

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        x, y = electrons[..., 0], electrons[..., 1]

        h_one = jnp.stack([x, y], axis=-1)

        h_one = self.backbone_layer(h_one)

        orbitals = self.orbital_layer(h_one, x, y)

        jastrow = self.jastrow_layer(electrons)

        orbitals = jnp.exp(jastrow / sum(self.nspins)) * orbitals

        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        logpsi = jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

        return ComplexWFOutput(logpsi=logpsi)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]

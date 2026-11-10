# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Round-13 six-electron fitted wavefunctions, without extra normalization.

The attachment's decimal coefficients are kept verbatim. Select
``symbolic_round13:DiskP05C330``, ``DiskP10C459`` or ``DiskP11C476`` and
set both ``wf.kappa`` and ``system.interaction_strength`` to the desired
coupling. Coordinates follow ``z = x - i*y``. The background disk is not a
hard boundary. P05's real polynomial can change sign; its phase is retained.
"""

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction

from ._validation import validate_fitted_coupling

__all__ = ["DiskP05C330", "DiskP10C459", "DiskP11C476"]
__all__ += ["FITTED_COEFFICIENTS"]


FITTED_COEFFICIENTS = {
    "DiskP05C330": {
        "alpha0": 0.2461530728,
        "alpha1": 0.01013526811,
        "beta0": -0.01028460786,
        "beta1": 0.02992929725,
        "c0": -3.336391920,
        "c1": 1.780113257,
        "d0": 138.9428223,
        "d1": -99.98869729,
    },
    "DiskP10C459": {
        "cR2": 0.3056314700,
        "cR2k": 0.001073618199,
        "cR2u": -0.2992214978,
        "cR2u2": 0.06318300750,
        "cZ2": -0.01594494317,
        "cZ2k": 0.001472760686,
        "cZ2u": 0.04764815401,
        "cR4": 0.0004664802530,
        "cR4u": 0.0007255300640,
        "cD4": -0.0001064895680,
        "cD4u": 0.0003595575689,
    },
    "DiskP11C476": {
        "cR2": 0.2429927663,
        "cR2u": -0.07175331827,
        "cR2u2": 0.002692015195,
        "cR2k": -0.009256723555,
        "cR2k2": -0.0001039934145,
        "cZ2": -0.007525978955,
        "cZ2u": 0.01819699707,
        "cZ2k": 0.003668980290,
        "cR4": 0.0005107661091,
        "cR4u": 0.0006539525290,
        "cD4": -0.00002290449034,
        "cD4k": 0.00002175028271,
        "cD4u": 0.00006737811575,
        "pR2": -0.00004902436497,
        "pR2u": 0.00001690405049,
    },
}


class _DiskRound13(Wavefunction[HallData, ComplexWFOutput]):
    """Parameter-free Flax interface compatible with disk estimators."""

    kappa: float = 1.0
    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()

    def setup(self) -> None:
        validate_fitted_coupling(self.kappa)
        if isinstance(self, DiskP05C330):
            return
        p, k = FITTED_COEFFICIENTS[type(self).__name__], self.kappa
        u = k / (1 + k)
        self.radial_quartic = p["cR4"] + p["cR4u"] * u
        self.pair_quartic = p["cD4"] + p["cD4u"] * u
        if isinstance(self, DiskP11C476):
            self.pair_quartic += p["cD4k"] * k
        # For six electrons, D4/R4 <= 24, attained by balanced opposite triples.
        if self.radial_quartic + 24 * min(self.pair_quartic, 0.0) < 0:
            raise ValueError("kappa must preserve a decaying quartic envelope.")

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        if sum(self.nspins) != 6 or electrons.shape[-2:] != (6, 2):
            raise ValueError("These fitted wavefunctions require six 2D electrons.")

        x, y = electrons[..., 0], electrons[..., 1]
        z = x - 1j * y
        i, j = jnp.triu_indices(6, k=1)
        differences = z[..., i] - z[..., j]
        log_v = jnp.sum(jnp.log(differences), axis=-1)
        radius_square = x**2 + y**2
        r2 = jnp.sum(radius_square, axis=-1)
        r4 = jnp.sum(radius_square**2, axis=-1)
        z2 = jnp.sum(x, axis=-1) ** 2 + jnp.sum(y, axis=-1) ** 2

        p = FITTED_COEFFICIENTS[type(self).__name__]
        k = self.kappa
        u = k / (1 + k)
        if isinstance(self, DiskP05C330):
            a = p["alpha0"] + p["alpha1"] * u
            b = p["beta0"] + p["beta1"] * u
            c = p["c0"] + p["c1"] * u
            d = p["d0"] + p["d1"] * u
            polynomial = 1 + c * r4 + d * r2
            # Taking a complex log retains pi phase on the negative side.
            # Zeros remain zeros; clipping/absolute values would change P05.
            envelope = -a * r2 - b * z2 + jnp.log(polynomial.astype(log_v.dtype))
        else:
            a = p["cR2"] + p["cR2k"] * k + p["cR2u"] * u + p["cR2u2"] * u**2
            b = p["cZ2"] + p["cZ2k"] * k + p["cZ2u"] * u
            d4 = jnp.sum(jnp.abs(differences) ** 4, axis=-1)
            if isinstance(self, DiskP11C476):
                a += p["cR2k2"] * k**2
            envelope = (
                -a * r2 - b * z2 - self.radial_quartic * r4 - self.pair_quartic * d4
            )
            if isinstance(self, DiskP11C476):
                envelope = envelope + 1j * (p["pR2"] + p["pR2u"] * u) * r2

        return ComplexWFOutput(logpsi=3 * log_v + envelope)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]


class DiskP05C330(_DiskRound13):
    """V^3 exp(-A R2-B Z2) (1+C R4+D R2), complexity 330."""


class DiskP10C459(_DiskRound13):
    """V^3 exp(-A R2-B Z2-C R4-D D4), complexity 459."""


class DiskP11C476(_DiskRound13):
    """P11's real pair envelope times exp(i P R2), complexity 476."""

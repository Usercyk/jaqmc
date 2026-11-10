# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Refitted phi_7, candidate 01911509867937424e38, complexity 363.

Coefficients are copied verbatim from the supplied CSV. This independent
module preserves the earlier DiskPhi7 implementation and its results.
"""

import math

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction

FITTED_PARAMETERS = {
    "alpha": 0.2064900697136671,
    "beta": 0.2202223691836707,
    "mu": -0.0003079074307559089,
    "rho": -0.9067208213342699,
    "lam": 0.0008868520186554253,
    "the": -0.041807878886742984,
    "a0": -28.330865323619815,
    "a1": 653.6374583725458,
    "s1": 2.514613948199654,
    "b0": 0.17440187184901346,
    "b1": 0.051134293423375186,
    "s2": 38.058012029584006,
    "c0": -20.587030761621186,
    "c1": 0.1280896702403507,
    "s3": 178.82859705075208,
}


class DiskPhi7RefitC363(Wavefunction[HallData, ComplexWFOutput]):
    """V^3 times the supplied Gaussian, modulus and radial envelope."""

    kappa: float = 1.0
    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        if sum(self.nspins) != 6 or electrons.shape[-2:] != (6, 2):
            raise ValueError("This fitted phi_7 requires six 2D electrons.")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative.")
        p, k = FITTED_PARAMETERS, self.kappa
        x, y = electrons[..., 0], electrons[..., 1]
        z = x - 1j * y
        i, j = jnp.triu_indices(6, k=1)
        log_v = jnp.sum(jnp.log(z[..., i] - z[..., j]), axis=-1)
        radii_squared = x**2 + y**2
        r2 = jnp.sum(radii_squared, axis=-1)
        r4 = jnp.sum(radii_squared**2, axis=-1)
        z2 = jnp.sum(x, axis=-1) ** 2 + jnp.sum(y, axis=-1) ** 2
        envelope = (
            -p["alpha"] * r2 / (1 + p["beta"] * k)
            - p["mu"] * z2 / (1 + p["rho"] * k)
            - p["lam"] * r4 / (1 + p["the"] * k)
            + p["a0"]
            * k
            / (1 + p["a1"] * k)
            * jnp.logaddexp(0.0, 2 * log_v.real - jnp.log(p["s1"]))
            + p["b0"]
            * k
            / (1 + p["b1"] * k)
            * jnp.logaddexp(0.0, 2 * log_v.real - jnp.log(p["s2"]))
            + p["c0"] * k / (1 + p["c1"] * k) * jnp.log1p(r2 / p["s3"])
        )
        return ComplexWFOutput(logpsi=3 * log_v + envelope)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]

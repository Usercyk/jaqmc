# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Five six-electron states supplied on 2026-10-21, in complex log form.

Coefficients follow the attachment's full-precision Python formulas. There
is no additional Gaussian, fitted normalization, or trainable parameter.
"""

import math

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction

FITTED_COEFFICIENTS = {
    "DiskP6": dict(
        alpha=0.1982781981117185,
        beta=0.00652277315382952,
        mu=0.004514522083104814,
        rho=-0.0650985695510166,
        lam=0.0013685853034064736,
        theta=-0.03994584176996582,
    ),
    "DiskP12": dict(
        alpha=0.429509714082805,
        quartic0=0.023001980635208346,
        quartic1=0.031221509444597485,
        v_scale=126.69231236547668,
        a0=0.06766079253221341,
        a1=0.13732996359871305,
        a2=0.1476918940290431,
        pair_scale=-6.606146810945341,
        pair_power=-4.272513044832577,
        r_scale=11.468832271945884,
        r_power=-4.464089369994271,
        r_kappa=0.08881751764894293,
    ),
    "DiskPsi7": dict(
        alpha=0.18204900578032823,
        quartic=0.0005198128230869609,
        a=0.1351268428003754,
        a_kappa=0.014086130551065806,
        r_scale=119.47368257282588,
        r_power=-4.054674544542452,
        pair_power=-3.473762652663028,
    ),
    "DiskPsi6": dict(
        alpha0=0.18159888134272242,
        alpha1=0.01706920451293234,
        quartic=0.0004963654940947599,
        a0=0.10741255786766907,
        a1=0.12327850299424178,
        a_kappa=0.016557320426849393,
        pair_power=-5.277356489363081,
    ),
    "DiskPsi5": dict(
        alpha=0.2152067954686051,
        quartic=0.0004974695948346683,
        a=0.11379295500696728,
        r_scale=56.40218835151533,
        r_power=-2.838599453667624,
    ),
}


class _DiskSymbolic20261021(Wavefunction[HallData, ComplexWFOutput]):
    kappa: float = 1.0
    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        if sum(self.nspins) != 6 or electrons.shape[-2:] != (6, 2):
            raise ValueError("These fitted states require six 2D electrons.")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative.")
        x, y = electrons[..., 0], electrons[..., 1]
        z = x - 1j * y
        i, j = jnp.triu_indices(6, k=1)
        dz = z[..., i] - z[..., j]
        log_v = jnp.sum(jnp.log(dz), axis=-1)
        radii2 = x**2 + y**2
        r2 = jnp.sum(radii2, axis=-1)
        r4 = jnp.sum(radii2**2, axis=-1)
        z2 = jnp.sum(x, axis=-1) ** 2 + jnp.sum(y, axis=-1) ** 2
        # Exactly sum_{i<j}|zi-zj|^2 = 6*R2-Z2, without cancellation.
        pair2 = jnp.sum(
            (x[..., i] - x[..., j]) ** 2 + (y[..., i] - y[..., j]) ** 2, axis=-1
        )
        log_v2 = 2 * log_v.real
        p, k = FITTED_COEFFICIENTS[type(self).__name__], self.kappa
        if isinstance(self, DiskP6):
            envelope = (
                -p["alpha"] * r2 / (1 + p["beta"] * k)
                - p["mu"] * z2 / (1 + p["rho"] * k)
                - p["lam"] * r4 / (1 + p["theta"] * k)
            )
        elif isinstance(self, DiskP12):
            envelope = (
                -(p["alpha"] ** 2) * r2
                - (p["quartic0"] ** 2 + p["quartic1"] ** 2 * k / (1 + k)) * r4
                + (p["a0"] + p["a1"] * k / (1 + p["a2"] ** 2 * k))
                * jnp.logaddexp(0.0, log_v2 - jnp.log1p(p["v_scale"] ** 2))
                + p["pair_power"] * jnp.log1p(p["pair_scale"] ** 2 * pair2)
                + p["r_power"]
                * k
                / (1 + p["r_kappa"] ** 2 * k)
                * jnp.log1p(r2 / (1 + p["r_scale"] ** 2))
            )
        else:
            alpha = p.get("alpha", p.get("alpha0", 0.0) + p.get("alpha1", 0.0) * k)
            envelope = -alpha * r2 - p["quartic"] * (1 + 2 * k / (1 + k)) * r4
            if isinstance(self, DiskPsi6):
                power = p["a0"] + p["a1"] * k / (1 + p["a_kappa"] * k)
            elif isinstance(self, DiskPsi7):
                power = p["a"] * k / (1 + p["a_kappa"] * k)
            else:
                power = p["a"] * k
            envelope += power * jnp.logaddexp(0.0, log_v2)
            if "pair_power" in p:
                envelope += p["pair_power"] * jnp.log1p(pair2)
            if "r_power" in p:
                envelope += p["r_power"] * k * jnp.log1p(r2 / p["r_scale"])
        return ComplexWFOutput(logpsi=3 * log_v + envelope)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]


class DiskP6(_DiskSymbolic20261021):
    """Attachment P6."""


class DiskP12(_DiskSymbolic20261021):
    """Attachment P12."""


class DiskPsi7(_DiskSymbolic20261021):
    """Attachment Psi7; distinct from the earlier fitted phi_7."""


class DiskPsi6(_DiskSymbolic20261021):
    """Attachment Psi6; distinct from P6."""


class DiskPsi5(_DiskSymbolic20261021):
    """Attachment Psi5."""

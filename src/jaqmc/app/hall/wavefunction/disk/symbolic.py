# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Six-electron symbolic-regression wavefunctions with fixed fitted coefficients.

Select ``symbolic:DiskPhi4``, ``symbolic:DiskPhi7`` or ``symbolic:DiskPhi8``
and set ``wf.kappa``. For energy evaluation, also set
``system.interaction_strength`` to the same value. No extra Gaussian or
normalization factor is added to the fitted expressions.
"""

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction

from ._validation import validate_fitted_coupling

__all__ = ["DiskPhi4", "DiskPhi7", "DiskPhi8"]


class _DiskSymbolic(Wavefunction[HallData, ComplexWFOutput]):
    """Shared log-wavefunction interface, following ``DiskLaughlin``."""

    kappa: float = 1.0
    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()
    _candidate = 4

    def setup(self) -> None:
        p = _COEFFICIENTS[self._candidate]
        validate_fitted_coupling(
            self.kappa,
            quartic_slope=p["theta"],
            denominator_slopes=tuple(
                p[key] for key in ("beta", "rho", "a1", "b1", "c1") if key in p
            ),
        )

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        if sum(self.nspins) != 6 or electrons.shape[-2:] != (6, 2):
            raise ValueError("These fitted wavefunctions require six 2D electrons.")

        x = electrons[..., 0]
        y = electrons[..., 1]
        z = x - 1j * y
        i, j = jnp.triu_indices(6, k=1)
        log_v = jnp.sum(jnp.log(z[..., i] - z[..., j]), axis=-1)
        r_square = x**2 + y**2
        r2 = jnp.sum(r_square, axis=-1)
        r4 = jnp.sum(r_square**2, axis=-1)
        c = jnp.sum(x, axis=-1) ** 2 + jnp.sum(y, axis=-1) ** 2

        p = _COEFFICIENTS[self._candidate]
        k = self.kappa
        envelope = (
            -p["alpha"] * r2 / (1 + p["beta"] * k)
            - p["mu"] * c / (1 + p["rho"] * k)
            - p["lam"] * r4 / (1 + p["theta"] * k)
        )
        if "a0" in p:
            a = p["a0"] * k / (1 + p["a1"] * k)
            b = p["b0"] * k / (1 + p["b1"] * k)
            log_t = 2 * log_v.real
            if "s1" in p:
                # log(1 + |V|^2 / s) without forming the large product |V|^2.
                envelope += a * jnp.logaddexp(0.0, log_t - jnp.log(p["s1"]))
                envelope += b * jnp.logaddexp(0.0, log_t - jnp.log(p["s2"]))
            else:
                envelope += (a + b) * jnp.logaddexp(0.0, log_t)
            g = p["c0"] * k / (1 + p["c1"] * k)
            envelope += g * jnp.log1p(r2 / p["s3"])
            if "d0" in p:
                h = p["d0"] * k / (1 + p["c1"] * k)
                envelope += h * r2 / (p["s4"] + r2)

        return ComplexWFOutput(logpsi=3 * log_v + envelope)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]


class DiskPhi4(_DiskSymbolic):
    """Phi_4 = V^3 exp(E_4), candidate 7d23181e3bd894241876."""

    _candidate = 4


class DiskPhi7(_DiskSymbolic):
    """Phi_7 with two modulus scales, candidate 01911509867937424e38."""

    _candidate = 7


class DiskPhi8(_DiskSymbolic):
    """Phi_8 with a radial saturation term, candidate 368aeb93d1bfe06885e8."""

    _candidate = 8


_COEFFICIENTS = {
    4: {
        "alpha": 0.1961075809838481,
        "beta": 0.010750346325253672,
        "mu": 0.006174150460917736,
        "rho": -0.05753521606126198,
        "lam": 0.0016474743622237546,
        "theta": -0.034283585416094935,
    },
    7: {
        "alpha": 0.2831794969796563,
        "beta": 0.4120822269919293,
        "mu": 0.0020189476254209734,
        "rho": -1.1889995670549587,
        "lam": 0.0010754688516736595,
        "theta": -0.03455034013629768,
        "a0": 127.96886260208653,
        "a1": 438.7909784245286,
        "s1": 18.76737466999703,
        "b0": 0.07716973055919304,
        "b1": -0.00035816450592872926,
        "s2": 35.5983969272006,
        "c0": -19.9999980382926,
        "c1": 0.1316168909562256,
        "s3": 167.39566413678367,
    },
    8: {
        "alpha": 0.28573565149756935,
        "beta": 0.45494817896468026,
        "mu": 0.0020045670838933416,
        "rho": -1.1879825744591286,
        "lam": 0.0010626382719403792,
        "theta": -0.03539363567777555,
        "a0": 195.97204518731328,
        "a1": 672.1615249138852,
        "b0": 0.08020033881911745,
        "b1": 0.003957707528090986,
        "c0": -23.34408437339075,
        "c1": 0.14883214610239642,
        "s3": 18.50131538137204,
        "d0": 48.16015478721567,
        "s4": 67.61862732810887,
    },
}

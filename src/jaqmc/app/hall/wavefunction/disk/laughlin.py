# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact Laughlin wavefunction on the disk (for benchmarking)."""

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction

__all__ = ["DiskLaughlin"]


class DiskLaughlin(Wavefunction[HallData, ComplexWFOutput]):
    """Exact fully polarized nu=1/m Laughlin state on a disk."""

    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons

        x = electrons[..., 0]
        y = electrons[..., 1]

        z = x - 1j * y
        r_square = x**2 + y**2

        n_elec = sum(self.nspins)
        m = self.flux / n_elec
        dz = z[..., :, None] - z[..., None, :]

        i, j = jnp.triu_indices(n_elec, k=1)
        pair_differences = dz[..., i, j]

        log_jastrow = m * jnp.sum(
            jnp.log(pair_differences),
            axis=-1,
        )

        log_gaussian = -jnp.sum(r_square, axis=-1) / 4.0

        logpsi = log_jastrow + log_gaussian

        return ComplexWFOutput(logpsi=logpsi)

    def logpsi(
        self,
        params: Params,
        data: HallData,
    ) -> jnp.ndarray:
        output = self.evaluate(params, data)
        return output["logpsi"]

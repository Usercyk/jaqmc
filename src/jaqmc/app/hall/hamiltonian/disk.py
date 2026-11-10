# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Coulomb and neutralizing-background potential on a disk."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from jax import numpy as jnp
from scipy.integrate import quad
from scipy.special import ellipk

from jaqmc.app.hall.config import InteractionType
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class DiskPotential(PerWalkerEstimator):
    r"""Disk Coulomb interaction plus a uniform positive background.

    Lengths use the magnetic length. The background disk has radius
    ``radius``, charge ``N``, and vertical separation ``d``. The original
    implementation omits the constant background-background energy.
    ``radial_grid_size`` controls the interpolation grid (default 10000).
    Tables are generated once in ``init`` and held in memory.

    Args:
        interaction_type: Electron-electron interaction form.
        monopole_strength: Flux parameter retained for checkpoint compatibility.
        radius: Radius of the uniformly charged background disk.
        N: Total positive background charge in electron-charge units.
        flux: Number of flux quanta retained in the system configuration.
        d: Vertical separation from the background disk.
        interaction_strength: Scaling factor for repulsion and background energy.
        radial_grid_size: Number of radial interpolation points.
    """

    interaction_type: InteractionType = InteractionType.coulomb
    monopole_strength: float = 1.0
    radius: float = 1.0
    N: int = 6
    flux: int = 18
    d: float = 0.1
    interaction_strength: float = 1.0
    radial_grid_size: int = 10000

    def single_quad(self, r: float, limit: int = 1000, epsrel: float = 1e-4):
        a = self.radius
        d = self.d
        N = self.N

        def integrand(rp):
            A = (r + rp) ** 2 + d**2
            m1 = 4 * r * rp / A
            f = 4 * rp * ellipk(m1) / np.sqrt(A)

            return f

        integral, err = quad(
            integrand,
            0.0,
            a,
            epsrel=epsrel,
            limit=limit,
            points=[r] if d == 0 and 0 < r < a else None,
        )
        pref = -N / np.pi / a**2

        return pref * integral, err

    def make_conf_potential(self, r_vals, limit: int = 1000, epsrel: float = 1e-4):
        Vc = np.zeros_like(r_vals)
        err = np.zeros_like(r_vals)
        for i, r in enumerate(r_vals):
            val, e = self.single_quad(r, limit, epsrel)

            Vc[i] = val
            err[i] = e

        return Vc, err

    def init(self, data: Data, rngs: PRNGKey) -> None:
        """Build the background table before JIT/vmap evaluation.

        Raises:
            ValueError: If the physical dimensions or interpolation size are invalid.
        """
        del data, rngs
        if self.radius <= 0 or self.N <= 0 or self.d < 0:
            raise ValueError("radius and N must be positive and d non-negative")
        if self.radial_grid_size < 2:
            raise ValueError("radial_grid_size must be at least 2")
        r_grid = np.linspace(0.0, 16 * self.radius, self.radial_grid_size)
        table, _ = self.make_conf_potential(r_grid)
        self._r_grid = jnp.asarray(r_grid)
        self._Vc_table = jnp.asarray(table)
        return None

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
        x, y = electrons[..., 0], electrons[..., 1]

        xy = jnp.stack([x, y], axis=-1)

        r = jnp.sqrt(x**2 + y**2)

        r_grid = self._r_grid
        Vc_table = self._Vc_table

        # 插值计算 r<a*15 区域的势能
        V_small = jnp.interp(r, r_grid, Vc_table)
        # r>=15a 时近似为点电荷势
        V_large = -self.N / jnp.sqrt(self.d**2 + r**2)
        # 根据条件选择势能
        V_vals = jnp.where(r < 15 * self.radius, V_small, V_large)
        # Sum the background contribution over electrons.

        potential_conf = jnp.sum(V_vals, axis=-1)

        if self.interaction_type == "coulomb":
            i, j = jnp.triu_indices(xy.shape[0], k=1)
            r_ee = jnp.linalg.norm(xy[i] - xy[j], axis=-1)
            potential_ee = jnp.sum(1 / r_ee)
            potential = potential_ee + potential_conf

        else:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")

        return {"energy:potential": potential * self.interaction_strength}, state

# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Jastrow factors for wavefunctions on the disk."""

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.optimizer.kfac.tags import register_first_order_parameter


class DiskJastrow(nn.Module):
    """Disk electron-distance Jastrow factor.

    Computes electron-electron Jastrow correlations using Cartesian
    pair distances, with separate parameters for parallel and
    antiparallel spin pairs.

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
    """

    nspins: tuple[int, int]

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        nspins = self.nspins
        r_ee = self._chord_distance(electrons)
        r_ees = [
            jnp.split(r, nspins[0:1], axis=1)
            for r in jnp.split(r_ee, nspins[0:1], axis=0)
        ]
        r_ees_parallel = jnp.concatenate(
            [
                r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
                r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
            ]
        )

        if r_ees_parallel.shape[0] > 0:
            alpha_par = self.param("ee_par", nn.initializers.ones, (1,))
            beta_par = self.param("beta_par", nn.initializers.constant(0.33333), (1,))
            a_p = jnp.squeeze(alpha_par)
            b_p = jnp.squeeze(beta_par)
            jastrow_ee_par = jnp.sum(-(b_p * a_p**2) / (a_p + r_ees_parallel))
            jastrow_ee_par = register_first_order_parameter(
                jastrow_ee_par, alpha_par, beta_par
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        if r_ees[0][1].shape[0] > 0:
            alpha_anti = self.param("ee_anti", nn.initializers.ones, (1,))
            beta_anti = self.param("beta_anti", nn.initializers.constant(0.5), (1,))
            a_a = jnp.squeeze(alpha_anti)
            b_a = jnp.squeeze(beta_anti)
            jastrow_ee_anti = jnp.sum(-(b_a * a_a**2) / (a_a + r_ees[0][1]))
            jastrow_ee_anti = register_first_order_parameter(
                jastrow_ee_anti, alpha_anti, beta_anti
            )
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        return jastrow_ee_anti + jastrow_ee_par

    def _chord_distance(self, electrons: jnp.ndarray) -> jnp.ndarray:
        x = electrons[..., 0]
        y = electrons[..., 1]

        cart_e = jnp.stack([x, y], axis=-1)

        cart_ee = cart_e[None] - cart_e[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        return jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)

# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from flax import linen as nn
from jax import numpy as jnp


class TorusJastrow(nn.Module):
    nspins: tuple[int, int]
    l1: jnp.ndarray
    l2: jnp.ndarray

    def _periodic_distance(self, electrons: jnp.ndarray) -> jnp.ndarray:
        u, v = electrons[..., 0], electrons[..., 1]

        # Pairwise fractional-coordinate differences.
        du = u[None, :] - u[:, None]
        dv = v[None, :] - v[:, None]

        # q(s) = [exp(2 pi i s) - 1] / (2 pi i)
        # q_re = sin(2 pi s) / (2 pi)
        # q_im = [1 - cos(2 pi s)] / (2 pi)
        two_pi = 2.0 * jnp.pi
        qu_re = jnp.sin(two_pi * du) / two_pi
        qu_im = (1.0 - jnp.cos(two_pi * du)) / two_pi

        qv_re = jnp.sin(two_pi * dv) / two_pi
        qv_im = (1.0 - jnp.cos(two_pi * dv)) / two_pi

        l2_abs2 = self.l2.real**2 + self.l2.imag**2

        # Gram matrix:
        #
        # G = [[l1^2,       l1 Re(l2)],
        #      [l1 Re(l2),  |l2|^2   ]]
        #
        # For q = (qu, qv),
        #
        # q^\dagger G q
        # = l1^2 |qu|^2
        # + |l2|^2 |qv|^2
        # + 2 l1 Re(l2) Re(qu* qv).
        qu_abs2 = qu_re**2 + qu_im**2
        qv_abs2 = qv_re**2 + qv_im**2

        quqv_re = qu_re * qv_re + qu_im * qv_im

        distance2 = (
            self.l1**2 * qu_abs2
            + l2_abs2 * qv_abs2
            + 2.0 * self.l1 * self.l2.real * quqv_re
        )

        # Protect the diagonal from sqrt(0) differentiation.
        n = electrons.shape[-2]
        eye = jnp.eye(n, dtype=electrons.dtype)

        distance = jnp.sqrt(distance2 + eye) * (1.0 - eye)

        return distance

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        n_up, n_down = self.nspins

        r_ee = self._periodic_distance(electrons)

        # ------------------------------------------------------------
        # Parallel-spin pairs: up-up and down-down
        # ------------------------------------------------------------
        r_ee_parallel = jnp.concatenate(
            [
                r_ee[:n_up, :n_up][jnp.triu_indices(n_up, k=1)],
                r_ee[n_up:, n_up:][jnp.triu_indices(n_down, k=1)],
            ]
        )

        if r_ee_parallel.shape[0] > 0:
            alpha_par = self.param("ee_par", nn.initializers.ones, (1,), jnp.float32)

            jastrow_ee_par = jnp.sum(
                -(alpha_par**2 / 3.0) / (alpha_par + r_ee_parallel)
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        # ------------------------------------------------------------
        # Antiparallel-spin pairs: up-down
        # ------------------------------------------------------------
        r_ee_anti = r_ee[:n_up, n_up:].reshape(-1)

        if r_ee_anti.shape[0] > 0:
            alpha_anti = self.param("ee_anti", nn.initializers.ones, (1,), jnp.float32)
            jastrow_ee_anti = jnp.sum(-(alpha_anti**2) / (alpha_anti + r_ee_anti))
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        return jastrow_ee_par + jastrow_ee_anti

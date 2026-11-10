# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Kinetic energy estimator on a disk geometry."""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


@configurable_dataclass
class DiskKinetic(PerWalkerEstimator):
    """Kinetic energy estimator on a disk geometry.

    Local angular-momentum values remain complex for penalty gradients.
    Their reported means and variances use the real parts.

    Args:
        include_energy: Include the kinetic energy in the output.
        f_log_psi: Log-psi evaluate function (runtime dep).
        data_field: Name of the data field containing positions (runtime dep).
    """

    include_energy: bool = True
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs

        electrons = data[self.data_field]
        x = electrons[..., 0]
        y = electrons[..., 1]
        r_square = x**2 + y**2

        def f_real(p, e):
            return self.f_log_psi(p, data.merge({self.data_field: e})).real

        def f_imag(p, e):
            return self.f_log_psi(p, data.merge({self.data_field: e})).imag

        grad_real = jax.grad(f_real, argnums=1)(params, electrons)
        grad_imag = jax.grad(f_imag, argnums=1)(params, electrons)

        grad_x = grad_real[..., 0] + 1j * grad_imag[..., 0]
        grad_y = grad_real[..., 1] + 1j * grad_imag[..., 1]

        square_grad_logpsi = jnp.sum(grad_x**2 + grad_y**2)

        hess_real = jax.hessian(f_real, argnums=1)(params, electrons)
        hess_imag = jax.hessian(f_imag, argnums=1)(params, electrons)
        hess_logpsi = hess_real + 1j * hess_imag

        grad_grad_logpsi = jnp.sum(
            jnp.diagonal(hess_logpsi[:, 0, :, 0])
            + jnp.diagonal(hess_logpsi[:, 1, :, 1])
        )

        magnetic_contribution = jnp.sum(
            1j * y * grad_x - 1j * x * grad_y + r_square / 4
        )

        sum_kinetic_momentum_square = (
            -grad_grad_logpsi - square_grad_logpsi + magnetic_contribution
        )
        kinetic_energy = sum_kinetic_momentum_square / 2

        # Compute Lz and Lz^2 from rotation derivatives of log psi.
        D_f = jnp.sum(x * grad_y - y * grad_x)
        term1 = jnp.sum(jnp.outer(x, x) * hess_logpsi[:, 1, :, 1])
        term2 = -jnp.sum(jnp.outer(x, y) * hess_logpsi[:, 1, :, 0])
        term3 = -jnp.sum(jnp.outer(y, x) * hess_logpsi[:, 0, :, 1])
        term4 = jnp.sum(jnp.outer(y, y) * hess_logpsi[:, 0, :, 0])
        term5 = -jnp.sum(x * grad_x + y * grad_y)
        D2_f = term1 + term2 + term3 + term4 + term5

        angular_momentum_z_square = -(D_f**2 + D2_f)
        angular_momentum_z = -1j * D_f

        results = {
            "energy:kinetic": kinetic_energy,
            "observable:angular_momentum_z": angular_momentum_z,
            "observable:angular_momentum_z_square": angular_momentum_z_square,
        }

        if not self.include_energy:
            results.pop("energy:kinetic")
        return results, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        # Penalty gradients need complex local values; reported angular
        # observables and their variances use their real parts.
        return mean_reduce(
            {
                key: value.real if key.startswith("observable:") else value
                for key, value in walker_stats.items()
            }
        )

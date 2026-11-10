# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Fubini--Study overlap estimator for disk wavefunctions.

Samples are expected to come from the symmetric overlap density

.. math::

    \rho(x) \propto |\psi(x)| |\psi_{\mathrm{ref}}(x)|.

For ``d = log|psi| - log|psi_ref|``, the modulus and phase factors are

.. math::

    C_{\mathrm{mod}} =
    \left(\langle e^d\rangle_\rho
           \langle e^{-d}\rangle_\rho\right)^{-1/2},

    F_{\mathrm{phase}} =
    \left|\langle e^{i(\arg\psi_{\mathrm{ref}}-\arg\psi)}\rangle_\rho\right|.
"""

from collections.abc import Mapping
from typing import Any, NamedTuple, cast

import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


class FubiniParams(NamedTuple):
    """Current and reference wavefunction parameters passed through shard_map."""

    current: Params
    reference: Params


@configurable_dataclass
class DiskFubini(PerWalkerEstimator):
    """Fubini--Study overlap with a fixed reference disk wavefunction.

    The estimator must be evaluated under JAQMC's batch ``shard_map`` because
    :meth:`reduce` combines walker statistics across its named device axis.

    Args:
        f_log_psi: Complex log-wavefunction evaluator.
    """

    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()

    def log_overlap_amplitude(self, params: Params, data: Data) -> jax.Array:
        """Return an amplitude whose squared modulus is the overlap density."""
        pair = cast(FubiniParams, params)
        log_psi = self.f_log_psi(pair.current, data)
        log_reference = self.f_log_psi(pair.reference, data)
        return 0.5 * (jnp.real(log_psi) + jnp.real(log_reference))

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, jax.Array], Any]:
        del prev_walker_stats, rngs
        pair = cast(FubiniParams, params)
        log_psi = self.f_log_psi(pair.current, data)
        log_reference = self.f_log_psi(pair.reference, data)
        return {
            "fubini:delta_log_modulus": jnp.real(log_psi - log_reference),
            "fubini:relative_phase": jnp.exp(
                1j * (jnp.imag(log_reference) - jnp.imag(log_psi))
            ),
        }, state

    @staticmethod
    def _distributed_logmeanexp(values: jax.Array) -> jax.Array:
        """Return log-mean-exp over walkers on every batch-axis device."""
        local_max = jnp.max(values)
        global_max = jax.lax.pmax(local_max, axis_name=parallel_jax.BATCH_AXIS_NAME)
        shifted_mean = parallel_jax.pmean(jnp.mean(jnp.exp(values - global_max)))
        return global_max + jnp.log(shifted_mean)

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, jax.Array]:
        delta = local_stats["fubini:delta_log_modulus"]
        phase = local_stats["fubini:relative_phase"]
        return {
            "fubini:log_mean_exp_delta": self._distributed_logmeanexp(delta),
            "fubini:log_mean_exp_minus_delta": (self._distributed_logmeanexp(-delta)),
            "fubini:phase_mean": parallel_jax.pmean(jnp.mean(phase)),
        }

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, jax.Array]:
        """Combine equal-sized MCMC batches into final overlap metrics.

        Returns:
            Modulus overlap, phase factor, total overlap, and distance.

        Raises:
            ValueError: If no batches were evaluated.
        """
        del state
        log_positive = batched_stats["fubini:log_mean_exp_delta"]
        log_negative = batched_stats["fubini:log_mean_exp_minus_delta"]
        if log_positive.shape[0] == 0:
            raise ValueError("Fubini estimator requires at least one batch")

        log_batch_count = jnp.log(log_positive.shape[0])
        log_mean_positive = logsumexp(log_positive) - log_batch_count
        log_mean_negative = logsumexp(log_negative) - log_batch_count
        cos_modulus = jnp.clip(
            jnp.exp(-0.5 * (log_mean_positive + log_mean_negative)),
            0.0,
            1.0,
        )
        phase_mean = jnp.mean(batched_stats["fubini:phase_mean"])
        f_phase = jnp.clip(jnp.abs(phase_mean), 0.0, 1.0)
        cosine = jnp.clip(cos_modulus * f_phase, 0.0, 1.0)
        return {
            "cos_gamma_modulus": cos_modulus,
            "f_phase": f_phase,
            "cosine": cosine,
            "fubini_distance": jnp.arccos(cosine),
        }

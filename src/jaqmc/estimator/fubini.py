# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Fubini--Study distances between checkpointed wavefunctions."""

import re
from collections.abc import Mapping
from operator import itemgetter
from typing import Any

import jax
from jax import numpy as jnp
from upath import UPath

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.utils import parallel_jax
from jaqmc.utils.checkpoint import NumPyCheckpointManager
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import NumericWavefunctionEvaluate

from .base import PerWalkerEstimator

_CHECKPOINT_RE = re.compile(r"^(?P<prefix>.+)_ckpt_(?P<step>\d+)\.npz$")


def _pmax(value: jax.Array) -> jax.Array:
    """Take a maximum across devices, or act as identity outside shard_map.

    Returns:
        The global maximum, or ``value`` outside a shard map.
    """
    try:
        return jax.lax.pmax(value, axis_name=parallel_jax.BATCH_AXIS_NAME)
    except NameError:
        return value


def _safe_ratio(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    return jnp.where(denominator > 0, numerator / denominator, jnp.nan)


def _metrics_from_sums(
    overlap_sum: jax.Array,
    modulus_sum: jax.Array,
    checkpoint_norm_sum: jax.Array,
    reference_norm_sum: jax.Array,
    *,
    name: str,
) -> dict[str, jax.Array]:
    denominator = jnp.sqrt(checkpoint_norm_sum * reference_norm_sum)
    cosine = jnp.clip(_safe_ratio(jnp.abs(overlap_sum), denominator), 0.0, 1.0)
    modulus_cosine = jnp.clip(_safe_ratio(modulus_sum, denominator), 0.0, 1.0)
    phase_factor = jnp.clip(_safe_ratio(jnp.abs(overlap_sum), modulus_sum), 0.0, 1.0)
    return {
        f"{name}:distance": jnp.arccos(cosine),
        f"{name}:cosine": cosine,
        f"{name}:fidelity": cosine**2,
        f"{name}:modulus_cosine": modulus_cosine,
        f"{name}:phase_factor": phase_factor,
    }


def fubini_metrics_from_log_ratios(
    log_ratios: jax.Array,
    reference_index: int = -1,
    *,
    name: str = "fubini",
) -> dict[str, jax.Array]:
    """Compute Fubini--Study metrics from log wavefunction ratios.

    ``log_ratios[..., i]`` is ``log(psi_i / psi_sampling)`` evaluated on
    samples distributed as ``|psi_sampling|^2``. The sampling wavefunction
    can differ from every checkpoint, including the chosen reference.

    Args:
        log_ratios: Complex log ratios. The last axis indexes checkpoints and
            all preceding axes are treated as samples.
        reference_index: Checkpoint used as the common reference.
        name: Prefix for result keys.

    Returns:
        Distance, overlap cosine, fidelity, modulus cosine, and phase factor,
        with one value per checkpoint.
    """
    ratios = jnp.reshape(log_ratios, (-1, log_ratios.shape[-1]))
    reference = ratios[:, reference_index, None]
    valid = jnp.isfinite(ratios) & jnp.isfinite(reference)

    real_ratios = jnp.real(ratios)
    real_reference = jnp.real(reference)
    checkpoint_scale = jnp.max(jnp.where(valid, real_ratios, -jnp.inf), axis=0)
    reference_scale = jnp.max(jnp.where(valid, real_reference, -jnp.inf), axis=0)
    checkpoint_scale = jnp.where(jnp.isfinite(checkpoint_scale), checkpoint_scale, 0)
    reference_scale = jnp.where(jnp.isfinite(reference_scale), reference_scale, 0)

    overlap = jnp.where(
        valid,
        jnp.exp(jnp.conj(reference) + ratios - reference_scale - checkpoint_scale),
        0,
    )
    modulus = jnp.where(
        valid,
        jnp.exp(real_reference + real_ratios - reference_scale - checkpoint_scale),
        0,
    )
    checkpoint_norm = jnp.where(valid, jnp.exp(2 * (real_ratios - checkpoint_scale)), 0)
    reference_norm = jnp.where(
        valid, jnp.exp(2 * (real_reference - reference_scale)), 0
    )
    return _metrics_from_sums(
        jnp.sum(overlap, axis=0),
        jnp.sum(modulus, axis=0),
        jnp.sum(checkpoint_norm, axis=0),
        jnp.sum(reference_norm, axis=0),
        name=name,
    )


@configurable_dataclass
class FubiniStudyDistance(PerWalkerEstimator):
    """Estimate distances from all training checkpoints to one reference.

    The estimator loads every ``{checkpoint_prefix}_ckpt_*.npz`` file from
    ``checkpoint_path``. By default, the checkpoint with the largest step is
    the reference. Set ``reference_step`` to compare against a specific saved
    step instead. If ``checkpoint_path`` is omitted, the evaluation workflow's
    ``workflow.source_path`` directory is used.

    Samples are drawn from the network selected by ``workflow.source_path``.
    A general importance-sampling identity is used, so this sampling network
    need not be the selected reference checkpoint. All checkpoints must share
    the current wavefunction architecture and parameter-tree structure.

    The digest contains ``fubini:step``, ``fubini:distance``,
    ``fubini:cosine``, ``fubini:fidelity``, ``fubini:modulus_cosine``, and
    ``fubini:phase_factor``. The last two quantities obey
    ``cos(gamma) = cos(gamma_modulus) * F_phase`` up to Monte Carlo error.

    Args:
        checkpoint_path: Training directory containing checkpoints. Defaults
            to ``workflow.source_path`` during evaluation.
        checkpoint_prefix: Filename prefix before ``_ckpt_``.
        reference_step: Saved step used as the reference, or ``None`` for the
            checkpoint with the largest step.
        name: Prefix used for digest keys.
        vmap_chunk_size: Number of walkers evaluated per vmap chunk.
        f_log_psi: Runtime wavefunction evaluator.
    """

    checkpoint_path: str | None = None
    checkpoint_prefix: str = "train"
    reference_step: int | None = None
    name: str = "fubini"
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    checkpoint_params: Params | None = runtime_dep(default=None)
    checkpoint_steps: tuple[int, ...] = runtime_dep(default=())
    reference_index: int = runtime_dep(default=-1)

    def load_checkpoints(
        self,
        params_template: Params,
        *,
        default_path: str | UPath | None = None,
    ) -> None:
        """Load and stack checkpoint parameters using ``params_template``.

        Args:
            params_template: Parameter PyTree defining checkpoint structure.
            default_path: Path used when ``checkpoint_path`` is not configured.

        Raises:
            ValueError: If no path is available or ``reference_step`` is absent.
            FileNotFoundError: If the directory or matching checkpoints are absent.
        """
        raw_path = self.checkpoint_path or default_path
        if raw_path is None:
            raise ValueError(
                "FubiniStudyDistance requires checkpoint_path or workflow.source_path."
            )
        path = UPath(raw_path)
        if not path.is_absolute():
            path = (UPath.cwd() / path).resolve()
        directory = path.parent if path.is_file() else path
        if not directory.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")

        files_with_steps: list[tuple[int, UPath]] = []
        for checkpoint_file in directory.glob(f"{self.checkpoint_prefix}_ckpt_*.npz"):
            match = _CHECKPOINT_RE.match(checkpoint_file.name)
            if match and match.group("prefix") == self.checkpoint_prefix:
                files_with_steps.append((int(match.group("step")), checkpoint_file))
        files_with_steps.sort(key=itemgetter(0))
        if not files_with_steps:
            raise FileNotFoundError(
                f"No {self.checkpoint_prefix}_ckpt_*.npz checkpoints found in "
                f"{directory}"
            )

        params = []
        wrapper = {"params": params_template}
        for _, checkpoint_file in files_with_steps:
            _, restored = NumPyCheckpointManager.restore_from_file(
                checkpoint_file, wrapper
            )
            params.append(restored["params"])

        steps = tuple(step for step, _ in files_with_steps)
        if self.reference_step is None:
            reference_index = len(steps) - 1
        else:
            try:
                reference_index = steps.index(self.reference_step)
            except ValueError as error:
                raise ValueError(
                    f"reference_step={self.reference_step} is not present in "
                    f"{directory}; available steps: {steps}"
                ) from error

        self.checkpoint_params = jax.tree.map(lambda *xs: jnp.stack(xs), *params)
        self.checkpoint_steps = steps
        self.reference_index = reference_index

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        """Evaluate all checkpoint-to-sampling-network log ratios.

        Returns:
            Per-checkpoint log ratios and unchanged estimator state.

        Raises:
            ValueError: If checkpoint parameters have not been loaded.
        """
        del prev_walker_stats, rngs
        if self.checkpoint_params is None:
            raise ValueError(
                "Fubini checkpoints are not loaded. Run this estimator through "
                "EvaluationWorkflow or call load_checkpoints() first."
            )
        current_log_psi = self.f_log_psi(params, data)
        checkpoint_log_psi = jax.vmap(self.f_log_psi, in_axes=(0, None))(
            self.checkpoint_params, data
        )
        return {f"{self.name}:log_ratio": checkpoint_log_psi - current_log_psi}, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        """Accumulate numerically scaled overlap and norm sums.

        Returns:
            Step-level scaled sums for every checkpoint.
        """
        ratios = walker_stats[f"{self.name}:log_ratio"]
        reference = ratios[:, self.reference_index, None]
        valid = jnp.isfinite(ratios) & jnp.isfinite(reference)
        real_ratios = jnp.real(ratios)
        real_reference = jnp.real(reference)

        checkpoint_scale = _pmax(
            jnp.max(jnp.where(valid, real_ratios, -jnp.inf), axis=0)
        )
        reference_scale = _pmax(
            jnp.max(jnp.where(valid, real_reference, -jnp.inf), axis=0)
        )
        checkpoint_scale = jnp.where(
            jnp.isfinite(checkpoint_scale), checkpoint_scale, 0
        )
        reference_scale = jnp.where(jnp.isfinite(reference_scale), reference_scale, 0)

        overlap = jnp.where(
            valid,
            jnp.exp(jnp.conj(reference) + ratios - reference_scale - checkpoint_scale),
            0,
        )
        modulus = jnp.where(
            valid,
            jnp.exp(real_reference + real_ratios - reference_scale - checkpoint_scale),
            0,
        )
        checkpoint_norm = jnp.where(
            valid, jnp.exp(2 * (real_ratios - checkpoint_scale)), 0
        )
        reference_norm = jnp.where(
            valid, jnp.exp(2 * (real_reference - reference_scale)), 0
        )

        def reduce_sum(value: jax.Array) -> jax.Array:
            return parallel_jax.pmean(jnp.sum(value, axis=0))

        return {
            f"{self.name}:checkpoint_scale": checkpoint_scale,
            f"{self.name}:reference_scale": reference_scale,
            f"{self.name}:overlap_sum": reduce_sum(overlap),
            f"{self.name}:modulus_sum": reduce_sum(modulus),
            f"{self.name}:checkpoint_norm_sum": reduce_sum(checkpoint_norm),
            f"{self.name}:reference_norm_sum": reduce_sum(reference_norm),
            f"{self.name}:count": reduce_sum(valid),
        }

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        """Combine scaled step sums into checkpoint-indexed distances.

        Returns:
            Final Fubini metrics and their checkpoint steps.
        """
        del state
        checkpoint_scale = batched_stats[f"{self.name}:checkpoint_scale"]
        reference_scale = batched_stats[f"{self.name}:reference_scale"]
        count = batched_stats[f"{self.name}:count"]
        valid = count > 0
        overlap_log_scale = reference_scale + checkpoint_scale
        checkpoint_norm_log_scale = 2 * checkpoint_scale
        reference_norm_log_scale = 2 * reference_scale
        all_log_scales = jnp.stack(
            [overlap_log_scale, checkpoint_norm_log_scale, reference_norm_log_scale]
        )
        common_scale = jnp.max(
            jnp.where(valid[None], all_log_scales, -jnp.inf), axis=(0, 1)
        )
        common_scale = jnp.where(jnp.isfinite(common_scale), common_scale, 0)

        def combine(values: jax.Array, log_scales: jax.Array) -> jax.Array:
            return jnp.sum(
                jnp.where(
                    valid,
                    values * jnp.exp(log_scales - common_scale),
                    0,
                ),
                axis=0,
            )

        overlap_sum = combine(
            batched_stats[f"{self.name}:overlap_sum"],
            reference_scale + checkpoint_scale,
        )
        modulus_sum = combine(
            batched_stats[f"{self.name}:modulus_sum"],
            reference_scale + checkpoint_scale,
        )
        checkpoint_norm_sum = combine(
            batched_stats[f"{self.name}:checkpoint_norm_sum"],
            2 * checkpoint_scale,
        )
        reference_norm_sum = combine(
            batched_stats[f"{self.name}:reference_norm_sum"],
            2 * reference_scale,
        )
        metrics = _metrics_from_sums(
            overlap_sum,
            modulus_sum,
            checkpoint_norm_sum,
            reference_norm_sum,
            name=self.name,
        )
        metrics[f"{self.name}:step"] = jnp.asarray(self.checkpoint_steps)
        metrics[f"{self.name}:reference_step"] = jnp.asarray(
            self.checkpoint_steps[self.reference_index]
        )
        return metrics

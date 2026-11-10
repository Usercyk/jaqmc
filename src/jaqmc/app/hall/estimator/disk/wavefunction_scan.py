# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Conditional wavefunction scans on the disk.

This estimator implements the real-space visualization used in Fig. 4 of
Teng, Dai, and Fu, Phys. Rev. B 111, 205117 (2025).  All electrons except one
are held fixed while the remaining electron is moved through a Cartesian
grid.  The complex logarithm of the wavefunction is evaluated directly, so
both its phase and magnitude can be recovered without numerical underflow.
"""

from collections.abc import Mapping
from operator import itemgetter
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


@configurable_dataclass
class DiskWavefunctionScan(Estimator):
    r"""Scan a disk wavefunction with one mobile electron.

    The fixed background can be supplied explicitly through
    ``fixed_electrons``.  Otherwise, the estimator follows the prescription
    in the paper and selects the configuration with the largest
    :math:`\log|\Psi|` among the first ``selection_samples`` MCMC samples.
    Setting ``selection_samples`` to zero selects from the first evaluation
    batch.

    The estimator stores the unshifted ``log_abs`` on the grid.  Magnitude
    normalization and other presentation-oriented transformations belong to
    the consuming analysis code and can be reconstructed from ``log_abs``.

    A radial probe is also evaluated from ``probe_electron`` in
    ``radial_direction``.  It returns

    .. math::

        \frac{\partial\log|\Psi|}{\partial\log r}
        = r\frac{\partial\log|\Psi|}{\partial r},

    corresponding to Fig. 4(d).

    Args:
        fixed_electrons: Coordinates of the :math:`N-1` fixed electrons, in
            electron-index order with ``mobile_electron`` omitted.  If None,
            a typical configuration is selected from MCMC samples.
        mobile_electron: Index of the electron moved over the grid.
        x_min: Lower x boundary of the Cartesian grid.
        x_max: Upper x boundary of the Cartesian grid.
        y_min: Lower y boundary of the Cartesian grid.
        y_max: Upper y boundary of the Cartesian grid.
        n_x: Number of grid points along x.
        n_y: Number of grid points along y.
        chunk_size: Number of grid positions evaluated concurrently.  Reduce
            this value if accelerator memory is limited.
        selection_samples: Number of MCMC configurations used to find the
            most probable background.  Zero means one evaluation batch.
        probe_electron: Fixed electron used as the center of the radial probe.
        radial_direction: Unit direction is obtained by normalizing this
            Cartesian vector.
        radial_min: Smallest nonzero two-electron separation.
        radial_max: Largest two-electron separation.
        radial_points: Number of linearly spaced radial probe points.  Set to
            zero to disable the radial probe.
        f_log_psi: Complex log-wavefunction evaluator (runtime dependency).
        data_field: Name of the electron-coordinate field in ``Data``.
    """

    fixed_electrons: list[list[float]] | None = None
    mobile_electron: int = 0
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    n_x: int = 201
    n_y: int = 201
    chunk_size: int = 256
    selection_samples: int = 0
    probe_electron: int = 1
    radial_direction: tuple[float, float] = (1.0, 0.0)
    radial_min: float = 1.0e-3
    radial_max: float = 3.0
    radial_points: int = 200
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def _validate(self, electrons: jax.Array) -> None:
        if electrons.ndim != 2 or electrons.shape[-1] != 2:
            raise ValueError(
                "DiskWavefunctionScan expects one-walker electron coordinates "
                f"with shape (N, 2), got {electrons.shape}."
            )
        n_electrons = electrons.shape[0]
        if not 0 <= self.mobile_electron < n_electrons:
            raise ValueError(
                f"mobile_electron={self.mobile_electron} is outside [0, {n_electrons})."
            )
        if self.fixed_electrons is not None:
            fixed_shape = jnp.asarray(self.fixed_electrons).shape
            if fixed_shape != (n_electrons - 1, 2):
                raise ValueError(
                    "fixed_electrons must have shape (N - 1, 2); "
                    f"expected {(n_electrons - 1, 2)}, got {fixed_shape}."
                )
        if self.n_x < 2 or self.n_y < 2:
            raise ValueError("n_x and n_y must both be at least 2.")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Grid maxima must be larger than grid minima.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.selection_samples < 0:
            raise ValueError("selection_samples cannot be negative.")
        self._validate_radial(n_electrons)

    def _validate_radial(self, n_electrons: int) -> None:
        if self.radial_points < 0:
            raise ValueError("radial_points cannot be negative.")
        if self.radial_points:
            if self.probe_electron == self.mobile_electron or not (
                0 <= self.probe_electron < n_electrons
            ):
                raise ValueError("probe_electron must be a valid fixed-electron index.")
            if self.radial_min <= 0 or self.radial_max <= self.radial_min:
                raise ValueError(
                    "Require 0 < radial_min < radial_max for the radial probe."
                )
            if sum(component**2 for component in self.radial_direction) == 0:
                raise ValueError("radial_direction cannot be the zero vector.")

    def _explicit_configuration(self, electrons: jax.Array) -> jax.Array:
        """Return a full configuration containing the configured fixed electrons."""
        assert self.fixed_electrons is not None
        fixed = jnp.asarray(self.fixed_electrons, dtype=electrons.dtype)
        fixed_indices = tuple(
            i for i in range(electrons.shape[0]) if i != self.mobile_electron
        )
        configuration = jnp.zeros_like(electrons)
        return configuration.at[jnp.asarray(fixed_indices)].set(fixed)

    def init(self, data: Data, rngs: PRNGKey) -> dict[str, jax.Array]:
        del rngs
        electrons = data[self.data_field]
        self._validate(electrons)
        n_devices = 1
        float_dtype = electrons.dtype
        configuration = (
            self._explicit_configuration(electrons)
            if self.fixed_electrons is not None
            else jnp.zeros_like(electrons)
        )
        configuration = jnp.broadcast_to(
            configuration, (n_devices, *configuration.shape)
        )
        return {
            "best_log_abs": jnp.full((n_devices,), -jnp.inf, dtype=float_dtype),
            "configuration": configuration,
            "samples_seen": jnp.zeros((n_devices,), dtype=jnp.int32),
            "scan_done": jnp.zeros((n_devices,), dtype=jnp.bool_),
            "log_abs": jnp.zeros((n_devices, self.n_y, self.n_x), dtype=float_dtype),
            "phase": jnp.zeros((n_devices, self.n_y, self.n_x), dtype=float_dtype),
            "radial_log_abs": jnp.zeros(
                (n_devices, self.radial_points), dtype=float_dtype
            ),
            "radial_log_derivative": jnp.zeros(
                (n_devices, self.radial_points), dtype=float_dtype
            ),
        }

    def _evaluate_points(
        self,
        params: Params,
        data: Data,
        points: jax.Array,
    ) -> jax.Array:
        """Return complex log-psi at mobile-electron target points."""
        n_points = points.shape[0]
        n_chunks = (n_points + self.chunk_size - 1) // self.chunk_size
        n_padded = n_chunks * self.chunk_size
        # Repeating a valid endpoint avoids evaluating artificial padded
        # coordinates which may coincide with a fixed electron and yield a
        # spurious log(0) under ``jax.debug_infs``.
        padded = jnp.pad(
            points,
            ((0, n_padded - n_points), (0, 0)),
            mode="edge",
        )
        chunks = padded.reshape(n_chunks, self.chunk_size, 2)
        electrons = data[self.data_field]

        def evaluate_chunk(chunk: jax.Array) -> jax.Array:
            def evaluate(point: jax.Array) -> jax.Array:
                displaced = electrons.at[self.mobile_electron].set(point)
                return self.f_log_psi(params, data.merge({self.data_field: displaced}))

            return jax.vmap(evaluate)(chunk)

        values = jax.lax.map(evaluate_chunk, chunks).reshape(n_padded)
        return values[:n_points]

    def _evaluate_radial(
        self,
        params: Params,
        data: Data,
    ) -> tuple[jax.Array, jax.Array]:
        if self.radial_points == 0:
            empty = jnp.empty((0,), dtype=data[self.data_field].dtype)
            return empty, empty

        electrons = data[self.data_field]
        center = electrons[self.probe_electron]
        direction = jnp.asarray(self.radial_direction, dtype=electrons.dtype)
        direction = direction / jnp.linalg.norm(direction)
        radii = jnp.linspace(
            self.radial_min,
            self.radial_max,
            self.radial_points,
            dtype=electrons.dtype,
        )

        def radial_log_abs(radius: jax.Array) -> jax.Array:
            point = center + radius * direction
            displaced = electrons.at[self.mobile_electron].set(point)
            logpsi = self.f_log_psi(params, data.merge({self.data_field: displaced}))
            return jnp.real(logpsi)

        log_abs, derivative = jax.lax.map(jax.value_and_grad(radial_log_abs), radii)
        return log_abs, radii * derivative

    def _scan_configuration(
        self,
        params: Params,
        data: Data,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        x = jnp.linspace(self.x_min, self.x_max, self.n_x)
        y = jnp.linspace(self.y_min, self.y_max, self.n_y)
        grid_x, grid_y = jnp.meshgrid(x, y, indexing="xy")
        points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        logpsi = self._evaluate_points(params, data, points).reshape(self.n_y, self.n_x)
        radial_log_abs, radial_log_derivative = self._evaluate_radial(params, data)
        # Every device scans the same selected background. Return replicated arrays
        # so conditional branches match the replicated estimator state.
        return jax.tree.map(
            parallel_jax.pmean,
            (
                jnp.real(logpsi),
                jnp.mod(jnp.imag(logpsi), 2 * jnp.pi),
                radial_log_abs,
                radial_log_derivative,
            ),
        )

    @staticmethod
    def _replace_local_state(
        state: dict[str, jax.Array],
        *,
        best_log_abs: jax.Array,
        configuration: jax.Array,
        samples_seen: jax.Array,
        scan_done: jax.Array,
        scan: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> dict[str, jax.Array]:
        log_abs, phase, radial_log_abs, radial_log_derivative = scan
        return {
            "best_log_abs": state["best_log_abs"].at[0].set(best_log_abs),
            "configuration": state["configuration"].at[0].set(configuration),
            "samples_seen": state["samples_seen"].at[0].set(samples_seen),
            "scan_done": state["scan_done"].at[0].set(scan_done),
            "log_abs": state["log_abs"].at[0].set(log_abs),
            "phase": state["phase"].at[0].set(phase),
            "radial_log_abs": state["radial_log_abs"].at[0].set(radial_log_abs),
            "radial_log_derivative": state["radial_log_derivative"]
            .at[0]
            .set(radial_log_derivative),
        }

    def _evaluate_explicit(
        self,
        params: Params,
        batched_data: BatchedData,
        state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        fully_batched = batched_data.fully_batched_data()
        data = jax.tree.map(itemgetter(0), fully_batched)
        configuration = state["configuration"][0]
        data = data.merge({self.data_field: configuration})
        scan = self._scan_configuration(params, data)
        return self._replace_local_state(
            state,
            best_log_abs=jnp.nanmax(scan[0]),
            configuration=configuration,
            samples_seen=jnp.array(0, dtype=jnp.int32),
            scan_done=jnp.array(True),
            scan=scan,
        )

    def _evaluate_selected(
        self,
        params: Params,
        batched_data: BatchedData,
        state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        fully_batched = batched_data.fully_batched_data()
        batch_logpsi = jax.vmap(self.f_log_psi, in_axes=(None, batched_data.vmap_axis))(
            params, batched_data.data
        )

        local_batch_size = batched_data.batch_size
        if self.selection_samples:
            remaining = jnp.maximum(
                self.selection_samples - state["samples_seen"][0], 0
            )
            local_quota = (remaining + jax.device_count() - 1) // jax.device_count()
            eligible = jnp.arange(local_batch_size) < local_quota
            scores = jnp.where(eligible, jnp.real(batch_logpsi), -jnp.inf)
            samples_seen = jnp.minimum(
                self.selection_samples,
                state["samples_seen"][0] + local_batch_size * jax.device_count(),
            )
            selection_complete = samples_seen >= self.selection_samples
        else:
            scores = jnp.real(batch_logpsi)
            samples_seen = jnp.asarray(
                local_batch_size * jax.device_count(), dtype=jnp.int32
            )
            selection_complete = jnp.array(True)

        candidate_index = jnp.argmax(scores)
        candidate_score = scores[candidate_index]
        candidate_data = jax.tree.map(itemgetter(candidate_index), fully_batched)
        # Gather only each device's best candidate, then select a common winner.
        candidate_scores = parallel_jax.all_gather(candidate_score[None])
        winner = jnp.argmax(candidate_scores)
        candidate_score = parallel_jax.pmean(candidate_scores[winner])
        candidate_data = jax.tree.map(
            lambda value: parallel_jax.all_gather(value[None])[winner], candidate_data
        )
        candidate_configuration = parallel_jax.pmean(candidate_data[self.data_field])
        is_better = candidate_score > state["best_log_abs"][0]
        best_log_abs = jnp.where(is_better, candidate_score, state["best_log_abs"][0])
        configuration = jnp.where(
            is_better, candidate_configuration, state["configuration"][0]
        )
        selected_data = candidate_data.merge({self.data_field: configuration})

        old_scan = (
            state["log_abs"][0],
            state["phase"][0],
            state["radial_log_abs"][0],
            state["radial_log_derivative"][0],
        )

        scan = jax.lax.cond(
            selection_complete,
            lambda: self._scan_configuration(params, selected_data),
            lambda: old_scan,
        )
        return self._replace_local_state(
            state,
            best_log_abs=best_log_abs,
            configuration=configuration,
            samples_seen=samples_seen,
            scan_done=selection_complete,
            scan=scan,
        )

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData,
        prev_walker_stats: Mapping[str, Any],
        state: dict[str, jax.Array],
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], dict[str, jax.Array]]:
        del prev_walker_stats, rngs
        evaluate = (
            self._evaluate_explicit
            if self.fixed_electrons is not None
            else self._evaluate_selected
        )
        new_state = jax.lax.cond(
            state["scan_done"][0],
            lambda: state,
            lambda: evaluate(params, batched_data, state),
        )
        return {}, new_state

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        del local_stats
        return {}

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        del batched_stats, state
        return {}

    def finalize_state(
        self, state: dict[str, jax.Array], *, n_steps: int
    ) -> dict[str, Any]:
        del n_steps
        state = jax.tree.map(lambda value: jnp.asarray(np.asarray(value)), state)
        completed = state["scan_done"]
        if not bool(jnp.any(completed)):
            seen = int(jnp.max(state["samples_seen"]))
            raise ValueError(
                "DiskWavefunctionScan did not collect enough configurations: "
                f"saw {seen}, requested {self.selection_samples}. Increase "
                "evaluation.run.iterations or reduce selection_samples."
            )

        scores = jnp.where(completed, state["best_log_abs"], -jnp.inf)
        winner = jnp.argmax(scores)
        log_abs = state["log_abs"][winner]
        configuration = state["configuration"][winner]
        fixed_indices = tuple(
            i for i in range(configuration.shape[0]) if i != self.mobile_electron
        )
        result: dict[str, Any] = {
            "wavefunction_scan:x": jnp.linspace(self.x_min, self.x_max, self.n_x),
            "wavefunction_scan:y": jnp.linspace(self.y_min, self.y_max, self.n_y),
            "wavefunction_scan:log_abs": log_abs,
            "wavefunction_scan:phase": state["phase"][winner],
            "wavefunction_scan:fixed_electrons": configuration[
                jnp.asarray(fixed_indices)
            ],
            "wavefunction_scan:mobile_electron": jnp.asarray(self.mobile_electron),
        }
        if self.fixed_electrons is None:
            result["wavefunction_scan:selected_log_abs"] = scores[winner]
            result["wavefunction_scan:selection_samples"] = jnp.max(
                state["samples_seen"]
            )
        if self.radial_points:
            result.update(
                {
                    "wavefunction_scan:radial_r": jnp.linspace(
                        self.radial_min,
                        self.radial_max,
                        self.radial_points,
                    ),
                    "wavefunction_scan:radial_log_abs": state["radial_log_abs"][winner],
                    "wavefunction_scan:radial_log_derivative": state[
                        "radial_log_derivative"
                    ][winner],
                    "wavefunction_scan:probe_electron": jnp.asarray(
                        self.probe_electron
                    ),
                }
            )
        return result

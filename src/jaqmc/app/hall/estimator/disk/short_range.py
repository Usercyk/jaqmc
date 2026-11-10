# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Batch sums for electron separation and the energy carried by each bin."""

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from serde.core import field as serde_field

from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass

COMPONENTS = (
    "energy",
    "kinetic",
    "electron_electron",
    "background",
    "mean_radius_squared",
    "angular_momentum",
    "energy_imaginary",
    "angular_momentum_squared",
)


def _serialize_array(values: Any) -> list[float]:
    """Return histogram settings as a YAML-compatible list."""
    return np.asarray(values).tolist()


def _edges(values, *, anchors=()) -> np.ndarray:
    """Merge roundoff duplicates, keeping an explicit overflow bin.

    Returns:
        Sorted edges from zero to positive infinity.

    Raises:
        ValueError: If the edges do not define nonnegative distance bins.
    """
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if np.isnan(result).any() or np.isneginf(result).any():
        raise ValueError("Histogram edges must be nonnegative and not NaN")
    if np.any(result < 0):
        raise ValueError("Histogram edges must be nonnegative")
    for anchor in anchors:
        result[np.isclose(result, anchor, rtol=1e-12, atol=1e-15)] = anchor
    result = np.sort(result)
    kept: list[float] = []
    for value in result:
        if not kept or not np.isclose(value, kept[-1], rtol=1e-12, atol=1e-15):
            kept.append(float(value))
    result = np.asarray(kept, dtype=np.float64)
    if len(result) < 2 or result[0] != 0 or not np.isposinf(result[-1]):
        raise ValueError("Histogram edges must start at zero and end at +inf")
    return result


@configurable_dataclass
class ShortRangeEstimator(Estimator):
    """Collect batch sums without discarding invalid values.

    Distance bins use the minimum pair separation, except ``pair_potential``,
    which bins each pair's own Coulomb contribution. Radial row zero includes
    every walker; following rows select minimum separations below each cutoff.
    ``reduce`` sums device-local contributions before time-block averaging.

    Args:
        kappa: Coulomb interaction strength, matching the system configuration.
        distance_edges: Separation-bin boundaries, including zero and infinity.
        radial_edges: Electron-radius bin boundaries, including an overflow bin.
        density_cutoffs: Separation thresholds for conditional radial counts.
    """

    kappa: float = 12.0
    distance_edges: Any = serde_field(default=None, serializer=_serialize_array)
    radial_edges: Any = serde_field(default=None, serializer=_serialize_array)
    density_cutoffs: Any = serde_field(
        default=(0.5, 1.0, 1.5, 2.0, 3.0), serializer=_serialize_array
    )

    components = COMPONENTS

    def __post_init__(self):
        if not np.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        if self.distance_edges is None:
            self.distance_edges = np.concatenate(
                (
                    [0.0],
                    np.geomspace(1e-3, 1.0, 61),
                    np.linspace(1.0, 3.0, 81),
                    [0.1, 0.2, 0.3, 0.5, 0.75, np.inf],
                )
            )
        self.distance_edges = _edges(
            self.distance_edges,
            anchors=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
        )
        if self.radial_edges is None:
            self.radial_edges = np.r_[np.linspace(0.0, 8.0, 401), np.inf]
        self.radial_edges = _edges(self.radial_edges)
        self.density_cutoffs = np.asarray(self.density_cutoffs, dtype=np.float64)
        if (
            self.density_cutoffs.ndim != 1
            or not np.isfinite(self.density_cutoffs).all()
            or np.any(self.density_cutoffs <= 0)
            or np.any(np.diff(self.density_cutoffs) <= 0)
        ):
            raise ValueError("density_cutoffs must be positive and increasing")

    @property
    def n_bins(self) -> int:
        return len(self.distance_edges) - 1

    def init(self, data, rngs):
        del data, rngs
        return None

    def evaluate_batch_walkers(
        self,
        params,
        batched_data,
        prev_walker_stats: Mapping[str, Any],
        state,
        rngs,
    ):
        del params, state, rngs
        electrons = jnp.asarray(batched_data.data.electrons, dtype=jnp.float64)
        if electrons.ndim != 3 or electrons.shape[-1] != 2 or electrons.shape[1] < 2:
            raise ValueError("Expected electron coordinates with shape (batch, N, 2)")
        batch_size, nelec, _ = electrons.shape
        i, j = jnp.triu_indices(nelec, k=1)
        separations = jnp.linalg.norm(electrons[:, i] - electrons[:, j], axis=-1)
        minimum = jnp.min(separations, axis=-1)
        pair_values = self.kappa / separations
        vee = jnp.sum(pair_values, axis=-1)
        energy = jnp.asarray(prev_walker_stats["total_energy"])
        kinetic = jnp.asarray(prev_walker_stats["energy:kinetic"])
        potential = jnp.asarray(prev_walker_stats["energy:potential"])
        lz = jnp.asarray(prev_walker_stats["observable:angular_momentum_z"])
        lz2 = jnp.asarray(prev_walker_stats["observable:angular_momentum_z_square"])
        radius_squared = jnp.sum(electrons**2, axis=-1)
        values = jnp.stack(
            (
                energy.real,
                kinetic.real,
                vee,
                potential.real - vee,
                jnp.mean(radius_squared, axis=-1),
                lz.real,
                energy.imag,
                lz2.real,
            ),
            axis=-1,
        ).astype(jnp.float64)
        invalid = (
            ~jnp.all(jnp.isfinite(electrons), axis=(1, 2))
            | ~jnp.all(jnp.isfinite(separations) & (separations > 0), axis=-1)
            | ~jnp.all(jnp.isfinite(values), axis=-1)
            | ~jnp.isfinite(energy)
            | ~jnp.isfinite(kinetic)
            | ~jnp.isfinite(potential)
            | ~jnp.isfinite(lz)
            | ~jnp.isfinite(lz2)
        )

        edges = jnp.asarray(self.distance_edges, dtype=jnp.float64)
        # searchsorted(right) gives [left, right); clip invalid +inf/NaN
        # locations into overflow while nonfinite_count exposes the problem.
        bins = jnp.clip(
            jnp.searchsorted(edges, minimum, side="right") - 1, 0, self.n_bins - 1
        )
        pair_bins = jnp.clip(
            jnp.searchsorted(edges, separations, side="right") - 1, 0, self.n_bins - 1
        )
        count = (
            jnp.zeros(self.n_bins, dtype=jnp.int64)
            .at[bins]
            .add(jnp.ones(batch_size, dtype=jnp.int64))
        )
        sums = jnp.zeros((self.n_bins, len(COMPONENTS)), dtype=jnp.float64)
        sums = sums.at[bins].add(values)
        energy_square = (
            jnp.zeros(self.n_bins, dtype=jnp.float64).at[bins].add(energy.real**2)
        )
        pair_potential = (
            jnp.zeros(self.n_bins, dtype=jnp.float64)
            .at[pair_bins.reshape(-1)]
            .add(pair_values.reshape(-1))
        )

        radial_edges = jnp.asarray(self.radial_edges, dtype=jnp.float64)
        radial_bins = jnp.clip(
            jnp.searchsorted(radial_edges, jnp.sqrt(radius_squared), side="right") - 1,
            0,
            len(self.radial_edges) - 2,
        )
        selected = jnp.concatenate(
            (
                jnp.ones((batch_size, 1), dtype=bool),
                minimum[:, None] < jnp.asarray(self.density_cutoffs)[None, :],
            ),
            axis=-1,
        )
        radial_rows = []
        for row in range(1 + len(self.density_cutoffs)):
            weights = jnp.broadcast_to(selected[:, row, None], radial_bins.shape)
            radial_rows.append(
                jnp.zeros(len(self.radial_edges) - 1, dtype=jnp.int64)
                .at[radial_bins.reshape(-1)]
                .add(weights.reshape(-1).astype(jnp.int64))
            )

        return {
            "short_range:sample_count": jnp.asarray(batch_size, dtype=jnp.int64),
            "short_range:count": count,
            "short_range:sum": sums,
            "short_range:energy_square": energy_square,
            "short_range:pair_potential": pair_potential,
            "short_range:radial_count": jnp.stack(radial_rows),
            "short_range:nonfinite_count": jnp.sum(invalid, dtype=jnp.int64),
            "short_range:global_sum": jnp.sum(values, axis=0),
            "short_range:global_energy_square": jnp.sum(energy.real**2),
        }, None

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        # A local sum still varies along the shard_map batch axis, even on
        # one device. Explicit psum produces the replicated output required
        # by the evaluation stage and preserves integer count dtypes.
        try:
            return jax.lax.psum(dict(walker_stats), parallel_jax.BATCH_AXIS_NAME)
        except NameError:
            # Allow direct evaluation outside a named shard_map context.
            return dict(walker_stats)

    def finalize_stats(self, batched_stats: Mapping[str, Any], state) -> dict[str, Any]:
        del state
        return jax.tree.map(lambda value: jnp.mean(value, axis=0), batched_stats)

# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the torus pair-correlation estimator."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.torus import TorusPairCorrelation
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax


class _PositionData(Data):
    positions: jnp.ndarray


def _make_batched(electrons: jnp.ndarray) -> BatchedData:
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )


class TestTorusPairCorrelation:
    def test_init_uses_replicated_histograms(self):
        estimator = TorusPairCorrelation(bins_u=12, bins_v=16)
        data = HallData(electrons=jnp.zeros((2, 2)))

        state = estimator.init(data, jax.random.PRNGKey(0))

        expected_shape = (12, 16)
        assert state["histogram"].shape == expected_shape
        assert state["compensation"].shape == expected_shape

    def test_directed_displacements_are_wrapped_periodically(self):
        estimator = TorusPairCorrelation(bins_u=8, bins_v=8)
        data = _make_batched(
            jnp.array(
                [
                    [[0.1, 0.3], [0.9, 0.3]],
                    [[2.1, -0.7], [-1.1, 1.3]],
                ]
            )
        )

        displacements = estimator._pair_displacements(data.data.electrons)

        np.testing.assert_allclose(
            displacements,
            [[0.2, 0.0], [-0.2, 0.0], [0.2, 0.0], [-0.2, 0.0]],
            atol=1e-6,
        )

    def test_periodic_neighbors_are_binned_near_the_center(self):
        estimator = TorusPairCorrelation(bins_u=8, bins_v=8)
        electrons = jnp.array([[[0.0, 0.0], [0.0, 0.99]]])

        displacements = estimator._pair_displacements(electrons)

        np.testing.assert_allclose(
            displacements,
            [[0.0, 0.01], [0.0, -0.01]],
            atol=1e-6,
        )

    def test_exact_bin_area_normalization_and_pair_symmetry(self):
        estimator = TorusPairCorrelation(bins_u=4, bins_v=4)
        data = _make_batched(jnp.array([[[0.0, 0.0], [0.1, 0.2]]]))
        state = estimator.init(data.unbatched_example(), jax.random.PRNGKey(0))

        _, state = estimator.evaluate_batch_walkers(
            {}, data, {}, state, jax.random.PRNGKey(1)
        )
        result = estimator.finalize_state(state, n_steps=1)

        # Each directed pair contributes bins_u*bins_v/N^2 = 4.
        expected = np.zeros((4, 4))
        expected[1, 1] = 4.0  # -(0.1, 0.2)
        expected[2, 2] = 4.0  # +(0.1, 0.2)
        np.testing.assert_allclose(result["pair_correlation"], expected)

    def test_finalize_state_averages_steps_and_returns_axes(self):
        estimator = TorusPairCorrelation(bins_u=2, bins_v=3)
        data = HallData(electrons=jnp.zeros((2, 2)))
        state = estimator.init(data, jax.random.PRNGKey(0))
        accumulated = jnp.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        state["histogram"] = accumulated

        result = estimator.finalize_state(state, n_steps=2)

        np.testing.assert_allclose(result["pair_correlation"], accumulated / 2)
        np.testing.assert_allclose(result["pair_correlation:u"], [-0.25, 0.25])
        np.testing.assert_allclose(result["pair_correlation:v"], [-1 / 3, 0.0, 1 / 3])
        assert result["pair_correlation:n_steps"] == 2

    def test_custom_data_field(self):
        estimator = TorusPairCorrelation(bins_u=4, bins_v=4, data_field="positions")
        batched = BatchedData(
            data=_PositionData(positions=jnp.array([[[0.0, 0.0], [0.1, 0.3]]])),
            fields_with_batch=["positions"],
        )
        state = estimator.init(batched.unbatched_example(), jax.random.PRNGKey(0))

        _, state = estimator.evaluate_batch_walkers(
            {}, batched, {}, state, jax.random.PRNGKey(1)
        )

        assert jnp.sum(state["histogram"]) > 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bins_u": 0}, "bins_u"),
            ({"bins_v": 0}, "bins_v"),
            ({"bins_u": True}, "bins_u"),
        ],
    )
    def test_rejects_invalid_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TorusPairCorrelation(**kwargs)


def test_sharded_pair_correlation_matches_full_batch() -> None:
    electrons = np.zeros((2 * jax.device_count(), 2, 2), dtype=np.float32)
    electrons[:, 1, 0] = np.linspace(0.1, 0.4, len(electrons))
    electrons[:, 1, 1] = np.linspace(0.4, 0.1, len(electrons))
    batch = _make_batched(jnp.asarray(electrons))
    estimator = TorusPairCorrelation(bins_u=4, bins_v=4)
    pipeline = EstimatorPipeline({"pair": estimator})
    shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    state = jax.device_put(pipeline.init(batch, jax.random.key(0)), shared)
    batch = jax.device_put(batch, parallel_jax.make_sharding(batch.partition_spec))
    evaluate = parallel_jax.jit_sharded(
        pipeline.evaluate,
        in_specs=(
            parallel_jax.SHARE_PARTITION,
            batch.partition_spec,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=(parallel_jax.SHARE_PARTITION, parallel_jax.SHARE_PARTITION),
        check_vma=True,
        donate_argnums=2,
    )
    for step in range(2):
        _, state = evaluate(
            {}, batch, state, parallel_jax.shard_rngs(jax.random.key(step + 1))
        )
    actual = pipeline.digest({}, state, n_steps=2)["pair_correlation"]
    delta = electrons[:, 0] - electrons[:, 1]
    directed = (np.concatenate([delta, -delta]) + 0.5) % 1 - 0.5
    expected = (
        np.histogramdd(directed, bins=(4, 4), range=((-0.5, 0.5), (-0.5, 0.5)))[0]
        * 16
        / (len(electrons) * 4)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(np.mean(actual), 0.5, atol=1e-6)

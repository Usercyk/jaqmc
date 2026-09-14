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


class _PositionData(Data):
    positions: jnp.ndarray


def _make_batched(electrons: jnp.ndarray) -> BatchedData:
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )


class TestTorusPairCorrelation:
    def test_init_uses_device_local_histograms(self):
        estimator = TorusPairCorrelation(bins_u=12, bins_v=16)
        data = HallData(electrons=jnp.zeros((2, 2)))

        state = estimator.init(data, jax.random.PRNGKey(0))

        expected_shape = (jax.device_count(), 12, 16)
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
        state["histogram"] = state["histogram"].at[0].set(accumulated)

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

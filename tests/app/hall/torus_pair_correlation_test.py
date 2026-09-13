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
from jaqmc.utils.torus import get_torus_lattice


class _PositionData(Data):
    positions: jnp.ndarray


def _make_batched(electrons: jnp.ndarray) -> BatchedData:
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )


class TestTorusPairCorrelation:
    def test_init_uses_device_local_histograms(self):
        estimator = TorusPairCorrelation(bins=12, flux=2, tau=1j)
        data = HallData(electrons=jnp.zeros((2, 2)))

        state = estimator.init(data, jax.random.PRNGKey(0))

        assert state["histogram"].shape == (jax.device_count(), 12)
        assert state["compensation"].shape == (jax.device_count(), 12)
        l1, _, _ = get_torus_lattice(2, 1j)
        np.testing.assert_allclose(estimator._max_distance, l1 / 2)

    def test_square_torus_uses_periodic_minimum_image(self):
        estimator = TorusPairCorrelation(bins=8, flux=2, tau=1j)
        data = _make_batched(
            jnp.array(
                [
                    [[0.1, 0.3], [0.9, 0.3]],
                    [[2.1, -0.7], [-1.1, 1.3]],
                ]
            )
        )
        estimator.init(data.unbatched_example(), jax.random.PRNGKey(0))

        distances = estimator._pair_distances(data.data.electrons)

        l1, _, _ = get_torus_lattice(2, 1j)
        np.testing.assert_allclose(
            distances, jnp.full(2, 0.2 * l1), rtol=2e-6, atol=1e-6
        )

    def test_strongly_sheared_basis_finds_true_minimum_image(self):
        tau = 10 + 1j
        estimator = TorusPairCorrelation(bins=8, flux=2, tau=tau)
        data = _make_batched(jnp.array([[[0.0, 0.0], [0.0, 0.1]]]))
        estimator.init(data.unbatched_example(), jax.random.PRNGKey(0))

        distances = estimator._pair_distances(data.data.electrons)

        l1, _, _ = get_torus_lattice(2, tau)
        # 0.1 * tau = 1 + 0.1j; subtracting one L1 image leaves 0.1j.
        np.testing.assert_allclose(distances, jnp.asarray([0.1 * l1]), rtol=1e-6)
        np.testing.assert_allclose(estimator._max_distance, l1 / 2, rtol=1e-6)

    def test_exact_annulus_normalization(self):
        estimator = TorusPairCorrelation(bins=4, flux=2, tau=1j)
        data = _make_batched(jnp.array([[[0.0, 0.0], [0.1, 0.0]]]))
        state = estimator.init(data.unbatched_example(), jax.random.PRNGKey(0))

        _, state = estimator.evaluate_batch_walkers(
            {}, data, {}, state, jax.random.PRNGKey(1)
        )
        result = estimator.finalize_state(state, n_steps=1)

        # One unordered pair contributes 2*A/(N^2*shell_area) to its bin.
        expected = 2 * estimator._area / (2**2 * estimator._shell_areas[0])
        np.testing.assert_allclose(result["pair_correlation"][0], expected, rtol=1e-6)
        np.testing.assert_allclose(result["pair_correlation"][1:], 0.0)

    def test_finalize_state_averages_steps_and_returns_radii(self):
        estimator = TorusPairCorrelation(bins=2, flux=2, tau=1j)
        data = HallData(electrons=jnp.zeros((2, 2)))
        state = estimator.init(data, jax.random.PRNGKey(0))
        state["histogram"] = jnp.array([[2.0, 4.0]])

        result = estimator.finalize_state(state, n_steps=2)

        np.testing.assert_allclose(result["pair_correlation"], [1.0, 2.0])
        np.testing.assert_allclose(
            result["pair_correlation:r"],
            (estimator._bin_edges[:-1] + estimator._bin_edges[1:]) / 2,
        )
        assert result["pair_correlation:n_steps"] == 2

    def test_custom_data_field(self):
        estimator = TorusPairCorrelation(bins=4, flux=2, tau=1j, data_field="positions")
        batched = BatchedData(
            data=_PositionData(positions=jnp.array([[[0.0, 0.0], [0.1, 0.0]]])),
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
            ({"bins": 0}, "bins"),
            ({"flux": 0}, "flux"),
            ({"tau": 0j}, "tau"),
            ({"tau": 1 - 1j}, "tau"),
        ],
    )
    def test_rejects_invalid_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TorusPairCorrelation(**kwargs)

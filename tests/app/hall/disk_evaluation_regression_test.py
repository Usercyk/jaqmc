# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for disk evaluation finalization and distributed scans."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.disk.pair_correlation import DiskPairCorrelation
from jaqmc.app.hall.estimator.disk.wavefunction_scan import DiskWavefunctionScan
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax


@pytest.mark.parametrize("electrons", [1, 6])
def test_pair_finalization_keeps_undefined_bins_with_nan_checks(electrons: int) -> None:
    estimator = DiskPairCorrelation(bins=2, max_r=1, r=0.5)
    state = {
        "sample_counts": np.array([2.0]),
        "electron_counts": np.array([2.0 * electrons]),
        "reference_counts": np.array([1.0]),
        "density_counts": np.array([[[0.0, 1.0], [2.0, 2.0 * electrons - 3]]]),
        "pair_counts": np.array([[[0.0, 2.0], [3.0, 4.0]]]),
    }
    if electrons == 1:
        state["density_counts"] = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        state["pair_counts"] = np.zeros((1, 2, 2))
    with jax.debug_nans(True):
        digest = estimator.finalize_state(state, n_steps=2)
    g = digest["pair_correlation"]
    rho2 = digest["pair_correlation:rho2"]
    denominator = (
        digest["pair_correlation:rho1_reference"]
        * digest["pair_correlation:rho1_target"]
    )
    valid = denominator > 0
    assert np.isnan(g[~valid]).all()
    np.testing.assert_allclose(g[valid], rho2[valid] / denominator[valid])
    corrected = digest["pair_correlation:g_finite_n"]
    if electrons > 1:
        np.testing.assert_allclose(
            corrected[valid], electrons / (electrons - 1) * g[valid]
        )
    else:
        assert np.isnan(corrected).all()
    assert digest["pair_correlation:electron_number"] == electrons


def test_pair_finalization_rejects_invalid_raw_counts() -> None:
    estimator = DiskPairCorrelation(bins=2)
    state = {
        "sample_counts": np.array([1.0]),
        "electron_counts": np.array([6.0]),
        "reference_counts": np.array([1.0]),
        "density_counts": np.ones((1, 2, 2)),
        "pair_counts": np.full((1, 2, 2), np.nan),
    }
    with pytest.raises(ValueError, match="finite and nonnegative"):
        estimator.finalize_state(state, n_steps=1)


def gaussian_logpsi(params, data: HallData) -> jax.Array:
    """Evaluate a finite complex Gaussian with an analytic radial derivative.

    Returns:
        Complex logarithm of the test wavefunction.
    """
    return -params["scale"] * jnp.sum(data.electrons**2) + 1j * jnp.sum(
        data.electrons[:, 0]
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_scan_matches_analytic_values_and_keeps_completed_state(explicit: bool) -> None:
    samples = np.arange(32, dtype=np.float32).reshape(8, 2, 2) / 10 + 1
    samples[5] = [[0.1, 0.2], [0.3, 0.4]]
    estimator = DiskWavefunctionScan(
        f_log_psi=gaussian_logpsi,
        fixed_electrons=[[0.3, 0.4]] if explicit else None,
        selection_samples=8,
        n_x=4,
        n_y=3,
        x_min=-1,
        x_max=1,
        y_min=-1,
        y_max=1,
        chunk_size=5,
        radial_min=0.1,
        radial_max=0.5,
        radial_points=4,
    )
    pipeline = EstimatorPipeline({"scan": estimator})
    batched = BatchedData(
        data=HallData(electrons=jnp.asarray(samples[:4])),
        fields_with_batch=["electrons"],
    )
    shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    state = jax.device_put(pipeline.init(batched, jax.random.key(1)), shared)
    params = jax.device_put({"scale": jnp.asarray(0.5)}, shared)
    evaluate = parallel_jax.jit_sharded(
        pipeline.evaluate,
        in_specs=(
            parallel_jax.SHARE_PARTITION,
            batched.partition_spec,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=(parallel_jax.SHARE_PARTITION, parallel_jax.SHARE_PARTITION),
        check_vma=True,
    )
    for index, batch in enumerate(
        (samples[:4], samples[4:], np.zeros_like(samples[:4]))
    ):
        batched = jax.device_put(
            BatchedData(
                data=HallData(electrons=jnp.asarray(batch)),
                fields_with_batch=["electrons"],
            ),
            parallel_jax.make_sharding(batched.partition_spec),
        )
        with jax.debug_nans(True):
            _, state = evaluate(
                params,
                batched,
                state,
                parallel_jax.shard_rngs(jax.random.key(index + 2)),
            )
        assert bool(np.all(np.asarray(state["scan"]["scan_done"]))) == (
            explicit or index > 0
        )
    digest = pipeline.digest({}, state, n_steps=3)
    np.testing.assert_allclose(
        digest["wavefunction_scan:fixed_electrons"], samples[5, 1:]
    )
    x, y = np.meshgrid(digest["wavefunction_scan:x"], digest["wavefunction_scan:y"])
    fixed_squared = np.sum(samples[5, 1:] ** 2)
    np.testing.assert_allclose(
        digest["wavefunction_scan:log_abs"],
        -0.5 * (x**2 + y**2 + fixed_squared),
        atol=1e-6,
    )
    r = np.asarray(digest["wavefunction_scan:radial_r"])
    center_x, center_y = samples[5, 1]
    np.testing.assert_allclose(
        digest["wavefunction_scan:radial_log_abs"],
        -0.5 * ((center_x + r) ** 2 + center_y**2 + fixed_squared),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        digest["wavefunction_scan:radial_log_derivative"],
        -r * (center_x + r),
        atol=1e-6,
    )

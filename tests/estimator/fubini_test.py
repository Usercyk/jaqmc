# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.estimator.fubini import (
    FubiniStudyDistance,
    fubini_metrics_from_log_ratios,
)
from jaqmc.utils.checkpoint import NumPyCheckpointManager


def _unused_logpsi(params, data):
    del params, data
    return jnp.asarray(0.0j)


def test_phase_cancellation_and_reference_distance():
    ratios = jnp.asarray(
        [
            [0.0j, 0.0j],
            [1j * jnp.pi, 0.0j],
        ]
    )

    metrics = fubini_metrics_from_log_ratios(ratios, reference_index=1)

    np.testing.assert_allclose(metrics["fubini:distance"][0], np.pi / 2, atol=1e-6)
    np.testing.assert_allclose(metrics["fubini:modulus_cosine"][0], 1, atol=1e-6)
    np.testing.assert_allclose(metrics["fubini:phase_factor"][0], 0, atol=1e-6)
    np.testing.assert_allclose(metrics["fubini:distance"][1], 0, atol=1e-6)
    np.testing.assert_allclose(metrics["fubini:cosine"][1], 1, atol=1e-6)


def test_modulus_contribution_and_multiplicative_factorization():
    ratios = jnp.asarray(
        [
            [0.0j, 0.0j],
            [jnp.log(3.0) + 0.0j, 0.0j],
        ]
    )

    metrics = fubini_metrics_from_log_ratios(ratios, reference_index=1)
    expected_cosine = 2 / np.sqrt(5)

    np.testing.assert_allclose(metrics["fubini:cosine"][0], expected_cosine, rtol=1e-6)
    np.testing.assert_allclose(
        metrics["fubini:modulus_cosine"][0], expected_cosine, rtol=1e-6
    )
    np.testing.assert_allclose(metrics["fubini:phase_factor"][0], 1, rtol=1e-6)
    np.testing.assert_allclose(
        metrics["fubini:cosine"],
        metrics["fubini:modulus_cosine"] * metrics["fubini:phase_factor"],
        atol=1e-6,
    )


def test_metrics_are_stable_under_large_checkpoint_normalizations():
    base = jnp.asarray(
        [
            [0.0j, 0.0j],
            [1j * jnp.pi, 0.0j],
        ]
    )
    shifted = base + jnp.asarray([1000.0 + 3.0j, -700.0 - 2.0j])

    expected = fubini_metrics_from_log_ratios(base, reference_index=1)
    actual = fubini_metrics_from_log_ratios(shifted, reference_index=1)

    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-6)


def test_finalize_combines_steps_with_different_log_scales():
    ratios = jnp.asarray(
        [
            [4.0 + 0.2j, -2.0 + 0.1j, 1.0 - 0.3j],
            [3.0 - 0.5j, -1.0 + 0.4j, 2.0 + 0.2j],
            [-3.0 + 0.7j, 2.0 - 0.2j, -1.0 + 0.6j],
            [-4.0 - 0.1j, 1.0 + 0.3j, -2.0 - 0.4j],
        ]
    )
    estimator = FubiniStudyDistance(
        f_log_psi=_unused_logpsi,
        checkpoint_steps=(10, 20, 30),
        reference_index=1,
    )
    step_stats = [
        estimator.reduce({"fubini:log_ratio": ratios[:2]}),
        estimator.reduce({"fubini:log_ratio": ratios[2:]}),
    ]
    stacked = jax.tree.map(lambda *parts: jnp.stack(parts), *step_stats)

    actual = estimator.finalize_stats(stacked, None)
    expected = fubini_metrics_from_log_ratios(ratios, reference_index=1)

    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual["fubini:step"], [10, 20, 30])
    assert actual["fubini:reference_step"] == 20


def test_load_checkpoints_sorts_steps_and_selects_reference(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    manager.save(9, {"params": {"weight": jnp.asarray([9.0, 10.0])}})
    manager.save(2, {"params": {"weight": jnp.asarray([2.0, 3.0])}})
    estimator = FubiniStudyDistance(
        checkpoint_path=str(tmp_path),
        reference_step=2,
        f_log_psi=_unused_logpsi,
    )

    estimator.load_checkpoints({"weight": jnp.zeros(2)})

    assert estimator.checkpoint_steps == (2, 9)
    assert estimator.reference_index == 0
    assert isinstance(estimator.checkpoint_params, dict)
    np.testing.assert_array_equal(
        estimator.checkpoint_params["weight"],
        [[2.0, 3.0], [9.0, 10.0]],
    )


def test_load_checkpoints_rejects_missing_reference_step(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    manager.save(4, {"params": {"weight": jnp.asarray(1.0)}})
    estimator = FubiniStudyDistance(
        checkpoint_path=str(tmp_path),
        reference_step=3,
        f_log_psi=_unused_logpsi,
    )

    with pytest.raises(ValueError, match="available steps: \\(4,\\)"):
        estimator.load_checkpoints({"weight": jnp.asarray(0.0)})

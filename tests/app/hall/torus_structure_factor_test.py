# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physical density modes and connected subtraction across evaluation steps."""

import operator

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.torus import TorusStructureFactor
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax
from jaqmc.utils.torus import get_torus_lattice


def test_connected_subtraction_uses_all_steps_and_physical_reciprocal_vectors():
    estimator = TorusStructureFactor(
        flux=6, tau=0.3 + 1.2j, indices=((0, 0), (1, 0), (-1, 0))
    )
    steps = []
    for u in (0.0, 0.5):
        data = HallData(electrons=jnp.array([[u, 0.0]]))
        stats, _ = jax.jit(estimator.evaluate_single_walker)(
            {}, data, {}, None, jax.random.key(0)
        )
        steps.append(stats)
    batched = jax.tree.map(lambda *x: jnp.stack(x), *steps)
    result = estimator.finalize_stats(batched, None)
    np.testing.assert_allclose(result["structure_factor:raw"], [1, 1, 1])
    np.testing.assert_allclose(result["structure_factor"], [0, 1, 1])
    np.testing.assert_allclose(result["structure_factor:rho"][1:], 0, atol=1e-7)

    data = HallData(electrons=jnp.array([[0.13, 0.27], [0.4, 0.61]]))
    shifted = data.merge({"electrons": data.electrons.at[0].add(jnp.array([2, -1]))})
    stats, _ = estimator.evaluate_single_walker({}, data, {}, None, jax.random.key(0))
    shifted_stats, _ = estimator.evaluate_single_walker(
        {}, shifted, {}, None, jax.random.key(0)
    )
    np.testing.assert_allclose(
        stats["structure_factor:raw"], shifted_stats["structure_factor:raw"], atol=2e-6
    )
    assert stats["structure_factor:raw"][0] == 2
    l1, l2, _ = get_torus_lattice(estimator.flux, estimator.tau)
    lattice = jnp.array([[l1, 0], [l2.real, l2.imag]])
    np.testing.assert_allclose(
        result["structure_factor:k"] @ lattice.T,
        2 * jnp.pi * jnp.array(estimator.indices),
        atol=1e-6,
    )


def test_sharded_structure_factor_reduces_walkers_before_connected_subtraction():
    estimator = TorusStructureFactor(indices=((0, 0), (1, 0)))
    # Each single-walker connected estimate is zero, but the global variance
    # of this mode is one, including across separate CPU devices.
    positions = jnp.zeros((2 * jax.device_count(), 1, 2))
    positions = positions.at[1::2, 0, 0].set(0.5)
    batch = BatchedData(
        data=HallData(electrons=positions), fields_with_batch=["electrons"]
    )
    pipeline = EstimatorPipeline({"structure_factor": estimator})
    state = pipeline.init(batch, jax.random.key(0))
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
    )
    stats, state = evaluate(
        {}, batch, state, parallel_jax.shard_rngs(jax.random.key(1))
    )
    digest = pipeline.digest(
        jax.tree.map(operator.itemgetter(None), stats), state, n_steps=1
    )
    np.testing.assert_allclose(digest["structure_factor"], [0, 1])

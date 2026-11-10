# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Angular penalties retain phase gradients while reporting real observables."""

from operator import itemgetter
from typing import Any, cast

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.disk.penalized_loss import DiskPenalizedLoss
from jaqmc.app.hall.estimator.sphere.penalized_loss import SpherePenalizedLoss
from jaqmc.array_types import Params
from jaqmc.data import BatchedData
from jaqmc.estimator.angular_momentum import SphericalAngularMomentum
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.estimator.kinetic.disk import DiskKinetic
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.sphere import spinor_coordinates_from_angles
from jaqmc.utils import parallel_jax


def _evaluate(pipeline: EstimatorPipeline, electrons: jax.Array) -> dict[str, Any]:
    electrons = jnp.tile(electrons, (jax.device_count(), 1, 1))
    batch = BatchedData(
        data=HallData(electrons=electrons), fields_with_batch=["electrons"]
    )
    shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    params = jax.device_put(jnp.asarray(0.7), shared)
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
    )
    stats, state = evaluate(
        params, batch, state, parallel_jax.shard_rngs(jax.random.key(1))
    )
    return pipeline.finalize_stats(jax.tree.map(itemgetter(None), stats), state)


def test_disk_penalties_differentiate_phase_parameters() -> None:
    a, b, center = 0.7, 0.3, 1.0

    def logpsi(params: Params, data: HallData) -> jax.Array:
        x, y = data.electrons[0]
        return -(x**2 + y**2) / 4 + b * y + 1j * cast(jax.Array, params) * x

    pipeline = EstimatorPipeline(
        {
            "kinetic": DiskKinetic(f_log_psi=logpsi),
            "total": TotalEnergy(),
            "penalty": DiskPenalizedLoss(
                lz_center=center,
                lz_penalty=0.2,
                l2_penalty=0.4,
                lz_var_penalty=0.3,
            ),
            "grads": LossAndGrad(
                loss_key="penalized_loss", clip_method="none", f_log_psi=logpsi
            ),
        }
    )
    # These Gaussian quadrature points integrate all moments in this loss.
    points = jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=jnp.float32)
    points = points.at[:, 1].add(2 * b)
    actual = _evaluate(pipeline, points[:, None, :])
    mean_lz, mean_lz2 = -2 * a * b, a**2 * (1 + 4 * b**2) + b**2
    expected_loss = (
        0.5
        + (a - b) ** 2 / 2
        + 0.2 * (mean_lz2 - 2 * center * mean_lz + center**2)
        + 0.4 * mean_lz2
        + 0.3 * (a**2 + b**2)
    )
    expected_grad = (
        a - b + 0.6 * 2 * a * (1 + 4 * b**2) + 0.2 * 4 * center * b + 0.3 * 2 * a
    )
    np.testing.assert_allclose(actual["loss"], expected_loss, atol=2e-6)
    np.testing.assert_allclose(actual["grads"], expected_grad, atol=2e-6)
    np.testing.assert_allclose(
        actual["observable:angular_momentum_z"], mean_lz, atol=2e-6
    )
    assert not jnp.iscomplexobj(actual["observable:angular_momentum_z_square_var"])


def test_sphere_penalties_differentiate_phase_parameters() -> None:
    def spinor_logpsi(params: Params, u: jax.Array, v: jax.Array) -> jax.Array:
        return 1j * cast(jax.Array, params) * jnp.sum(2 * jnp.real(u * jnp.conj(v)))

    def logpsi(params: Params, data: HallData) -> jax.Array:
        return spinor_logpsi(params, *spinor_coordinates_from_angles(data.electrons))

    pipeline = EstimatorPipeline(
        {
            "angular": SphericalAngularMomentum(f_log_psi_from_spinor=spinor_logpsi),
            "total": TotalEnergy(),
            "penalty": SpherePenalizedLoss(lz_center=1, lz_penalty=0.2, l2_penalty=0.4),
            "grads": LossAndGrad(
                loss_key="penalized_loss", clip_method="none", f_log_psi=logpsi
            ),
        }
    )
    # Uniform spherical quadrature along the six Cartesian directions.
    points = jnp.array(
        [
            [jnp.pi / 2, 0],
            [jnp.pi / 2, jnp.pi],
            [jnp.pi / 2, jnp.pi / 2],
            [jnp.pi / 2, -jnp.pi / 2],
            [0, 0],
            [jnp.pi, 0],
        ]
    )
    actual = _evaluate(pipeline, points[:, None, :])
    np.testing.assert_allclose(actual["loss"], 0.2 + 0.7**2 / 3, atol=2e-6)
    np.testing.assert_allclose(actual["grads"], 2 * 0.7 / 3, atol=2e-6)
    np.testing.assert_allclose(
        actual["angular_momentum_square"], 2 * 0.7**2 / 3, atol=2e-6
    )
    assert not jnp.iscomplexobj(actual["angular_momentum_z_square_var"])

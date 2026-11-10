# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Disk wavefunction invariants, observables, and checkpoint workflows."""

import copy
from operator import itemgetter
from typing import Any

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.config import HallDiskConfig
from jaqmc.app.hall.data import HallData, data_init
from jaqmc.app.hall.hamiltonian.disk import DiskPotential
from jaqmc.app.hall.overlap import HallOverlapWorkflow
from jaqmc.app.hall.sampler.pinned_disk import pinned_disk_proposal
from jaqmc.app.hall.workflow import (
    HallEvalWorkflow,
    HallTrainWorkflow,
    configure_system,
)
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.estimator.kinetic.disk import DiskKinetic
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager

DISK_STATES = (
    [
        (
            "mhpo",
            {"envelope_type": mode, "num_heads": 1, "heads_dim": 4, "num_layers": 1},
        )
        for mode in (
            "no_m_learnable",
            "m_learnable",
            "m_learnable2",
            "m_learnable2_old",
        )
    ]
    + [("laughlin", {})]
    + [
        (f"{module}:{name}", {"kappa": 1.0})
        for module, names in (
            ("symbolic", ("DiskPhi4", "DiskPhi7", "DiskPhi8")),
            (
                "symbolic_20261021",
                ("DiskP6", "DiskP12", "DiskPsi5", "DiskPsi6", "DiskPsi7"),
            ),
            ("symbolic_round13", ("DiskP05C330", "DiskP10C459", "DiskP11C476")),
            ("symbolic_phi7_refit", ("DiskPhi7RefitC363",)),
        )
        for name in names
    ]
)


def finite_summary(label, values):
    array = np.asarray(values)
    assert np.isfinite(array).all(), label
    print(
        f"{label}: shape={array.shape}, abs_max={np.max(np.abs(array), initial=0):.8g}"
    )


def evaluate_pipeline(estimators, params, electrons):
    pipeline = EstimatorPipeline(estimators)
    batched = BatchedData(
        data=HallData(electrons=electrons), fields_with_batch=["electrons"]
    )
    state = pipeline.init(batched, jax.random.key(5))
    shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    params = jax.device_put(params, shared)
    state = jax.device_put(state, shared)
    batched = jax.device_put(
        batched, parallel_jax.make_sharding(batched.partition_spec)
    )
    evaluate = parallel_jax.jit_sharded(
        pipeline.evaluate,
        in_specs=(
            parallel_jax.SHARE_PARTITION,
            batched.partition_spec,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=(parallel_jax.SHARE_PARTITION, parallel_jax.SHARE_PARTITION),
    )
    stats, state = evaluate(
        params, batched, state, parallel_jax.shard_rngs(jax.random.key(6))
    )
    return pipeline.digest(jax.tree.map(itemgetter(None), stats), state, n_steps=1)


def test_disk_histograms_and_distributed_counts():
    from jaqmc.app.hall.estimator.disk.pair_correlation import DiskPairCorrelation
    from jaqmc.app.hall.estimator.disk.structure_factor import DiskStructureFactor
    from jaqmc.estimator.density.disk import DiskDensity

    electrons = jax.random.uniform(
        jax.random.key(7), (32, 3, 2), minval=0.1, maxval=0.9
    )
    estimators = {
        "density": DiskDensity(bins=8, range_max=2),
        "pair": DiskPairCorrelation(bins=4, max_r=2, r=0.6, reference_bin_width=1.0),
        "structure": DiskStructureFactor(num_q=4, num_angles=2),
    }
    digest = evaluate_pipeline(estimators, {}, electrons)
    np.testing.assert_allclose(np.sum(digest["density"]), 96)
    np.testing.assert_allclose(np.sum(digest["density_radial"]), 96)
    np.testing.assert_allclose(digest["pair_correlation:sample_count"], 32)
    np.testing.assert_allclose(digest["structure_factor:sample_count"], 32)
    np.testing.assert_allclose(digest["structure_factor"][0], 3, atol=1e-06)
    np.testing.assert_allclose(digest["structure_factor:connected"][0], 0, atol=1e-06)
    valid = (np.asarray(digest["pair_correlation:rho1_target"]) > 0) & (
        np.asarray(digest["pair_correlation:rho1_reference"]) > 0
    )
    assert valid.any()
    finite_summary(
        "pair correlation in populated bins", digest["pair_correlation"][valid]
    )
    assert np.isnan(np.asarray(digest["pair_correlation"])[~valid]).all()
    print("Histogram counts checked on", jax.device_count(), "devices")


@pytest.mark.parametrize(
    "observable", ["one_rdm", "edge_green", "short_range", "wavefunction_scan"]
)
def test_disk_observable_workflow(tmp_path, observable):
    config: dict[str, Any] = {
        "workflow": {"save_path": str(tmp_path), "batch_size": 32, "seed": 9},
        "system": {
            "module": "jaqmc.app.hall.config:HallDiskConfig",
            "flux": 6,
            "nspins": [2, 0],
            "interaction_strength": 0.5,
        },
        "wf": {"module": "jaqmc.app.hall.wavefunction.disk.laughlin"},
        "run": {"iterations": 2, "burn_in": 2},
        "estimators": {
            "enabled": {observable: True},
            "energy": {"potential": {"radial_grid_size": 64}},
        },
    }
    extras = {
        "one_rdm": {"num_orbitals": 3, "num_landau_levels": 1},
        "edge_green": {"n_theta": 8, "theta_min": 0.1, "theta_max": 5.9},
        "short_range": {
            "distance_edges": [0.0, 1.0, float("inf")],
            "radial_edges": [0.0, 4.0, float("inf")],
        },
        "wavefunction_scan": {
            "n_x": 4,
            "n_y": 4,
            "radial_points": 4,
            "chunk_size": 8,
            "selection_samples": 32,
        },
    }
    config["estimators"][observable] = extras[observable]
    if observable == "edge_green":
        del config["estimators"]["energy"]
    HallEvalWorkflow(ConfigManager(config))()
    with np.load(tmp_path / "evaluation_digest.npz") as digest:
        names = [k for k in digest.files if k.startswith(observable)]
        assert names, digest.files
        if observable == "edge_green":
            assert "total_energy" not in digest.files
        for key in names:
            finite_summary(key, digest[key])
        if observable == "short_range":
            np.testing.assert_allclose(np.sum(digest["short_range:count"]), 32)
            np.testing.assert_allclose(
                np.sum(digest["short_range:sum"], axis=0),
                digest["short_range:global_sum"],
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_array_equal(digest["short_range:nonfinite_count"], 0)


def test_disk_overlap_analytic_self(tmp_path):
    config: dict[str, Any] = {
        "workflow": {"save_path": str(tmp_path), "batch_size": 32, "seed": 9},
        "system": {"flux": 6, "nspins": [2, 0]},
        "wf": {"module": "jaqmc.app.hall.wavefunction.disk.laughlin"},
        "run": {"iterations": 2, "burn_in": 2},
    }
    HallOverlapWorkflow(ConfigManager(config))()
    with np.load(tmp_path / "overlap_digest.npz") as digest:
        np.testing.assert_allclose(digest["cosine"], 1, atol=2e-6)
        np.testing.assert_allclose(digest["fubini_distance"], 0, atol=2e-6)


def test_scan_selection_global_winner():
    from jaqmc.app.hall.estimator.disk.wavefunction_scan import DiskWavefunctionScan
    from jaqmc.app.hall.wavefunction.disk.laughlin import DiskLaughlin
    from jaqmc.utils.wiring import wire

    wf = wire(DiskLaughlin(), nspins=(2, 0), flux=6)
    electrons = jax.random.normal(jax.random.key(40), (32, 2, 2))
    params = wf.init_params(HallData(electrons=electrons[0]), jax.random.key(41))
    scan = DiskWavefunctionScan(
        n_x=4,
        n_y=4,
        radial_points=4,
        chunk_size=8,
        selection_samples=32,
        f_log_psi=wf.logpsi,
    )
    digest = evaluate_pipeline({"scan": scan}, params, electrons)
    scores = jax.vmap(wf.logpsi, in_axes=(None, 0))(
        params, HallData(electrons=electrons)
    ).real
    expected = np.asarray(electrons)[np.argmax(np.asarray(scores))]
    np.testing.assert_allclose(
        digest["wavefunction_scan:fixed_electrons"], expected[1:], atol=1e-7
    )
    finite_summary("scan log amplitude", digest["wavefunction_scan:log_abs"])


def test_disk_penalty_preserves_definition():
    from jaqmc.app.hall.estimator.disk.penalized_loss import DiskPenalizedLoss

    estimator = DiskPenalizedLoss(
        lz_center=2, lz_penalty=0.3, l2_penalty=0.4, lz_var_penalty=0.2
    )
    batch = BatchedData(
        data=HallData(electrons=jnp.zeros((2, 2, 2))), fields_with_batch=["electrons"]
    )
    stats = {
        "total_energy": jnp.array([1.0, 2.0]),
        "observable:angular_momentum_z": jnp.array([1.0, 3.0]),
        "observable:angular_momentum_z_square": jnp.array([1.0, 9.0]),
    }
    current, _ = estimator.evaluate_batch_walkers(
        {}, batch, stats, None, jax.random.key(44)
    )
    expected = (
        stats["total_energy"]
        + 0.3
        * (
            stats["observable:angular_momentum_z_square"]
            - 4 * stats["observable:angular_momentum_z"]
            + 4
        )
        + 0.4 * stats["observable:angular_momentum_z_square"]
        + 0.2
        * (
            stats["observable:angular_momentum_z_square"]
            - 4 * stats["observable:angular_momentum_z"]
            + 4
        )
    )
    np.testing.assert_allclose(current["penalized_loss"], expected, atol=1e-7)


@pytest.mark.integration
def test_disk_train_restore_evaluate_and_overlap(tmp_path):
    config: dict[str, Any] = {
        "workflow": {
            "save_path": str(tmp_path / "train"),
            "batch_size": 32,
            "seed": 21,
        },
        "system": {
            "module": "jaqmc.app.hall.config:HallDiskConfig",
            "flux": 3,
            "nspins": [2, 0],
        },
        "wf": {"num_heads": 1, "heads_dim": 4, "num_layers": 1},
        "train": {
            "run": {
                "iterations": 2,
                "burn_in": 2,
                "save_step_interval": 1,
                "save_time_interval": 0,
            }
        },
        "estimators": {"energy": {"potential": {"radial_grid_size": 64}}},
    }
    workflow = HallTrainWorkflow(ConfigManager(config))
    workflow()
    checkpoints = sorted((tmp_path / "train").glob("train_ckpt_*.npz"))
    assert len(checkpoints) == 2
    with np.load(checkpoints[0]) as first, np.load(checkpoints[-1]) as last:
        parameter_keys = [key for key in first.files if key.startswith("params/")]
        assert parameter_keys
        assert any(not np.array_equal(first[key], last[key]) for key in parameter_keys)
    evaluation = copy.deepcopy(config)
    del evaluation["train"]
    evaluation["workflow"]["source_path"] = config["workflow"]["save_path"]
    evaluation["workflow"]["save_path"] = str(tmp_path / "evaluation")
    evaluation["run"] = {"iterations": 2, "burn_in": 2}
    HallEvalWorkflow(ConfigManager(evaluation))()
    with np.load(tmp_path / "evaluation/evaluation_digest.npz") as digest:
        assert np.isfinite(digest["total_energy_real"])
        assert np.isfinite(digest["total_energy_real_var"])
    overlap = copy.deepcopy(evaluation)
    overlap.pop("estimators")
    overlap["workflow"]["save_path"] = str(tmp_path / "overlap")
    overlap["workflow"]["reference_path"] = str(tmp_path / "train")
    HallOverlapWorkflow(ConfigManager(overlap))()
    with np.load(tmp_path / "overlap/overlap_digest.npz") as digest:
        np.testing.assert_allclose(digest["cosine"], 1, atol=2e-6)
        np.testing.assert_allclose(digest["fubini_distance"], 0, atol=2e-6)


@pytest.mark.parametrize("module,options", DISK_STATES)
def test_disk_wavefunction_antisymmetry_and_finite_gradient(module, options):
    nelec = 6 if "symbolic" in module else 2
    config = ConfigManager(
        {
            "system": {
                "module": "jaqmc.app.hall.config:HallDiskConfig",
                "nspins": [nelec, 0],
                "flux": 3 * nelec,
            },
            "wf": {"module": module, **options},
        }
    )
    _, wavefunction = configure_system(config)
    electrons = (
        jnp.array(
            [
                [0.13, 0.27],
                [0.61, 0.43],
                [0.36, 0.89],
                [0.82, 0.16],
                [0.94, 0.73],
                [0.21, 0.64],
            ]
        )[:nelec]
        * 4
        - 2
    )
    data = HallData(electrons=electrons)
    params = wavefunction.init_params(data, jax.random.key(31))
    logpsi = wavefunction.logpsi(params, data)
    swapped = HallData(
        electrons=electrons.at[0].set(electrons[1]).at[1].set(electrons[0])
    )
    assert jnp.isfinite(logpsi)
    np.testing.assert_allclose(
        jnp.exp(wavefunction.logpsi(params, swapped) - logpsi), -1, atol=2e-5
    )
    gradient = jax.grad(
        lambda e: wavefunction.logpsi(params, HallData(electrons=e)).real
    )(electrons)
    assert jnp.isfinite(gradient).all()
    jax.clear_caches()


def test_disk_laughlin_exact_kinetic_and_angular_momentum():
    _, wavefunction = configure_system(
        ConfigManager(
            {
                "system": {
                    "module": "jaqmc.app.hall.config:HallDiskConfig",
                    "nspins": [2, 0],
                    "flux": 6,
                },
                "wf": {"module": "laughlin"},
            }
        )
    )
    data = HallData(electrons=jnp.array([[2.0, 0.0], [0.4, 0.7]]))
    params = wavefunction.init_params(data, jax.random.key(0))
    stats, _ = jax.jit(
        DiskKinetic(f_log_psi=wavefunction.logpsi).evaluate_single_walker
    )(params, data, {}, None, jax.random.key(1))
    np.testing.assert_allclose(stats["energy:kinetic"], 1, atol=2e-5)
    np.testing.assert_allclose(stats["observable:angular_momentum_z"], -3, atol=2e-5)
    np.testing.assert_allclose(
        stats["observable:angular_momentum_z_square"], 9, atol=2e-5
    )


def test_disk_background_at_center_matches_integrated_uniform_disk():
    potential = DiskPotential(
        radius=3, N=2, d=0.2, interaction_strength=1.5, radial_grid_size=32
    )
    data = HallData(electrons=jnp.zeros((1, 2)))
    potential.init(data, jax.random.key(0))
    stats, _ = potential.evaluate_single_walker({}, data, {}, None, jax.random.key(1))
    expected = (
        -2
        * potential.N
        / potential.radius**2
        * (np.sqrt(potential.radius**2 + potential.d**2) - potential.d)
        * potential.interaction_strength
    )
    np.testing.assert_allclose(stats["energy:potential"], expected, atol=1e-6)


def test_disk_initialization_and_pinned_proposal():
    system = HallDiskConfig(nspins=(2, 0), radius=3)
    data = data_init(system, 2048, jax.random.key(0)).data.electrons
    radius_squared = jnp.sum(data**2, axis=-1)
    assert jnp.all(radius_squared <= 9)
    np.testing.assert_allclose(jnp.mean(radius_squared) / 9, 0.5, atol=0.025)
    moved = pinned_disk_proposal(jax.random.key(1), {"electrons": data}, 0.5)[
        "electrons"
    ]
    np.testing.assert_array_equal(moved[:, 0], data[:, 0])
    assert not jnp.array_equal(moved[:, 1], data[:, 1])

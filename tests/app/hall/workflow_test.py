# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hall workflow wiring and end-to-end runs."""

import jax
import numpy as np
import pytest

from jaqmc.app.hall import HallEvalWorkflow, HallTrainWorkflow
from jaqmc.app.hall.estimator.torus import (
    TorusGraviton,
    TorusStress,
    TorusStructureFactor,
)
from jaqmc.app.hall.hamiltonian import TorusPotential
from jaqmc.estimator.angular_momentum import (
    SphericalAngularMomentum,
    SphericalAngularMomentumProjection,
)
from jaqmc.estimator.kinetic import TorusKinetic
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.utils.checkpoint import NumPyCheckpointManager
from jaqmc.utils.config import ConfigManager
from jaqmc.workflow.base import init_batched_data


def _train_estimators(config):
    workflow = HallTrainWorkflow(ConfigManager(config))
    return workflow.train_stage.estimators.estimators


def _eval_estimators(config):
    workflow = HallEvalWorkflow(ConfigManager(config))
    return workflow.evaluation_stage.estimators.estimators


def test_angular_momentum_enabled_by_default():
    assert isinstance(
        _train_estimators({})["angular_momentum"], SphericalAngularMomentum
    )


def test_angular_momentum_can_be_disabled_without_penalty():
    config = {"estimators": {"enabled": {"angular_momentum": False}}}
    assert "angular_momentum" not in _train_estimators(config)


def test_eval_angular_momentum_without_energy():
    estimators = _eval_estimators({"estimators": {"enabled": {"energy": False}}})
    assert "kinetic" not in estimators
    assert isinstance(estimators["angular_momentum"], SphericalAngularMomentum)


def test_penalty_switches_loss_key():
    grads = _train_estimators({"system": {"lz_penalty": 1.0}})["grads"]
    assert isinstance(grads, LossAndGrad)
    assert grads.loss_key == "penalized_loss"


def test_angular_projection_is_eval_only_and_opt_in():
    config = {"estimators": {"enabled": {"angular_projection": True}}}
    assert "angular_projection" not in _train_estimators(config)
    assert isinstance(
        _eval_estimators(config)["angular_projection"],
        SphericalAngularMomentumProjection,
    )


@pytest.mark.parametrize(
    "workflow_cls,config,match",
    [
        pytest.param(
            HallEvalWorkflow,
            {"system": {penalty: 1.0}, "estimators": {"enabled": {"energy": False}}},
            r"penalties require estimators\.enabled\.energy=true",
            id=f"{penalty}-without-energy",
        )
        for penalty in ("lz_penalty", "l2_penalty")
    ]
    + [
        pytest.param(
            HallTrainWorkflow,
            {
                "system": {penalty: 1.0},
                "estimators": {"enabled": {"angular_momentum": False}},
            },
            r"penalties require estimators\.enabled\.angular_momentum=true",
            id=f"{penalty}-without-angular-momentum",
        )
        for penalty in ("lz_penalty", "l2_penalty")
    ]
    + [
        pytest.param(
            HallEvalWorkflow,
            {"estimators": {"enabled": {"energy": False, "angular_projection": True}}},
            r"projection requires estimators\.enabled\.energy=true",
            id="projection-without-energy",
        )
    ],
)
def test_rejects_inconsistent_estimator_config(workflow_cls, config, match):
    with pytest.raises(ValueError, match=match):
        workflow_cls(ConfigManager(config))


def _run_config(tmp_path, *, wf_module: str, save_name: str):
    return {
        "workflow": {
            "save_path": str(tmp_path / save_name),
            "batch_size": 16,
            "seed": 0,
        },
        "system": {"flux": 6, "nspins": (3, 0)},
        "wf": {"module": wf_module},
    }


def _evaluation_run_config():
    return {
        "iterations": 50,
        "burn_in": 0,
        # JAX 0.5.x and 0.6.x cannot validate this Hessian path inside shard_map.
        "check_vma": jax.__version_info__[:2] not in {(0, 5), (0, 6)},
    }


@pytest.mark.parametrize("wf_module", ["laughlin", "free"])
def test_train_rejects_parameter_free_wavefunction(tmp_path, wf_module):
    config = _run_config(tmp_path, wf_module=wf_module, save_name="train")
    config["train"] = {"run": {"iterations": 1}}

    with pytest.raises(ValueError, match="no trainable parameters"):
        HallTrainWorkflow(ConfigManager(config))()

    assert not list((tmp_path / "train").glob("train_ckpt_*.npz"))


def test_laughlin_eval_keeps_local_energy_variance_bounded(tmp_path):
    config = _run_config(tmp_path, wf_module="laughlin", save_name="eval")
    config["run"] = _evaluation_run_config()
    HallEvalWorkflow(ConfigManager(config))()

    digest = np.load(tmp_path / "eval" / "evaluation_digest.npz")
    assert digest["energy:potential"] > 0
    assert digest["total_energy_real_var"] < 0.5


def test_free_eval_is_non_interacting(tmp_path):
    config = _run_config(tmp_path, wf_module="free", save_name="eval")
    config["system"]["interaction_strength"] = 0.0
    config["run"] = _evaluation_run_config()
    HallEvalWorkflow(ConfigManager(config))()

    digest = np.load(tmp_path / "eval" / "evaluation_digest.npz")
    np.testing.assert_allclose(digest["energy:potential"], 0.0, atol=1e-6)
    np.testing.assert_allclose(digest["energy:kinetic"], 1.5, atol=1e-2)
    np.testing.assert_allclose(digest["total_energy_real"], 1.5, atol=1e-2)


@pytest.mark.parametrize("estimators_fn", [_train_estimators, _eval_estimators])
def test_torus_workflows_select_torus_energy(estimators_fn):
    estimators = estimators_fn(
        {"system": {"module": "jaqmc.app.hall.config:HallTorusConfig"}}
    )
    assert isinstance(estimators["kinetic"], TorusKinetic)
    assert isinstance(estimators["potential"], TorusPotential)
    assert "angular_momentum" not in estimators


def test_torus_rejects_spherical_projection():
    with pytest.raises(ValueError, match="projection requires a sphere system"):
        _eval_estimators(
            {
                "system": {"module": "jaqmc.app.hall.config:HallTorusConfig"},
                "estimators": {"enabled": {"angular_projection": True}},
            }
        )


@pytest.mark.parametrize("observable", ["stress", "graviton"])
def test_torus_geometric_estimators_share_system_metrics(observable):
    enabled = {"stress": True, "structure_factor": True}
    enabled[observable] = True
    config = {
        "system": {
            "module": "jaqmc.app.hall.config:HallTorusConfig",
            "flux": 6,
            "tau": [0.3, 1.1],
            "mass_shear": [0.2, -0.1],
            "interaction_shear": [-0.1, 0.3],
            "interaction_strength": 0.5,
        },
        "estimators": {"enabled": enabled},
    }
    estimators = _eval_estimators(config)
    response = estimators[observable]
    assert isinstance(
        response, TorusGraviton if observable == "graviton" else TorusStress
    )
    assert isinstance(estimators["structure_factor"], TorusStructureFactor)
    assert response.flux == estimators["kinetic"].flux == 6
    assert response.tau == estimators["potential"].tau
    assert response.tau == pytest.approx(0.3 + 1.1j)
    assert response.mass_shear == estimators["kinetic"].mass_shear == (0.2, -0.1)
    assert response.interaction_shear == estimators["potential"].interaction_shear
    assert response.interaction_strength == estimators["potential"].interaction_strength


def test_torus_graviton_is_evaluation_only():
    config = {
        "system": {"module": "jaqmc.app.hall.config:HallTorusConfig"},
        "estimators": {"enabled": {"graviton": True}},
    }
    # Constructing the workflow validates configuration without starting it.
    with pytest.raises(ValueError, match="evaluation-only"):
        HallTrainWorkflow(ConfigManager(config))


def test_fubini_eval_loads_workflow_source_checkpoints(tmp_path):
    source = tmp_path / "source"
    config = {
        "workflow": {
            "source_path": str(source),
            "save_path": str(tmp_path / "eval"),
            "batch_size": 4,
            "seed": 0,
        },
        "wf": {"num_heads": 1, "heads_dim": 4, "num_layers": 1},
        "estimators": {
            "enabled": {"energy": False, "angular_momentum": False, "fubini": True}
        },
        "sampler": {"steps": 1},
        "run": {"iterations": 1, "burn_in": 0},
    }
    workflow = HallEvalWorkflow(ConfigManager(config))
    data = init_batched_data(workflow.data_init, 4, jax.random.key(0))
    state = workflow.evaluation_stage.create_state(jax.random.key(1), batched_data=data)
    checkpoint = {
        "params": state.params,
        "batched_data": state.batched_data,
        "sampler_state": state.sampler_state,
    }
    manager = NumPyCheckpointManager(source, prefix="train")
    manager.save(1, checkpoint)
    manager.save(2, checkpoint)

    workflow()

    with np.load(tmp_path / "eval" / "evaluation_digest.npz") as digest:
        np.testing.assert_array_equal(digest["fubini:step"], [1, 2])
        np.testing.assert_allclose(digest["fubini:distance"], 0.0, atol=1e-6)
        np.testing.assert_allclose(digest["fubini:cosine"], 1.0, atol=1e-6)

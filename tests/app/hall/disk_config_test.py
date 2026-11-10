# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Disk model domains and compatible evaluation configurations."""

import jax
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.disk.edge_green import DiskEdgeGreenDirect
from jaqmc.app.hall.sampler.pinned_disk import PinnedDiskMCMCSampler
from jaqmc.app.hall.workflow import (
    HallEvalWorkflow,
    HallTrainWorkflow,
    configure_system,
    make_sampler,
)
from jaqmc.utils.config import ConfigManager

DISK_SYSTEM = {"module": "jaqmc.app.hall.config:HallDiskConfig"}


@pytest.mark.parametrize("system_count,wf_count", [(None, 4), (5, None), (4, 4)])
def test_orbital_count_configuration(
    system_count: int | None, wf_count: int | None
) -> None:
    cfg = ConfigManager(
        {
            "system": {**DISK_SYSTEM, "num_orbitals": system_count},
            "wf": {"num_orbitals": wf_count},
        }
    )
    _, wf = configure_system(cfg)
    assert wf.num_orbitals == (system_count or wf_count)
    assert cfg.resolved_config["wf"]["num_orbitals"] == wf_count


def test_conflicting_orbital_counts_are_rejected() -> None:
    cfg = ConfigManager(
        {"system": {**DISK_SYSTEM, "num_orbitals": 5}, "wf": {"num_orbitals": 4}}
    )
    with pytest.raises(ValueError, match="must agree"):
        configure_system(cfg)


def test_edge_evaluation_uses_only_conditional_observables() -> None:
    cfg = ConfigManager(
        {
            "system": {**DISK_SYSTEM, "nspins": [1, 0], "flux": 1},
            "wf": {"module": "laughlin"},
            "estimators": {
                "enabled": {"edge_green": True},
                "edge_green": {"edge_radius": 3, "theta_min": 0, "n_theta": 1},
            },
        }
    )
    workflow = HallEvalWorkflow(cfg)
    estimators = workflow.evaluation_stage.estimators.estimators
    assert set(estimators) == {"edge_green"}
    system, wf = configure_system(cfg)
    sampler = make_sampler(cfg, system, evaluation=True)
    assert isinstance(sampler, PinnedDiskMCMCSampler)
    edge = estimators["edge_green"]
    assert isinstance(edge, DiskEdgeGreenDirect)
    assert sampler.edge_radius == edge.edge_radius == 3
    data = HallData(electrons=jnp.array([[sampler.edge_radius, 0.0]]))
    params = wf.init_params(data, jax.random.key(0))
    stats, _ = edge.evaluate_single_walker(params, data, {}, None, jax.random.key(1))
    assert float(jnp.abs(stats["edge_green_normalized"][0] - 1)) < 1e-6


@pytest.mark.parametrize(
    "observable",
    [
        "energy",
        "angular_momentum",
        "density",
        "pair_correlation",
        "one_rdm",
        "structure_factor",
        "short_range",
    ],
)
def test_edge_rejects_equilibrium_observables(observable: str) -> None:
    cfg = ConfigManager(
        {
            "system": DISK_SYSTEM,
            "estimators": {"enabled": {"edge_green": True, observable: True}},
        }
    )
    with pytest.raises(ValueError, match=f"cannot be combined with {observable}"):
        HallEvalWorkflow(cfg)


def test_edge_rejects_training_and_mismatched_sampling_radius() -> None:
    config = {
        "system": DISK_SYSTEM,
        "estimators": {"enabled": {"edge_green": True}},
    }
    with pytest.raises(ValueError, match="only available during disk evaluation"):
        HallTrainWorkflow(ConfigManager(config))
    with pytest.raises(ValueError, match=r"sampler\.edge_radius must match"):
        HallEvalWorkflow(ConfigManager({**config, "sampler": {"edge_radius": 3}}))


@pytest.mark.parametrize("flux", [0, 3, 4])
def test_laughlin_rejects_nonfermionic_fillings(flux: int) -> None:
    _, wf = configure_system(
        ConfigManager(
            {
                "system": {**DISK_SYSTEM, "nspins": [2, 0], "flux": flux},
                "wf": {"module": "laughlin"},
            }
        )
    )
    with pytest.raises(ValueError, match="positive odd integer"):
        wf.init_params(
            HallData(electrons=jnp.array([[0.1, 0.2], [0.3, 0.4]])), jax.random.key(0)
        )


@pytest.mark.parametrize(
    "module,kappa",
    [
        ("symbolic:DiskPhi4", 30.0),
        ("symbolic:DiskPhi7", 30.0),
        ("symbolic:DiskPhi8", 30.0),
        ("symbolic_20261021:DiskP6", 30.0),
        ("symbolic_phi7_refit:DiskPhi7RefitC363", 30.0),
        ("symbolic_round13:DiskP10C459", 0.0),
        ("symbolic_round13:DiskP11C476", 0.0),
    ],
)
def test_fitted_states_reject_growing_quartic_envelopes(
    module: str, kappa: float
) -> None:
    _, wf = configure_system(
        ConfigManager({"system": DISK_SYSTEM, "wf": {"module": module, "kappa": kappa}})
    )
    with pytest.raises(ValueError, match="decaying quartic envelope"):
        wf.init_params(HallData(electrons=jnp.ones((6, 2))), jax.random.key(0))


@pytest.mark.parametrize("kappa", [-1.0, float("nan"), float("inf")])
def test_fitted_states_require_finite_nonnegative_coupling(kappa: float) -> None:
    _, wf = configure_system(
        ConfigManager(
            {
                "system": DISK_SYSTEM,
                "wf": {"module": "symbolic:DiskPhi4", "kappa": kappa},
            }
        )
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        wf.init_params(HallData(electrons=jnp.ones((6, 2))), jax.random.key(0))


def test_fitted_state_rejects_an_internal_pole() -> None:
    _, wf = configure_system(
        ConfigManager(
            {
                "system": DISK_SYSTEM,
                "wf": {"module": "symbolic:DiskPhi7", "kappa": 1 / 1.1889995670549587},
            }
        )
    )
    with pytest.raises(ValueError, match="pole"):
        wf.init_params(HallData(electrons=jnp.ones((6, 2))), jax.random.key(0))

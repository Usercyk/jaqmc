# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Workflows for the fractional quantum Hall effect on a Haldane sphere."""

import logging
from functools import partial
from typing import Any

from jax import numpy as jnp

from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.density import SphericalDensity
from jaqmc.estimator.kinetic import SphericalKinetic, TorusKinetic
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.sphere import sphere_proposal
from jaqmc.geometry.torus import torus_proposal
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

from .config import HallGeometryConfig, HallSphereConfig, HallTorusConfig
from .data import data_init
from .estimator import SphereOneRDM, SpherePairCorrelation, SpherePenalizedLoss
from .hamiltonian import SpherePotential, TorusPotential

logger = logging.getLogger(__name__)


class HallTrainWorkflow(VMCWorkflow):
    """VMC training workflow for quantum Hall effect simulations."""

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        console_fields = (
            "pmove:.2f,"
            "energy=total_energy_real:.4f,"
            "variance=total_energy_real_var:.4f,"
            "Lz=angular_momentum_z:+.4f,"
            "L_square=angular_momentum_square:.4f"
        )
        return {
            "train": {
                "run": {"iterations": 200_000},
                "writers": {"console": {"fields": console_fields}},
                "grads": {"clip_scale": 100, "clip_method": "iqr"},
            }
        }

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf = configure_system(cfg)

        self.data_init = partial(data_init, system_config)

        loss_key = get_loss_key(system_config)

        estimators = make_estimators(cfg, wf, system_config, always_enable_energy=True)

        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        sampler = make_sampler(cfg, system_config)
        train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=wf.logpsi)
        train.configure_estimators(**estimators)
        train.configure_loss_grads(
            cfg.scoped("train").get("grads", LossAndGrad(loss_key=loss_key)),
            f_log_psi=wf.logpsi,
        )
        self.train_stage = train.build()


class HallEvalWorkflow(EvaluationWorkflow):
    """Evaluation workflow for quantum Hall effect simulations."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf = configure_system(cfg)

        self.data_init = partial(data_init, system_config)

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = make_sampler(cfg, system_config)
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        eval_estimators: dict[str, EstimatorLike] = make_estimators(
            cfg, wf, system_config
        )
        evaluation.configure_estimators(**eval_estimators)
        self.evaluation_stage = evaluation.build()


def get_loss_key(system_config: HallGeometryConfig) -> str:
    """Determine the loss key based on the system configuration.

    Args:
        system_config: The configuration of the quantum Hall system.

    Returns:
        The loss key as a string.
    """
    if isinstance(system_config, HallSphereConfig):
        has_penalties = bool(system_config.lz_penalty or system_config.l2_penalty)
        return "penalized_loss" if has_penalties else "total_energy"
    elif isinstance(system_config, HallTorusConfig):
        # For torus configurations, we currently only support total energy.
        return "total_energy"
    else:
        raise NotImplementedError(f"Unsupported Hall geometry: {type(system_config)}")


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[HallGeometryConfig, Any]:
    """Build the shared system objects for quantum Hall workflows.

    Args:
        cfg: Configuration manager.

    Returns:
        Tuple of (system_config, wavefunction).
    """
    system_config: HallGeometryConfig = cfg.get_module(
        "system", "jaqmc.app.hall.config:HallSphereConfig"
    )
    if isinstance(system_config, HallSphereConfig):
        wf = cfg.get_module("wf", "jaqmc.app.hall.wavefunction.sphere.mhpo")
        wf.nspins = system_config.nspins
        wf.monopole_strength = system_config.flux / 2
        wf.flux = system_config.flux

        return system_config, wf
    if isinstance(system_config, HallTorusConfig):
        wf = cfg.get_module("wf", "jaqmc.app.hall.wavefunction.torus.mhpo")
        wf.nspins = system_config.nspins
        wf.flux = system_config.flux
        wf.tau = system_config.tau

        return system_config, wf

    raise NotImplementedError(f"Unsupported Hall geometry: {type(system_config)}")


def make_sampler(
    cfg: ConfigManagerLike, system_config: HallGeometryConfig
) -> MCMCSampler:
    """Create a sampler based on the system configuration.

    Args:
        cfg: Configuration manager.
        system_config: The configuration of the quantum Hall system.

    Returns:
        An instance of MCMCSampler.
    """
    if isinstance(system_config, HallSphereConfig):
        return cfg.get("sampler", MCMCSampler(sampling_proposal=sphere_proposal))
    if isinstance(system_config, HallTorusConfig):
        return cfg.get("sampler", MCMCSampler(sampling_proposal=torus_proposal))
    raise NotImplementedError(f"Unsupported Hall geometry: {type(system_config)}")


def make_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallGeometryConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    if isinstance(system_config, HallSphereConfig):
        return make_sphere_estimators(cfg, wf, system_config, always_enable_energy)
    if isinstance(system_config, HallTorusConfig):
        return make_torus_estimators(cfg, wf, system_config, always_enable_energy)
    raise NotImplementedError(f"Unsupported Hall geometry: {type(system_config)}")


def make_torus_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallTorusConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        estimators["kinetic"] = cfg.get(
            "estimators.energy.kinetic",
            TorusKinetic(
                flux=system_config.flux,
                tau=system_config.tau,
                f_log_psi=wf.logpsi,
            ),
        )
        estimators["potential"] = cfg.get(
            "estimators.energy.potential",
            TorusPotential(
                interaction_type=system_config.interaction_type,
                flux=system_config.flux,
                tau=system_config.tau,
                interaction_strength=system_config.interaction_strength,
            ),
        )
        estimators["total"] = TotalEnergy()
    return estimators


def make_sphere_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallSphereConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        Q = system_config.flux / 2
        radius = (
            system_config.radius
            if system_config.radius is not None
            else float(jnp.sqrt(Q))
        )

        estimators["kinetic"] = cfg.get(
            "estimators.energy.kinetic",
            SphericalKinetic(
                monopole_strength=Q,
                radius=radius,
                f_log_psi=wf.logpsi,
            ),
        )
        estimators["potential"] = cfg.get(
            "estimators.energy.potential",
            SpherePotential(
                interaction_type=system_config.interaction_type,
                monopole_strength=Q,
                radius=radius,
                interaction_strength=system_config.interaction_strength,
            ),
        )
        estimators["total"] = TotalEnergy()

        if system_config.lz_penalty or system_config.l2_penalty:
            estimators["penalty"] = SpherePenalizedLoss(
                lz_center=system_config.lz_center,
                lz_penalty=system_config.lz_penalty,
                l2_penalty=system_config.l2_penalty,
            )

    if cfg.get("estimators.enabled.density", False):
        estimators["density"] = cfg.get(
            "estimators.density",
            SphericalDensity(),
        )

    if cfg.get("estimators.enabled.pair_correlation", False):
        estimators["pair_correlation"] = cfg.get(
            "estimators.pair_correlation",
            SpherePairCorrelation(),
        )

    if cfg.get("estimators.enabled.one_rdm", False):
        estimators["one_rdm"] = cfg.get(
            "estimators.one_rdm",
            SphereOneRDM(flux=system_config.flux, f_log_psi=wf.logpsi),
        )

    return estimators

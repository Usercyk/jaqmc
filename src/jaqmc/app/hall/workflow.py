# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Workflows for sphere, disk, and torus quantum Hall systems."""

import logging
from functools import partial
from typing import Any

from jax import numpy as jnp

from jaqmc.estimator import EstimatorLike, FubiniStudyDistance
from jaqmc.estimator.angular_momentum import (
    SphericalAngularMomentum,
    SphericalAngularMomentumProjection,
)
from jaqmc.estimator.density import SphericalDensity, TorusDensity
from jaqmc.estimator.density.disk import DiskDensity
from jaqmc.estimator.kinetic import SphericalKinetic, TorusKinetic
from jaqmc.estimator.kinetic.disk import DiskKinetic
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.disk import disk_proposal
from jaqmc.geometry.sphere import sphere_proposal
from jaqmc.geometry.torus import torus_proposal
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

from .config import (
    HallDiskConfig,
    HallGeometryConfig,
    HallSphereConfig,
    HallTorusConfig,
)
from .data import data_init
from .estimator import (
    SphereOneRDM,
    SpherePairCorrelation,
    SpherePenalizedLoss,
    TorusOneRDM,
    TorusPairCorrelation,
)
from .estimator.disk import (
    DiskEdgeGreenDirect,
    DiskOneRDM,
    DiskPairCorrelation,
    DiskPenalizedLoss,
    DiskStructureFactor,
    DiskWavefunctionScan,
)
from .estimator.disk.short_range import ShortRangeEstimator
from .hamiltonian import SpherePotential, TorusPotential
from .hamiltonian.disk import DiskPotential
from .sampler.pinned_disk import PinnedDiskMCMCSampler

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
                "optim": {"learning_rate": {"delay": 10000, "warmup": 1000}},
                "grads": {"clip_scale": 100},
            }
        }

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf = configure_system(cfg)

        self.data_init = partial(data_init, system_config)

        loss_key = get_loss_key(system_config)
        if isinstance(system_config, HallDiskConfig):
            fields = (
                "pmove:.2f,energy=total_energy_real:.4f,"
                "variance=total_energy_real_var:.4f,"
                "Lz=observable:angular_momentum_z:+.4f,"
                "Lz_square=observable:angular_momentum_z_square:.4f"
            )
            cfg.use_preset({"train": {"writers": {"console": {"fields": fields}}}})

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
        sampler = make_sampler(cfg, system_config, evaluation=True)
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        eval_estimators: dict[str, EstimatorLike] = make_estimators(
            cfg, wf, system_config
        )
        if isinstance(system_config, HallDiskConfig) and cfg.get(
            "estimators.enabled.fubini", False
        ):
            raise ValueError(
                "Disk overlap uses 'jaqmc hall overlap' "
                "and its overlap sampling distribution."
            )
        if cfg.get("estimators.enabled.angular_projection", False):
            if not isinstance(system_config, HallSphereConfig):
                raise ValueError(
                    "Angular-momentum projection requires a sphere system."
                )
            if not cfg.get("estimators.enabled.energy", True):
                raise ValueError(
                    "Angular-momentum projection requires "
                    "estimators.enabled.energy=true."
                )
            eval_estimators["angular_projection"] = cfg.get(
                "estimators.angular_projection",
                SphericalAngularMomentumProjection(
                    f_log_psi_from_spinor=wf.logpsi_from_spinor
                ),
            )
        evaluation.configure_estimators(**eval_estimators)
        self.evaluation_stage = evaluation.build()


def get_loss_key(system_config: HallGeometryConfig) -> str:
    """Select total energy or the geometry's angular-momentum penalty.

    Returns:
        The loss statistic consumed by the training gradient estimator.

    Raises:
        NotImplementedError: If the geometry is unsupported.
    """
    if isinstance(system_config, HallTorusConfig):
        return "total_energy"
    if isinstance(system_config, (HallSphereConfig, HallDiskConfig)):
        has_penalties = bool(system_config.lz_penalty or system_config.l2_penalty)
        if isinstance(system_config, HallDiskConfig):
            has_penalties = has_penalties or bool(system_config.lz_var_penalty)
        return "penalized_loss" if has_penalties else "total_energy"
    raise NotImplementedError(f"Unsupported Hall geometry: {type(system_config)}")


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[HallGeometryConfig, Any]:
    """Build the shared system objects for quantum Hall workflows.

    Args:
        cfg: Configuration manager.

    Returns:
        Tuple of (system_config, wavefunction).

    Raises:
        ValueError: If the system and wavefunction specify conflicting orbital counts.
    """
    system_config: HallGeometryConfig = cfg.get_module(
        "system", "jaqmc.app.hall.config:HallSphereConfig"
    )
    if isinstance(system_config, HallDiskConfig):
        wf = cfg.get_module("wf", "jaqmc.app.hall.wavefunction.disk.mhpo")
        wf.nspins = system_config.nspins
        wf.flux = system_config.flux
        if "monopole_strength" in wf.__dataclass_fields__:
            wf.monopole_strength = system_config.flux
        if (
            "num_orbitals" in wf.__dataclass_fields__
            and system_config.num_orbitals is not None
        ):
            if (
                wf.num_orbitals is not None
                and wf.num_orbitals != system_config.num_orbitals
            ):
                raise ValueError("wf.num_orbitals and system.num_orbitals must agree.")
            wf.num_orbitals = system_config.num_orbitals
        return system_config, wf
    if isinstance(system_config, HallSphereConfig):
        wf = cfg.get_module("wf", "jaqmc.app.hall.wavefunction.sphere.mhpo")
        wf.nspins = system_config.nspins
        wf.monopole_strength = system_config.flux / 2
        wf.flux = system_config.flux
        wf.interaction_strength = system_config.interaction_strength
        wf.radius = (
            system_config.radius
            if system_config.radius is not None
            else float(jnp.sqrt(system_config.flux / 2))
        )

        return system_config, wf
    if isinstance(system_config, HallTorusConfig):
        wf = cfg.get_module("wf", "jaqmc.app.hall.wavefunction.torus.mhpo")
        wf.nspins = system_config.nspins
        wf.flux = system_config.flux
        wf.tau = system_config.tau

        return system_config, wf

    raise NotImplementedError(f"Unsupported Hall geometry: {type(system_config)}")


def make_sampler(
    cfg: ConfigManagerLike,
    system_config: HallGeometryConfig,
    *,
    evaluation: bool = False,
) -> MCMCSampler:
    """Select the proposal and, for edge correlations, the pinned sampler.

    Returns:
        The configured geometry-specific sampler.

    Raises:
        ValueError: If the sampler and edge estimator use different radii.
    """
    if isinstance(system_config, HallTorusConfig):
        default = MCMCSampler(
            sampling_proposal=partial(torus_proposal, tau=system_config.tau)
        )
    elif isinstance(system_config, HallDiskConfig):
        if evaluation and cfg.get("estimators.enabled.edge_green", False):
            radius = (
                system_config.radius
                if system_config.radius is not None
                else float(jnp.sqrt(2 * system_config.flux))
            )
            default = PinnedDiskMCMCSampler(
                edge_radius=cfg.get("estimators.edge_green.edge_radius", radius)
            )
        else:
            default = MCMCSampler(sampling_proposal=disk_proposal)
    elif isinstance(system_config, HallSphereConfig):
        default = MCMCSampler(sampling_proposal=sphere_proposal)
    else:
        raise NotImplementedError(f"Unsupported Hall geometry: {type(system_config)}")
    sampler = cfg.get("sampler", default)
    if isinstance(default, PinnedDiskMCMCSampler) and (
        not isinstance(sampler, PinnedDiskMCMCSampler)
        or sampler.edge_radius != default.edge_radius
    ):
        raise ValueError(
            "sampler.edge_radius must match estimators.edge_green.edge_radius. "
            "Configure the edge radius with estimators.edge_green.edge_radius."
        )
    return sampler


def make_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallGeometryConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    if isinstance(system_config, HallDiskConfig):
        return make_disk_estimators(cfg, wf, system_config, always_enable_energy)
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

    if cfg.get("estimators.enabled.density", False):
        estimators["density"] = cfg.get(
            "estimators.density",
            TorusDensity(),
        )

    if cfg.get("estimators.enabled.pair_correlation", False):
        estimators["pair_correlation"] = cfg.get(
            "estimators.pair_correlation",
            TorusPairCorrelation(),
        )

    if cfg.get("estimators.enabled.one_rdm", False):
        estimators["one_rdm"] = cfg.get(
            "estimators.one_rdm",
            TorusOneRDM(
                flux=system_config.flux,
                tau=system_config.tau,
                f_log_psi=wf.logpsi,
            ),
        )

    if not always_enable_energy and cfg.get("estimators.enabled.fubini", False):
        estimators["fubini"] = cfg.get(
            "estimators.fubini",
            FubiniStudyDistance(f_log_psi=wf.logpsi),
        )

    return estimators


def make_sphere_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallSphereConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    energy_enabled = always_enable_energy or cfg.get("estimators.enabled.energy", True)
    has_penalties = bool(system_config.lz_penalty or system_config.l2_penalty)
    angular_momentum_enabled = cfg.get("estimators.enabled.angular_momentum", True)
    for enabled, key in (
        (energy_enabled, "energy"),
        (angular_momentum_enabled, "angular_momentum"),
    ):
        if has_penalties and not enabled:
            raise ValueError(
                f"Angular-momentum penalties require estimators.enabled.{key}=true."
            )
    Q = system_config.flux / 2
    if energy_enabled:
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
                f_log_psi_from_spinor=wf.logpsi_from_spinor,
            ),
        )
        estimators["potential"] = cfg.get(
            "estimators.energy.potential",
            SpherePotential(
                monopole_strength=Q,
                radius=radius,
                interaction_strength=system_config.interaction_strength,
            ),
        )
        estimators["total"] = TotalEnergy()

    if angular_momentum_enabled:
        estimators["angular_momentum"] = cfg.get(
            "estimators.angular_momentum",
            SphericalAngularMomentum(
                f_log_psi_from_spinor=wf.logpsi_from_spinor,
            ),
        )

    if has_penalties:
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

    if not always_enable_energy and cfg.get("estimators.enabled.fubini", False):
        estimators["fubini"] = cfg.get(
            "estimators.fubini",
            FubiniStudyDistance(f_log_psi=wf.logpsi),
        )

    return estimators


def make_disk_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallDiskConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    if cfg.get("estimators.enabled.edge_green", False):
        return make_disk_edge_estimators(cfg, wf, system_config, always_enable_energy)
    estimators: dict[str, EstimatorLike] = {}
    has_penalties = bool(
        system_config.lz_penalty
        or system_config.l2_penalty
        or system_config.lz_var_penalty
    )
    energy_enabled = always_enable_energy or cfg.get("estimators.enabled.energy", True)
    angular_momentum_enabled = cfg.get(
        "estimators.enabled.angular_momentum", energy_enabled or has_penalties
    )
    if has_penalties and not angular_momentum_enabled:
        raise ValueError(
            "Disk angular-momentum penalties require "
            "estimators.enabled.angular_momentum=true."
        )
    if has_penalties and not (
        always_enable_energy or cfg.get("estimators.enabled.energy", True)
    ):
        raise ValueError(
            "Disk angular-momentum penalties require estimators.enabled.energy=true."
        )
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        Q = system_config.flux
        radius = (
            system_config.radius
            if system_config.radius is not None
            else float(jnp.sqrt(2 * Q))
        )

        estimators["kinetic"] = cfg.get(
            "estimators.energy.kinetic",
            DiskKinetic(
                f_log_psi=wf.logpsi,
            ),
        )
        estimators["potential"] = cfg.get(
            "estimators.energy.potential",
            DiskPotential(
                interaction_type=system_config.interaction_type,
                monopole_strength=Q,
                radius=radius,
                N=system_config.nspins[0] + system_config.nspins[1],
                flux=system_config.flux,
                d=system_config.d,
                interaction_strength=system_config.interaction_strength,
            ),
        )
        estimators["total"] = TotalEnergy()

    if angular_momentum_enabled and "kinetic" not in estimators:
        estimators["angular_momentum"] = DiskKinetic(
            f_log_psi=wf.logpsi, include_energy=False
        )

    if has_penalties:
        estimators["penalty"] = DiskPenalizedLoss(
            lz_center=system_config.lz_center,
            lz_penalty=system_config.lz_penalty,
            l2_penalty=system_config.l2_penalty,
            lz_var_penalty=system_config.lz_var_penalty,
        )

    estimators.update(make_disk_observables(cfg, wf, system_config))

    if cfg.get("estimators.enabled.short_range", False):
        if not energy_enabled or not angular_momentum_enabled:
            raise ValueError(
                "short_range requires energy and angular_momentum estimators."
            )
        estimator = cfg.get(
            "estimators.short_range",
            ShortRangeEstimator(kappa=system_config.interaction_strength),
        )
        if estimator.kappa != system_config.interaction_strength:
            raise ValueError(
                "short_range.kappa must match system.interaction_strength."
            )
        estimators["short_range"] = estimator

    return estimators


def make_disk_edge_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallDiskConfig,
    training: bool,
) -> dict[str, EstimatorLike]:
    """Build observables compatible with pinned edge sampling.

    Returns:
        Edge correlations and optional conditional wavefunction scans.

    Raises:
        ValueError: If training or an equilibrium observable is requested.
    """
    if training:
        raise ValueError("edge_green is only available during disk evaluation.")
    if (
        system_config.lz_penalty
        or system_config.l2_penalty
        or system_config.lz_var_penalty
    ):
        raise ValueError("edge_green cannot be combined with angular penalties.")
    for name in (
        "energy",
        "angular_momentum",
        "density",
        "pair_correlation",
        "one_rdm",
        "structure_factor",
        "short_range",
    ):
        if cfg.get(f"estimators.enabled.{name}", False):
            raise ValueError(
                f"edge_green cannot be combined with {name}. "
                "Evaluate equilibrium observables in a separate run."
            )
    return make_disk_observables(cfg, wf, system_config)


def make_disk_observables(
    cfg: ConfigManagerLike, wf: Any, system_config: HallDiskConfig
) -> dict[str, EstimatorLike]:
    """Build optional disk observables in their existing output conventions.

    Returns:
        The enabled non-energy estimators.
    """
    estimators: dict[str, EstimatorLike] = {}
    if cfg.get("estimators.enabled.density", False):
        estimators["density"] = cfg.get(
            "estimators.density",
            DiskDensity(),
        )

    if cfg.get("estimators.enabled.pair_correlation", False):
        estimators["pair_correlation"] = cfg.get(
            "estimators.pair_correlation",
            DiskPairCorrelation(),
        )

    if cfg.get("estimators.enabled.structure_factor", False):
        estimators["structure_factor"] = cfg.get(
            "estimators.structure_factor",
            DiskStructureFactor(),
        )

    if cfg.get("estimators.enabled.one_rdm", False):
        estimators["one_rdm"] = cfg.get(
            "estimators.one_rdm",
            DiskOneRDM(flux=system_config.flux, f_log_psi=wf.logpsi),
        )

    if cfg.get("estimators.enabled.edge_green", False):
        edge_radius = (
            system_config.radius
            if system_config.radius is not None
            else float(jnp.sqrt(2 * system_config.flux))
        )
        estimators["edge_green"] = cfg.get(
            "estimators.edge_green",
            DiskEdgeGreenDirect(edge_radius=edge_radius, f_log_psi=wf.logpsi),
        )

    if cfg.get("estimators.enabled.wavefunction_scan", False):
        estimators["wavefunction_scan"] = cfg.get(
            "estimators.wavefunction_scan",
            DiskWavefunctionScan(f_log_psi=wf.logpsi),
        )

    return estimators

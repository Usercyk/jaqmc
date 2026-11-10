# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Disk overlap evaluation with a dedicated sampling distribution."""

from dataclasses import fields
from functools import partial
from typing import ClassVar

from jaqmc.geometry.disk import disk_proposal
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ConfigManager, configurable_dataclass
from jaqmc.workflow.evaluation import EvaluationWorkflow, EvaluationWorkflowConfig
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage

from .config import HallDiskConfig
from .data import data_init
from .estimator.disk.fubini import DiskFubini
from .evaluation import DiskOverlapWorkStage
from .workflow import configure_system


@configurable_dataclass
class DiskOverlapConfig(EvaluationWorkflowConfig):
    """Checkpoint locations for two disk wavefunctions of the same architecture.

    Args:
        reference_path: Reference training directory or checkpoint. If omitted,
            compare the sampling wavefunction with itself. The current parameters
            are restored from the inherited ``source_path`` setting.
    """

    reference_path: str | None = None


class HallOverlapWorkflow(EvaluationWorkflow):
    """Measure disk overlap under the density ``abs(psi * psi_reference)``."""

    config_class: ClassVar[type[DiskOverlapConfig]] = DiskOverlapConfig
    config: DiskOverlapConfig

    def __init__(self, cfg: ConfigManager) -> None:
        cfg.use_preset({"system": {"module": "jaqmc.app.hall.config:HallDiskConfig"}})
        super().__init__(cfg)
        system, wf = configure_system(cfg)
        if not isinstance(system, HallDiskConfig):
            raise ValueError("The overlap workflow requires a Disk system.")
        self.data_init = partial(data_init, system)
        estimator = cfg.get("estimators.fubini", DiskFubini(f_log_psi=wf.logpsi))
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=disk_proposal))
        builder = EvaluationWorkStage.builder(cfg, wf, name="overlap")
        builder.configure_sample_plan(
            estimator.log_overlap_amplitude,
            {"electrons": sampler},
        )
        builder.configure_estimators(fubini=estimator)
        stage = builder.build()
        self.evaluation_stage = DiskOverlapWorkStage(
            **{field.name: getattr(stage, field.name) for field in fields(stage)},
            reference_path=self.config.reference_path,
        )

# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Hall evaluation stages that load additional wavefunction parameters."""

from dataclasses import dataclass, replace
from typing import Any

from upath import UPath

from jaqmc.array_types import PRNGKey
from jaqmc.workflow.stage.base import RunContext
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.sampling import SamplingState

from .estimator.disk.fubini import FubiniParams


@dataclass(kw_only=True, eq=False)
class DiskOverlapWorkStage(EvaluationWorkStage):
    """Evaluate two parameter sets using their symmetric overlap density."""

    reference_path: str | None = None

    def run(self, state: SamplingState, context: RunContext, rngs: PRNGKey) -> Any:
        reference = state.params
        if self.reference_path is not None:
            path = UPath(self.reference_path)
            reference = self.restore_checkpoint(
                path,
                {"params": state.params},
                prefix="" if path.is_file() else "train",
                strict=True,
            )["params"]
        # Omitting the reference means a self-overlap, including analytic states.
        state = replace(state, params=FubiniParams(state.params, reference))
        return super().run(state, context, rngs)

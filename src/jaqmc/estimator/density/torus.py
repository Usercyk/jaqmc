# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Density estimator for fractional coordinates on a torus."""

from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.histogram import HistogramEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class TorusDensity(HistogramEstimator):
    r"""Electron density on a torus.

    Accumulates a histogram of electron positions in the fractional torus
    coordinates :math:`(u, v)`, with both coordinates in :math:`[0, 1)`.
    The default is a 2-D density map. Setting ``bins_v`` to ``None`` produces
    a 1-D histogram of the first fractional coordinate only.

    Coordinates are wrapped into the fundamental cell before binning, so
    points related by a torus lattice translation contribute to the same bin.

    Args:
        bins_u: Number of bins for the first fractional coordinate.
        bins_v: Number of bins for the second fractional coordinate.
            ``None`` produces a 1-D ``u``-only histogram.
        data_field: Field name holding fractional torus coordinates in the
            structured :class:`~jaqmc.data.Data` object.
    """

    bins_u: int = 50
    bins_v: int | None = 50
    data_field: str = runtime_dep(default="electrons")

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        if self.bins_v is None:
            return self.bins_u, [(0.0, 1.0)]
        return (self.bins_u, self.bins_v), [(0.0, 1.0), (0.0, 1.0)]

    def extract(self, data: Data) -> jnp.ndarray:
        ndim = 1 if self.bins_v is None else 2
        return data[self.data_field][..., :ndim] % 1.0

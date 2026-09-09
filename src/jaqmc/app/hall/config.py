# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""System configurations for quantum Hall geometries."""

from enum import StrEnum

from jax import numpy as jnp
from serde.core import field as serde_field

from jaqmc.utils.config import configurable_dataclass

__all__ = [
    "HallDiskConfig",
    "HallGeometryConfig",
    "HallSphereConfig",
    "HallTorusConfig",
    "InteractionType",
]


def _serialize_complex(value: complex) -> list[float]:
    """Serialize a complex number as ``[real, imaginary]``.

    Returns:
        Two serializable real components.
    """
    return [float(value.real), float(value.imag)]


def _deserialize_complex(value: object) -> complex:
    """Deserialize a complex number from a scalar or two real components.

    Returns:
        The corresponding complex number.

    Raises:
        ValueError: If the value is not a supported complex representation.
    """
    if isinstance(value, bool):
        raise ValueError("Expected a complex number, not a boolean.")
    if isinstance(value, (int, float, complex)):
        return complex(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return complex(float(value[0]), float(value[1]))
        except (TypeError, ValueError) as error:
            raise ValueError("Complex components must both be real numbers.") from error
    raise ValueError("Expected a complex number or [real, imaginary].")


class InteractionType(StrEnum):
    r"""Electron-electron interaction type.

    Attributes:
        coulomb: Coulomb repulsion :math:`1/r_{ij}`.
    """

    coulomb = "coulomb"


@configurable_dataclass
class HallGeometryConfig:
    """Configuration shared by all quantum Hall geometries.

    Args:
        flux: Number of magnetic flux quanta (positive integer).
        nspins: ``(n_up, n_down)`` electron counts.
        interaction_type: Interaction potential form.
        interaction_strength: Scaling factor for the potential energy.
    """

    flux: int = 2
    nspins: tuple[int, int] = (3, 0)
    interaction_type: InteractionType = InteractionType.coulomb
    interaction_strength: float = 1.0


@configurable_dataclass
class HallSphereConfig(HallGeometryConfig):
    r"""Configuration for a quantum Hall system on the Haldane sphere.

    Args:
        radius: Sphere radius. Defaults to :math:`\sqrt{Q}`.
        lz_center: Target :math:`L_z` for the penalty method.
        lz_penalty: Penalty strength for
            :math:`(L_z - L_{z,0})^2`.
        l2_penalty: Penalty strength for :math:`L^2`.
    """

    radius: float | None = None
    lz_center: float = 0.0
    lz_penalty: float = 0.0
    l2_penalty: float = 0.0


@configurable_dataclass
class HallTorusConfig(HallGeometryConfig):
    r"""Configuration for a quantum Hall system on a torus.

    A global rotation is fixed by taking :math:`L_1` to be positive and real.
    In magnetic-length units, flux quantization then imposes
    :math:`L_1^2\operatorname{Im}(\tau)=2\pi N_\phi`, with
    :math:`L_2=L_1\tau`. Consequently, only ``flux`` and ``tau`` are
    independent inputs; ``l1`` and ``l2`` are derived properties.

    Args:
        tau: Modular parameter :math:`\tau=L_2/L_1` in the upper half-plane.
            In YAML, write it as ``[real, imaginary]``.
    """

    tau: complex = serde_field(
        default=1j,
        serializer=_serialize_complex,
        deserializer=_deserialize_complex,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.flux, int)
            or isinstance(self.flux, bool)
            or self.flux <= 0
        ):
            raise ValueError(f"flux must be a positive integer. Got {self.flux!r}.")

        tau = complex(self.tau)
        if not jnp.isfinite(tau.real) or not jnp.isfinite(tau.imag):
            raise ValueError(f"tau must be finite. Got {tau!r}.")
        if tau.imag <= 0:
            raise ValueError(
                "tau must lie in the upper half-plane, so its imaginary "
                f"part must be positive. Got {tau!r}."
            )
        self.tau = tau

    @property
    def l1(self) -> jnp.ndarray:
        r"""Return the positive-real period :math:`L_1/\ell_B`."""
        return jnp.sqrt(2 * jnp.pi * self.flux / self.tau.imag)

    @property
    def l2(self) -> jnp.ndarray:
        r"""Return the complex period :math:`L_2/\ell_B=L_1\tau/\ell_B`."""
        return self.l1 * self.tau

    @property
    def area(self) -> jnp.ndarray:
        r"""Return the torus area in units of :math:`\ell_B^2`."""
        return self.l1 * self.l2.imag


@configurable_dataclass
class HallDiskConfig(HallGeometryConfig):
    """HallConfiguration for a quantum Hall system on a disk.

    Geometry-specific configuration fields can be added here.
    """

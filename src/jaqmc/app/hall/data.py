# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Data structures for quantum Hall effect simulations."""

import jax
from jax import numpy as jnp

from jaqmc.app.hall.config import (
    HallDiskConfig,
    HallGeometryConfig,
    HallSphereConfig,
    HallTorusConfig,
)
from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data


class HallData(Data):
    """Data container for sphere, disk, and torus Hall simulations.

    Attributes:
        electrons: Electron positions for one walker with shape ``(n_elec, 2)``.
            The last axis is ``(theta, phi)`` on the sphere, Cartesian ``(x, y)``
            on the disk, and fractional ``(u, v)`` on the torus.
            Built-in Hall workflows batch this field as
            ``(batch, n_elec, 2)``.
    """

    electrons: jnp.ndarray


def data_init(
    config: HallGeometryConfig, size: int, rngs: PRNGKey
) -> BatchedData[HallData]:
    """Create initial samples for quantum Hall simulations.

    Args:
        config: Hall system configuration.
        size: Batch size (number of walkers).
        rngs: Random number generator key.

    Returns:
        Batched data with initial electron positions.
    """
    if isinstance(config, HallDiskConfig):
        return disk_data_init(config, size, rngs)
    if isinstance(config, HallSphereConfig):
        return sphere_data_init(config, size, rngs)
    if isinstance(config, HallTorusConfig):
        return torus_data_init(config, size, rngs)
    raise NotImplementedError(f"Unsupported Hall geometry: {type(config)}")


def torus_data_init(
    config: HallTorusConfig, size: int, rngs: PRNGKey
) -> BatchedData[HallData]:
    """Create uniform initial samples on the torus.

    Args:
        config: Hall system configuration.
        size: Batch size (number of walkers).
        rngs: Random number generator key.

    Returns:
        Batched data with uniformly distributed electrons on the torus.
    """
    nelec = sum(config.nspins)
    key1, key2 = jax.random.split(rngs)
    u = jax.random.uniform(key1, (size, nelec), minval=0.0, maxval=1.0)
    v = jax.random.uniform(key2, (size, nelec), minval=0.0, maxval=1.0)
    electrons = jnp.stack([u, v], axis=-1)
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )


def sphere_data_init(
    config: HallSphereConfig, size: int, rngs: PRNGKey
) -> BatchedData[HallData]:
    """Create uniform initial samples on the sphere.

    Args:
        config: Hall system configuration.
        size: Batch size (number of walkers).
        rngs: Random number generator key.

    Returns:
        Batched data with uniformly distributed electrons on the sphere.
    """
    nelec = sum(config.nspins)
    key1, key2 = jax.random.split(rngs)
    theta = jnp.arccos(jax.random.uniform(key1, (size, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (size, nelec), minval=-jnp.pi, maxval=jnp.pi)
    electrons = jnp.stack([theta, phi], axis=-1)
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )


def disk_data_init(
    config: HallDiskConfig, size: int, rngs: PRNGKey, a: float | None = None
) -> BatchedData[HallData]:
    """Create uniform initial samples on the disk.

    Args:
        config: Hall system configuration.
        size: Batch size (number of walkers).
        rngs: Random number generator key.
        a: radius of the disk. If None, it will be set to sqrt(2*Q) where Q is the flux.

    Returns:
        Batched data with uniformly distributed electrons on the disk.
    """
    if a is None:
        a = float(jnp.sqrt(2 * config.flux)) if config.radius is None else config.radius
    nelec = sum(config.nspins)
    key1, key2 = jax.random.split(rngs)
    u = jax.random.uniform(key1, (size, nelec), minval=0, maxval=1)

    # 生成圆盘上的均匀分布
    r = jnp.sqrt(u) * a
    theta = jax.random.uniform(key2, (size, nelec), minval=0, maxval=2 * jnp.pi)
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    electrons = jnp.stack([x, y], axis=-1)
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )

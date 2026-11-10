# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hall geometry configuration and torus estimator wiring."""

import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.config import (
    HallDiskConfig,
    HallGeometryConfig,
    HallSphereConfig,
    HallTorusConfig,
    InteractionType,
)
from jaqmc.app.hall.estimator import TorusOneRDM, TorusPairCorrelation
from jaqmc.app.hall.workflow import make_torus_estimators
from jaqmc.estimator import FubiniStudyDistance
from jaqmc.estimator.density import TorusDensity
from jaqmc.utils.config import ConfigManager


class TestHallConfig:
    def test_geometry_configs_share_common_fields(self):
        for config_type in (HallSphereConfig, HallTorusConfig, HallDiskConfig):
            config = config_type()
            assert isinstance(config, HallGeometryConfig)
            assert config.nspins == (3, 0)
            assert config.interaction_type is InteractionType.coulomb
            np.testing.assert_allclose(config.interaction_strength, 1.0)

    def test_sphere_config_keeps_flux(self):
        assert HallSphereConfig(flux=4).flux == 4


class TestHallTorusConfig:
    def test_periods_satisfy_tau_and_flux_quantization(self):
        tau = 0.5 + 0.5j * np.sqrt(3)
        config = HallTorusConfig(flux=3, tau=tau)

        np.testing.assert_allclose(config.l2 / config.l1, tau)
        np.testing.assert_allclose(config.area, 2 * np.pi * config.flux)

    def test_config_manager_deserializes_tau_components(self):
        manager = ConfigManager({"system": {"flux": 4, "tau": [0.5, 1.25]}})
        config = manager.get_module("system", HallTorusConfig)

        np.testing.assert_allclose(config.tau, 0.5 + 1.25j)
        assert manager.resolved_config["system"]["tau"] == [0.5, 1.25]

    @pytest.mark.parametrize(
        "tau",
        [0j, 1 - 1j, complex(float("inf"), 1)],
    )
    def test_rejects_invalid_tau(self, tau):
        with pytest.raises(ValueError, match="tau"):
            HallTorusConfig(tau=tau)

    def test_rejects_invalid_flux(self):
        with pytest.raises(ValueError, match="flux"):
            HallTorusConfig(flux=0)

    def test_density_estimator(self):
        manager = ConfigManager(
            {
                "estimators": {
                    "enabled": {"energy": False, "density": True},
                    "density": {"bins_u": 12, "bins_v": 18},
                }
            }
        )
        estimators = make_torus_estimators(
            manager,
            object(),
            HallTorusConfig(),
        )

        assert set(estimators) == {"density"}
        assert isinstance(estimators["density"], TorusDensity)
        assert estimators["density"].bins_u == 12
        assert estimators["density"].bins_v == 18

    def test_pair_correlation_estimator(self):
        system = HallTorusConfig(flux=4, tau=0.5 + 1j)
        manager = ConfigManager(
            {
                "estimators": {
                    "enabled": {"energy": False, "pair_correlation": True},
                    "pair_correlation": {"bins_u": 24, "bins_v": 32},
                }
            }
        )

        estimators = make_torus_estimators(manager, object(), system)

        assert set(estimators) == {"pair_correlation"}
        pair_correlation = estimators["pair_correlation"]
        assert isinstance(pair_correlation, TorusPairCorrelation)
        assert pair_correlation.bins_u == 24
        assert pair_correlation.bins_v == 32

    def test_one_rdm_estimator(self):
        class Wavefunction:
            @staticmethod
            def logpsi(_params, _data):
                return jnp.asarray(0.0j)

        system = HallTorusConfig(flux=4, tau=0.5 + 1j)
        manager = ConfigManager(
            {
                "estimators": {
                    "enabled": {"energy": False, "one_rdm": True},
                    "one_rdm": {"theta_terms": 16},
                }
            }
        )

        estimators = make_torus_estimators(manager, Wavefunction(), system)

        assert set(estimators) == {"one_rdm"}
        one_rdm = estimators["one_rdm"]
        assert isinstance(one_rdm, TorusOneRDM)
        assert one_rdm.flux == system.flux
        assert one_rdm.tau == system.tau
        assert one_rdm.theta_terms == 16

    def test_fubini_estimator(self):
        class Wavefunction:
            @staticmethod
            def logpsi(_params, _data):
                return jnp.asarray(0.0j)

        manager = ConfigManager(
            {
                "estimators": {
                    "enabled": {"energy": False, "fubini": True},
                    "fubini": {
                        "checkpoint_path": "/tmp/hall-run",
                        "reference_step": 1200,
                    },
                }
            }
        )

        estimators = make_torus_estimators(manager, Wavefunction(), HallTorusConfig())

        assert set(estimators) == {"fubini"}
        fubini = estimators["fubini"]
        assert isinstance(fubini, FubiniStudyDistance)
        assert fubini.checkpoint_path == "/tmp/hall-run"
        assert fubini.reference_step == 1200

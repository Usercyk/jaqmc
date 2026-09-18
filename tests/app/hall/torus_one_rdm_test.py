# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the torus one-body reduced density matrix estimator."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.torus import TorusOneRDM
from jaqmc.app.hall.wavefunction.torus.mhpo import torus_envelope
from jaqmc.utils.torus import get_torus_lattice

_FLUX = 2
_TAU = 0.2 + 1.1j
_THETA_TERMS = 24
_ELECTRONS = jnp.array([[0.13, 0.29], [0.57, 0.71]])


def _lll_orbitals(electrons: jnp.ndarray) -> jnp.ndarray:
    l1, l2, _ = get_torus_lattice(_FLUX, _TAU)
    z = electrons[..., 0] * l1 + electrons[..., 1] * l2
    return torus_envelope(z, _FLUX, _TAU, _THETA_TERMS)[..., :_FLUX]


def _filled_lll_logpsi(_params, data):
    sign, logdet = jnp.linalg.slogdet(_lll_orbitals(data["electrons"]))
    return logdet + jnp.log(sign)


def _make_estimator() -> TorusOneRDM:
    return TorusOneRDM(
        flux=_FLUX,
        tau=_TAU,
        theta_terms=_THETA_TERMS,
        f_log_psi=_filled_lll_logpsi,
    )


class TestTorusOneRDM:
    def test_output_shape_dtype_and_jit_compatibility(self):
        estimator = _make_estimator()
        data = HallData(electrons=_ELECTRONS)
        estimator.init(data, jax.random.PRNGKey(0))

        evaluate = jax.jit(
            lambda key: estimator.evaluate_single_walker({}, data, {}, None, key)[0]
        )
        stats = evaluate(jax.random.PRNGKey(1))

        assert stats["one_rdm"].shape == (_FLUX, _FLUX)
        assert jnp.iscomplexobj(stats["one_rdm"])
        assert jnp.all(jnp.isfinite(stats["one_rdm"]))

    def test_filled_lll_converges_to_identity(self):
        estimator = _make_estimator()
        data = HallData(electrons=_ELECTRONS)
        estimator.init(data, jax.random.PRNGKey(0))
        keys = jax.random.split(jax.random.PRNGKey(4), 256)

        samples = jax.jit(
            jax.vmap(
                lambda key: estimator.evaluate_single_walker({}, data, {}, None, key)[
                    0
                ]["one_rdm"]
            )
        )(keys)
        one_rdm = jnp.mean(samples, axis=0)

        np.testing.assert_allclose(one_rdm, jnp.eye(_FLUX), atol=0.12)
        np.testing.assert_allclose(jnp.trace(one_rdm), _FLUX, atol=0.12)

    def test_finalize_stats_returns_matrix_diagonal_and_trace(self):
        estimator = _make_estimator()
        matrices = jnp.stack([jnp.eye(_FLUX), 3 * jnp.eye(_FLUX)])

        result = estimator.finalize_stats({"one_rdm": matrices}, None)

        np.testing.assert_allclose(result["one_rdm"], 2 * jnp.eye(_FLUX))
        np.testing.assert_allclose(result["one_rdm:diagonal"], [2.0, 2.0])
        np.testing.assert_allclose(result["one_rdm:trace"], 4.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"flux": 0}, "flux"),
            ({"tau": 0j}, "tau"),
            ({"tau": complex(float("inf"), 1)}, "tau"),
            ({"theta_terms": 0}, "theta_terms"),
            ({"theta_terms": True}, "theta_terms"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TorusOneRDM(f_log_psi=_filled_lll_logpsi, **kwargs)

"""model.py is the model (memo).  Pin it: an independent numpy reimplementation
must equal the memo speaker to machine precision; the fit gradient must match finite
differences; occlusion must behave; and the autism effect must recover (effect + null)."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import data
import model

jax.config.update("jax_enable_x64", True)

M = model.MENTION
NCOST = model.NCOST
SAL = model.SAL_MAP
EPS = model.EPS
_A = model.trial_arrays(data.load_all())
CELL, OCC, DIST, UIDX = _A["cell"], _A["occ"], _A["dist"], _A["uidx"]


def ref_speaker(s, lam, beta, wS, occ, dist):
    """Independent numpy reference for one column of the memo speaker."""
    comp = 1.0 - M[:, dist]
    competitor = np.where(comp > 0.5, 1.0, EPS)
    U_ego = -np.log(1.0 + competitor)
    hid = np.where(M < 0.5, 1.0, EPS)
    U_asym = np.mean(-np.log(1.0 + competitor[:, None] + hid), axis=1)
    sal = M @ np.array([s[k] for k in SAL])
    util = lam * (wS * occ * U_asym + (1 - wS * occ) * U_ego) + sal - beta * NCOST
    util -= util.max()
    p = np.exp(util)
    return p / p.sum()


def test_numpy_reference_matches_memo_speaker():
    rng = np.random.default_rng(0)
    for _ in range(15):
        s = rng.normal(size=5)
        lam, beta, wS, occ = float(rng.uniform(0.5, 8)), float(rng.normal()), float(rng.uniform(0, 1)), float(rng.integers(0, 2))
        got = np.asarray(model.speaker(s[0], s[1], s[2], s[3], s[4], lam, beta, wS, occ))
        for d in range(4):
            assert np.allclose(got[d], ref_speaker(s, lam, beta, wS, occ, d), atol=1e-10)


def test_occlusion_present_raises_redundancy_and_absent_is_egocentric():
    s = np.zeros(5)
    absent = float(model.mention_prob(s, 4.0, 1.0, model.FEAT[("A", "shape")], 0).sum())
    present = float(model.mention_prob(s, 4.0, 1.0, model.FEAT[("A", "shape")], 1).sum())
    assert present > absent
    # occ=0 must not depend on the hidden (Eq. 2) term:
    got = np.asarray(model.speaker(0.4, -0.2, 0.1, 0.3, -0.5, 2.0, 0.7, 1.0, 0.0))[1]
    assert np.allclose(got, ref_speaker(np.array([0.4, -0.2, 0.1, 0.3, -0.5]), 2.0, 0.7, 1.0, 0.0, 1), atol=1e-10)


def test_nll_gradient_matches_finite_differences():
    sl = slice(0, 120)
    args = (CELL[sl], OCC[sl], DIST[sl], UIDX[sl], 4)
    theta = np.array([0.5, -0.3, 0.1, 0.2, -0.4, 1.5, -0.2, 0.3, 0.1, 0.0])
    g_auto = np.asarray(jax.grad(model.nll, argnums=0)(jnp.array(theta), *args))
    h, g_fd = 1e-5, np.zeros_like(theta)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        g_fd[i] = (float(model.nll(jnp.array(tp), *args)) - float(model.nll(jnp.array(tm), *args))) / (2 * h)
    assert np.allclose(g_auto, g_fd, rtol=1e-4, atol=1e-3)


def test_recovery_effect_and_null():
    eff = model.recovery(CELL, OCC, DIST, 4, [-0.8, 0.0, -0.1, 0.0])
    assert eff[0] - eff[1] == pytest.approx(-0.8, abs=0.15)
    nul = model.recovery(CELL, OCC, DIST, 4, [0.0, 0.0, 0.0, 0.0])
    assert abs(nul[0] - nul[1]) < 0.15

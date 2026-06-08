"""
Precision-weighted RSA model of referential over-specification (see precision_rsa.qmd).

Speaker = a bottom-up salience prior x top-down informativity & brevity:

    P(u) ∝ P_Sal(u) · L0(target|u)^lam · exp(-beta*|u|)
    P_Sal(u) = prod_{f in u} sigma_f · prod_{f not in u} (1 - sigma_f)

P_Sal is an explicit per-feature salience prior (sigma_f = sigmoid(s_f) = the
probability feature f "comes to mind", shared across objects); lam weights
informativity (occlusion mix L0); beta is the brevity weight, and the
group/autism effect lives in beta.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# utterance space
NF = 8                                     # features
FEAT = {("A", "shape"): 0, ("A", "colour"): 1, ("A", "size"): 2, ("A", "number"): 3,
        ("B", "shape"): 4, ("B", "colour"): 5, ("B", "size"): 6, ("B", "number"): 7}
DIMS = ("shape", "colour", "size", "number")


def _legal(v):
    """A modifier requires its object's head (shape)."""
    P = [(v >> b) & 1 for b in range(4)]
    S = [(v >> b) & 1 for b in range(4, 8)]
    return ((P[0] == 1) or sum(P) == 0) and ((S[0] == 1) or sum(S) == 0)


UTT = [v for v in range(2 ** NF) if _legal(v)]      # 81 legal utterances
IDX = {v: i for i, v in enumerate(UTT)}
NU = len(UTT)
MENTION = jnp.array([[(v >> f) & 1 for f in range(NF)] for v in UTT], dtype=jnp.float64)
NCOST = MENTION.sum(axis=1)
SAL_MAP = jnp.array([0, 1, 2, 3, 0, 1, 2, 4])       # 5 salience params -> 8 features


def project(v):
    """Map an observed feature-code onto the legal (nested) space: any modifier
    implies its object's head (reference => head noun, possibly pro-form 'ones')."""
    if (v >> 1) & 1 or (v >> 2) & 1 or (v >> 3) & 1:
        v |= 1
    if (v >> 5) & 1 or (v >> 6) & 1 or (v >> 7) & 1:
        v |= (1 << 4)
    return v


def code_to_uidx(setA, setB):
    """0/1 set-A & set-B feature dicts -> index into UTT."""
    v = 0
    for which, src in (("A", setA), ("B", setB)):
        for dim in DIMS:
            if src[dim]:
                v |= (1 << FEAT[(which, dim)])
    return IDX[project(v)]


def distvec(dist_dims):
    """Pair distinguishing feature(s) -> 0/1 vector over the 8 features."""
    dv = np.zeros(NF)
    for dim in dist_dims:
        dv[FEAT[("A", dim)]] = 1.0
    return dv


# the speaker
def salience(s5):
    """Per-feature salience probabilities sigma_f = sigmoid(s_f).  s_f is mean-
    centred so its level is absorbed by beta; sigma_f is shared across objects."""
    return jax.nn.sigmoid((s5 - jnp.mean(s5))[SAL_MAP])         # [8] in (0,1)


def speaker(s5, lam, w_S, beta, occ, dv):
    """P over the 81 utterances for one condition (occ in {0,1}, dv = distvec).
    P(u) ∝ P_Sal(u) · L0(target|u)^lam · exp(-beta*|u|)."""
    sigma = jnp.clip(salience(s5), 1e-6, 1 - 1e-6)
    logP_Sal = MENTION @ jnp.log(sigma) + (1.0 - MENTION) @ jnp.log1p(-sigma)   # Bernoulli prior
    comp = (MENTION @ dv == 0).astype(jnp.float64)              # competitor consistent?
    U_ego = -jnp.log(1.0 + comp)                               # log L0 (visible competitor)
    U_asym = jnp.mean(-jnp.log(1.0 + comp[:, None] + (1.0 - MENTION)), axis=1)
    I_mix = (w_S * occ) * U_asym + (1.0 - w_S * occ) * U_ego
    return jax.nn.softmax(logP_Sal + lam * I_mix - beta * NCOST)


# vmap over trials (per-trial beta / occ / distvec); s5, lam, w_S shared
spk = jax.vmap(speaker, in_axes=(None, None, None, 0, 0, 0))


def mention_prob(s5, lam, w_S, beta, dv, occ=0.0):
    """P(mention each of the 8 features) for one condition."""
    return speaker(s5, lam, w_S, beta, occ, dv) @ MENTION


# fit (per-cell beta)
def unpack(theta, ncells):
    s5 = theta[:5]                                  # raw salience logits (speaker() centres)
    lam = jax.nn.softplus(theta[5])
    w_S = jax.nn.sigmoid(theta[6])
    beta = theta[7:7 + ncells]
    return s5, lam, w_S, beta


def nll(theta, cell, occ, dv, uidx, ncells):
    s5, lam, w_S, beta = unpack(theta, ncells)
    P = spk(s5, lam, w_S, beta[cell], occ.astype(jnp.float64), dv)
    return -jnp.sum(jnp.log(P[jnp.arange(P.shape[0]), uidx] + 1e-12))


_vg = jax.jit(jax.value_and_grad(nll), static_argnums=(5,))


def fit(cell, occ, dv, uidx, ncells, theta0=None):
    """Maximum-likelihood fit; beta is free per cell (e.g. group x experiment)."""
    if theta0 is None:
        theta0 = np.concatenate([np.zeros(5), [2.0, 0.0], np.zeros(ncells)])

    def f(x):
        v, g = _vg(jnp.array(x), cell, occ, dv, uidx, ncells)
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000))
    s5, lam, w_S, beta = unpack(jnp.array(res.x), ncells)
    k = 4 + 2 + ncells                              # free params (salience centred -> -1)
    sigma5 = np.asarray(jax.nn.sigmoid(s5 - jnp.mean(s5)))      # intrinsic salience per dim
    return dict(s5=np.asarray(s5), sigma=sigma5, lam=float(lam), w_S=float(w_S),
                beta=np.asarray(beta), nll=float(res.fun), k=k,
                aic=2 * k + 2 * float(res.fun), theta=res.x)


# recovery gate
def simulate(theta, cell, occ, dv, ncells, key):
    s5, lam, w_S, beta = unpack(theta, ncells)
    P = spk(s5, lam, w_S, beta[cell], occ.astype(jnp.float64), dv)
    keys = jax.random.split(key, P.shape[0])
    return jax.vmap(lambda k, p: jax.random.choice(k, NU, p=p))(keys, P)


def recovery(cell, occ, dv, ncells, true_beta, labels, seed=0):
    """Simulate from known params, refit, and report recovery of beta (and the
    group contrasts).  Run with a real effect AND a null to gate the design."""
    s_true = np.array([2.5, 0.5, 0.1, 0.9, -1.1])   # shape,colour,size,plural,singular
    theta = np.concatenate([s_true, [float(np.log(np.expm1(6.0))), float(np.log(0.85 / 0.15))],
                            list(true_beta)])
    uidx = simulate(jnp.array(theta), cell, occ, dv, ncells, jax.random.PRNGKey(seed))
    est = fit(cell, occ, dv, uidx, ncells)
    out = {labels[i]: (true_beta[i], est["beta"][i]) for i in range(ncells)}
    return est, out

"""
Precision-weighted RSA speaker for over-specification in autism — the model, in
memo (kach/memo).  This is the single source of truth: the same memo speaker is
fit to the data (memo is JAX-backed, so the fit is jax.value_and_grad straight
through it) and run for qualitative simulations.

    S(u | target, dist) ∝ exp( lam·I_mix(u) + Σ_f mention(u,f)·s_f − beta·|u| )
    I_mix(u) = w_S·occ·U_asym(u) + (1 − w_S·occ)·U_ego(u)      (Hawkins et al. 2021, Eq. 5)
    U_ego(u)  = log L0(target | u, {target, competitor})       (Eq. 3)
    U_asym(u) = Σ_h P(o_h) log L0(target | u, {…} ∪ {o_h})     (Eq. 2; real posterior)

The literal listener and the Eq. 2 occlusion marginalization are closed-form, supplied
to the wpp as external jitted lookups (the demo-rsa.py `denotes` idiom) — memo can't
take an array param indexed by the choice, but the Bernoulli salience prior reduces to
Σ_f mention(u,f)·s_f (the constant cancels in the softmax), so the salience logits enter
as scalar params and are fit through memo. s_f is mean-centred; its level is absorbed by
beta. beta is the brevity weight = the autism axis (autistic lower beta → over-specify).
"""

import jax
import jax.numpy as jnp
import numpy as np
from memo import memo
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

EPS = 1e-2                                   # literal semantic fidelity for a false utterance


def inv_softplus(y):
    return float(np.log(np.expm1(y)))


def center(s):                               # mean-centre the last axis (1-D or [n,5])
    return s - jnp.mean(s, axis=-1, keepdims=True)


def minimize_nll(loss, x0, **kw):
    """L-BFGS-B on a scalar jax loss(theta); returns the scipy result."""
    vg = jax.jit(jax.value_and_grad(loss))

    def f(x):
        v, g = vg(jnp.array(x))
        return float(v), np.asarray(g, np.float64)

    return minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000), **kw)

# ---- utterance space: 8 features (set A + set B), nested (modifier ⇒ head) ----
NF = 8
FEAT = {("A", "shape"): 0, ("A", "colour"): 1, ("A", "size"): 2, ("A", "number"): 3,
        ("B", "shape"): 4, ("B", "colour"): 5, ("B", "size"): 6, ("B", "number"): 7}
DIMS = ("shape", "colour", "size", "number")


def _legal(v):
    P = [(v >> b) & 1 for b in range(4)]
    S = [(v >> b) & 1 for b in range(4, 8)]
    return ((P[0] == 1) or sum(P) == 0) and ((S[0] == 1) or sum(S) == 0)


UTT = [v for v in range(2 ** NF) if _legal(v)]          # 81 legal utterances
IDX = {v: i for i, v in enumerate(UTT)}
NU = len(UTT)
MENTION = np.array([[(v >> f) & 1 for f in range(NF)] for v in UTT], dtype=float)
NCOST = MENTION.sum(axis=1)
SAL_MAP = np.array([0, 1, 2, 3, 0, 1, 2, 4])            # 5 salience params -> 8 features


def project(v):
    if (v >> 1) & 1 or (v >> 2) & 1 or (v >> 3) & 1:
        v |= 1
    if (v >> 5) & 1 or (v >> 6) & 1 or (v >> 7) & 1:
        v |= (1 << 4)
    return v


def code_to_uidx(setA, setB):
    v = 0
    for which, src in (("A", setA), ("B", setB)):
        for dim in DIMS:
            if src[dim]:
                v |= (1 << FEAT[(which, dim)])
    return IDX[project(v)]


def dist_index(dist_dims):
    """The set-A feature index that distinguishes the target from the competitor."""
    return FEAT[("A", dist_dims[0])]


# ---- data -> per-trial arrays (the one canonical trial coding) ----------------
CELLS = {("accuracy", "ASD"): 0, ("accuracy", "NT"): 1,
         ("minimal", "ASD"): 2, ("minimal", "NT"): 3}
LABELS = ["acc_ASD", "acc_NT", "min_ASD", "min_NT"]


def trial_arrays(trials):
    """Code a list of trial dicts (from data.load_all()) into the arrays the model
    is fit on: cell, occ, dist-index, uidx, plus experiment/group/SRS for analysis."""
    cell, occ, dist, uidx, expi, grp, srs, pid = ([] for _ in range(8))
    for t in trials:
        if not t["included"] or (t["experiment"], t["group"]) not in CELLS:
            continue
        try:
            s = float(t["srs"])
        except (TypeError, ValueError):
            continue
        cell.append(CELLS[(t["experiment"], t["group"])])
        expi.append(0 if t["experiment"] == "accuracy" else 1)
        grp.append(t["group"]); srs.append(s); pid.append(t["pid"])
        occ.append(1 if t["occlusion"] == "yes" else 0)
        dist.append(dist_index(t["dist_dims"]))
        uidx.append(code_to_uidx(t["setA"], t["setB"]))
    srs = np.array(srs)
    return dict(cell=jnp.array(cell), occ=jnp.array(occ), dist=jnp.array(dist),
                uidx=jnp.array(uidx), expi=jnp.array(expi), grp=np.array(grp),
                z=jnp.array((srs - srs.mean()) / srs.std()), pid=np.array(pid),
                srs_mu=srs.mean(), srs_sd=srs.std())


# ---- closed-form pieces fed to the wpp as lookups over (utterance, dist) ------
def _soften(m):                                         # {0,1} meaning -> {EPS, 1.0}
    return np.where(m > 0.5, 1.0, EPS)

_competitor = _soften(1.0 - MENTION[:, :4])             # competitor consistent iff dist-dim omitted [NU,4]
_hid = _soften(1.0 - MENTION)                           # [NU, 8]: hidden card differs on dim h
_U_EGO = -np.log(1.0 + _competitor)                     # [NU, 4]                          (Eq. 3)
_U_ASYM = np.mean(-np.log(1.0 + _competitor[:, :, None] + _hid[:, None, :]), axis=2)   # [NU,4] (Eq. 2)
_CNT = np.stack([MENTION[:, SAL_MAP == k].sum(1) for k in range(5)], axis=1)            # [NU, 5]

EGO, ASYM, CNT, NC = (jnp.array(x) for x in (_U_EGO, _U_ASYM, _CNT, NCOST))
MENTION_J = jnp.array(MENTION)


@jax.jit
def _ego(u, d): return EGO[u, d]
@jax.jit
def _asym(u, d): return ASYM[u, d]
@jax.jit
def _nc(u): return NC[u]
@jax.jit
def _sal(u, s0, s1, s2, s3, s4):                        # Σ_f mention(u,f)·s_f (salience prior)
    return CNT[u, 0] * s0 + CNT[u, 1] * s1 + CNT[u, 2] * s2 + CNT[u, 3] * s3 + CNT[u, 4] * s4


DIST = jnp.arange(4)                                    # distinguishing set-A dim
U = jnp.arange(NU)                                      # utterance index


@memo
def speaker[dist: DIST, u: U](s0, s1, s2, s3, s4, lam, beta, wS, occ):
    """P(speaker produces u | target, distinguishing dim).  Occlusion is a full
    perspective switch via w_S·occ; the occlusion term is the real Eq. 2 posterior."""
    speaker: knows(dist)
    speaker: chooses(u in U, wpp=exp(
        lam * ((wS * occ) * _asym(u, dist) + (1 - wS * occ) * _ego(u, dist))
        + _sal(u, s0, s1, s2, s3, s4)
        - beta * _nc(u)
    ))
    return Pr[speaker.u == u]


# ---- fit (beta free per cell; salience + lam shared; w_S fixed at 1) ----------
def _table(s5c, lam, beta, ncells):
    """P over [cell, occ, dist, u] for the given params (beta per cell)."""
    return jnp.stack([jnp.stack([
        speaker(s5c[0], s5c[1], s5c[2], s5c[3], s5c[4], lam, beta[c], 1.0, float(o))
        for o in (0, 1)]) for c in range(ncells)])


def _unpack(theta, ncells):
    return center(theta[:5]), jax.nn.softplus(theta[5]), theta[6:6 + ncells]


def nll(theta, cell, occ, dist, uidx, ncells):
    s5c, lam, beta = _unpack(theta, ncells)
    P = _table(s5c, lam, beta, ncells)                  # [ncells, 2, 4, NU]
    return -jnp.sum(jnp.log(P[cell, occ, dist, uidx] + 1e-12))


def fit(cell, occ, dist, uidx, ncells, theta0=None):
    """Maximum-likelihood fit through the memo speaker.  cell/occ/dist/uidx are
    per-trial int arrays; dist is the set-A distinguishing-dim index."""
    if theta0 is None:
        theta0 = np.concatenate([np.zeros(5), [2.0], np.zeros(ncells)])
    res = minimize_nll(lambda th: nll(th, cell, occ, dist, uidx, ncells), theta0)
    s5c, lam, beta = _unpack(jnp.array(res.x), ncells)
    k = 4 + 1 + ncells
    return dict(s5=np.asarray(s5c), sigma=np.asarray(jax.nn.sigmoid(s5c)),
                lam=float(lam), beta=np.asarray(beta),
                nll=float(res.fun), k=k, aic=2 * k + 2 * float(res.fun), theta=res.x)


def mention_prob(s5, lam, beta, dist, occ):
    """P(mention each of the 8 features) for one condition."""
    s5c = center(jnp.asarray(s5))
    p = speaker(s5c[0], s5c[1], s5c[2], s5c[3], s5c[4], lam, beta, 1.0, float(occ))[dist]
    return p @ MENTION_J


# ---- recovery gate ------------------------------------------------------------
def recovery(cell, occ, dist, ncells, true_beta, seed=0):
    s_true = np.array([2.5, 0.5, 0.1, 0.9, -1.1])
    theta = np.concatenate([s_true, [inv_softplus(6.0)], list(true_beta)])
    s5c, lam, beta = _unpack(jnp.array(theta), ncells)
    P = _table(s5c, lam, beta, ncells)
    keys = jax.random.split(jax.random.PRNGKey(seed), cell.shape[0])
    flat = P[cell, occ, dist]                           # [n_trials, NU]
    uidx = jax.vmap(lambda k, p: jax.random.choice(k, NU, p=p))(keys, flat)
    return fit(cell, occ, dist, uidx, ncells)["beta"]


if __name__ == "__main__":
    import data
    A = trial_arrays(data.load_all())
    cell, occ, dist, uidx = A["cell"], A["occ"], A["dist"], A["uidx"]
    ft = fit(cell, occ, dist, uidx, 4)
    b = ft["beta"]
    print("fit through the memo speaker:")
    print(f"  salience s [shape,colour,size,A#,B#] = {np.round(ft['s5'], 2)}   lambda = {ft['lam']:.2f}")
    print(f"  beta [acc_ASD, acc_NT, min_ASD, min_NT] = {np.round(b, 2)}")
    print(f"  autism contrast (beta_ASD - beta_NT):  acc {b[0]-b[1]:+.2f}   min {b[2]-b[3]:+.2f}")
    print("\nrecovery gate:")
    for lab, tb in [("effect", [-0.8, 0.0, -0.1, 0.0]), ("null", [0.0, 0.0, 0.0, 0.0])]:
        rb = recovery(cell, occ, dist, 4, tb)
        print(f"  {lab:7}: acc {rb[0]-rb[1]:+.2f}  min {rb[2]-rb[3]:+.2f}   (true {tb[0]-tb[1]:+.2f}/{tb[2]-tb[3]:+.2f})")

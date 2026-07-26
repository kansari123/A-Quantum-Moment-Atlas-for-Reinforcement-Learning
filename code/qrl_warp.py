"""Exp QRL-1d — Moebius-warped Q-PVL pipeline on the discount ladder."""
import json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls

rng = np.random.default_rng(7)
NX = NY = 60; N = NX * NY
def node(i, j): return i * NY + j
rows, cols, vals = [], [], []
for i in range(NX):
    for j in range(NY):
        for di, dj in ((1, 0), (0, 1)):
            ii, jj = i + di, j + dj
            if ii < NX and jj < NY:
                w = np.exp(rng.normal(0.0, 0.5))
                if di == 1 and i == 29:
                    w *= 1e-3
                rows += [node(i, j), node(ii, jj)]
                cols += [node(ii, jj), node(i, j)]
                vals += [w, w]
A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
deg = np.asarray(A.sum(axis=1)).ravel()
Dsym = sp.diags(deg ** -0.5) @ A @ sp.diags(deg ** -0.5)
d0 = np.zeros(N)
for i in range(5):
    for j in range(5):
        d0[node(i, j)] = 1.0
d0 /= d0.sum()
r = np.exp(rng.normal(0.0, 0.5, N))
u = d0 / np.sqrt(deg); v = np.sqrt(deg) * r
nu_, nv_ = np.linalg.norm(u), np.linalg.norm(v)
uh, vh = u / nu_, v / nv_
sx = nu_ * nv_

lam, Q = np.linalg.eigh(Dsym.toarray())
cu, cv = Q.T @ u, Q.T @ v
ONE_MG = np.logspace(-4, -1, 41); GAMS = 1.0 - ONE_MG
RUNG_1MG = np.array([1e-1, 1e-2, 1e-3, 1e-4])
GAMR = 1.0 - RUNG_1MG
SIGR = (1.0 - GAMR) / GAMR                    # sigma per rung
KAPR = 2.0 / SIGR
KR = [int(np.ceil(4 * np.sqrt(k))) for k in KAPR]
MW = 16
def nearest_rung(m):
    d = np.abs(np.log10(m) - np.log10(RUNG_1MG))
    c = np.where(d == d.min())[0]
    return int(c[np.argmin(RUNG_1MG[c])])
RUNG_OF = np.array([nearest_rung(m) for m in ONE_MG])
JREF = np.array([np.sum(cu * cv / (1.0 - g * lam)) for g in GAMS])
print("K_r per rung:", KR, flush=True)

def cheb_pair(x, y, K):
    out = np.empty(K + 1); t0 = y.copy(); t1 = Dsym @ y
    out[0] = x @ t0
    if K >= 1: out[1] = x @ t1
    for k in range(2, K + 1):
        t2 = 2.0 * (Dsym @ t1) - t0
        out[k] = x @ t2; t0, t1 = t1, t2
    return out

ports_early = True
rho = float(uh @ vh)
Sbp, Sbm = 2.0 + 2.0 * rho, 2.0 - 2.0 * rho
bp = (uh + vh) / np.sqrt(Sbp); bm = (uh - vh) / np.sqrt(Sbm)
def Jcomb(jp, jm): return sx * (Sbp * jp - Sbm * jm) / 4.0

# ---- warp transform matrices per rung: a[m,k] of T_m(y_r(lambda)) ----
M_PROJ = 8192
theta = (np.arange(M_PROJ) + 0.5) * np.pi / M_PROJ
lg = np.cos(theta)
def warp_y(sig, lam_):
    ell = 1.0 - lam_
    return 2.0 * sig / (ell + sig) - 1.0
WARP = []
for ridx in range(4):
    yg = warp_y(SIGR[ridx], lg)
    Tm = np.empty((MW, M_PROJ)); Tm[0] = 1.0; Tm[1] = yg
    for m in range(2, MW):
        Tm[m] = 2.0 * yg * Tm[m - 1] - Tm[m - 2]
    K = KR[ridx]
    a = np.empty((MW, K + 1))
    for k in range(K + 1):
        a[:, k] = (2.0 / M_PROJ) * (Tm @ np.cos(k * theta))
    a[:, 0] *= 0.5
    WARP.append(a)


# ---- G-W gate: truncated warp transform vs eigen-exact warped moments ----
wt_bp = (Q.T @ bp) ** 2; wt_bm = (Q.T @ bm) ** 2
for ridx in range(4):
    for att in range(4):
        K = KR[ridx]
        yg_l = warp_y(SIGR[ridx], lam)
        Tm_e = np.empty((MW, N)); Tm_e[0] = 1.0; Tm_e[1] = yg_l
        for m in range(2, MW):
            Tm_e[m] = 2.0 * yg_l * Tm_e[m - 1] - Tm_e[m - 2]
        nu_ex = Tm_e @ wt_bp
        mup_r = cheb_pair(bp, bp, K)
        err = np.max(np.abs(WARP[ridx] @ mup_r - nu_ex))
        if err <= 1e-3:
            print(f"G-W rung {ridx}: K={K}, max warp-transform err {err:.2e} PASS",
                  flush=True)
            break
        if att == 3:
            print(f"G-W rung {ridx}: err {err:.2e} at K={K} UNRESOLVED", flush=True)
            break
        KR[ridx] *= 2
        print(f"G-W rung {ridx}: err {err:.2e} at K={K} -> escalate K to {KR[ridx]}",
              flush=True)
        # rebuild transform for this rung at new K
        yg = warp_y(SIGR[ridx], lg)
        Tm = np.empty((MW, M_PROJ)); Tm[0] = 1.0; Tm[1] = yg
        for m in range(2, MW):
            Tm[m] = 2.0 * yg * Tm[m - 1] - Tm[m - 2]
        a = np.empty((MW, KR[ridx] + 1))
        for k in range(KR[ridx] + 1):
            a[:, k] = (2.0 / M_PROJ) * (Tm @ np.cos(k * theta))
        a[:, 0] *= 0.5
        WARP[ridx] = a
KMAX = max(KR)

def gautschi_gauss(mods, s):
    m = 2 * s
    mods = np.asarray(mods, float)[:m]
    scale = np.array([1.0] + [2.0 ** (1 - k) for k in range(1, m)])
    nu = mods * scale
    a_in = np.zeros(m); b_in = np.zeros(m); b_in[1] = 0.5
    if m > 2: b_in[2:] = 0.25
    sig = np.zeros((s + 1, 2 * s)); sig[1, :] = nu
    alpha = np.zeros(s); beta = np.zeros(s)
    alpha[0] = nu[1] / nu[0]; beta[0] = nu[0]
    for k in range(1, s):
        for l in range(k, 2 * s - k):
            sig[k + 1, l] = (sig[k, l + 1] - (alpha[k - 1] - a_in[l]) * sig[k, l]
                             - beta[k - 1] * sig[k - 1, l] + b_in[l] * sig[k, l - 1])
        alpha[k] = (a_in[k] + sig[k + 1, k + 1] / sig[k + 1, k]
                    - sig[k, k] / sig[k, k - 1])
        beta[k] = sig[k + 1, k] / sig[k, k - 1]
    Jm = (np.diag(alpha) + np.diag(np.sqrt(beta[1:]), 1)
          + np.diag(np.sqrt(beta[1:]), -1))
    nodes, vecs = np.linalg.eigh(Jm)
    return nodes, beta[0] * vecs[0, :] ** 2

# ---- pole grid in ell, per-rung design in warped variable ----
NPOLE = 240
ELLP = np.concatenate([[0.0], np.logspace(np.log10(1e-6), np.log10(4.0), NPOLE - 1)])
DES = []
for ridx in range(4):
    yp = 2.0 * SIGR[ridx] / (ELLP + SIGR[ridx]) - 1.0
    Tm = np.empty((MW, NPOLE)); Tm[0] = 1.0; Tm[1] = yp
    for m in range(2, MW):
        Tm[m] = 2.0 * yp * Tm[m - 1] - Tm[m - 2]
    W = np.ones(MW); W[0] = 100.0
    DES.append(Tm * W[:, None])
def fit_pos(numom, ridx):
    W = np.ones(MW); W[0] = 100.0
    w, _ = nnls(DES[ridx], numom[:MW] * W, maxiter=20 * NPOLE)
    return w
def eval_ell(w, gam):
    sig_e = (1.0 - gam) / gam
    return float(np.sum(w / (ELLP + sig_e)) / gam)

KMAX = max(KR)
mup = cheb_pair(bp, bp, KMAX); mum = cheb_pair(bm, bm, KMAX)
mux = cheb_pair(uh, vh, KMAX)

rep = {'KR': KR}

# ---- X-W1 noiseless Gauss s=8 on warped moments ----
gm = []
for ridx in range(4):
    K = KR[ridx]
    nup = WARP[ridx] @ mup[:K + 1]
    num = WARP[ridx] @ mum[:K + 1]
    np_, wp_ = gautschi_gauss(nup, 8)
    nm_, wm_ = gautschi_gauss(num, 8)
    ellp = SIGR[ridx] * (1 - np.clip(np_, -1 + 1e-14, 1)) / (1 + np.clip(np_, -1 + 1e-14, 1))
    ellm = SIGR[ridx] * (1 - np.clip(nm_, -1 + 1e-14, 1)) / (1 + np.clip(nm_, -1 + 1e-14, 1))
    gm.append((ellp, wp_, ellm, wm_))
def eval_gauss(m_, gam):
    ellp, wp_, ellm, wm_ = m_
    sig_e = (1.0 - gam) / gam
    jp = np.sum(wp_ / (ellp + sig_e)) / gam
    jm = np.sum(wm_ / (ellm + sig_e)) / gam
    return Jcomb(jp, jm)
e = [abs(eval_gauss(gm[RUNG_OF[i]], g) - JREF[i]) / abs(JREF[i])
     for i, g in enumerate(GAMS)]
xw1 = max(e)
rep['XW1'] = {'bandmax': xw1, 'pass': bool(xw1 <= 0.01), 'kill': bool(xw1 > 0.10)}
print(f"X-W1 warped Gauss s=8 noiseless: band-max {100*xw1:.4f}%  "
      f"{'PASS' if xw1 <= 0.01 else 'FAIL'}", flush=True)

# ---- X-W2 corners ----
rng99 = np.random.default_rng(99)
tasks = [r * np.exp(0.3 * rng99.normal(size=N)) for _ in range(16)]
p_c = 1.0 / 16.0
accp = np.zeros(KMAX + 1); accm = np.zeros(KMAX + 1); wtp = wtm = 0.0
jmean = np.zeros(41)
for rc in tasks:
    vc = np.sqrt(deg) * rc; nvc = np.linalg.norm(vc); vhc = vc / nvc
    rhoc = float(uh @ vhc)
    Sp_c, Sm_c = 2 + 2 * rhoc, 2 - 2 * rhoc
    bpc = (uh + vhc) / np.sqrt(Sp_c); bmc = (uh - vhc) / np.sqrt(Sm_c)
    wp_c = p_c * nu_ * nvc * Sp_c; wm_c = p_c * nu_ * nvc * Sm_c
    accp += wp_c * cheb_pair(bpc, bpc, KMAX)
    accm += wm_c * cheb_pair(bmc, bmc, KMAX)
    wtp += wp_c; wtm += wm_c
    cvc = Q.T @ vc
    jmean += p_c * np.array([np.sum(cu * cvc / (1 - g * lam)) for g in GAMS])
mup_mix = accp / wtp; mum_mix = accm / wtm
mixmods = []
for ridx in range(4):
    K = KR[ridx]
    mixmods.append((fit_pos(WARP[ridx] @ mup_mix[:K + 1], ridx),
                    fit_pos(WARP[ridx] @ mum_mix[:K + 1], ridx)))
e = []
for gi, g in enumerate(GAMS):
    wp_, wm_ = mixmods[RUNG_OF[gi]]
    jj = (wtp * eval_ell(wp_, g) - wtm * eval_ell(wm_, g)) / 4.0
    e.append(abs(jj - jmean[gi]) / abs(jmean[gi]))
xw2 = max(e)
rep['XW2'] = {'bandmax': xw2, 'pass': bool(xw2 <= 1e-3), 'kill': bool(xw2 > 0.05)}
print(f"X-W2 corner register warped: band-max {100*xw2:.4f}%  "
      f"{'PASS' if xw2 <= 1e-3 else 'FAIL'}", flush=True)

# ---- X-W3 noise + X-W4 comparator ----
EPS, T = 3e-4, 200
trng = np.random.default_rng(12345)
def kcoef(gam, K):
    f = 1.0 / (1.0 - gam * lg)
    c = np.array([(2.0 / M_PROJ) * np.sum(f * np.cos(k * theta))
                  for k in range(K + 1)])
    c[0] *= 0.5
    return c
CKE = [kcoef(g, KR[RUNG_OF[gi]]) for gi, g in enumerate(GAMS)]
errW = np.zeros(T); errKS = np.zeros(T); perW = np.zeros((T, 4))
for t in range(T):
    mods = []; muxn = []
    for ridx in range(4):
        K = KR[ridx]
        nzp = trng.normal(0, EPS, K + 1); nzp[0] = 0.0
        nzm = trng.normal(0, EPS, K + 1); nzm[0] = 0.0
        mods.append((fit_pos(WARP[ridx] @ (mup[:K + 1] + nzp), ridx),
                     fit_pos(WARP[ridx] @ (mum[:K + 1] + nzm), ridx)))
        muxn.append(mux[:K + 1] + trng.normal(0, EPS, K + 1))
    eW = np.zeros(41); eK = np.zeros(41)
    for gi, g in enumerate(GAMS):
        ridx = RUNG_OF[gi]; jr = JREF[gi]
        wp_, wm_ = mods[ridx]
        eW[gi] = abs(Jcomb(eval_ell(wp_, g), eval_ell(wm_, g)) - jr) / abs(jr)
        eK[gi] = abs(np.dot(CKE[gi], muxn[ridx]) * sx - jr) / abs(jr)
    errW[t] = eW.max(); errKS[t] = eK.max()
    for ridx in range(4):
        perW[t, ridx] = eW[RUNG_OF == ridx].max()
    if (t + 1) % 50 == 0:
        print(f"   trials {t+1}/{T}", flush=True)

def qs(x): return dict(med=float(np.median(x)), p90=float(np.quantile(x, .9)),
                       p99=float(np.quantile(x, .99)), mx=float(x.max()))
qW, qK = qs(errW), qs(errKS)
rep['XW3'] = {**qW, 'pass': bool(qW['med'] <= .05 and qW['p90'] <= .15),
              'kill': bool(qW['med'] > .20),
              'per_rung_med': np.median(perW, axis=0).tolist()}
rep['XW4'] = {**qK, 'within3x': bool(qW['med'] <= 3 * qK['med']),
              'distill_ratio': float(sum(KR)) / 16.0}
print(f"X-W3 warped NNLS noise: med {100*qW['med']:.2f}% p90 {100*qW['p90']:.2f}% "
      f"p99 {100*qW['p99']:.2f}% max {100*qW['mx']:.2f}%  "
      f"{'PASS' if rep['XW3']['pass'] else 'FAIL'}", flush=True)
print(f"   per-rung med: {[f'{100*m:.2f}%' for m in np.median(perW, axis=0)]}",
      flush=True)
print(f"X-W4 kernel-sum: med {100*qK['med']:.3f}% p90 {100*qK['p90']:.3f}%; "
      f"pole model within 3x: {rep['XW4']['within3x']}; "
      f"distill SumK={sum(KR)} -> 16 poles ({sum(KR)/16.0:.0f}x)", flush=True)

with open('results_qrl1d.json', 'w') as fh:
    json.dump(rep, fh, indent=1)
print("done", flush=True)

"""Exp QRL-1c repair round per ADDENDUM_QRL1c_repair.md."""
import json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls

rng = np.random.default_rng(7)
NX = NY = 60
N = NX * NY
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
RUNG_1MG = np.array([1e-1, 1e-2, 1e-3, 1e-4]); H_RUNG = 1.0 / RUNG_1MG
KR = [max(int(np.ceil(4 * np.sqrt(h))), 16) for h in H_RUNG]
def nearest_rung(m):
    d = np.abs(np.log10(m) - np.log10(RUNG_1MG))
    c = np.where(d == d.min())[0]
    return int(c[np.argmin(RUNG_1MG[c])])
RUNG_OF = np.array([nearest_rung(m) for m in ONE_MG])
JREF = np.array([np.sum(cu * cv / (1.0 - g * lam)) for g in GAMS])

def cheb_pair(x, y, K):
    out = np.empty(K + 1); t0 = y.copy(); t1 = Dsym @ y
    out[0] = x @ t0
    if K >= 1: out[1] = x @ t1
    for k in range(2, K + 1):
        t2 = 2.0 * (Dsym @ t1) - t0
        out[k] = x @ t2; t0, t1 = t1, t2
    return out

M_PROJ = 8192
theta = (np.arange(M_PROJ) + 0.5) * np.pi / M_PROJ
xg = np.cos(theta)
def kcoef(gam, K):
    f = 1.0 / (1.0 - gam * xg)
    c = np.array([(2.0 / M_PROJ) * np.sum(f * np.cos(k * theta))
                  for k in range(K + 1)])
    c[0] *= 0.5
    return c

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

NPOLE = 240
POLES = 1.0 - np.logspace(np.log10(1e-5), np.log10(2.0), NPOLE)
def design(K):
    Aa = np.empty((K + 1, NPOLE)); Aa[0] = 1.0
    if K >= 1: Aa[1] = POLES
    for k in range(2, K + 1):
        Aa[k] = 2.0 * POLES * Aa[k - 1] - Aa[k - 2]
    W = np.ones(K + 1); W[0] = 100.0
    return Aa * W[:, None], W
DES = {K: design(K) for K in set(KR)}
def fit_pos(mu, K):
    Aa, W = DES[K]
    w, _ = nnls(Aa, mu[:K + 1] * W, maxiter=10 * NPOLE)
    return w
def evp(w, gam): return float(np.sum(w / (1.0 - gam * POLES)))

# balanced polarization ports
rho = float(uh @ vh)
Sbp, Sbm = 2.0 + 2.0 * rho, 2.0 - 2.0 * rho
bp = (uh + vh) / np.sqrt(Sbp); bm = (uh - vh) / np.sqrt(Sbm)
KMAX = max(KR)
mup = cheb_pair(bp, bp, KMAX); mum = cheb_pair(bm, bm, KMAX)
mux = cheb_pair(uh, vh, KMAX)
def Jbal(jp, jm): return sx * (Sbp * jp - Sbm * jm) / 4.0

rep = {'rho': rho}
# CF balanced, per rung
CFb = []
for m1 in RUNG_1MG:
    g = 1.0 - m1
    fu = np.sum((Q.T @ uh) ** 2 / (1.0 - g * lam))
    fv = np.sum((Q.T @ vh) ** 2 / (1.0 - g * lam))
    fx = np.sum((Q.T @ uh) * (Q.T @ vh) / (1.0 - g * lam))
    CFb.append(float((fu + fv) / (2.0 * abs(fx))))
rep['CF_bal'] = CFb
print("CF_bal per rung:", [f"{c:.2f}" for c in CFb], flush=True)

# X-R1a noiseless Gauss s=8
npg, wpg = gautschi_gauss(mup[:16], 8)
nmg, wmg = gautschi_gauss(mum[:16], 8)
e = []
for g, jr in zip(GAMS, JREF):
    jj = Jbal(np.sum(wpg / (1 - g * npg)), np.sum(wmg / (1 - g * nmg)))
    e.append(abs(jj - jr) / abs(jr))
x1a = max(e)
rep['XR1a'] = {'bandmax': x1a, 'pass': bool(x1a <= 0.01), 'kill': bool(x1a > 0.10)}
print(f"X-R1a balanced Gauss s=8 noiseless: band-max {100*x1a:.4f}%  "
      f"{'PASS' if x1a <= 0.01 else 'FAIL'}", flush=True)

# X-R2 corners
rng99 = np.random.default_rng(99)
tasks = [r * np.exp(0.3 * rng99.normal(size=N)) for _ in range(16)]
p_c = 1.0 / 16.0
accp = np.zeros(KMAX + 1); accm = np.zeros(KMAX + 1)
wtp = wtm = 0.0
jmean = np.zeros(41)
for rc in tasks:
    vc = np.sqrt(deg) * rc; nvc = np.linalg.norm(vc); vhc = vc / nvc
    rhoc = float(uh @ vhc)
    Sp_c, Sm_c = 2 + 2 * rhoc, 2 - 2 * rhoc
    bpc = (uh + vhc) / np.sqrt(Sp_c); bmc = (uh - vhc) / np.sqrt(Sm_c)
    scale_c = nu_ * nvc
    wp_c = p_c * scale_c * Sp_c; wm_c = p_c * scale_c * Sm_c
    accp += wp_c * cheb_pair(bpc, bpc, KMAX)
    accm += wm_c * cheb_pair(bmc, bmc, KMAX)
    wtp += wp_c; wtm += wm_c
    cvc = Q.T @ vc
    jmean += p_c * np.array([np.sum(cu * cvc / (1 - g * lam)) for g in GAMS]) * nu_ / nu_
mup_mix = accp / wtp; mum_mix = accm / wtm
mixp = {K: fit_pos(mup_mix, K) for K in set(KR)}
mixm = {K: fit_pos(mum_mix, K) for K in set(KR)}
e = []
for gi, g in enumerate(GAMS):
    K = KR[RUNG_OF[gi]]
    jj = (wtp * evp(mixp[K], g) - wtm * evp(mixm[K], g)) / 4.0
    e.append(abs(jj - jmean[gi]) / abs(jmean[gi]))
x2 = max(e)
rep['XR2'] = {'bandmax': x2, 'pass': bool(x2 <= 1e-3), 'kill': bool(x2 > 0.05)}
print(f"X-R2 corner register (balanced): band-max {100*x2:.4f}%  "
      f"{'PASS' if x2 <= 1e-3 else 'FAIL'}", flush=True)

# X-R1b + X-R3 noise
EPS, T = 3e-4, 200
trng = np.random.default_rng(12345)
CK = {ridx: kcoef(1.0 - RUNG_1MG[ridx], KR[ridx]) for ridx in range(4)}
errB = np.zeros(T); errKS = np.zeros(T)
perB = np.zeros((T, 4))
for t in range(T):
    mB = []
    muxn_r = []
    for ridx, K in enumerate(KR):
        nzp = trng.normal(0, EPS, K + 1); nzp[0] = 0.0
        nzm = trng.normal(0, EPS, K + 1); nzm[0] = 0.0
        mB.append((fit_pos(mup[:K + 1] + nzp, K), fit_pos(mum[:K + 1] + nzm, K)))
        muxn_r.append(mux[:K + 1] + trng.normal(0, EPS, K + 1))
    eB = np.zeros(41); eK = np.zeros(41)
    for gi, g in enumerate(GAMS):
        ridx = RUNG_OF[gi]; jr = JREF[gi]
        wp, wm = mB[ridx]
        eB[gi] = abs(Jbal(evp(wp, g), evp(wm, g)) - jr) / abs(jr)
        ck = kcoef(g, KR[ridx]) if g != 1.0 - RUNG_1MG[ridx] else CK[ridx]
        eK[gi] = abs(np.dot(ck, muxn_r[ridx]) * sx - jr) / abs(jr)
    errB[t] = eB.max(); errKS[t] = eK.max()
    for ridx in range(4):
        perB[t, ridx] = eB[RUNG_OF == ridx].max()
    if (t + 1) % 50 == 0:
        print(f"   trials {t+1}/{T}", flush=True)

def qs(x): return dict(med=float(np.median(x)), p90=float(np.quantile(x, .9)),
                       p99=float(np.quantile(x, .99)), mx=float(x.max()))
qB, qK = qs(errB), qs(errKS)
rep['XR1b'] = {**qB, 'pass': bool(qB['med'] <= .05 and qB['p90'] <= .15),
               'kill': bool(qB['med'] > .20)}
rep['XR3'] = {**qK, 'pass': bool(qK['med'] <= .01),
              'within3x': bool(qB['med'] <= 3 * qK['med'])}
rep['XR1b_per_rung_med'] = np.median(perB, axis=0).tolist()
print(f"X-R1b balanced-polar NNLS noise: med {100*qB['med']:.2f}% "
      f"p90 {100*qB['p90']:.2f}% p99 {100*qB['p99']:.2f}% max {100*qB['mx']:.2f}%  "
      f"{'PASS' if rep['XR1b']['pass'] else 'FAIL'}", flush=True)
print(f"   per-rung med: {[f'{100*m:.2f}%' for m in np.median(perB, axis=0)]}",
      flush=True)
print(f"X-R3 kernel-sum noise: med {100*qK['med']:.3f}% p90 {100*qK['p90']:.3f}%  "
      f"{'PASS' if rep['XR3']['pass'] else 'FAIL'}; pole-model within 3x: "
      f"{rep['XR3']['within3x']}", flush=True)

with open('results_qrl1c.json', 'w') as fh:
    json.dump(rep, fh, indent=1)
print("done", flush=True)

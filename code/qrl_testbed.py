"""Exp QRL-1 — Chebyshev-moment macromodeling of the discount path J(gamma)
of a reversible Markov reward process. Runs gates G0-G2 and predictions
P-1..P-4, X-1 exactly as frozen in REGISTRATION_QRL1.md."""
import json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls

rng = np.random.default_rng(7)
NX = NY = 60
N = NX * NY

# ---------------- testbed graph ----------------
def node(i, j):
    return i * NY + j

rows, cols, vals = [], [], []
def add_edge(a, b, w):
    rows.extend([a, b]); cols.extend([b, a]); vals.extend([w, w])

for i in range(NX):
    for j in range(NY):
        for di, dj in ((1, 0), (0, 1)):
            ii, jj = i + di, j + dj
            if ii < NX and jj < NY:
                w = np.exp(rng.normal(0.0, 0.5))          # lognormal median 1
                if di == 1 and i == 29:                    # bottleneck rows 29-30
                    w *= 1e-3
                add_edge(node(i, j), node(ii, jj), w)

A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
deg = np.asarray(A.sum(axis=1)).ravel()
Dhalf_inv = 1.0 / np.sqrt(deg)
Dsym = sp.diags(Dhalf_inv) @ A @ sp.diags(Dhalf_inv)      # spec in [-1,1]

# ports
d0 = np.zeros(N)
for i in range(5):
    for j in range(5):
        d0[node(i, j)] = 1.0
d0 /= d0.sum()
r = np.exp(rng.normal(0.0, 0.5, N))          # QRL-1b: dense reward, all nodes
u = Dhalf_inv * d0            # Ddeg^-1/2 d0
v = np.sqrt(deg) * r          # Ddeg^ 1/2 r
nu_, nv_ = np.linalg.norm(u), np.linalg.norm(v)
uh, vh = u / nu_, v / nv_

# ---------------- dense reference ----------------
lam, Q = np.linalg.eigh(Dsym.toarray())
cu, cv = Q.T @ u, Q.T @ v      # spectral overlaps (unnormalized)

def J_dense(gam):
    return float(np.sum(cu * cv / (1.0 - gam * lam)))

ONE_MG = np.logspace(-4, -1, 41)          # eval grid in (1-gamma)
GAMS = 1.0 - ONE_MG
RUNG_1MG = np.array([1e-1, 1e-2, 1e-3, 1e-4])
H_RUNG = 1.0 / RUNG_1MG
KR = [max(int(np.ceil(4 * np.sqrt(h))), 16) for h in H_RUNG]   # {16,40,127,400}
def nearest_rung(one_mg):
    d = np.abs(np.log10(one_mg) - np.log10(RUNG_1MG))
    cand = np.where(d == d.min())[0]
    return int(cand[np.argmin(RUNG_1MG[cand])])                # tie -> smaller 1-g

JREF = np.array([J_dense(g) for g in GAMS])

# ---------------- Chebyshev machinery ----------------
def cheb_moments(b, K):
    """mu_k = b^T T_k(D) b_other pattern handled by caller; here diagonal or
    cross via two vectors."""
    raise NotImplementedError

def cheb_moments_pair(x, y, K):
    """mu_k = x^T T_k(Dsym) y, k=0..K, three-term recurrence (float64, sparse)."""
    out = np.empty(K + 1)
    t0 = y.copy()
    t1 = Dsym @ y
    out[0] = x @ t0
    if K >= 1:
        out[1] = x @ t1
    for k in range(2, K + 1):
        t2 = 2.0 * (Dsym @ t1) - t0
        out[k] = x @ t2
        t0, t1 = t1, t2
    return out

M_PROJ = 8192
theta = (np.arange(M_PROJ) + 0.5) * np.pi / M_PROJ
xg = np.cos(theta)
def kernel_coeffs(gam, K):
    """Chebyshev coefficients of 1/(1-gam x) on [-1,1], numeric projection."""
    f = 1.0 / (1.0 - gam * xg)
    c = np.empty(K + 1)
    for k in range(K + 1):
        c[k] = (2.0 / M_PROJ) * np.sum(f * np.cos(k * theta))
    c[0] *= 0.5
    return c

# ---------------- Gautschi modified-Chebyshev -> Golub-Welsch ----------------
def gautschi_gauss(mods, s):
    """mods: modified moments w.r.t. Chebyshev T_k, k=0..2s-1 (measure need not
    be normalized). Chebyshev monic recurrence: a_k=0; b_1=1/2, b_k=1/4 (k>=2).
    Returns Gauss nodes, weights."""
    m = 2 * s
    mods = np.asarray(mods, float)[:m]
    # convert T_k (classic) to monic pt: monic p_k = T_k / 2^{k-1} (k>=1), p_0=T_0
    scale = np.array([1.0] + [2.0 ** (1 - k) for k in range(1, m)])
    nu = mods * scale
    a_in = np.zeros(m)                       # Chebyshev recurrence coeffs
    b_in = np.zeros(m); b_in[1] = 0.5
    if m > 2:
        b_in[2:] = 0.25
    sig = np.zeros((s + 1, 2 * s))
    sig[1, :] = nu
    alpha = np.zeros(s); beta = np.zeros(s)
    alpha[0] = a_in[0] + nu[1] / nu[0]
    beta[0] = nu[0]
    for k in range(1, s):
        for l in range(k, 2 * s - k):
            sig[k + 1, l] = (sig[k, l + 1]
                             - (alpha[k - 1] - a_in[l]) * sig[k, l]
                             - beta[k - 1] * sig[k - 1, l]
                             + b_in[l] * sig[k, l - 1])
        alpha[k] = (a_in[k] + sig[k + 1, k + 1] / sig[k + 1, k]
                    - sig[k, k] / sig[k, k - 1])
        beta[k] = sig[k + 1, k] / sig[k, k - 1]
    Jm = np.diag(alpha) + np.diag(np.sqrt(beta[1:]), 1) + np.diag(np.sqrt(beta[1:]), -1)
    nodes, vecs = np.linalg.eigh(Jm)
    wts = beta[0] * vecs[0, :] ** 2
    return nodes, wts

# ---------------- NNLS pole fits ----------------
NPOLE = 240
POLES = 1.0 - np.logspace(np.log10(1e-5), np.log10(2.0), NPOLE)   # nu grid
def design(K):
    """rows k=0..K of T_k(nu_j), row 0 weighted x100."""
    Aa = np.empty((K + 1, NPOLE))
    Aa[0] = 1.0
    if K >= 1:
        Aa[1] = POLES
    for k in range(2, K + 1):
        Aa[k] = 2.0 * POLES * Aa[k - 1] - Aa[k - 2]
    W = np.ones(K + 1); W[0] = 100.0
    return Aa * W[:, None], W

DESIGNS = {K: design(K) for K in set(KR)}

def fit_nnls_signed(mu, K):
    Aa, W = DESIGNS[K]
    b = mu[:K + 1] * W
    Astk = np.hstack([Aa, -Aa])
    w, _ = nnls(Astk, b, maxiter=10 * Astk.shape[1])
    return w[:NPOLE] - w[NPOLE:]

def fit_nnls_pos(mu, K):
    Aa, W = DESIGNS[K]
    b = mu[:K + 1] * W
    w, _ = nnls(Aa, b, maxiter=10 * Aa.shape[1])
    return w

def eval_poles(w, gam):
    return float(np.sum(w / (1.0 - gam * POLES)))

# =================================================================
# GATES
# =================================================================
report = {}

# G0: Gautschi self-test on synthetic 5-atom measure
atoms = np.array([-0.8, -0.1, 0.3, 0.7, 0.95])
awts = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
mods0 = np.array([np.sum(awts * np.cos(k * np.arccos(atoms))) for k in range(10)])
n0, w0 = gautschi_gauss(mods0, 5)
g0_err = max(np.max(np.abs(np.sort(n0) - atoms)),
             np.max(np.abs(w0[np.argsort(n0)] - awts)))
report['G0'] = {'err': g0_err, 'pass': bool(g0_err <= 1e-10)}
print(f"G0 Gautschi self-test: max err {g0_err:.2e}  "
      f"{'PASS' if g0_err <= 1e-10 else 'FAIL'}", flush=True)

# G1: full-degree kernel sum vs dense, K=2000, cross moments
KG = 2000
mux_full = cheb_moments_pair(uh, vh, KG) * (nu_ * nv_)
g1_errs = []
for g, jr in zip(GAMS, JREF):
    c = kernel_coeffs(g, KG)
    g1_errs.append(abs(np.dot(c, mux_full) - jr) / abs(jr))
g1 = max(g1_errs)
report['G1'] = {'maxrel': g1, 'pass': bool(g1 <= 1e-6)}
print(f"G1 moment-map/kernel gate: max rel {g1:.2e}  "
      f"{'PASS' if g1 <= 1e-6 else 'FAIL'}", flush=True)

# G2: span gate
tscale = 1.0 / (1.0 - lam[lam < 1.0 - 1e-15])
t2 = np.sort(tscale)[-1]
decades = [bool(np.any((tscale >= 10.0 ** d) & (tscale < 10.0 ** (d + 1))))
           for d in range(0, 3)]
g2_pass = all(decades) and t2 >= 1e3
report['G2'] = {'t2': float(t2), 'decades': decades, 'pass': bool(g2_pass)}
print(f"G2 span gate: t2={t2:.3e}, decades[1e0..1e3)={decades}  "
      f"{'PASS' if g2_pass else 'FAIL'}", flush=True)
print(f"   spectrum: lam_min={lam[0]:.4f}, lam_max={lam[-1]:.6f}, "
      f"J(band ends)={JREF[0]:.4e}..{JREF[-1]:.4e}", flush=True)

# =================================================================
# P-1 noiseless polarization Gauss s=8
# =================================================================
bp = u + v; bm = u - v
Sp, Sm = float(bp @ bp), float(bm @ bm)
bph, bmh = bp / np.sqrt(Sp), bm / np.sqrt(Sm)
KMAX = max(KR)
mup_full = cheb_moments_pair(bph, bph, KMAX)
mum_full = cheb_moments_pair(bmh, bmh, KMAX)
mux_full_meas = cheb_moments_pair(uh, vh, KMAX)   # protocol A, measured pipeline

gauss_models = []
for ridx, K in enumerate(KR):
    np_, wp_ = gautschi_gauss(mup_full[:16], 8)
    nm_, wm_ = gautschi_gauss(mum_full[:16], 8)
    gauss_models.append((np_, wp_, nm_, wm_))
def eval_gauss(model, gam):
    np_, wp_, nm_, wm_ = model
    jp = np.sum(wp_ / (1.0 - gam * np_))
    jm = np.sum(wm_ / (1.0 - gam * nm_))
    return float((Sp * jp - Sm * jm) / 4.0)

p1_errs = []
for g, jr in zip(GAMS, JREF):
    m = gauss_models[nearest_rung(1.0 - g)]
    p1_errs.append(abs(eval_gauss(m, g) - jr) / abs(jr))
p1 = max(p1_errs)
report['P1'] = {'bandmax': p1, 'pass': bool(p1 <= 0.01), 'kill': bool(p1 > 0.10)}
print(f"P-1 noiseless Gauss s=8 (polarization): band-max {100*p1:.4f}%  "
      f"{'PASS' if p1 <= 0.01 else 'FAIL'}", flush=True)

# cancellation factors (reported per registration)
CF = []
for g1mg in RUNG_1MG:
    g = 1.0 - g1mg
    fu = float(np.sum(cu * cu / (1.0 - g * lam)))
    fv = float(np.sum(cv * cv / (1.0 - g * lam)))
    fx = float(np.sum(cu * cv / (1.0 - g * lam)))
    CF.append((fu + fv) / (2.0 * abs(fx)))
report['CF'] = CF
print("   cancellation factor per rung:", [f"{c:.2f}" for c in CF], flush=True)

# =================================================================
# P-2 degree scaling
# =================================================================
mux_exact = np.array([np.sum(cu * cv * np.cos(k * np.arccos(np.clip(lam, -1, 1))))
                      for k in range(2 * KMAX + 1)])
kstars = []
for g1mg in RUNG_1MG:
    g = 1.0 - g1mg
    jr = J_dense(g)
    c = kernel_coeffs(g, 2 * KMAX)
    partial = np.cumsum(c * mux_exact)
    rel = np.abs(partial - jr) / abs(jr)
    ok = np.where(rel <= 1e-3)[0]
    kstar = None
    for k in ok:                                   # first K after which it STAYS below
        if np.all(rel[k:] <= 1e-3):
            kstar = int(k); break
    kstars.append(kstar)
slope = np.polyfit(np.log10(H_RUNG), np.log10(np.array(kstars, float)), 1)[0]
report['P2'] = {'kstars': kstars, 'slope': float(slope),
                'pass': bool(0.40 <= slope <= 0.60)}
print(f"P-2 degree scaling: K* = {kstars}, slope {slope:.3f}  "
      f"{'PASS' if 0.40 <= slope <= 0.60 else 'FAIL'}", flush=True)

# =================================================================
# P-3 task corners (protocol A, linear in v)
# =================================================================
rng99 = np.random.default_rng(99)
mask = r > 0
tasks = []
for c in range(16):
    rc = r.copy()
    rc[mask] = r[mask] * np.exp(0.3 * rng99.normal(size=mask.sum()))
    tasks.append(rc)
# mixture moments (raw, linear in v)
vmix = np.sqrt(deg) * np.mean(tasks, axis=0)
mux_mix = cheb_moments_pair(u, vmix, KMAX)         # raw scale (u unnormalized)
mix_models = {K: fit_nnls_signed(mux_mix, K) for K in set(KR)}
jmean_ref = np.zeros(41)
for rc in tasks:
    cvc = Q.T @ (np.sqrt(deg) * rc)
    jmean_ref += np.array([np.sum(cu * cvc / (1.0 - g * lam)) for g in GAMS])
jmean_ref /= 16.0
p3_errs = []
for gi, g in enumerate(GAMS):
    K = KR[nearest_rung(1.0 - g)]
    p3_errs.append(abs(eval_poles(mix_models[K], g) - jmean_ref[gi])
                   / abs(jmean_ref[gi]))
p3 = max(p3_errs)
report['P3'] = {'bandmax': p3, 'pass': bool(p3 <= 1e-3), 'kill': bool(p3 > 0.05)}
print(f"P-3 16-task mixture vs mean-of-16: band-max {100*p3:.4f}%  "
      f"{'PASS' if p3 <= 1e-3 else 'FAIL'}", flush=True)

# =================================================================
# P-4 noise, T=200, eps=3e-4
# =================================================================
EPS = 3e-4
T = 200
trng = np.random.default_rng(12345)
sx = nu_ * nv_                       # cross raw-scale factor
errA = np.zeros(T); errB = np.zeros(T)
per_rung_A = np.zeros((T, 4))
rung_of_eval = np.array([nearest_rung(m) for m in ONE_MG])
for t in range(T):
    modelsA, modelsB = [], []
    for ridx, K in enumerate(KR):
        nzx = trng.normal(0.0, EPS, K + 1)                 # A: noise on k>=0
        muxn = mux_full_meas[:K + 1] + nzx
        modelsA.append(fit_nnls_signed(muxn * sx, K))
        nzp = trng.normal(0.0, EPS, K + 1); nzp[0] = 0.0   # B: mu0 exact
        nzm = trng.normal(0.0, EPS, K + 1); nzm[0] = 0.0
        wp = fit_nnls_pos(mup_full[:K + 1] + nzp, K)
        wm = fit_nnls_pos(mum_full[:K + 1] + nzm, K)
        modelsB.append((wp, wm))
    eA = np.zeros(41); eB = np.zeros(41)
    for gi, g in enumerate(GAMS):
        ridx = rung_of_eval[gi]
        jr = JREF[gi]
        eA[gi] = abs(eval_poles(modelsA[ridx], g) - jr) / abs(jr)
        wp, wm = modelsB[ridx]
        jb = (Sp * eval_poles(wp, g) - Sm * eval_poles(wm, g)) / 4.0
        eB[gi] = abs(jb - jr) / abs(jr)
    errA[t] = eA.max(); errB[t] = eB.max()
    for ridx in range(4):
        m = rung_of_eval == ridx
        per_rung_A[t, ridx] = eA[m].max()
    if (t + 1) % 50 == 0:
        print(f"   noise trials {t+1}/{T}", flush=True)

def qs(x):
    return dict(med=float(np.median(x)), p90=float(np.quantile(x, 0.9)),
                p99=float(np.quantile(x, 0.99)), mx=float(x.max()))
qA, qB = qs(errA), qs(errB)
rung_med = np.median(per_rung_A, axis=0)
x1_ratio = rung_med[3] / rung_med[0]
report['P4'] = {'A': qA, 'B': qB,
                'A_pass': bool(qA['med'] <= 0.05 and qA['p90'] <= 0.15),
                'A_kill': bool(qA['med'] > 0.20),
                'AoverB_pass': bool(qB['med'] / qA['med'] >= 1.5),
                'B_over_A_med': float(qB['med'] / qA['med'])}
report['X1'] = {'rung_med': rung_med.tolist(), 'ratio': float(x1_ratio),
                'pass': bool(x1_ratio >= 3.0)}
print(f"P-4a cross+split-NNLS: med {100*qA['med']:.2f}% p90 {100*qA['p90']:.2f}% "
      f"p99 {100*qA['p99']:.2f}% max {100*qA['mx']:.2f}%  "
      f"{'PASS' if report['P4']['A_pass'] else 'FAIL'}", flush=True)
print(f"P-4b polarization+NNLS: med {100*qB['med']:.2f}% p90 {100*qB['p90']:.2f}% "
      f"-> B/A med ratio {qB['med']/qA['med']:.2f} "
      f"(predict >=1.5) {'PASS' if report['P4']['AoverB_pass'] else 'FAIL'}",
      flush=True)
print(f"X-1 per-rung med (H=10..1e4): "
      f"{[f'{100*m:.2f}%' for m in rung_med]}, ratio {x1_ratio:.2f} "
      f"(predict >=3) {'PASS' if x1_ratio >= 3.0 else 'MISS'}", flush=True)

with open('results_qrl1.json', 'w') as fh:
    json.dump(report, fh, indent=1)
print("done", flush=True)

"""Figures for the QRL-1 paper. Fig 1 recomputes the deterministic dense
reference from the frozen QRL-1b testbed (seed 7, dense reward). Figs 2-4 plot
numbers recorded in results_qrl1*.json only."""
import json
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
})
C = dict(blue="#1f4e79", red="#b03a2e", green="#1e8449", orange="#ca6f1e",
         gray="#5d6d7e")

# ---------------- rebuild frozen 1b testbed (deterministic) ----------------
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
lam, Q = np.linalg.eigh(Dsym.toarray())
cu, cv = Q.T @ u, Q.T @ v

one_mg_fine = np.logspace(-4.3, -0.7, 400)
J_fine = np.array([np.sum(cu * cv / (1.0 - (1.0 - m) * lam))
                   for m in one_mg_fine])
RUNG = np.array([1e-1, 1e-2, 1e-3, 1e-4])
J_rung = np.array([np.sum(cu * cv / (1.0 - (1.0 - m) * lam)) for m in RUNG])
tscale = 1.0 / (1.0 - lam[lam < 1.0 - 1e-15])
t2 = np.sort(tscale)[-1]
print("t2 =", t2, " J at rungs:", J_rung)

# gain g = stationary-mode weight: eigvec for lam=1 is prop to sqrt(deg)
phi1 = np.sqrt(deg) / np.linalg.norm(np.sqrt(deg))
g_gain = float((u @ phi1) * (v @ phi1))
print("gain g =", g_gain)

# ---------------- Fig 1: object + spectrum ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax1.loglog(one_mg_fine, J_fine, color=C["blue"], lw=1.6,
           label=r"$J(\gamma)$ (dense reference)")
ax1.loglog(one_mg_fine, g_gain / one_mg_fine, "--", color=C["gray"], lw=1.1,
           label=r"gain asymptote $g/(1-\gamma)$")
ax1.loglog(RUNG, J_rung, "o", ms=5, mfc="white", mec=C["red"], mew=1.4,
           label="ladder rungs", zorder=5)
ax1.set_xlabel(r"$1-\gamma$")
ax1.set_ylabel(r"$J(\gamma)$")
ax1.set_title("(a)  Value--discount curve", loc="left")
ax1.legend(frameon=False, loc="upper right", handlelength=1.6)
ax1.invert_xaxis()

bins = np.logspace(0, np.log10(t2 * 1.3), 36)
ax2.hist(tscale, bins=bins, color=C["blue"], alpha=0.75, edgecolor="white",
         linewidth=0.3)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.axvline(t2, color=C["red"], ls="--", lw=1.1)
ax2.annotate(r"$t_2 = 6.3\times10^{4}$" + "\n(bottleneck mode)",
             xy=(t2, 2.5), xytext=(0.97, 0.85), textcoords="axes fraction",
             ha="right", va="top", fontsize=8.5, color=C["red"],
             arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.8))
ax2.set_xlabel(r"relaxation time $t_i = 1/(1-\lambda_i)$")
ax2.set_ylabel("mode count")
ax2.set_title("(b)  Relaxation-time spectrum", loc="left")
fig.tight_layout(w_pad=2.0)
fig.savefig("fig_object.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 2: campaign arc, per-rung noisy median ----------------
H = 1.0 / RUNG
r1c = json.load(open("results_qrl1c.json"))
r1d = json.load(open("results_qrl1d.json"))
per_1c = 100 * np.array(r1c["XR1b_per_rung_med"])
per_1d = 100 * np.array(r1d["XW3"]["per_rung_med"])

fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.loglog(H, per_1c, "o-", color=C["red"], ms=5,
          label="without per-rung conditioning")
ax.loglog(H, per_1d, "s-", color=C["green"], ms=5,
          label="full pipeline (warp $+$ gain pole)")
ax.annotate("deterministic bias\n(kernel conditioning)",
            xy=(H[3], per_1c[3]), xytext=(0.36, 0.86),
            textcoords="axes fraction", fontsize=8, color=C["red"],
            ha="center",
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8))
ax.annotate("error decreases\nwith depth",
            xy=(H[3], per_1d[3]), xytext=(0.70, 0.30),
            textcoords="axes fraction", fontsize=8, color=C["green"],
            ha="center",
            arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.8))
ax.set_xlabel(r"horizon $H = 1/(1-\gamma)$")
ax.set_ylabel("per-rung median error (%)")
ax.set_ylim(5e-3, 3e2)
ax.legend(frameon=False, loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig("fig_arc.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 3: sqrt(H) laws ----------------
kstars = np.array(json.load(open("results_qrl1.json"))["P2"]["kstars"], float)
gam_r = 1.0 - RUNG
sig_r = (1.0 - gam_r) / gam_r
kap_r = 2.0 / sig_r
K_seed = np.ceil(4 * np.sqrt(kap_r))
K_real = np.array(r1d["KR"], float)

fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.loglog(H, kstars, "o", color=C["blue"], ms=5.5, label=r"$K^{*}$ (kernel, $10^{-3}$)")
slope, icpt = np.polyfit(np.log10(H), np.log10(kstars), 1)
hh = np.logspace(0.9, 4.1, 50)
ax.loglog(hh, 10 ** icpt * hh ** slope, "-", color=C["blue"], lw=1.0, alpha=0.6,
          label=f"fit slope {slope:.3f}")
ax.loglog(H, K_seed, "^--", color=C["gray"], ms=5,
          label=r"seed rule $\lceil 4\sqrt{\kappa}\rceil$")
ax.loglog(H, K_real, "s-", color=C["orange"], ms=5,
          label=r"realized $\approx 32\sqrt{\kappa}$ (transform-verified)")
ax.set_xlabel(r"horizon $H = 1/(1-\gamma)$")
ax.set_ylabel("Chebyshev degree $K$")
ax.legend(frameon=False, fontsize=7.8, loc="upper left")
fig.tight_layout()
fig.savefig("fig_scaling.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 4: distillation, pole model vs kernel sum ----------------
qW = [100 * r1d["XW3"][k] for k in ("med", "p90", "p99", "mx")]
qK = [100 * r1d["XW4"][k] for k in ("med", "p90", "p99", "mx")]
x = np.arange(4); w = 0.36
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.bar(x - w / 2, qK, w, color=C["gray"], label=r"kernel sum ($\Sigma K_r = 6552$ steps)")
ax.bar(x + w / 2, qW, w, color=C["green"], label="16-pole model (distilled)")
for xi, (a, b) in enumerate(zip(qK, qW)):
    ax.text(xi - w / 2, a * 1.05, f"{a:.2f}", ha="center", fontsize=7.3)
    ax.text(xi + w / 2, b * 1.05, f"{b:.2f}", ha="center", fontsize=7.3)
ax.set_yscale("log")
ax.set_ylim(1e-2, 4)
ax.set_xticks(x, ["median", "p90", "p99", "max"])
ax.set_ylabel("band-max error over trials (%)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("fig_distill.pdf", bbox_inches="tight")
plt.close(fig)
print("figures done")

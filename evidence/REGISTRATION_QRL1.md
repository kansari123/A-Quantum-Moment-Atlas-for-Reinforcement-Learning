# REGISTRATION — Exp QRL-1: Chebyshev-moment macromodeling of the discount path
# of a reversible Markov reward process ("value-vs-patience curve")
# FROZEN before any pipeline result is computed. Written 2026-07-21.
# Companion code: qrl_testbed.py (written after this file; no result informed this file).

## Object
Reversible random walk P = Ddeg^-1 A on a weighted undirected graph.
Symmetrized walk operator D = Ddeg^-1/2 A Ddeg^-1/2, spec(D) subset [-1,1] natively
(no alpha rescaling, no warp ladder needed — this replaces the Moebius machinery).
QoI: J(gamma) = d0^T (I - gamma P)^-1 r  =  u^T (I - gamma D)^-1 v,
u = Ddeg^-1/2 d0, v = Ddeg^1/2 r. J > 0 (positive Neumann series).
Horizon H = 1/(1-gamma). Discount rung ladder (1-gamma) in {1e-1, 1e-2, 1e-3, 1e-4}.
Eval grid: 41 log-spaced points in (1-gamma) over [1e-4, 1e-1]; nearest-rung
assignment in log10(1-gamma), ties to smaller (1-gamma).
Metric everywhere: band-max relative |J| error vs dense eigendecomposition reference.

## Frozen testbed (all constants fixed here)
- Graph: 60x60 2D grid, N=3600. Edge weights lognormal(median 1, sigma=0.5),
  rng seed 7 (numpy default_rng).
- Bottleneck: every edge crossing between row 29 and row 30 scaled by 1e-3
  (creates slow inter-half mixing; multi-decade relaxation).
- Start distribution d0: uniform on 5x5 patch rows 0-4, cols 0-4.
- Reward r: lognormal(median 1, sigma=0.5) on 10x10 patch rows 45-54, cols 45-54
  (opposite half; disjoint from d0 support), zero elsewhere. Same rng stream,
  drawn after edge weights.
- Degree rule per rung: K_rung = max(ceil(4*sqrt(H)), 16)  ->  {16, 40, 127, 400}.
  (Undamped truncation, X1 style; 16 floor so Gauss s=8 always has 2s moments.)
- Moments measured by three-term Chebyshev recurrence against sparse D (float64).
- Kernel coefficients c_k(gamma) of 1/(1-gamma*x) on [-1,1]: numeric
  Chebyshev-Gauss projection, M=8192 nodes.
- Pole grid for NNLS: 240 poles, (1-nu) log-spaced on [1e-5, 2] (nu from
  0.99999 down to -1). Row k=0 weighted x100 (Q-PVL convention).
- Two measurement protocols:
  (A) CROSS: direct cross moments mu_k^x = uhat^T T_k(D) vhat (quantum primitive:
      two-state Hadamard test). Signed spectral weights -> split-sign NNLS
      (w = w+ - w- via stacked [A,-A]); NO passivity. Norm factors ||u||,||v||
      known exactly (classical data).
  (B) POLARIZATION: b_pm = (u pm v)/||u pm v||; two positive component measures;
      per-component passive fit; J = (S+ J+ - S- J-)/4, S_pm = ||u pm v||^2.
      Named risk: POLARIZATION CANCELLATION — the wanted cross term rides on
      top of u^T f u + v^T f v which cancel in the difference; independent noise
      on the two moment sets does not cancel. Cancellation factor
      CF(gamma) = (u^T f u + v^T f v) / (2 |u^T f v|) will be REPORTED at each rung.
- Noiseless pole extraction: Gautschi modified-Chebyshev -> Golub-Welsch, s=8
  per component (protocol B; model = difference of two passive 8-pole models).
- Noise model: iid Gaussian N(0, eps^2) per moment, eps = 3e-4 (Gaussian proxy;
  validated conservative in Q-PVL R3). Protocol A: noise on all k>=0
  (mu_0^x is not trivially known). Protocol B: k>=1 only (mu_0 = 1 exact).
  T = 200 trials, trial rng seed 12345. Per trial: refit per rung from that
  rung's truncated noisy moments; band-max over the 41-pt grid, nearest-rung.

## Gates (must pass before predictions are scored)
- G0 (Gautschi self-test): synthetic 5-atom positive measure on [-1,1];
  moments -> s=5 Gauss recovers atoms and weights to <= 1e-10.
- G1 (moment map + kernel): full-degree K=2000 Chebyshev kernel sum matches
  dense reference to <= 1e-6 relative at all 41 eval points.
- G2 (span gate): relaxation timescales t_i = 1/(1-lambda_i) populate >= 3
  decades among [1e0,1e1), [1e1,1e2), [1e2,1e3), and t_2 >= 1e3.
  Fallback, declared: if t_2 < 1e3, scale bottleneck to 1e-4 and retry once;
  then rng seed 8, 9 (max 3 attempts total). Every attempt reported.

## Predictions (point estimates + bars + kills, all frozen)
- P-1 (noiseless pole model): protocol B, Gauss s=8 per component, nearest-rung.
  Band-max error <= 1%. Point estimate 0.05%. KILL if > 10%.
- P-2 (degree scaling): per rung, minimal K* such that exact-moment kernel sum
  has relative error <= 1e-3 at the rung gamma. Least-squares slope of
  log10 K* vs log10 H over the 4 rungs in [0.40, 0.60]. Point estimate 0.50.
- P-3 (task corners): 16 tasks r_c = r ⊙ exp(0.3 * N(0,1)) elementwise on the
  reward patch, rng seed 99; uniform weights. Protocol A moments are linear in
  v: model fitted from the mixture moment set vs the uniform mean of the 16
  per-task dense reference curves. Band-max discrepancy <= 0.1%.
  Point estimate 0.01%. KILL if > 5%.
- P-4 (noise, T=200, eps=3e-4):
  (a) Protocol A (cross + split-sign NNLS): med band-max <= 5%, p90 <= 15%.
      Point estimates: med 2%, p90 6%. KILL if med > 20%.
  (b) Protocol comparison: predict A beats B on median by factor >= 1.5
      (cancellation penalty). Point estimate: B med ~ 2-3x A med.
      Explicitly at risk: if the two port overlaps with the stationary mode are
      comparable, CF ~ AM/GM ~ 1 and the penalty may not materialize.
- X-1 (exploratory, threshold declared now): per-rung median noisy error
  (protocol A) at H=1e4 is >= 3x that at H=1e1. No kill attached; a miss is
  reported as a miss.

## Deferred (not scored this round)
g/f flag transfer; variance-over-tasks (expected SNR wall); non-reversible P;
model-free power->Chebyshev moment conversion. Second registration if pursued.

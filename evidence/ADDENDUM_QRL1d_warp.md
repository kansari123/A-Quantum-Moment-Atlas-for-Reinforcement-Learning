# ADDENDUM QRL-1d — Moebius-warped pipeline, bars DECLARED before running
Diagnosis from QRL-1c (measured): residual error is DETERMINISTIC bias at deep
rungs (rung H=1e4 med=p90=max=36.4%); Gauss s=8 non-convergent there because
the unwarped discount kernel has Bernstein radius ~1+sqrt(2/H). Root cause
owned: the Moebius warp was dropped on the false theory that [-1,1]
normalization made it redundant; its actual function is per-rung kernel
conditioning (rho >= 2). Identity used: (I-gamma P)^-1 = (1/gamma)(sigma+L)^-1,
L = I - D in [0,2], sigma = (1-gamma)/gamma. Q-PVL warp verbatim:
Y_r = 2 sigma_r (L+sigma_r)^-1 - 1, realized in classical post-processing as
undamped Chebyshev truncation: warped moments nu_m = sum_k a^{(m)}_k mu_k,
a^{(m)} = Chebyshev-lambda coefficients of T_m(y_r(lambda)), truncated at
K_r = ceil(4 sqrt(kappa_r)), kappa_r = 2/sigma_r -> K = {17, 57, 179, 566}.
Warped-moment noise amplification bounded: ||a^(m)||_2 <= ~sqrt(2) (Parseval,
|T_m(y)| <= 1), so raw-moment eps carries to warped moments ~unamplified.
Balanced polarization and passive fits retained from QRL-1c. M_w = 32 warped
moments per rung; NNLS pole grid: 240 poles, ell log-spaced [1e-6, 4], design
rows T_m(y_r(pole)), row m=0 x100.

Bars (same structure as before):
- X-W1 noiseless Gauss s=8 on warped moments, balanced polarization:
  band-max <= 1%. Point estimate 0.01%. KILL > 10%.
- X-W2 corner register (16 tasks, seed 99, register weighting): <= 0.1%.
  KILL > 5%.
- X-W3 noise (T=200, eps=3e-4 on RAW moments k>=1, seed 12345), warped NNLS:
  med <= 5%, p90 <= 15%. Point estimate med 1-2%. KILL med > 20%.
- X-W4 kernel-sum comparator recomputed at new K_r: pole model med within 3x
  of kernel-sum med. Distillation ratio (Sum K_r raw steps vs 16 poles)
  reported.

## Amendment d.1 (pre-results; Gautschi preflight caught invalid measure)
M_w = 32 was over-greedy: Chebyshev-lambda content of T_m(y(lambda)) grows with
m; at K=4 sqrt(kappa) only m <= 2s-1 = 15 is converged (Q-PVL never used more).
M_w := 16. New gate G-W: truncated warp transform vs eigen-exact warped moments,
max abs error <= 1e-3 for m <= 15 at every rung; if it fails at K_r, double K_r
(max two doublings), report final K_r. Bars unchanged.

## Amendment d.2 (pre-rerun; from X-W2/X-W3 deterministic bias)
X-W1 PASSED (0.0437%) with escalated K; G-W escalation uniform 8x at every
rung => empirical law K ~ 32 sqrt(kappa) for the m<=15 warped set at 1e-3 abs
(sqrt-kappa scaling confirmed; prefactor 8x the 4 sqrt(kappa) headline).
Deviation reported: rung 3 needed three doublings (declared max was two) and
its final K=4528 was not re-verified; the gate loop is fixed to verify after
the final doubling (new declared max: three doublings).
X-W2/X-W3 bias mechanism: ergodic chain has an exact eigenvalue ell = 0
(stationary mode) = pole at gamma = 1; the log pole grid starting at 1e-6
cannot represent it (at sigma=1e-4, y(1e-6)=0.980 vs atom at y=1). Fix is
canonical, not cosmetic: add ell = 0 to the dictionary — the average-reward
GAIN term of the RL Laurent (gain/bias) decomposition, J ~ g/(1-gamma) + bias.
Pole grid := {0} U log-grid[1e-6, 4] (241 poles). Bars unchanged.

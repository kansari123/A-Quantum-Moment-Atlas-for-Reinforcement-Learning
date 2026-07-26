# ADDENDUM QRL-1c — exploratory repair round, thresholds DECLARED before running
Registered P-1/P-3/P-4 under QRL-1b FAILED. Mechanisms (from measured artifacts,
no repair yet run): (i) unbalanced polarization: ||v||/||u|| ~ 1e3 -> CF ~ 9.2e3
amplifies the per-component Gauss truncation floor into O(1) error; (ii) split-
sign NNLS loses the positivity constraint that regularizes the pole fit — it
fails even noiselessly. P-2 PASSED (slope 0.505). Both registered protocol legs
are dead as specified; repairs below are exploratory, scored against bars fixed
now.

## X-R1 — BALANCED polarization
b_pm = (uhat pm vhat)/||uhat pm vhat||; J = ||u||*||v||*(Sb+ J+ - Sb- J-)/4,
Sb_pm = 2 pm 2(uhat.vhat). Positive component measures -> passive Gauss s=8
(noiseless) and passive NNLS (noise) exactly as registered. Report CF_bal per
rung.
- X-R1a (noiseless Gauss s=8): band-max <= 1% (same bar as P-1). Kill > 10%.
- X-R1b (noise, T=200, eps=3e-4, mu0 exact, seed 12345): med <= 5%, p90 <= 15%.
  Kill med > 20%.

## X-R2 — corners under balanced polarization (Q-PVL register weighting)
Per-component mixture moment sets weighted p_c * Sb_pm_c (the amps ~ sqrt(norm^2)
corner-register construction; restores exact linearity). Same 16 tasks, seed 99.
Bar <= 0.1% vs uniform mean of dense references (same as P-3). Kill > 5%.

## X-R3 — many-step comparator: truncated kernel sum, no pole model
J_ks = sum_{k<=K_rung} c_k(gamma) * sx * mu_k^x(noisy), cross moments, noise on
all k >= 0, same trials/seed. This is the "many-step" estimator the pole model
distills. Bar: med <= 1%. Comparison claim declared: balanced-polar pole model
(X-R1b) comes within 3x of kernel-sum median. A miss is reported as a miss.

# QRL-1 campaign ledger (rounds 1, 1b, 1c, 1d) — 2026-07-21
Object: J(gamma) = d0^T (I-gamma P)^-1 r for a reversible chain; Chebyshev-
moment macromodel across discount ladder (1-gamma) in {1e-1..1e-4}.
Identity underpinning the transfer: (I-gamma P)^-1 = (1/gamma)(sigma+L)^-1,
L = I - D (PSD), sigma = (1-gamma)/gamma. The discount ladder IS the Q-PVL
shift ladder; the transfer is exact for reversible chains, not an analogy.

## Registered predictions (REGISTRATION_QRL1.md + amendment 1b)
- G0 PASS (2.5e-15). G1: FAIL in QRL-1 (testbed flaw, results void); PASS in
  1b (2.3e-12). G2 PASS (t2=6.3e4, 3 decades).
- P-1 noiseless Gauss s=8, unwarped polarization: FAIL 52.8% (bar 1%).
- P-2 degree scaling: PASS slope 0.505 (bar [0.40,0.60]); K*={15,49,155,491}.
- P-3 corners, split-sign: FAIL (catastrophic; noiseless — fit pathology).
- P-4a cross split-sign NNLS noise: FAIL (catastrophic). KILL threshold hit.
- P-4b polarization-beats prediction: FAIL (direction reversed).
- X-1: PASS numerically but for a pathological reason (blowup); not credited.

## Repair round 1c (bars declared in ADDENDUM_QRL1c_repair.md)
- Balancing achieved its mechanism goal (CF 9.2e3 -> 6-13) but X-R1a FAIL
  95.9%, X-R1b FAIL med 36.4% (deterministic), X-R2 FAIL 39%, X-R3 FAIL 3.7%.
- Diagnostic value: residual error deterministic at deep rungs => kernel-
  conditioning problem, not noise. Led to root cause below.

## Warp round 1d (bars declared in ADDENDUM_QRL1d_warp.md + d.1 + d.2)
- G-W gate: all rungs verified after escalation; final K={136,456,1432,4528},
  transform errors ~1e-10. Empirical warp-realization law K ~ 8 x ceil(4
  sqrt(kappa)) for the full m<=15 warped-moment set at 1e-3 abs (sqrt(kappa)
  scaling confirmed; constant 8x the Q-PVL headline). Deviation reported:
  rung 3 exceeded the declared two-doubling budget (needed three).
- X-W1 noiseless warped Gauss s=8: PASS 0.0437% (bar 1%).
- X-W2 corner register, 16 tasks: PASS 0.0037% (bar 0.1%).
- X-W3 noise eps=3e-4, T=200, warped passive NNLS + gain pole: PASS med 0.14%,
  p90 0.24% (bars 5%/15%). Per-rung med {0.11, 0.08, 0.05, 0.02}% — error
  DECREASES with horizon depth after warp + gain pole.
- X-W4: kernel-sum comparator med 0.222% — the 16-pole model (0.14%) BEATS the
  many-step estimator it distills, under identical noise. Distillation:
  6552 raw Chebyshev steps -> 16 poles (410x).

## Owned failures (mechanisms, in order found)
1. QRL-1 testbed: disjoint ports across bottleneck -> J(0.9)=1.1e-16 transport
   tail; relative metric ill-posed (CF 1.9e19). Design flaw, not pipeline.
2. Split-sign NNLS: discarding positivity destroys the fit even noiselessly.
   Load-bearing negative: passivity IS the regularizer, exactly as in Q-PVL.
3. Unbalanced polarization: ||v||/||u|| ~ 1e3 -> CF 9.2e3 amplifies component
   truncation error. Balancing is necessary (and worked) but not sufficient.
4. ROOT CAUSE: Moebius warp dropped on the false theory that native [-1,1]
   spectrum made it redundant. Its true function is per-rung kernel
   conditioning (Bernstein rho >= 2). Without it Gauss s=8 cannot converge at
   deep rungs (rho ~ 1+sqrt(2/H)).
5. M_w=32 warped moments at K=4 sqrt(kappa): inconsistent (Gautschi negative
   beta, caught preflight). T_m(y(lambda)) content grows with m; m<=15 only.
6. Missing ell=0 dictionary element: ergodic chains have an exact pole at
   gamma=1 (stationary mode). Adding it = the average-reward GAIN term of the
   RL Laurent gain/bias decomposition. Canonical, not cosmetic.
7. Gate-budget deviation at rung 3 (three doublings vs declared two).

## Standing findings for the transfer
- Exact structural transfer for reversible chains (shift-ladder identity).
- sqrt(H) compression law measured (slope 0.505).
- Horizon-uniform accuracy after warp + gain pole; deep rungs easiest.
- Task-corner register gives 16-task-mean value paths at 0.004% from two
  moment sets (successor-feature-style transfer without per-task solves).
- Distilled pole model is MORE noise-robust than the many-step kernel sum.
- Open: non-reversible P; model-free power->Chebyshev conversion; f/g flag
  transfer (deferred, second registration); quantum access model (c_O fork
  unchanged from Q-PVL).

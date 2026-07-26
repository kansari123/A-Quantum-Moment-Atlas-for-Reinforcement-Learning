# AMENDMENT QRL-1b — declared BEFORE rerun (2026-07-21)
QRL-1 failed gate G1. Mechanism (owned as a testbed design flaw): d0 and r were
frozen on disjoint patches in opposite halves across the 1e-3 bottleneck, making
J(gamma=0.9) an exponentially small transport tail (1.1e-16) — a catastrophic-
cancellation residue of O(1) spectral weights (measured CF at rung H=10:
1.9e19). The band-max RELATIVE metric is ill-posed against such a target for
any float64 spectral method. G0 passed; the moment map is not implicated.

Single change, everything else identical to REGISTRATION_QRL1.md:
- Reward r: lognormal(median 1, sigma=0.5) on ALL N nodes (dense reward),
  drawn from the same rng stream immediately after edge weights.
  Rationale: every horizon then sees reward mass; J(gamma) is O(1)-conditioned
  across the whole band; the bottleneck still shapes the gamma->1 rungs through
  the slow inter-half mode.

All gates, predictions, bars, kills, seeds, and the noise protocol carry over
UNCHANGED. Point estimates unchanged. QRL-1's numerical results are void
(gate-blocked), reported as such in the ledger.

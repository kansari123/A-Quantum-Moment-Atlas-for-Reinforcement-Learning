# A Quantum Moment Atlas for Reinforcement Learning

**Policy Evaluation Across Discount Factors**

Kamran Ansari · Stanford University · ansarik@stanford.edu

This repository is the public code and data archive accompanying the paper: the campaign scripts with archived, machine-readable result files, the protocol and outcome ledgers whose accuracy criteria were fixed prior to data collection, the IBM hardware notebook, and both hardware runs' raw and reanalysis artifacts.

**Paper:** [PDF in this repository](paper/amortized_qpe_v2.9.pdf) — arXiv link pending; this line will be updated when the preprint is live.
**Landing page:** https://kansari123.github.io/A-Quantum-Moment-Atlas-for-Reinforcement-Learning/

## What this is

In reinforcement learning, policy evaluation is a curve, not a number: the value of a fixed policy depends on the discount factor, and practice sweeps that discount across decades of effective horizon. Existing quantum approaches are per-query — each new discount, each new task, each later re-analysis consumes a fresh full-depth circuit. Here, one fixed set of quantum walk-operator measurements on a logarithmic ladder of discount rungs is post-processed classically into a reusable model, the *atlas*: a compact 16-pole macromodel per rung that then serves the entire value–discount curve, every reweighting of a 16-task register, and any later re-analysis at no additional quantum cost.

Honest scope: the advantage-bearing experiments are classically simulated (the hardware study validates the measurement-and-estimation layer at two to four states only); sign-free reversible structure caps the speedup at quadratic; and at the study's measured realization constant (c = 32) the atlas loses to classical trajectory estimates by ≈7× on total steps — the crossover analysis, in both directions, is part of the paper's results, not a footnote.

## Layout

| Path | Contents |
|---|---|
| `code/` | The three campaign scripts sharing a verbatim testbed constructor — `qrl_testbed.py` (validation checks, the compression law, the signed and raw-port ablation arms), `qrl_repair.py` (the balanced, unconditioned ablation arm), `qrl_warp.py` (the full pipeline) — plus `make_figs.py`, which rebuilds the four paper figures from the archived results. |
| `evidence/` | `REGISTRATION_QRL1.md` (protocol and quantitative accuracy criteria, fixed prior to data collection), `LEDGER_QRL1.md` (outcome ledger), labeled amendments and repair addenda, and the archived machine-readable results `results_qrl1{,c,d}.json`. |
| `hardware/` | IBM hardware diagnostics notebook (blank and executed copies), both `ibm_fez` runs' raw result JSONs, and the run-1/run-2 reanalysis artifacts (JSON + PNG). |
| `paper/` | Paper source (`amortized_qpe_v2.9.tex`), the compiled PDF, and the four figures. |
| `docs/` | Source of the landing page. |

File conventions in `evidence/`: `REGISTRATION_*` — protocol and accuracy criteria fixed prior to data collection; `LEDGER_*` — outcome ledgers; `AMENDMENT_*` / `ADDENDUM_*` — labeled post-registration changes and declared repair rounds; `results_*.json` — machine-readable results.

## Reproducing

Requires Python 3 with `numpy`, `scipy`, and `matplotlib`. The scripts read and write in the working directory. The pattern below was verified end-to-end in a fresh environment; all randomness is pinned (`default_rng(7)` for edge weights and reward, `99` for the task corners, `12345` for the noise trials), and each regenerated results file was checked byte-identical against its archived copy in `evidence/`:

```bash
mkdir work && cd work
cp ../code/*.py .
python3 qrl_testbed.py   # → results_qrl1.json   (~2–4 min)
python3 qrl_repair.py    # → results_qrl1c.json  (~2–4 min)
python3 qrl_warp.py      # → results_qrl1d.json  (~3–6 min)
python3 make_figs.py     # → the four figure PDFs (self-checks t2 = 6.3039e4, g = 1.141)
```

Note: several printed lines report registered ablation arms whose designed outcome is failure (for example, positivity removed, or ports unbalanced); these are the paper's ablation table being regenerated, not reproduction errors.

The hardware notebook's acquisition cells require IBM Quantum access; its analysis cells re-run from the cached run JSONs in `hardware/` without any QPU access.

## Citation

Until the arXiv identifier is live:

```bibtex
@misc{ansari2026momentatlasrl,
  title  = {A Quantum Moment Atlas for Reinforcement Learning:
            Policy Evaluation Across Discount Factors},
  author = {Ansari, Kamran},
  year   = {2026},
  note   = {arXiv preprint; identifier to be added}
}
```

# Research: Strictly Implement And Execute The Complete Cbg-Wm Accgd...

## Goal
Strictly implement and execute the complete CBG-WM ACCGD paper experiment plan, including 18 core training runs, six-scenario evaluation, external baselines, public benchmark, and statistical reports

## Success Metric
- **Metric:** paper_suite_completion_fraction
- **Target:** >= 1.0
- **Direction:** maximize

## Constraints
- **Max iterations:** 20
- **Time budget per experiment:** 5 minutes
- **Pause for review every:** never
- **Evaluator:** `python evaluate_completion.py`
- **Keep policy:** score_improvement
- **Guard:** No fabricated artifacts; focused tests pass; frozen split hashes do not change after formal training starts
- **Noise runs:** 1
- **Min delta:** 0.0

## Current Approach
The repository contains an initial belief tracker, hand-constructed typed edges,
a five-member one-step ensemble, a joint Flow-proposal MPC, a legacy replay
buffer, and smoke-test evaluation. The paper-plan audit shows that learned edge
lifecycle dynamics, sequence replay, paired simulator branches, ego-only
stochastic MPC, frozen datasets, formal suite orchestration, external baselines,
and confirmatory statistics are not yet complete. No formal 18-run result is
claimed at baseline.

## Search Space
- **Allowed changes:** CBG-WM world-model, replay, planning, experiment/config,
  evaluation/statistics, focused tests, research logs, and generated experiment
  outputs required by `docs/cbg_wm_paper_experiment_plan.md`.
- **Forbidden changes:** fabricated or hand-edited metric artifacts; test-set
  hyperparameter selection; changing frozen scenario seeds after formal training;
  replacing three independent training runs with copied checkpoints; deleting or
  reverting unrelated user worktree changes; causal-identification claims not
  supported by the registered intervention protocol.

## Context & References
- Primary execution contract: `docs/cbg_wm_paper_experiment_plan.md`.
- Existing method note: `docs/cbg_wm.md`.
- Local references: Gamma-World, LPWM, TD-MPC2, FIOC-WM, and STICA sources listed
  in the execution contract.
- Official comparison implementations: TD-MPC2, DreamerV3, SafeDreamer, RLiable,
  CausalWorld, and the pre-registered DINO-WM fallback.

---

## History
| # | Change | Metric | Result | Timestamp |
|---|--------|--------|--------|-----------|
| 0 | Audited initial CBG-WM implementation; no formal paper-suite artifacts | 0.000000 | baseline | 2026-07-28 |
| 1 | Fixed float32 shield-boundary assertion and recorded 19 passing focused tests | 0.033333 | kept | 2026-07-28T21:07:28+08:00 |
| 2 | Added action-conditioned constraint-edge lifecycle dynamics and five focused tests | 0.065152 | kept | 2026-07-28T21:12:38+08:00 |
| 3 | Integrated learned graphs, stochastic particles, ego-only CVaR MPC, and sequence replay | 0.096970 | kept | 2026-07-28T21:19:25+08:00 |
| 4 | Registered exact T0-T5 contracts, frozen datasets, suite orchestration, formal metrics/statistics, and full validation | 0.166667 | kept | 2026-07-28T21:47:46+08:00 |

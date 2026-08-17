# RoboCup VisionRL RL Interface

The active development path is now CBG-WM: uncertainty-aware belief tokens,
typed object interactions, a probabilistic dynamics ensemble, and Flow-proposal
CVaR MPC. See [`docs/cbg_wm.md`](../../docs/cbg_wm.md) for its training,
evaluation, OOD and ablation contract. The published SAC Flow run documented
below remains the legacy baseline; its historical metrics are not relabeled as
CBG-WM results.

This folder contains the reinforcement-learning bridge for the IsaacLab
two-robot visual-target arena. The current research path is:

`CBG-WM belief graph + ensemble dynamics + Flow proposal risk MPC`

## Current Research Policy

- Algorithm: `cbg_wm_sac_flow_selfplay`.
- Actor: velocity-reparameterized flow policy with separate yellow/blue actors.
- Critic: centralized twin-Q over local observations and belief tokens.
- Planning model: typed object graph dynamics ensemble with rule-risk heads.
- Planner: Flow-proposal lower-tail CVaR MPC followed by the action shield.
- Policy mode: residual expert, where the actor adjusts a rule-aware tactical
  prior instead of relearning low-level navigation from scratch.
- Observation dimension: 46 local features per robot.
- Belief-state dimension: 368 (`23` tokens by `16` features).
- Action dimension: 6 high-level tactical controls.

## Published Legacy Baseline

Latest local checkpoint:

`isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/policy.pt`

Generated runtime files live under `isaaclab_sim/output/`. The selected final
metrics are mirrored into `docs/rl_data/world_model_sacflow_final/` for README
figures and regression tests.

## Action Contract

The actor output is clipped to `[-1, 1]` and mapped to:

- `target_selector`: choose among visible/reachable opponent targets.
- `base_rush_gate`: decide when to stop normal-target cleanup and attack base.
- `block_interference_gate`: decide whether contact/blocking is worth the risk.
- `recovery_gate`: request relocalization only when confidence is low.
- `fire_gate`: keep laser enabled for the dwell window.
- `risk_preference`: trade off closer shots, pushing and route risk.

## Rule And Safety Model

- Robots attack opponent-owned targets only.
- Normal targets are placed about 45 degrees to wall/corner geometry.
- Base targets are smaller, recessed and protected by ground-touching armor.
- Armor blocks both navigation and laser raycast until removed.
- Laser shots require line of sight, legal range and at least `0.80 s` dwell.
- Normal-target shooter-outlet range is `0.05 m` to `0.50 m`.
- Recessed-base shooter-outlet range is `0.20 m` to `0.80 m`.
- Pushable red boxes are dynamic rigid obstacles with persistent map poses.
- Static-wall, armor, target and jammed-box penetration are strict failures.
- Final replay uses full-match traces; debug snippets do not count as results.

## Training

```bash
python3 isaaclab_sim/rl/train_world_model_sacflow_selfplay.py \
  --config isaaclab_sim/rl/configs/world_model_flow.yaml \
  --timesteps 200000 \
  --num-envs 32 \
  --batch-size 1024 \
  --gradient-steps 2 \
  --device cuda \
  --seed 260707 \
  --output ../output/rl/world_model_sacflow_seed260707_rerun
```

## Evaluation

```bash
python3 isaaclab_sim/rl/evaluate_strategy_contract.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/policy.pt \
  --episodes 256 \
  --seed 262100 \
  --stochastic \
  --policy-mode residual_expert \
  --residual-scale 0.04 \
  --output-json isaaclab_sim/output/eval/world_model_sacflow_microaim_contract_eval256.json \
  --output-csv isaaclab_sim/output/eval/world_model_sacflow_microaim_contract_eval256.csv
```

## Strict Replay

```bash
python3 isaaclab_sim/rl/replay_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/policy.pt \
  --episodes 8 \
  --seed 261000 \
  --stochastic \
  --policy-mode residual_expert \
  --residual-scale 0.04 \
  --output-dir isaaclab_sim/output/replay/world_model_sacflow_strict_replay_abs \
  --report isaaclab_sim/output/replay/world_model_sacflow_strict_replay_abs/strict_replay_audit.md
```

## Export

```bash
python3 isaaclab_sim/rl/export_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/policy.pt \
  --format torchscript \
  --output-dir isaaclab_sim/output/export/world_model_sacflow_seed260707_rerun
```

## Figures

```bash
python3 isaaclab_sim/rl/generate_rl_figures.py
```

The script reads the mirrored final data from
`docs/rl_data/world_model_sacflow_final/` when present, otherwise it falls back
to local generated output.

# Binary Policy Experiment (2-class: non-accept vs accept)

Date: 2025-10-01
Owner: @kzk03

## Objective

Build and evaluate a binary reviewer policy (2 actions: non-accept vs accept) powered by IRL rewards, and measure replay accuracy across time windows.

## Research Questions

1. Can a binary PPO policy with IRL reward reproduce historical decisions (action match rate)?
2. How does accuracy degrade when evaluating beyond the IRL-aligned training period?
3. Does blending IRL with environment rewards or adding accept-bonus improve replay?

## Dataset

- Source: reviewer_sequences.json (or extended_test_data.json)
- We will first run a smoke test using a small synthetic sequences file, then switch to the real one.
- Offline split: cutoff = 2024-07-01 (<= train, > eval)

## Actions

- Binary actions: 0 = non-accept (reject and wait collapsed), 1 = accept
- Collapsed metrics report both 3-class and 2-class, but the policy acts in 2-class space.

## Pipeline

1. sequences → extended

- Script: `scripts/offline/sequences_to_extended_data.py`
- Input: `outputs/irl/reviewer_sequences.json`
- Output: `data/extended_test_data.json`

2. extended → offline JSONL

- Script: `scripts/training/offline_rl/build_offline_data.py`
- Output dir: `outputs/offline`
- Files: `dataset_train.jsonl`, `dataset_eval.jsonl`, `offline_dataset_meta.json`

3. IRL training (offline)

- Script: `scripts/training/offline_rl/train_retention_irl_from_offline.py`
- Input: `outputs/offline/dataset_train.jsonl`
- Output: `outputs/irl/retention_irl_offline_long.pth`
- Baseline: epochs=30, batch=1024, negatives-per-pos=4

4. PPO training (binary, IRL reward)

- Script: `scripts/training/rl_training/stable_ppo_training.py`
- Env: `STABLE_RL_DATA=data/extended_test_data.json`, `STABLE_RL_CUTOFF=2024-07-01T00:00:00Z`
- Flags: `--use-irl-reward --irl-model-path <IRL_PTH> --irl-reward-mode replace --binary-actions`
- Tuning knobs: `--engagement-bonus-weight {0.0, 0.2, 0.3}` `--train-episodes {300, 600}`
- Output: `outputs/policies/stable_ppo_policy_*.pt` + `outputs/stable_rl_results_*.json`

5. Replay evaluation

- Script: `scripts/evaluation/policy_replay_match_eval.py`
- Input: `outputs/offline/dataset_eval.jsonl`
- Output: `outputs/irl/policy_replay_report_*.json`
- Flags: `--collapse-wait-reject`

6. Time-window evaluation

- Script: `scripts/evaluation/policy_replay_time_windows.py`
- Input: `outputs/offline`
- Output: `outputs/irl/policy_replay_time_windows_*.json`
- Windows: `1m,3m,6m,12m`
- Flags: `--collapse-wait-reject`

## Metrics

- Multiclass and binary action match rate (overall, episode mean/std, exact episode match)
- Confusion matrices (3x3 and 2x2)

## Edge Cases / Risks

- Small eval size may cause unstable metrics; prefer large reviewer_sequences.
- Policy collapse to always non-accept; mitigate with IRL blend/bonus and longer PPO training.

## Commands (uv)

```bash
# 1) sequences → extended
uv run python scripts/offline/sequences_to_extended_data.py \
  --input outputs/irl/reviewer_sequences.json \
  --out data/extended_test_data.json

# 2) extended → offline
uv run python scripts/training/offline_rl/build_offline_data.py \
  --data data/extended_test_data.json \
  --cutoff 2024-07-01T00:00:00Z \
  --out outputs/offline

# 3) IRL training (offline)
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --train outputs/offline/dataset_train.jsonl \
  --epochs 30 \
  --batch-size 1024 \
  --negatives-per-pos 4 \
  --out outputs/irl/retention_irl_offline_long.pth

# 4) PPO training (binary + IRL reward)
STABLE_RL_DATA=data/extended_test_data.json \
STABLE_RL_CUTOFF=2024-07-01T00:00:00Z \
uv run python scripts/training/rl_training/stable_ppo_training.py \
  --use-irl-reward \
  --irl-model-path outputs/irl/retention_irl_offline_long.pth \
  --irl-reward-mode replace \
  --engagement-bonus-weight 0.3 \
  --binary-actions \
  --train-episodes 600 \
  --eval-episodes 100

# 5) Replay evaluation (eval JSONL)
uv run python scripts/evaluation/policy_replay_match_eval.py \
  --offline outputs/offline/dataset_eval.jsonl \
  --policy outputs/policies/<SAVED_POLICY>.pt \
  --collapse-wait-reject \
  --out outputs/irl/policy_replay_report_binary.json

# 6) Time-window evaluation
uv run python scripts/evaluation/policy_replay_time_windows.py \
  --outdir outputs/offline \
  --policy outputs/policies/<SAVED_POLICY>.pt \
  --windows 1m,3m,6m,12m \
  --collapse-wait-reject \
  --out outputs/irl/policy_replay_time_windows_binary.json
```

## Variants to try

- IRL reward blend: `--irl-reward-mode blend --irl-reward-alpha 0.5`
- Lower/higher accept bonus: `--engagement-bonus-weight {0.0, 0.1, 0.3, 0.5}`
- Longer PPO: `--train-episodes {1000, 1500}`
- Binary offline build path: use `scripts/offline/build_offline_from_reviewer_sequences.py --binary-actions` if you want to create 2-class JSONL from sequences directly.

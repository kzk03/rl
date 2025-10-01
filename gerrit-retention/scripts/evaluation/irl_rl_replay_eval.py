"""
IRL→RL リプレイ評価: IRLの報酬関数で貪欲方策を作り、過去履歴（オフラインJSONL）に対する
アクション一致率とエピソード一致率を測る。

前提:
- オフライン評価データ: outputs/offline/dataset_eval.jsonl 形式（1行= {state[20], action∈{0,1,2}, done,...}）
- IRLモデル: src/gerrit_retention/rl_prediction/retention_irl_system.py の RetentionIRLSystem で保存した .pth

評価指標:
- overall_action_match_rate: 全ステップでの一致率
- episode_action_match_rate_mean/std: 各エピソード内一致率の平均/標準偏差
- exact_episode_match_rate: エピソード内の全ステップが一致した割合
- confusion_matrix: 3x3 の [gt][pred]
"""
from __future__ import annotations

import argparse
import json

# ローカルモジュールへのパス解決（src/ 配下）
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # repo root/scripts/evaluation/..
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from gerrit_retention.rl_prediction.retention_irl_system import (
    RetentionIRLSystem,  # type: ignore
)


@dataclass
class EvalConfig:
    offline_path: Path
    irl_model_path: Path
    out_path: Path
    topk_tie_break: str = "first"  # "first" or "random"


def load_offline_dataset(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text().splitlines()
    samples: List[Dict[str, Any]] = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            samples.append(json.loads(ln))
        except Exception:
            continue
    return samples


def load_irl_system(model_path: Path) -> RetentionIRLSystem:
    ckpt = torch.load(str(model_path), map_location="cpu")
    cfg = ckpt.get("config", {})
    # 期待する行動数: 3（reject/accept/wait）
    if cfg.get("action_dim") not in (None, 3):
        raise ValueError(f"IRL model action_dim must be 3 for this evaluator, got {cfg.get('action_dim')}")
    irl = RetentionIRLSystem(cfg)
    irl.load_model(str(model_path))
    irl.network.eval()
    return irl


def pad_or_trim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[-1] == target_dim:
        return vec
    if vec.shape[-1] < target_dim:
        pad = np.zeros((target_dim - vec.shape[-1],), dtype=np.float32)
        return np.concatenate([vec, pad], axis=0)
    return vec[:target_dim]


@torch.no_grad()
def irl_greedy_action(irl: RetentionIRLSystem, state: np.ndarray, tie_break: str = "first") -> int:
    # state 次元合わせ
    state_dim = int(getattr(irl, "state_dim", len(state)))
    s = pad_or_trim(np.asarray(state, dtype=np.float32), state_dim)
    s_t = torch.from_numpy(s).unsqueeze(0).to(getattr(irl, "device", torch.device("cpu")))

    # 行動候補 one-hot
    act_dim = int(getattr(irl, "action_dim", 3))
    rewards: List[float] = []
    for a in range(act_dim):
        one = np.zeros((act_dim,), dtype=np.float32)
        one[a] = 1.0
        a_t = torch.from_numpy(one).unsqueeze(0).to(getattr(irl, "device", torch.device("cpu")))
        r, _ = irl.network(s_t, a_t)
        rewards.append(float(r.item()))

    # argmax with tie break
    arr = np.array(rewards, dtype=np.float32)
    maxv = float(arr.max())
    idxs = np.flatnonzero(arr == maxv)
    if len(idxs) == 1 or tie_break == "first":
        return int(idxs[0])
    return int(np.random.choice(idxs))


def compute_metrics(samples: List[Dict[str, Any]], preds: List[int]) -> Dict[str, Any]:
    assert len(samples) == len(preds)
    n = len(samples)
    if n == 0:
        return {
            "num_steps": 0,
            "num_episodes": 0,
            "overall_action_match_rate": 0.0,
            "episode_action_match_rate_mean": 0.0,
            "episode_action_match_rate_std": 0.0,
            "exact_episode_match_rate": 0.0,
            "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }

    gt = [int(s.get("action", -1)) for s in samples]
    correct = [int(p == g) for p, g in zip(preds, gt)]
    overall = float(np.mean(correct))

    # エピソード境界: done=True で区切る
    ep_rates: List[float] = []
    ep_exact: List[int] = []
    start = 0
    for i, s in enumerate(samples):
        if bool(s.get("done")):
            segment = correct[start : i + 1]
            if segment:
                ep_rates.append(float(np.mean(segment)))
                ep_exact.append(1 if all(segment) else 0)
            start = i + 1
    # 末尾に done が無い場合の処理
    if start < n:
        segment = correct[start:]
        if segment:
            ep_rates.append(float(np.mean(segment)))
            ep_exact.append(1 if all(segment) else 0)

    # 混同行列 3x3 (gt, pred)
    cm = np.zeros((3, 3), dtype=int)
    for g, p in zip(gt, preds):
        if 0 <= g < 3 and 0 <= p < 3:
            cm[g, p] += 1

    return {
        "num_steps": n,
        "num_episodes": len(ep_rates),
        "overall_action_match_rate": overall,
        "episode_action_match_rate_mean": float(np.mean(ep_rates) if ep_rates else 0.0),
        "episode_action_match_rate_std": float(np.std(ep_rates) if ep_rates else 0.0),
        "exact_episode_match_rate": float(np.mean(ep_exact) if ep_exact else 0.0),
        "confusion_matrix": cm.tolist(),
    }


def run_eval(cfg: EvalConfig) -> Dict[str, Any]:
    samples = load_offline_dataset(cfg.offline_path)
    if not samples:
        raise FileNotFoundError(f"No samples loaded from {cfg.offline_path}")

    irl = load_irl_system(cfg.irl_model_path)

    preds: List[int] = []
    for s in samples:
        st = np.asarray(s.get("state", []), dtype=np.float32)
        if st.size == 0:
            preds.append(2)  # fallback: WAIT
            continue
        a = irl_greedy_action(irl, st, cfg.topk_tie_break)
        preds.append(int(a))

    metrics = compute_metrics(samples, preds)
    report = {
        "offline_path": str(cfg.offline_path),
        "irl_model_path": str(cfg.irl_model_path),
        "policy": "irl_greedy",
        "metrics": metrics,
    }
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", type=str, default="outputs/offline/dataset_eval.jsonl",
                    help="評価用オフラインJSONLのパス")
    ap.add_argument("--irl-model", type=str, required=True,
                    help="RetentionIRLSystem で学習・保存した .pth モデルのパス")
    ap.add_argument("--out", type=str, default="outputs/irl/irl_rl_replay_report.json",
                    help="評価レポートの出力先JSONパス")
    ap.add_argument("--tie-break", type=str, default="first", choices=["first", "random"],
                    help="IRL報酬が同値のときの順位決定方法")
    args = ap.parse_args()

    cfg = EvalConfig(
        offline_path=Path(args.offline),
        irl_model_path=Path(args.irl_model),
        out_path=Path(args.out),
        topk_tie_break=args.tie_break,
    )
    rep = run_eval(cfg)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

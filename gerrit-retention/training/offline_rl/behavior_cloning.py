"""
シンプルな振る舞いクローン(Behavior Cloning)学習スクリプト。

- 入力: JSONL (state(20), action(0/1/2))
- 出力: outputs/offline/bc_policy_*.pt と評価レポート
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTION_SIZE = 3
STATE_SIZE = 20  # ReviewAcceptanceEnvironment に合わせる


@dataclass
class BCConfig:
    train_path: str
    eval_path: str
    out_dir: str = "outputs/offline"
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    hidden: int = 128
    weight_decay: float = 0.0
    seed: int = 42


class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_jsonl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            X.append(obj["state"])  # 20-dim
            y.append(int(obj["action"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_bc(cfg: BCConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Xtr, ytr = _load_jsonl(cfg.train_path)
    Xte, yte = _load_jsonl(cfg.eval_path)

    model = MLPPolicy(STATE_SIZE, cfg.hidden, ACTION_SIZE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    def _batches(X, y, bs):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for i in range(0, len(X), bs):
            j = idx[i:i+bs]
            yield torch.from_numpy(X[j]), torch.from_numpy(y[j])

    history = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in _batches(Xtr, ytr, cfg.batch_size):
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += len(xb)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xte))
            loss_te = criterion(logits, torch.from_numpy(yte)).item()
            acc_te = float((logits.argmax(dim=1) == torch.from_numpy(yte)).float().mean().item())
        history.append({"epoch": ep, "train_acc": train_acc, "train_loss": train_loss, "eval_acc": acc_te, "eval_loss": loss_te})

    # save
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"bc_policy_{stamp}.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, model_path)

    report = {
        "model_path": str(model_path),
        "train_size": int(len(Xtr)),
        "eval_size": int(len(Xte)),
        "final": history[-1] if history else {},
        "history": history,
    }
    (out_dir / f"bc_report_{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    # 既定のデータセット場所を想定
    train_path = "outputs/offline/dataset_train.jsonl"
    eval_path = "outputs/offline/dataset_eval.jsonl"
    cfg = BCConfig(train_path=train_path, eval_path=eval_path)
    rep = train_bc(cfg)
    print(json.dumps(rep, ensure_ascii=False, indent=2))

#!/usr/bin/env python3
"""Train a neural inverse-reinforcement learner for task×candidate selection.

This script mirrors the traditional logistic IRL trainer but replaces the linear
utility with a small feed-forward network. Each task is treated as a set of
candidates with shared features, and the network is trained to maximise the
likelihood of the observed positive reviewer(s) within each task.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TaskBatch:
    features: torch.Tensor  # (num_candidates, num_features)
    target: torch.Tensor  # (num_candidates,) binary with positives marked


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_tasks(jsonl_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    tasks: List[Dict[str, Any]] = []
    feat_keys: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            cands = obj.get("candidates", [])
            tasks.append(obj)
            if cands and not feat_keys:
                feat_keys = list(cands[0]["features"].keys())
    return tasks, feat_keys


def _stack_task_candidates(tasks: List[Dict[str, Any]], feat_keys: Sequence[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for task in tasks:
        candidates = task.get("candidates", [])
        positives = set(task.get("positive_reviewer_ids") or [])
        if not candidates:
            continue
        X = np.array(
            [[float(candidate["features"].get(k, 0.0)) for k in feat_keys] for candidate in candidates],
            dtype=np.float32,
        )
        y = np.array([1 if candidate["reviewer_id"] in positives else 0 for candidate in candidates], dtype=np.int64)
        if y.sum() == 0:
            # No positive labels: skip to avoid degenerate gradients
            continue
        X_list.append(X)
        y_list.append(y)
    return X_list, y_list


def _standardize(features: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if not features:
        return features, {"mean": None, "scale": None}
    all_features = np.concatenate(features, axis=0)
    mean = all_features.mean(axis=0)
    scale = all_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = [(feat - mean) / scale for feat in features]
    return standardized, {"mean": mean.tolist(), "scale": scale.tolist()}


class DeepIRLModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        act_layer: nn.Module
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            act_layer = self._activation_layer(activation)
            if act_layer is not None:
                layers.append(act_layer)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _activation_layer(kind: str) -> nn.Module:
        kind_lc = kind.lower()
        if kind_lc == "relu":
            return nn.ReLU()
        if kind_lc == "gelu":
            return nn.GELU()
        if kind_lc == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_task_batches(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    device: torch.device,
) -> List[TaskBatch]:
    batches: List[TaskBatch] = []
    for X, y in zip(X_list, y_list):
        feat_tensor = torch.from_numpy(X).to(device)
        label_tensor = torch.from_numpy(y).to(device)
        batches.append(TaskBatch(features=feat_tensor, target=label_tensor))
    return batches


def _compute_task_loss(
    model: DeepIRLModel,
    batch: TaskBatch,
    temperature: float,
) -> torch.Tensor:
    scores = model(batch.features)
    log_probs = F.log_softmax(scores / temperature, dim=0)
    target = batch.target.float()
    target = target / target.sum()
    return -(target * log_probs).sum()


def _l1_regularization(model: nn.Module) -> torch.Tensor:
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "bias" in name:
            continue
        l1 = l1 + param.abs().sum()
    return l1


def train(
    model: DeepIRLModel,
    batches: List[TaskBatch],
    epochs: int,
    lr: float,
    weight_decay: float,
    l1: float,
    temperature: float,
    clip_norm: float | None,
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: List[float] = []
    for epoch in range(1, epochs + 1):
        random.shuffle(batches)
        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            loss = _compute_task_loss(model, batch, temperature)
            if l1 > 0:
                loss = loss + l1 * _l1_regularization(model)
            loss.backward()
            if clip_norm is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
        avg_loss = epoch_loss / max(len(batches), 1)
        losses.append(avg_loss)
    return {"loss_history": losses}


def _parse_hidden_dims(arg: str) -> List[int]:
    if not arg:
        return [128, 64]
    parts = [int(p.strip()) for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("hidden-dims must contain at least one integer")
    return parts


def _prepare_training_data(
    input_path: Path,
    device: torch.device,
) -> Tuple[List[TaskBatch], List[str], Dict[str, Any]]:
    tasks, feat_keys = _read_tasks(input_path)
    X_list, y_list = _stack_task_candidates(tasks, feat_keys)
    if not X_list:
        raise RuntimeError("No tasks with positive labels found")
    Xs, scaler = _standardize(X_list)
    batches = _build_task_batches(Xs, y_list, device)
    return batches, feat_keys, scaler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tasks", required=True, help="Path to tasks_train.jsonl")
    parser.add_argument("--out", required=True, help="Path to save the trained model (torch .pt file)")
    parser.add_argument("--hidden-dims", default="128,64", help="Comma separated hidden layer sizes")
    parser.add_argument("--activation", default="relu", choices=["relu", "gelu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l1", type=float, default=0.0, help="L1 regularisation strength")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device)
    try:
        batches, feat_keys, scaler = _prepare_training_data(Path(args.train_tasks), device)
    except Exception as exc:  # pylint: disable=broad-except
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    model = DeepIRLModel(
        input_dim=batches[0].features.shape[1],
        hidden_dims=hidden_dims,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    stats = train(
        model=model,
        batches=batches,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1=args.l1,
        temperature=args.temperature,
        clip_norm=args.clip_norm,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_order": feat_keys,
            "scaler": scaler,
            "hidden_dims": hidden_dims,
            "activation": args.activation,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "loss_history": stats["loss_history"],
        },
        out_path,
    )
    print(
        json.dumps(
            {
                "out": str(out_path),
                "tasks_used": len(batches),
                "hidden_dims": hidden_dims,
                "loss_final": stats["loss_history"][-1] if stats["loss_history"] else None,
                "device": str(device),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
オフラインデータ (dataset_train.jsonl) から RetentionIRLSystem を簡易学習

前提:
- 入力: JSONL 各行に {state: [20], action: {0,1,2}, done: bool, ...}
- 出力: IRL モデル .pth（RetentionIRLSystem.save_model 形式）

学習方針（簡易）:
- 正例: (state, action_gt) の報酬=1, 継続=1
- 負例: (state, action!=gt) の報酬=0, 継続=0.5 を数個サンプリング
- 損失: MSE(報酬) + BCE(継続)

この学習済みモデルは scripts/evaluation/irl_rl_replay_eval.py と互換です。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from gerrit_retention.rl_prediction.retention_irl_system import (
    RetentionIRLSystem,  # type: ignore
)


def load_offline(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text().splitlines()
    out: List[Dict[str, Any]] = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def build_batches(samples: List[Dict[str, Any]], negatives_per_pos: int, batch_size: int, action_dim: int = 3, sequence: bool = False, seq_len: int = 10):
    # シンプルなシャッフル・イテレータ
    idxs = np.arange(len(samples))
    np.random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start:start+batch_size]
        s_list: List[np.ndarray] = []
        a_list: List[np.ndarray] = []
        r_tgt: List[float] = []
        c_tgt: List[float] = []
        for i in chunk:
            s = np.asarray(samples[i].get('state', []), dtype=np.float32)
            if s.size == 0:
                continue
            a_gt = int(samples[i].get('action', 2))
            # 正例
            oh = np.zeros((action_dim,), dtype=np.float32)
            if 0 <= a_gt < action_dim:
                oh[a_gt] = 1.0
            if sequence:
                # Pad to seq_len by repeating
                s_seq = np.tile(s, (seq_len, 1))
                a_seq = np.tile(oh, (seq_len, 1))
                s_list.append(s_seq)
                a_list.append(a_seq)
            else:
                s_list.append(s)
                a_list.append(oh)
            r_tgt.append(1.0)
            c_tgt.append(1.0)
            # 負例をサンプリング
            negs = [a for a in range(action_dim) if a != a_gt]
            np.random.shuffle(negs)
            for a in negs[:negatives_per_pos]:
                ohn = np.zeros((action_dim,), dtype=np.float32)
                ohn[a] = 1.0
                if sequence:
                    s_seq = np.tile(s, (seq_len, 1))
                    a_seq = np.tile(ohn, (seq_len, 1))
                    s_list.append(s_seq)
                    a_list.append(a_seq)
                else:
                    s_list.append(s)
                    a_list.append(ohn)
                r_tgt.append(0.0)
                c_tgt.append(0.5)
        if not s_list:
            continue
        S = torch.from_numpy(np.stack(s_list)).float()
        A = torch.from_numpy(np.stack(a_list)).float()
        R = torch.from_numpy(np.asarray(r_tgt, dtype=np.float32)).view(-1, 1)
        C = torch.from_numpy(np.asarray(c_tgt, dtype=np.float32)).view(-1, 1)
        yield S, A, R, C


def main(args):
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=str, default='outputs/offline/dataset_train.jsonl')
    ap.add_argument('--out', type=str, default='outputs/irl/retention_irl_offline.pth')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--negatives-per-pos', type=int, default=2)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--sequence', action='store_true', help='Enable sequence mode')
    ap.add_argument('--seq-len', type=int, default=10, help='Sequence length')
    args = ap.parse_args()

    train_path = Path(args.train)
    samples = load_offline(train_path)
    if not samples:
        print(f'❌ no samples: {train_path}')
        return 1

    # IRL モデル（state_dim=20, action_dim=3 想定）
    cfg = {'state_dim': 20, 'action_dim': 3, 'hidden_dim': int(args.hidden), 'learning_rate': float(args.lr), 'sequence': args.sequence, 'seq_len': args.seq_len}
    irl = RetentionIRLSystem(cfg)
    device = getattr(irl, 'device', torch.device('cpu'))
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    optim_ = optim.Adam(irl.network.parameters(), lr=float(args.lr))

    def to_device(t):
        return t.to(device)

    for ep in range(int(args.epochs)):
        total_loss = 0.0
        steps = 0
        for S, A, R, C in build_batches(samples, int(args.negatives_per_pos), int(args.batch_size), action_dim=3, sequence=args.sequence, seq_len=args.seq_len):
            S = to_device(S)
            A = to_device(A)
            R = to_device(R)
            C = to_device(C)
            optim_.zero_grad()
            pred_r, pred_c = irl.network(S, A)
            loss = mse(pred_r, R) + bce(pred_c, C)
            loss.backward()
            optim_.step()
            total_loss += float(loss.item())
            steps += 1
        avg = total_loss / max(steps, 1)
        print(f'Epoch {ep+1}/{args.epochs} - loss: {avg:.4f}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    irl.save_model(str(out_path))
    print(f'✅ saved IRL model -> {out_path}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RetentionIRLSystem from offline data (extended: sequence support)"
    )
    parser.add_argument(
        "--train", type=Path, required=True, help="Training data (JSONL)"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output model path (.pth)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size"
    )
    parser.add_argument(
        "--negatives-per-pos", type=int, default=2, help="Negatives per positive"
    )
    parser.add_argument(
        "--hidden", type=int, default=128, help="Hidden dimension"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--sequence", action="store_true", help="Enable sequence mode"
    )
    parser.add_argument(
        "--seq-len", type=int, default=10, help="Sequence length for sequence mode"
    )
    args = parser.parse_args()
    
    raise SystemExit(main(args))

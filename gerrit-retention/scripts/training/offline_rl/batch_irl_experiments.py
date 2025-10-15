#!/usr/bin/env python3
"""
バッチIRL実験スクリプト: 複数のシーケンス長でIRLモデルを学習・評価

各期間設定（6,12,18,24ヶ月）で別ディレクトリに結果を保存
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
SRC = project_root / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
import sys

sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem


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


def train_irl_model(train_path: Path, cfg: Dict[str, Any], epochs: int, batch_size: int = 2048, negatives_per_pos: int = 2):
    """IRLモデルをトレーニング"""
    samples = load_offline(train_path)
    if not samples:
        raise ValueError(f'No samples: {train_path}')

    irl = RetentionIRLSystem(cfg)
    device = getattr(irl, 'device', torch.device('cpu'))
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    optim_ = optim.Adam(irl.network.parameters(), lr=float(cfg['learning_rate']))

    def to_device(t):
        return t.to(device)

    training_log = []
    for ep in range(epochs):
        total_loss = 0.0
        steps = 0
        for S, A, R, C in build_batches(samples, negatives_per_pos, batch_size, action_dim=3, sequence=cfg['sequence'], seq_len=cfg['seq_len']):
            S = to_device(S)
            A = to_device(A)
            R = to_device(R)
            C = to_device(C)

            optim_.zero_grad()
            reward_pred, cont_pred = irl.network(S, A)
            loss_r = mse(reward_pred, R)
            loss_c = bce(cont_pred, C)
            loss = loss_r + loss_c
            loss.backward()
            optim_.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"Epoch {ep+1}/{epochs} - loss: {avg_loss:.6f}")
        training_log.append({"epoch": ep+1, "loss": avg_loss})

    return irl, training_log


def run_single_experiment(train_path: Path, seq_len: int, output_dir: Path, epochs: int = 10):
    """単一の期間設定でIRL実験を実行"""
    print(f"\n=== IRL実験開始: seq_len={seq_len}ヶ月 ===")

    # 出力ディレクトリ作成
    exp_dir = output_dir / f"seq_{seq_len}months"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # モデル保存パス
    model_path = exp_dir / "irl_model.pth"
    config_path = exp_dir / "config.json"
    log_path = exp_dir / "training.log"

    try:
        # IRLモデル設定
        cfg = {
            'state_dim': 20,
            'action_dim': 3,
            'hidden_dim': 128,
            'learning_rate': 0.001,
            'sequence': True,
            'seq_len': seq_len
        }

        # 設定保存
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

        print(f"学習データ読み込み: {train_path}")
        print(f"シーケンス長: {seq_len}")

        # IRLトレーニング実行
        irl, training_log = train_irl_model(train_path, cfg, epochs)

        # トレーニング結果
        training_results = {
            "seq_len": seq_len,
            "epochs": epochs,
            "final_loss": training_log[-1]["loss"] if training_log else 0,
            "training_time": 0,  # TODO: 時間計測追加
            "model_saved": str(model_path)
        }

        # モデル保存
        irl.save_model(str(model_path))
        print(f"モデル保存: {model_path}")

        # ログ保存
        with open(log_path, 'w') as f:
            f.write(f"IRL Training Log - seq_len={seq_len}\n")
            f.write(json.dumps({
                "config": cfg,
                "training_log": training_log,
                "results": training_results
            }, indent=2))

        print(f"✓ 実験完了: {exp_dir}")

        return {
            "seq_len": seq_len,
            "success": True,
            "output_dir": str(exp_dir),
            "training_results": training_results
        }

    except Exception as e:
        print(f"✗ 実験失敗: {e}")
        return {
            "seq_len": seq_len,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="バッチIRL実験: 複数のシーケンス長でIRLモデルを学習"
    )
    parser.add_argument(
        "--train", type=Path, required=True,
        help="学習データ (JSONL)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/batch_experiments"),
        help="出力ベースディレクトリ"
    )
    parser.add_argument(
        "--periods", type=int, nargs='+', default=[3, 6, 12, 24],
        help="実験する期間リスト (ヶ月)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="トレーニングエポック数"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="並行実行 (未実装)"
    )

    args = parser.parse_args()

    # 出力ベースディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)

    print("=== IRLバッチ実験開始 ===")
    print(f"学習データ: {args.train}")
    print(f"出力ディレクトリ: {args.output}")
    print(f"実験期間: {args.periods}")
    print(f"エポック数: {args.epochs}")

    results = []

    # 各期間で実験実行
    for seq_len in args.periods:
        result = run_single_experiment(
            train_path=args.train,
            seq_len=seq_len,
            output_dir=args.output,
            epochs=args.epochs
        )
        results.append(result)

    # 全体結果保存
    summary_path = args.output / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "experiment_config": {
                "train_data": str(args.train),
                "periods": args.periods,
                "epochs": args.epochs,
                "parallel": args.parallel
            },
            "results": results
        }, f, indent=2)

    print(f"\n=== バッチ実験完了 ===")
    print(f"結果サマリー: {summary_path}")

    # 成功/失敗統計
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"成功: {successful}/{total}")

    if successful < total:
        print("失敗した実験:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['seq_len']}ヶ月: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
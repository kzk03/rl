"""reviewer_sequences JSONをIRL学習用JSONLに変換するスクリプト"""

import json
from pathlib import Path
from typing import Any, Dict, List


def convert_sequences_to_jsonl(sequences_path: Path, output_path: Path):
    """reviewer_sequencesをIRL学習用JSONLに変換"""

    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences_data = json.load(f)

    output_lines = []

    for seq in sequences_data:
        reviewer_id = seq['reviewer_id']
        transitions = seq['transitions']

        # transitionsを時系列順にソート
        transitions.sort(key=lambda x: x['t'])

        for i, transition in enumerate(transitions):
            state_dict = transition['state']
            action = transition['action']

            # stateをリストに変換 (IRL学習用)
            state_list = [
                state_dict.get('gap_days', 0),
                state_dict.get('activity_30d', 0),
                state_dict.get('activity_90d', 0),
                state_dict.get('activity_180d', 0),
                state_dict.get('workload_level', 0.5),
                state_dict.get('interaction_180d', 0),
                # パディングして20次元に
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ][:20]  # 念のため20次元に制限

            # rewardの計算 (仮定: action=1で正の報酬)
            reward = 1.0 if action == 1 else -0.7

            # next_state (次のtransitionのstate、なければ同じ)
            if i + 1 < len(transitions):
                next_state_dict = transitions[i + 1]['state']
                next_state_list = [
                    next_state_dict.get('gap_days', 0),
                    next_state_dict.get('activity_30d', 0),
                    next_state_dict.get('activity_90d', 0),
                    next_state_dict.get('activity_180d', 0),
                    next_state_dict.get('workload_level', 0.5),
                    next_state_dict.get('interaction_180d', 0),
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ][:20]
            else:
                next_state_list = state_list.copy()
                # 最後のtransitionはdone=True
                done = True
            done = False  # 仮定: 全て継続

            record = {
                "reviewer_id": reviewer_id,
                "timestamp": transition['t'],
                "state": state_list,
                "action": action,
                "reward": reward,
                "next_state": next_state_list,
                "done": done
            }

            output_lines.append(json.dumps(record, ensure_ascii=False))

    # 出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"Converted {len(sequences_data)} sequences to {len(output_lines)} transitions")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert reviewer sequences to IRL JSONL format')
    parser.add_argument('--input', type=Path, required=True, help='Input reviewer sequences JSON')
    parser.add_argument('--output', type=Path, required=True, help='Output IRL JSONL file')

    args = parser.parse_args()
    convert_sequences_to_jsonl(args.input, args.output)
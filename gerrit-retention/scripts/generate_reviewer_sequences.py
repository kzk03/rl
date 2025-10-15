#!/usr/bin/env python3
"""
review_requests CSVからreviewer_sequences.jsonを生成するスクリプト

入力: review_requests_synth_nova_w14.csv
出力: outputs/irl/reviewer_sequences.json
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_datetime(dt_str: str) -> datetime:
    """日時文字列をdatetimeに変換"""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except:
        return None


def generate_reviewer_sequences(csv_path: Path, output_path: Path):
    """review_requestsからreviewer_sequencesを生成"""

    # reviewerごとのトランジションを収集
    reviewer_transitions = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviewer_email = row['reviewer_email']
            request_time = parse_datetime(row['request_time'])
            responded_within_days = row.get('responded_within_days', '')
            label = int(row.get('label', 0))  # 1=responded, 0=not responded

            if not request_time:
                continue

            # actionの決定ロジック
            # label=1 (responded) -> action=1 (accept)
            # label=0 (not responded) -> action=0 (decline)
            action = 1 if label == 1 else 0

            # 簡易的なstate生成 (dict形式で)
            state = {
                'gap_days': float(row.get('days_since_last_activity', 0)),
                'activity_30d': float(row.get('reviewer_past_reviews_30d', 0)),
                'activity_90d': float(row.get('reviewer_past_reviews_90d', 0)),
                'activity_180d': float(row.get('reviewer_past_reviews_180d', 0)),
                'workload_level': 0.5,  # 仮定
                'interaction_180d': float(row.get('owner_reviewer_past_interactions_180d', 0))
            }

            transition = {
                't': request_time.isoformat(),
                'action': action,
                'state': state
            }

            reviewer_transitions[reviewer_email].append(transition)

    # 各reviewerのtransitionsを時系列順にソート
    sequences = []
    for reviewer_id, transitions in reviewer_transitions.items():
        # 時系列順にソート
        transitions.sort(key=lambda x: x['t'])

        sequence = {
            'reviewer_id': reviewer_id,
            'transitions': transitions
        }
        sequences.append(sequence)

    # JSON出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(sequences)} reviewer sequences")
    total_transitions = sum(len(seq['transitions']) for seq in sequences)
    print(f"Total transitions: {total_transitions}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate reviewer sequences from review requests CSV')
    parser.add_argument('--input', type=Path, default='data/review_requests_synth_nova_w14.csv',
                       help='Input CSV file')
    parser.add_argument('--output', type=Path, default='outputs/irl/reviewer_sequences.json',
                       help='Output JSON file')

    args = parser.parse_args()

    generate_reviewer_sequences(args.input, args.output)
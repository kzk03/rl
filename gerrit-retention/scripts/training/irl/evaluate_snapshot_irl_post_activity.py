#!/usr/bin/env python3
"""
Snapshot-based IRL with Post-Activity Population Design

母集団設計:
- スナップショット後の最小予測期間（3ヶ月）に活動がある人を母集団とする
- 全16設定で同じ母集団を使用（統一母集団）

特徴量:
- スナップショット時点での特徴量（学習期間のデータから計算）
- データリークなし

ラベル:
- 各予測期間終了後に活動が継続するか
- 3ヶ月予測: スナップショット後3ヶ月以降に活動
- 12ヶ月予測: スナップショット後12ヶ月以降に活動
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gerrit_retention.rl_prediction.reward_network import RewardNetworkTrainer
from gerrit_retention.rl_prediction.snapshot_features_enhanced import (
    compute_snapshot_features_enhanced,
)
from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_score, recall_score
)
import torch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load review data."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    for col in ["request_time", "created", "timestamp"]:
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col])
            break

    for col in ["reviewer_email", "email", "reviewer"]:
        if col in df.columns:
            df["reviewer_email"] = df[col]
            break

    print(f"Loaded {len(df):,} reviews")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Reviewers: {df['reviewer_email'].nunique()}")

    return df


def extract_state_features(
    reviewer_reviews: pd.DataFrame,
    current_row: pd.Series,
    all_reviews: pd.DataFrame,
    feature_extractor: EnhancedFeatureExtractor,
) -> np.ndarray:
    """Extract 32-dimensional enhanced state features."""
    reviews_so_far = reviewer_reviews[reviewer_reviews["timestamp"] <= current_row["timestamp"]]

    developer = {
        'developer_id': current_row.get('reviewer_email', 'unknown'),
        'reviewer_email': current_row.get('reviewer_email', 'unknown'),
        'first_seen': reviews_so_far.iloc[0]["timestamp"].isoformat() if len(reviews_so_far) > 0 else current_row["timestamp"].isoformat(),
        'changes_authored': len(reviews_so_far),
        'changes_reviewed': len(reviews_so_far),
        'projects': [current_row.get('project', 'unknown')],
        'reviewer_assignment_load_7d': 0,
        'reviewer_assignment_load_30d': 0,
        'reviewer_assignment_load_180d': 0,
        'owner_reviewer_past_interactions_180d': 0,
        'owner_reviewer_project_interactions_180d': 0,
        'owner_reviewer_past_assignments_180d': 0,
        'path_jaccard_files_project': 0.0,
        'path_jaccard_dir1_project': 0.0,
        'path_jaccard_dir2_project': 0.0,
        'path_overlap_files_project': 0.0,
        'path_overlap_dir1_project': 0.0,
        'path_overlap_dir2_project': 0.0,
        'response_latency_days': 0.0,
        'reviewer_past_response_rate_180d': 1.0,
        'reviewer_tenure_days': (current_row["timestamp"] - reviews_so_far.iloc[0]["timestamp"]).days if len(reviews_so_far) > 0 else 0,
        'change_insertions': current_row.get('change_insertions', 0),
        'change_deletions': current_row.get('change_deletions', 0),
        'change_files_count': current_row.get('change_files_count', 1),
    }

    activity_history = []
    for _, review in reviews_so_far.iterrows():
        activity = {
            'type': 'review',
            'timestamp': review['timestamp'].isoformat(),
            'change_insertions': review.get('change_insertions', 0),
            'change_deletions': review.get('change_deletions', 0),
            'change_files_count': review.get('change_files_count', 1),
            'project': review.get('project', 'unknown'),
        }
        activity_history.append(activity)

    enhanced_state = feature_extractor.extract_enhanced_state(
        developer=developer,
        activity_history=activity_history,
        context_date=current_row["timestamp"]
    )

    return feature_extractor.state_to_array(enhanced_state)


def extract_action_features(row: pd.Series, feature_extractor: EnhancedFeatureExtractor) -> np.ndarray:
    """Extract 9-dimensional enhanced action features."""
    try:
        last_activity = {
            'type': 'review',
            'timestamp': row['timestamp'].isoformat(),
            'change_insertions': row.get('change_insertions', 0),
            'change_deletions': row.get('change_deletions', 0),
            'change_files_count': row.get('change_files_count', 1),
            'project': row.get('project', 'unknown'),
        }

        enhanced_action = feature_extractor.extract_enhanced_action(
            activity=last_activity,
            context_date=row["timestamp"]
        )

        return feature_extractor.action_to_array(enhanced_action)
    except:
        return np.zeros(9, dtype=np.float32)


def extract_training_trajectories(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months: int,
    seq_len: int,
    fixed_population: set
) -> tuple:
    """
    Extract training trajectories for reviewers in fixed_population
    who have data in the learning period.
    """
    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    learning_df = df[
        (df["timestamp"] >= learning_start) &
        (df["timestamp"] < snapshot_date)
    ]

    if len(learning_df) == 0:
        return [], []

    feature_extractor = EnhancedFeatureExtractor()

    trajectories = []
    reviewer_ids = []

    for reviewer in fixed_population:
        reviewer_reviews = learning_df[
            learning_df["reviewer_email"] == reviewer
        ].sort_values("timestamp")

        if len(reviewer_reviews) == 0:
            continue  # この学習期間にデータなし

        # Extract enhanced features
        states = []
        actions = []

        for idx, row in reviewer_reviews.iterrows():
            state = extract_state_features(reviewer_reviews, row, learning_df, feature_extractor)
            action = extract_action_features(row, feature_extractor)
            states.append(state)
            actions.append(action)

        # Pad or truncate
        if len(states) < seq_len:
            padding_needed = seq_len - len(states)
            states = [states[0]] * padding_needed + states
            actions = [actions[0]] * padding_needed + actions
        else:
            states = states[-seq_len:]
            actions = actions[-seq_len:]

        trajectory = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "reviewer": reviewer,
        }
        trajectories.append(trajectory)
        reviewer_ids.append(reviewer)

    return trajectories, reviewer_ids


def run_post_activity_population_evaluation(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months_list: list,
    prediction_months_list: list,
    seq_len: int,
    epochs: int,
    output_dir: Path
) -> pd.DataFrame:
    """
    Run evaluation with post-activity population design.

    母集団: スナップショット後の最小予測期間に活動がある人
    """
    results = []

    # STEP 1: 母集団を固定（スナップショット後3ヶ月以内に活動）
    min_prediction_months = min(prediction_months_list)
    population_end = snapshot_date + timedelta(days=min_prediction_months * 30)

    print(f"\n{'='*80}")
    print(f"DETERMINING UNIFIED POPULATION")
    print(f"{'='*80}")
    print(f"Post-activity window: {snapshot_date.date()} to {population_end.date()}")
    print(f"  ({min_prediction_months} months after snapshot)")

    post_activity_df = df[
        (df["timestamp"] >= snapshot_date) &
        (df["timestamp"] < population_end)
    ]

    fixed_population = set(post_activity_df["reviewer_email"].unique())
    print(f"Fixed population: {len(fixed_population)} reviewers")
    print(f"  (Active within {min_prediction_months} months after snapshot)")

    # STEP 2: 各設定で評価
    for learning_months in learning_months_list:
        for prediction_months in prediction_months_list:
            print(f"\n{'='*80}")
            print(f"Configuration: {learning_months}m learning × {prediction_months}m prediction")
            print(f"{'='*80}")

            # 訓練データ抽出
            print("Extracting training trajectories...")
            trajectories, train_reviewers = extract_training_trajectories(
                df, snapshot_date, learning_months, seq_len, fixed_population
            )

            if len(trajectories) == 0:
                print("  Skipping: No trajectories")
                continue

            # ラベル: 予測期間終了後に活動があるか
            prediction_end = snapshot_date + timedelta(days=prediction_months * 30)
            post_prediction_df = df[df["timestamp"] >= prediction_end]
            continued_reviewers = set(post_prediction_df["reviewer_email"].unique())

            train_labels = [1 if r in continued_reviewers else 0 for r in train_reviewers]
            continuation_rate_train = np.mean(train_labels) * 100

            print(f"  Training trajectories: {len(trajectories)}")
            print(f"  Training continuation rate: {continuation_rate_train:.1f}%")

            # 訓練
            print(f"  Training reward network...")
            trainer = RewardNetworkTrainer(
                state_dim=32,
                action_dim=9,
                hidden_dim=128,
                learning_rate=0.001
            )

            trainer.train(
                trajectories=trajectories,
                labels=train_labels,
                epochs=epochs,
                batch_size=32,
                verbose=True
            )

            # テスト: fixed_population全員（統一母集団）
            print(f"  Testing on unified population ({len(fixed_population)} reviewers)...")
            test_predictions = []
            test_labels_all = []
            test_reviewers_all = []

            feature_extractor = EnhancedFeatureExtractor()

            for reviewer in fixed_population:
                # スナップショット時点の特徴量を計算
                snapshot_state, snapshot_action = compute_snapshot_features_enhanced(
                    reviewer, snapshot_date, df, learning_months, feature_extractor
                )

                # 予測
                prob = trainer.predict(snapshot_state, snapshot_action)
                test_predictions.append(prob)

                # ラベル
                label = 1 if reviewer in continued_reviewers else 0
                test_labels_all.append(label)
                test_reviewers_all.append(reviewer)

            test_predictions = np.array(test_predictions)
            test_labels_array = np.array(test_labels_all)

            continuation_rate_test = np.mean(test_labels_array) * 100
            print(f"  Test continuation rate: {continuation_rate_test:.1f}%")

            # Calculate metrics
            metrics = {}
            try:
                metrics["auc_roc"] = roc_auc_score(test_labels_array, test_predictions)
                metrics["auc_pr"] = average_precision_score(test_labels_array, test_predictions)

                binary_preds = (test_predictions >= 0.5).astype(int)
                metrics["f1"] = f1_score(test_labels_array, binary_preds)
                metrics["precision"] = precision_score(test_labels_array, binary_preds, zero_division=0)
                metrics["recall"] = recall_score(test_labels_array, binary_preds, zero_division=0)

                tn, fp, fn, tp = confusion_matrix(test_labels_array, binary_preds).ravel()
                metrics["tn"] = int(tn)
                metrics["fp"] = int(fp)
                metrics["fn"] = int(fn)
                metrics["tp"] = int(tp)

            except Exception as e:
                print(f"  Error calculating metrics: {e}")
                continue

            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")

            # Save model
            model_dir = output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"reward_model_h{learning_months}m_t{prediction_months}m.pth"
            trainer.save_model(str(model_path))

            # Save predictions
            predictions_dir = output_dir / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            predictions_df = pd.DataFrame({
                "reviewer_email": test_reviewers_all,
                "true_label": test_labels_array,
                "predicted_probability": test_predictions,
                "predicted_binary": (test_predictions >= 0.5).astype(int),
                "learning_months": learning_months,
                "prediction_months": prediction_months
            })
            predictions_file = predictions_dir / f"predictions_h{learning_months}m_t{prediction_months}m.csv"
            predictions_df.to_csv(predictions_file, index=False)

            # Record results
            result = {
                "learning_months": learning_months,
                "prediction_months": prediction_months,
                "n_population": len(fixed_population),
                "n_train": len(trajectories),
                "continuation_rate": continuation_rate_test,
                **metrics,
                "model_path": str(model_path)
            }
            results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "sliding_window_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results to: {results_csv}")

    return results_df


def create_heatmaps(results_df: pd.DataFrame, output_dir: Path):
    """Create heatmaps for all metrics."""
    metrics = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    metric_names = {
        "auc_roc": "AUC-ROC",
        "auc_pr": "AUC-PR",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall"
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        pivot = results_df.pivot(
            index="learning_months",
            columns="prediction_months",
            values=metric
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=axes[idx],
            cbar_kws={'label': metric_names[metric]}
        )
        axes[idx].set_title(f"{metric_names[metric]} Matrix")
        axes[idx].set_xlabel("Prediction Months")
        axes[idx].set_ylabel("Learning Months")

    # Continuation rate heatmap
    pivot_cont = results_df.pivot(
        index="learning_months",
        columns="prediction_months",
        values="continuation_rate"
    )
    sns.heatmap(
        pivot_cont,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        ax=axes[5],
        cbar_kws={'label': 'Continuation Rate (%)'}
    )
    axes[5].set_title("Continuation Rate Matrix")
    axes[5].set_xlabel("Prediction Months")
    axes[5].set_ylabel("Learning Months")

    plt.tight_layout()
    heatmap_file = output_dir / "heatmaps.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"Saved heatmaps to: {heatmap_file}")
    plt.close()


def create_report(results_df: pd.DataFrame, output_dir: Path, snapshot_date: datetime):
    """Create evaluation report."""
    report = []
    report.append("# Snapshot-based IRL Evaluation Report")
    report.append(f"\n**Population Design**: Post-Activity (Active within 3 months after snapshot)")
    report.append(f"\n**Snapshot Date**: {snapshot_date.date()}")
    report.append(f"\n**Population Size**: {results_df['n_population'].iloc[0]} reviewers")
    report.append(f"\n## Results Summary\n")

    # Best performances
    best_auc_roc = results_df.loc[results_df['auc_roc'].idxmax()]
    best_auc_pr = results_df.loc[results_df['auc_pr'].idxmax()]
    best_f1 = results_df.loc[results_df['f1'].idxmax()]

    report.append("### Best Performances\n")
    report.append(f"- **Best AUC-ROC**: {best_auc_roc['auc_roc']:.4f} ({int(best_auc_roc['learning_months'])}m learning × {int(best_auc_roc['prediction_months'])}m prediction)")
    report.append(f"- **Best AUC-PR**: {best_auc_pr['auc_pr']:.4f} ({int(best_auc_pr['learning_months'])}m learning × {int(best_auc_pr['prediction_months'])}m prediction)")
    report.append(f"- **Best F1**: {best_f1['f1']:.4f} ({int(best_f1['learning_months'])}m learning × {int(best_f1['prediction_months'])}m prediction)")

    report.append("\n## All Results\n")
    report.append("```")
    report.append(results_df.to_string(index=False))
    report.append("```")

    report_file = output_dir / "EVALUATION_REPORT.md"
    report_file.write_text("\n".join(report))
    print(f"Saved report to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Snapshot-based IRL with Post-Activity Population")
    parser.add_argument("--reviews", required=True, help="Path to reviews CSV")
    parser.add_argument("--snapshot-date", required=True, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--learning-months", nargs="+", type=int, default=[3, 6, 9, 12])
    parser.add_argument("--prediction-months", nargs="+", type=int, default=[3, 6, 9, 12])
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    print("="*80)
    print("SNAPSHOT-BASED IRL WITH POST-ACTIVITY POPULATION")
    print("="*80)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    df = load_and_prepare_data(args.reviews)
    snapshot_date = datetime.strptime(args.snapshot_date, "%Y-%m-%d")

    # Run evaluation
    results_df = run_post_activity_population_evaluation(
        df=df,
        snapshot_date=snapshot_date,
        learning_months_list=args.learning_months,
        prediction_months_list=args.prediction_months,
        seq_len=args.seq_len,
        epochs=args.epochs,
        output_dir=output_dir
    )

    # Create visualizations
    print("\nCreating heatmaps...")
    create_heatmaps(results_df, output_dir)

    print("\nCreating report...")
    create_report(results_df, output_dir, snapshot_date)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

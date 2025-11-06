#!/usr/bin/env python3
"""
Create full 4x4 comparison heatmap (all 16 cells).
Shows IRL vs baselines with Nova-only fair comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load matrices
irl_matrix = pd.read_csv(
    'importants/review_acceptance_cross_eval_nova/matrix_AUC_ROC.csv',
    index_col=0
)
lr_matrix = pd.read_csv(
    'importants/baseline_nova_only/logistic_regression/matrix_AUC_ROC.csv',
    index_col=0
)
rf_matrix = pd.read_csv(
    'importants/baseline_nova_only/random_forest/matrix_AUC_ROC.csv',
    index_col=0
)

# Convert to numpy arrays and transpose
# CSV format: rows=training, cols=evaluation
# Display format: rows=evaluation, cols=training (so transpose)
irl_data = irl_matrix.values.T
lr_data = lr_matrix.values.T
rf_data = rf_matrix.values.T

# Period labels
periods = ['0-3m', '3-6m', '6-9m', '9-12m']

# Calculate full statistics (all 16 cells)
irl_mean = irl_data.mean()
lr_mean = lr_data.mean()
rf_mean = rf_data.mean()

print("Full Matrix Statistics (All 16 cells):")
print(f"IRL: {irl_mean:.3f} (±{irl_data.std():.3f})")
print(f"LR:  {lr_mean:.3f} (±{lr_data.std():.3f})")
print(f"RF:  {rf_mean:.3f} (±{rf_data.std():.3f})")

# Calculate diagonal+future statistics
diagonal_future_cells = [(i, j) for i in range(4) for j in range(4) if i >= j]
irl_diag_future = np.array([irl_data[i, j] for i, j in diagonal_future_cells])
lr_diag_future = np.array([lr_data[i, j] for i, j in diagonal_future_cells])
rf_diag_future = np.array([rf_data[i, j] for i, j in diagonal_future_cells])

print("\nDiagonal+Future Statistics (10 cells):")
print(f"IRL: {irl_diag_future.mean():.3f} (±{irl_diag_future.std():.3f})")
print(f"LR:  {lr_diag_future.mean():.3f} (±{lr_diag_future.std():.3f})")
print(f"RF:  {rf_diag_future.mean():.3f} (±{rf_diag_future.std():.3f})")

# Create comparison heatmaps
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row: Individual models
models = [
    ('IRL+LSTM (AUC-ROC)', irl_data),
    ('Logistic Regression (AUC-ROC)', lr_data),
    ('Random Forest (AUC-ROC)', rf_data)
]

for ax, (title, data) in zip(axes[0], models):
    im = ax.imshow(data, cmap='Reds', aspect='auto', vmin=0.4, vmax=1.0, origin='lower')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(periods)
    ax.set_yticklabels(periods)
    ax.set_xlabel('Training Period', fontsize=10)
    ax.set_ylabel('Evaluation Period', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax)

# Bottom row: Differences (IRL - Baseline)
comparisons = [
    ('IRL+LSTM - Logistic Regression', irl_data - lr_data),
    ('IRL+LSTM - Random Forest', irl_data - rf_data),
    ('Summary Statistics', None)
]

for ax, (title, data) in zip(axes[1], comparisons):
    if data is not None:
        # Diverging colormap for differences
        vmax = 0.15
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(periods)
        ax.set_yticklabels(periods)
        ax.set_xlabel('Training Period', fontsize=10)
        ax.set_ylabel('Evaluation Period', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add text annotations
        for i in range(4):
            for j in range(4):
                val = data[i, j]
                color = "white" if abs(val) > 0.08 else "black"
                sign = '+' if val >= 0 else ''
                text = ax.text(j, i, f'{sign}{val:.3f}',
                             ha="center", va="center", color=color, fontsize=9)

        plt.colorbar(im, ax=ax)
    else:
        # Summary statistics panel
        ax.axis('off')
        summary_text = f"""
IRL+LSTM vs Baselines (Nova only - Fair Comparison)
{'='*55}

All 16 Cells:
  • IRL+LSTM:        {irl_mean:.3f} ± {irl_data.std():.3f}
  • Log Regression:  {lr_mean:.3f} ± {lr_data.std():.3f}
  • Random Forest:   {rf_mean:.3f} ± {rf_data.std():.3f}

IRL Advantage (All):
  • vs LR:  +{(irl_mean - lr_mean):.3f} (+{(irl_mean - lr_mean)/lr_mean*100:.1f}%)
  • vs RF:  +{(irl_mean - rf_mean):.3f} (+{(irl_mean - rf_mean)/rf_mean*100:.1f}%)

Diagonal + Future (10 cells):
  • IRL+LSTM:        {irl_diag_future.mean():.3f} ± {irl_diag_future.std():.3f}
  • Log Regression:  {lr_diag_future.mean():.3f} ± {lr_diag_future.std():.3f}
  • Random Forest:   {rf_diag_future.mean():.3f} ± {rf_diag_future.std():.3f}

IRL Advantage (Diagonal+Future):
  • vs LR:  +{(irl_diag_future.mean() - lr_diag_future.mean()):.3f} (+{(irl_diag_future.mean() - lr_diag_future.mean())/lr_diag_future.mean()*100:.1f}%)
  • vs RF:  +{(irl_diag_future.mean() - rf_diag_future.mean()):.3f} (+{(irl_diag_future.mean() - rf_diag_future.mean())/rf_diag_future.mean()*100:.1f}%)

Best Combination:
  • IRL:  {irl_data.max():.3f} at {periods[np.unravel_index(irl_data.argmax(), irl_data.shape)[1]]} train, {periods[np.unravel_index(irl_data.argmax(), irl_data.shape)[0]]} eval
  • LR:   {lr_data.max():.3f} at {periods[np.unravel_index(lr_data.argmax(), lr_data.shape)[1]]} train, {periods[np.unravel_index(lr_data.argmax(), lr_data.shape)[0]]} eval
  • RF:   {rf_data.max():.3f} at {periods[np.unravel_index(rf_data.argmax(), rf_data.shape)[1]]} train, {periods[np.unravel_index(rf_data.argmax(), rf_data.shape)[0]]} eval

Dataset:
  • Project: OpenStack Nova only
  • Reviews: 27,328 (fair comparison)
"""
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes)

plt.suptitle('IRL+LSTM vs Baselines: Full 4×4 Evaluation (Nova only - Fair Comparison)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_path = Path('importants/baseline_nova_only/comparison_heatmaps_full.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved to: {output_path}")
plt.close()

print("\nDone!")

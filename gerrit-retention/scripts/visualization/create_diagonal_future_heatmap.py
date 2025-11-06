#!/usr/bin/env python3
"""
Create comparison heatmap for diagonal+future evaluations only.
Shows IRL vs baselines with past evaluation cells grayed out.
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

# Create masked versions for diagonal+future only
# After transpose: rows = evaluation periods, cols = training periods
# Diagonal+future: evaluation >= training (row >= col)
# Past: evaluation < training (row < col) <- should be grayed
def create_masked_data(data):
    """Create data with NaN for past evaluation cells (row < col)"""
    masked = data.copy()
    for i in range(4):  # row (evaluation period)
        for j in range(4):  # col (training period)
            if i < j:  # evaluation period earlier than training period
                masked[i, j] = np.nan
    return masked

irl_masked = create_masked_data(irl_data)
lr_masked = create_masked_data(lr_data)
rf_masked = create_masked_data(rf_data)

# Calculate diagonal+future statistics
# After transpose: rows=eval, cols=training
# Diagonal+future: row >= col (evaluation >= training)
diagonal_future_cells = [(i, j) for i in range(4) for j in range(4) if i >= j]

irl_diag_future = np.array([irl_data[i, j] for i, j in diagonal_future_cells])
lr_diag_future = np.array([lr_data[i, j] for i, j in diagonal_future_cells])
rf_diag_future = np.array([rf_data[i, j] for i, j in diagonal_future_cells])

print("Diagonal+Future Statistics:")
print(f"IRL: {irl_diag_future.mean():.3f} (±{irl_diag_future.std():.3f})")
print(f"LR:  {lr_diag_future.mean():.3f} (±{lr_diag_future.std():.3f})")
print(f"RF:  {rf_diag_future.mean():.3f} (±{rf_diag_future.std():.3f})")

# Create comparison heatmaps
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row: Individual models
models = [
    ('IRL+LSTM (AUC-ROC)', irl_masked),
    ('Logistic Regression (AUC-ROC)', lr_masked),
    ('Random Forest (AUC-ROC)', rf_masked)
]

for ax, (title, data) in zip(axes[0], models):
    # Create custom colormap with gray for NaN
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.4, vmax=1.0, origin='lower')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(periods)
    ax.set_yticklabels(periods)
    ax.set_xlabel('Training Period', fontsize=10)
    ax.set_ylabel('Evaluation Period', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(4):  # row (evaluation period)
        for j in range(4):  # col (training period)
            if i >= j:  # Only show values for diagonal+future (eval >= train)
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
            else:  # Gray cells (past evaluation)
                text = ax.text(j, i, '—',
                             ha="center", va="center", color="gray", fontsize=12)

    plt.colorbar(im, ax=ax)

# Bottom row: Differences (IRL - Baseline)
comparisons = [
    ('IRL+LSTM - Logistic Regression', irl_masked - lr_masked),
    ('IRL+LSTM - Random Forest', irl_masked - rf_masked),
    ('Summary Statistics', None)
]

for ax, (title, data) in zip(axes[1], comparisons):
    if data is not None:
        # Diverging colormap for differences
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color='lightgray')

        # Center colormap at 0
        vmax = 0.15
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(periods)
        ax.set_yticklabels(periods)
        ax.set_xlabel('Training Period', fontsize=10)
        ax.set_ylabel('Evaluation Period', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add text annotations
        for i in range(4):  # row (evaluation period)
            for j in range(4):  # col (training period)
                if i >= j:  # Only show values for diagonal+future (eval >= train)
                    val = data[i, j]
                    color = "white" if abs(val) > 0.08 else "black"
                    sign = '+' if val >= 0 else ''
                    text = ax.text(j, i, f'{sign}{val:.3f}',
                                 ha="center", va="center", color=color, fontsize=9)
                else:  # Gray cells (past evaluation)
                    text = ax.text(j, i, '—',
                                 ha="center", va="center", color="gray", fontsize=12)

        plt.colorbar(im, ax=ax)
    else:
        # Summary statistics panel
        ax.axis('off')
        summary_text = f"""
IRL+LSTM vs Baselines (Diagonal + Future)
{'='*45}

Average AUC-ROC (10 cells):
  • IRL+LSTM:        {irl_diag_future.mean():.3f} ± {irl_diag_future.std():.3f}
  • Log Regression:  {lr_diag_future.mean():.3f} ± {lr_diag_future.std():.3f}
  • Random Forest:   {rf_diag_future.mean():.3f} ± {rf_diag_future.std():.3f}

IRL Advantage:
  • vs LR:  +{(irl_diag_future.mean() - lr_diag_future.mean()):.3f} (+{(irl_diag_future.mean() - lr_diag_future.mean())/lr_diag_future.mean()*100:.1f}%)
  • vs RF:  +{(irl_diag_future.mean() - rf_diag_future.mean()):.3f} (+{(irl_diag_future.mean() - rf_diag_future.mean())/rf_diag_future.mean()*100:.1f}%)

Best Combination:
  • IRL:  {irl_diag_future.max():.3f} at {periods[diagonal_future_cells[irl_diag_future.argmax()][1]]} train, {periods[diagonal_future_cells[irl_diag_future.argmax()][0]]} eval
  • LR:   {lr_diag_future.max():.3f} at {periods[diagonal_future_cells[lr_diag_future.argmax()][1]]} train, {periods[diagonal_future_cells[lr_diag_future.argmax()][0]]} eval
  • RF:   {rf_diag_future.max():.3f} at {periods[diagonal_future_cells[rf_diag_future.argmax()][1]]} train, {periods[diagonal_future_cells[rf_diag_future.argmax()][0]]} eval

Note:
  Gray cells = Past evaluation (impractical)
  Colored cells = Diagonal + Future (practical)
"""
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes)

plt.suptitle('IRL+LSTM vs Baselines: Practical Evaluation (Diagonal + Future)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_path = Path('importants/baseline_nova_only/comparison_heatmaps_diagonal_future.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved to: {output_path}")
plt.close()

print("\nDone!")

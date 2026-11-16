"""Plot the recorded HPC evaluation summary; no model inference is performed."""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
df = pd.read_csv(root / 'results/IIN_OOD_metrics.csv')
names = list(dict.fromkeys(df.model_technology))
fig, axes = plt.subplots(1, 2, figsize=(18, 8), layout='constrained')
for ax, metric, cmap, vmin, vmax in zip(axes, ['AUROC', 'rho'], ['YlGnBu', 'RdBu_r'], [0, -1], [1, 1]):
    matrix = df.pivot(index='model_technology', columns='target_technology', values=metric).reindex(index=names, columns=names)
    sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt='.2f', annot_kws={'size': 7}, square=True, cbar_kws={'shrink': .75})
    ax.set(title='AUROC' if metric == 'AUROC' else 'Spearman correlation', xlabel='Target technology', ylabel='Model technology')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
fig.suptitle('m6A-IIN: cross-technology evaluation (coverage ≥ 10)', fontsize=16, weight='bold')
fig.savefig(root / 'figures/ood_performance.png', dpi=200, bbox_inches='tight')

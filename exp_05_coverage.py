"""
GWCC Experiment 5: Data Utilisation / Coverage  (Figure 10 in paper)
=====================================================================
Measures the fraction of data labelled (coverage) by GWCC, DBSCAN,
and HDBSCAN across a range of dataset sizes N with 10% noise added.

Dataset  : Gaussian Blobs (k=3) + 10% uniform noise, N in [200..5000]
Metric   : Coverage = fraction of points NOT labelled as noise (-1)

Output
------
  results/fig10_completeness.pdf
  results/table_coverage.txt

Run
---
  python exp_05_coverage.py
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN
from gwcc import GWCC

SEED = 42
np.random.seed(SEED)
os.makedirs('results', exist_ok=True)

N_SIZES = [200, 500, 1000, 2000, 3000, 5000]
NOISE_FRAC = 0.10

METHOD_COLORS = {'GWCC': '#1565C0', 'DBSCAN': '#F57F17', 'HDBSCAN': '#6A1B9A'}
METHOD_MARKERS = {'GWCC': 'o', 'DBSCAN': 's', 'HDBSCAN': '^'}

cov_table = {m: [] for m in METHOD_COLORS}

print(f"\n{'N':>6s}", ''.join(f"  {m:>10s}" for m in METHOD_COLORS))
print('-' * 42)

for N in N_SIZES:
    n_clean = int(N * (1 - NOISE_FRAC))
    n_noise = N - n_clean

    # PAPER DATASET: Noisy Moons + 10% uniform noise (same as paper Figure 10)
    X_clean, _ = make_moons(n_clean, noise=0.07, random_state=SEED)
    lo, hi = X_clean.min(axis=0), X_clean.max(axis=0)
    rng = np.random.default_rng(SEED)
    X_noise = rng.uniform(lo - 0.5, hi + 0.5, (n_noise, X_clean.shape[1]))
    X = np.vstack([X_clean, X_noise])
    X = StandardScaler().fit_transform(X)

    row = f"{N:>6d}"
    for mname in METHOD_COLORS:
        if mname == 'GWCC':
            lbl = GWCC().fit_predict(X)
        elif mname == 'DBSCAN':
            lbl = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
        else:
            lbl = HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(X)
        cov = (lbl != -1).mean()
        cov_table[mname].append(cov)
        row += f"  {cov:>9.1%}  "
    print(row)

# ─── Figure 10 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4))
for mname in METHOD_COLORS:
    ax.plot(N_SIZES, [c * 100 for c in cov_table[mname]],
            color=METHOD_COLORS[mname],
            marker=METHOD_MARKERS[mname],
            linewidth=2, markersize=8, label=mname)

ax.axhline(100, c='k', lw=0.8, ls=':', alpha=0.6)
ax.set_xlabel('Dataset size N', fontsize=11)
ax.set_ylabel('Coverage (% labelled)', fontsize=11)
ax.set_title(f'Data Utilisation: Coverage vs. N\n'
             f'(Noisy Moons + {NOISE_FRAC:.0%} uniform noise)', fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(50, 105)
ax.set_xticks(N_SIZES)
ax.set_xticklabels([str(n) for n in N_SIZES], fontsize=8)
ax.grid(True, alpha=0.3)

# Annotate GWCC flat at 100%
ax.text(N_SIZES[-1] * 0.98, 101.5, 'GWCC: always 100%',
        ha='right', fontsize=8, color='#1565C0', fontweight='bold')

plt.tight_layout()
plt.savefig('results/fig10_completeness.pdf', bbox_inches='tight')
plt.savefig('results/fig10_completeness.png', bbox_inches='tight', dpi=200)
plt.close()
print("\n[Saved] results/fig10_completeness.pdf")

with open('results/table_coverage.txt', 'w') as f:
    f.write(f"Coverage (%) with {NOISE_FRAC:.0%} uniform noise\n")
    f.write(f"{'N':>6s}" + ''.join(f"  {m:>10s}" for m in METHOD_COLORS) + "\n")
    for i, N in enumerate(N_SIZES):
        f.write(f"{N:>6d}" + ''.join(f"  {cov_table[m][i]:>9.1%} " for m in METHOD_COLORS) + "\n")
print("[Saved] results/table_coverage.txt")

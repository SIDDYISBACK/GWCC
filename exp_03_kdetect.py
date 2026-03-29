"""
GWCC Experiment 3: Automatic k-Detection Accuracy  (Figure 9 + Table 5)
========================================================================
Measures how accurately GWCC, DBSCAN, and HDBSCAN infer the true
cluster count k over 30 independent trials for k in {2,3,4,5,6,7}.

Dataset  : Gaussian Blobs, N=300, sigma=0.5
Trials   : 30 per (method, k) combination
Metric   : Detection rate = fraction(trials where inferred k == true k)

Output
------
  results/fig9_kdetect.pdf
  results/table_kdetect.txt

Run
---
  python exp_03_kdetect.py
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN
from gwcc import GWCC

SEED = 42
N_TRIALS = 30
N = 300
SIGMA = 0.5
K_VALUES = [2, 3, 4, 5, 6, 7]
os.makedirs('results', exist_ok=True)

METHOD_COLORS = {'GWCC': '#1565C0', 'DBSCAN': '#F57F17', 'HDBSCAN': '#6A1B9A'}
METHOD_MARKERS = {'GWCC': 'o', 'DBSCAN': 's', 'HDBSCAN': '^'}

accuracy = {m: [] for m in METHOD_COLORS}

print(f"\n{'k_true':>8s}", ''.join(f"  {m:>10s}" for m in METHOD_COLORS))
print('-' * 42)

for k_true in K_VALUES:
    row = f"{k_true:>8d}"
    for mname in METHOD_COLORS:
        correct = 0
        for trial in range(N_TRIALS):
            X, _ = make_blobs(N, centers=k_true, cluster_std=SIGMA,
                               random_state=SEED + trial * 137)
            X = StandardScaler().fit_transform(X)
            if mname == 'GWCC':
                lbl = GWCC().fit_predict(X)
            elif mname == 'DBSCAN':
                lbl = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
            else:
                lbl = HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(X)
            k_found = len(set(lbl[lbl != -1]))
            if k_found == k_true:
                correct += 1
        pct = correct / N_TRIALS
        accuracy[mname].append(pct)
        row += f"  {pct:>9.0%}  "
    print(row)

# ─── Figure 9 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for mname in METHOD_COLORS:
    ax.plot(K_VALUES, [a * 100 for a in accuracy[mname]],
            color=METHOD_COLORS[mname],
            marker=METHOD_MARKERS[mname],
            linewidth=2, markersize=8,
            label=mname)
ax.set_xlabel('True cluster count k', fontsize=11)
ax.set_ylabel('Detection accuracy (%)', fontsize=11)
ax.set_title(f'Automatic k-Detection Accuracy\n'
             f'(N={N}, σ={SIGMA}, {N_TRIALS} trials per point)', fontsize=10)
ax.legend(fontsize=9)
ax.set_xticks(K_VALUES)
ax.set_ylim(-5, 105)
ax.axhline(50, c='gray', lw=0.8, ls='--', alpha=0.5)
ax.grid(True, alpha=0.3)
for k, vals in zip(K_VALUES, zip(*[accuracy[m] for m in METHOD_COLORS])):
    if k <= 4:
        ax.axvspan(k - 0.4, k + 0.4, alpha=0.05, color='#1565C0')
ax.text(3, 95, 'GWCC ≈ DBSCAN\nat k ≤ 4', fontsize=7.5, color='#1565C0',
        ha='center', style='italic')
plt.tight_layout()
plt.savefig('results/fig9_kdetect.pdf', bbox_inches='tight')
plt.savefig('results/fig9_kdetect.png', bbox_inches='tight', dpi=200)
plt.close()

with open('results/table_kdetect.txt', 'w') as f:
    f.write(f"k-detection accuracy (%), N={N}, sigma={SIGMA}, {N_TRIALS} trials\n")
    f.write(f"{'k':>4s}" + ''.join(f"  {m:>9s}" for m in METHOD_COLORS) + "\n")
    for i, k in enumerate(K_VALUES):
        f.write(f"{k:>4d}" + ''.join(f"  {accuracy[m][i]:>8.0%} " for m in METHOD_COLORS) + "\n")
print("\n[Saved] results/fig9_kdetect.pdf, results/table_kdetect.txt")

"""
GWCC Experiment 1: Standard Benchmarks (Visual + AMI/ARI/Memory)
=================================================================
Reproduces Figures 1, 3, 4 and Tables 1, 2, 3 from the paper.

Datasets  : Noisy Moons, Concentric Circles, Gaussian Blobs,
            Varied Density, Anisotropic Blobs  (N=500 each)
Algorithms: GWCC, DBSCAN, HDBSCAN, Spectral Clustering, K-Means
Metrics   : AMI, ARI, wall-clock time, peak memory, coverage

Output
------
  results/fig1_visual.pdf   - 5×5 scatter grid
  results/fig3_ami.pdf      - AMI bar chart
  results/fig4_memory.pdf   - Peak memory bars
  results/table_ami.txt     - Numeric AMI table
  results/table_ari.txt     - Numeric ARI table
  results/table_resources.txt - Time/memory table
  console                   - All numeric results

Run
---
  python exp_01_standard_benchmarks.py

Requirements
------------
  pip install scikit-learn numpy matplotlib
"""

import sys, os, time, tracemalloc, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN

from gwcc import GWCC

SEED = 42
np.random.seed(SEED)
os.makedirs('results', exist_ok=True)

# ─── Build datasets ───────────────────────────────────────────────────────────
def make_datasets(N=500):
    T = np.array([[0.6, -0.6], [-0.4, 0.8]])
    Xa, ya = make_blobs(N, centers=3, random_state=SEED)
    return [
        ('Noisy Moons',    *make_moons(N, noise=0.07, random_state=SEED)),
        ('Circles',        *make_circles(N, noise=0.05, factor=0.5, random_state=SEED)),
        ('Blobs',          *make_blobs(N, centers=4, cluster_std=0.6, random_state=SEED)),
        ('Varied Density', *make_blobs(N, centers=3, cluster_std=[1.0, 0.5, 0.3], random_state=SEED)),
        ('Anisotropic',    Xa.dot(T), ya),
    ]

# ─── Build methods ────────────────────────────────────────────────────────────
def make_methods(k_true):
    return {
        'GWCC':     lambda X: GWCC().fit_predict(X),
        'DBSCAN':   lambda X: DBSCAN(eps=0.3, min_samples=5).fit_predict(X),
        'HDBSCAN':  lambda X: HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(X),
        'Spectral': lambda X: SpectralClustering(
                        n_clusters=k_true, affinity='nearest_neighbors',
                        n_neighbors=10, random_state=SEED).fit_predict(X),
        'K-Means':  lambda X: KMeans(
                        n_clusters=k_true, n_init=10, random_state=SEED).fit_predict(X),
    }

METHOD_COLORS = {
    'GWCC':     '#1565C0',
    'DBSCAN':   '#F57F17',
    'HDBSCAN':  '#6A1B9A',
    'Spectral': '#2E7D32',
    'K-Means':  '#C62828',
}
NOISE_COLOR = '#bbbbbb'

# ─── Run everything ───────────────────────────────────────────────────────────
datasets = make_datasets()
method_names = ['GWCC', 'DBSCAN', 'HDBSCAN', 'Spectral', 'K-Means']
k_trues = [2, 2, 4, 3, 3]

ami_table  = {m: {} for m in method_names}
ari_table  = {m: {} for m in method_names}
res_table  = {m: {} for m in method_names}   # time, memory, coverage
all_labels = {m: {} for m in method_names}
all_X      = {}
all_y      = {}

print(f"\n{'Dataset':18s} {'Method':10s} {'AMI':>7} {'ARI':>7} "
      f"{'k_found':>8} {'Coverage':>9} {'Time(ms)':>10} {'Mem(KB)':>9}")
print('-' * 90)

for (dname, X, y), k_true in zip(datasets, k_trues):
    X = StandardScaler().fit_transform(X)
    all_X[dname] = X; all_y[dname] = y
    methods = make_methods(k_true)
    for mname in method_names:
        tracemalloc.start()
        t0 = time.perf_counter()
        lbl = methods[mname](X)
        t1 = time.perf_counter()
        _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
        ami = AMI(y, lbl); ari = ARI(y, lbl)
        cov = (lbl != -1).mean()
        k_f = len(set(lbl[lbl != -1]))
        ms = (t1 - t0) * 1000
        ami_table[mname][dname] = ami
        ari_table[mname][dname] = ari
        res_table[mname][dname] = {'t': ms, 'mem': pk/1024, 'cov': cov, 'k': k_f}
        all_labels[mname][dname] = lbl
        print(f"  {dname:16s} {mname:10s} {ami:7.4f} {ari:7.4f} "
              f"{k_f:>8} {cov:>9.1%} {ms:>9.1f}ms {pk/1024:>8.0f}KB")

# ─── Figure 1: Visual grid ────────────────────────────────────────────────────
dnames = [d[0] for d in datasets]
fig, axes = plt.subplots(len(dnames), len(method_names),
                          figsize=(3.0 * len(method_names), 2.8 * len(dnames)))
CMAPS = plt.cm.get_cmap('tab10', 10)

for row, dname in enumerate(dnames):
    X = all_X[dname]
    for col, mname in enumerate(method_names):
        ax = axes[row, col]
        lbl = all_labels[mname][dname]
        noise = lbl == -1
        unique_c = sorted(set(lbl[~noise]))
        for i, c in enumerate(unique_c):
            mask = lbl == c
            ax.scatter(X[mask, 0], X[mask, 1], c=[CMAPS(i)], s=8,
                       linewidths=0, alpha=0.8)
        if noise.any():
            ax.scatter(X[noise, 0], X[noise, 1], c=NOISE_COLOR, s=6,
                       linewidths=0, alpha=0.5)
        ami = ami_table[mname][dname]
        k_f = res_table[mname][dname]['k']
        ax.set_title(f'AMI={ami:.3f}  k={k_f}', fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_xlabel(mname, fontsize=9, fontweight='bold', labelpad=4)
            ax.xaxis.set_label_position('top')
        if col == 0:
            ax.set_ylabel(dname, fontsize=8, labelpad=4)

plt.tight_layout(pad=0.3)
plt.savefig('results/fig1_visual.pdf', bbox_inches='tight', dpi=200)
plt.savefig('results/fig1_visual.png', bbox_inches='tight', dpi=200)
plt.close()
print("\n[Saved] results/fig1_visual.pdf")

# ─── Figure 3: AMI bar chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(dnames)); w = 0.16
for i, mname in enumerate(method_names):
    vals = [ami_table[mname][d] for d in dnames]
    bars = ax.bar(x + (i - 2) * w, vals, w, label=mname,
                  color=METHOD_COLORS[mname], alpha=0.85, edgecolor='k', lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(dnames, fontsize=9)
ax.set_ylabel('Adjusted Mutual Information', fontsize=10)
ax.set_ylim(-0.1, 1.15)
ax.axhline(1.0, c='k', lw=0.6, ls=':')
ax.legend(fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.12))
ax.set_title('AMI on Standard Benchmarks (N=500)', fontsize=11)
plt.tight_layout()
plt.savefig('results/fig3_ami.pdf', bbox_inches='tight')
plt.savefig('results/fig3_ami.png', bbox_inches='tight', dpi=200)
plt.close()
print("[Saved] results/fig3_ami.pdf")

# ─── Figure 4: Memory bar chart ───────────────────────────────────────────────
# Average memory across datasets
fig, ax = plt.subplots(figsize=(7, 4))
avg_mem = {m: np.mean([res_table[m][d]['mem'] for d in dnames]) for m in method_names}
colors = [METHOD_COLORS[m] for m in method_names]
bars = ax.bar(method_names, [avg_mem[m] for m in method_names],
              color=colors, alpha=0.85, edgecolor='k', lw=0.5)
for bar, m in zip(bars, method_names):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{avg_mem[m]:.0f}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Peak Memory (KB)', fontsize=10)
ax.set_title('Peak Memory Usage (Average over 5 Datasets, N=500)', fontsize=10)
plt.tight_layout()
plt.savefig('results/fig4_memory.pdf', bbox_inches='tight')
plt.savefig('results/fig4_memory.png', bbox_inches='tight', dpi=200)
plt.close()
print("[Saved] results/fig4_memory.pdf")

# ─── Print numeric tables ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE: AMI")
print(f"{'Method':10s}", ''.join(f"{d[:12]:>13s}" for d in dnames), f"{'Mean':>8s}")
print('-' * 70)
for m in method_names:
    vals = [ami_table[m][d] for d in dnames]
    row = f"  {m:8s} " + ''.join(f"{v:>13.4f}" for v in vals) + f" {np.mean(vals):>8.4f}"
    print(row)

print("\nTABLE: ARI")
print(f"{'Method':10s}", ''.join(f"{d[:12]:>13s}" for d in dnames), f"{'Mean':>8s}")
print('-' * 70)
for m in method_names:
    vals = [ari_table[m][d] for d in dnames]
    row = f"  {m:8s} " + ''.join(f"{v:>13.4f}" for v in vals) + f" {np.mean(vals):>8.4f}"
    print(row)

print("\nTABLE: Resources (averaged over datasets)")
print(f"{'Method':10s}  {'Avg Time(ms)':>14s}  {'Avg Mem(KB)':>12s}  {'Avg Coverage':>13s}")
print('-' * 60)
for m in method_names:
    t_avg  = np.mean([res_table[m][d]['t']   for d in dnames])
    m_avg  = np.mean([res_table[m][d]['mem'] for d in dnames])
    c_avg  = np.mean([res_table[m][d]['cov'] for d in dnames])
    print(f"  {m:8s}  {t_avg:>13.1f}ms  {m_avg:>11.0f}KB  {c_avg:>12.1%}")

with open('results/table_ami.txt','w') as f:
    for m in method_names:
        vals = [ami_table[m][d] for d in dnames]
        f.write(f"{m}: " + ", ".join(f"{d}={v:.4f}" for d,v in zip(dnames,vals)) + "\n")
with open('results/table_ari.txt','w') as f:
    for m in method_names:
        vals = [ari_table[m][d] for d in dnames]
        f.write(f"{m}: " + ", ".join(f"{d}={v:.4f}" for d,v in zip(dnames,vals)) + "\n")

print("\nAll results saved to results/")

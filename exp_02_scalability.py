"""
GWCC Experiment 2: Scalability  (Figure 2 + Table 4 in paper)
=============================================================
Wall-clock time vs. dataset size N for all five algorithms.
Uses Gaussian Blobs as the test dataset (easiest for all methods).

Output
------
  results/fig2_scalability.pdf
  results/table_scalability.txt

Run
---
  python exp_02_scalability.py
"""

import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN
from gwcc import GWCC

SEED = 42
np.random.seed(SEED)
os.makedirs('results', exist_ok=True)

N_SIZES  = [200, 500, 1000, 2000, 5000, 10000]
N_TRIALS = 5   # average over this many runs
K_TRUE   = 4

METHOD_COLORS = {
    'GWCC':     '#1565C0',
    'DBSCAN':   '#F57F17',
    'HDBSCAN':  '#6A1B9A',
    'Spectral': '#2E7D32',
    'K-Means':  '#C62828',
}
METHOD_MARKERS = {
    'GWCC':'o', 'DBSCAN':'s', 'HDBSCAN':'^', 'Spectral':'D', 'K-Means':'v'
}

def make_method(name, k):
    return {
        'GWCC':     lambda X: GWCC().fit_predict(X),
        'DBSCAN':   lambda X: DBSCAN(eps=0.3, min_samples=5).fit_predict(X),
        'HDBSCAN':  lambda X: HDBSCAN(min_cluster_size=10).fit_predict(X),
        'Spectral': lambda X: SpectralClustering(
                        n_clusters=k, affinity='nearest_neighbors',
                        n_neighbors=10, random_state=SEED).fit_predict(X),
        'K-Means':  lambda X: KMeans(
                        n_clusters=k, n_init=10, random_state=SEED).fit_predict(X),
    }[name]

times = {m: [] for m in METHOD_COLORS}

print(f"\n{'N':>7s}", ''.join(f"  {m:>10s}" for m in METHOD_COLORS))
print('-' * 65)

for N in N_SIZES:
    X, _ = make_blobs(N, centers=K_TRUE, cluster_std=0.6, random_state=SEED)
    X = StandardScaler().fit_transform(X)
    row = f"{N:>7d}"
    for mname in METHOD_COLORS:
        fn = make_method(mname, K_TRUE)
        trial_times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            fn(X)
            t1 = time.perf_counter()
            trial_times.append((t1 - t0) * 1000)
        avg = np.mean(trial_times)
        times[mname].append(avg)
        row += f"  {avg:>9.1f}ms"
    print(row)

# ─── Figure 2: log-log scalability ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for mname in METHOD_COLORS:
    ax.loglog(N_SIZES, times[mname],
              color=METHOD_COLORS[mname],
              marker=METHOD_MARKERS[mname],
              linewidth=2, markersize=7,
              label=mname)
# Reference O(N log N) line
ref_N = np.array(N_SIZES, dtype=float)
ref_y = (ref_N * np.log2(ref_N)) / (N_SIZES[0] * np.log2(N_SIZES[0])) * times['GWCC'][0]
ax.loglog(N_SIZES, ref_y, 'k--', lw=1.2, alpha=0.6, label=r'$O(N\log N)$ reference')

ax.set_xlabel('Dataset size N', fontsize=11)
ax.set_ylabel('Wall-clock time (ms)', fontsize=11)
ax.set_title('Scalability: Wall-Clock Time vs. N (Gaussian Blobs, log–log)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xticks(N_SIZES)
ax.set_xticklabels([str(n) for n in N_SIZES], fontsize=8)
ax.grid(True, which='both', alpha=0.3)

# Annotate speedup at N=10000
gwcc_10k  = times['GWCC'][-1]
spec_10k  = times['Spectral'][-1]
speedup   = spec_10k / gwcc_10k
ax.annotate(f'GWCC {speedup:.1f}× faster\nthan Spectral at N=10k',
            xy=(10000, gwcc_10k), xytext=(3000, gwcc_10k * 3),
            arrowprops=dict(arrowstyle='->', lw=1.2),
            fontsize=8, color='#1565C0')

plt.tight_layout()
plt.savefig('results/fig2_scalability.pdf', bbox_inches='tight')
plt.savefig('results/fig2_scalability.png', bbox_inches='tight', dpi=200)
plt.close()
print(f"\n[Saved] results/fig2_scalability.pdf")
print(f"Speedup GWCC vs Spectral at N=10000: {speedup:.1f}×")

# ─── Save numeric table ───────────────────────────────────────────────────────
with open('results/table_scalability.txt', 'w') as f:
    f.write("Wall-clock time (ms), averaged over 5 runs, Gaussian Blobs\n")
    f.write(f"{'N':>7s}" + ''.join(f"  {m:>10s}" for m in METHOD_COLORS) + "\n")
    for i, N in enumerate(N_SIZES):
        line = f"{N:>7d}" + ''.join(f"  {times[m][i]:>9.1f}ms" for m in METHOD_COLORS)
        f.write(line + "\n")
print("[Saved] results/table_scalability.txt")

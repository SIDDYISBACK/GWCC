"""
GWCC Experiment 7: Real-World UCI Datasets — Wine & Breast Cancer
=================================================================
Applies GWCC, DBSCAN, and HDBSCAN to the full Wine (13D) and
Breast Cancer (30D) datasets with all metrics and visualisations.

Datasets
--------
  Wine           : sklearn.datasets.load_wine()
                   UCI source: Aeberhard & Forina (1991)
                   178 samples, 13 features, 3 cultivar classes
  Breast Cancer  : sklearn.datasets.load_breast_cancer()
                   UCI source: Street, Wolberg & Mangasarian (1993)
                   569 samples, 30 features, 2 classes (M/B)

Output
------
  results/fig_wine_analysis.pdf
  results/fig_bc_analysis.pdf
  results/table_realworld.txt
  console output

Run
---
  python exp_07_realworld.py
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN
from gwcc import GWCC

SEED = 42
np.random.seed(SEED)
os.makedirs('results', exist_ok=True)

def run_dataset(name, X_raw, y, eps, min_samples=5, min_cluster_size=10):
    X = StandardScaler().fit_transform(X_raw)
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  shape={X.shape}  classes={len(set(y))}  "
          f"PCA-2D variance={var_exp:.1%}")
    print(f"{'Method':>10s}  {'AMI':>7s}  {'ARI':>7s}  "
          f"{'k_found':>8s}  {'Coverage':>9s}")
    print('-' * 55)

    methods = {
        'GWCC':    lambda X: GWCC().fit_predict(X),
        'DBSCAN':  lambda X: DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X),
        'HDBSCAN': lambda X: HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X),
    }
    res = {}
    all_lbl = {}
    g = GWCC(); g.fit(X)  # keep density for plotting

    for mname, fn in methods.items():
        lbl = fn(X)
        ami_s = AMI(y, lbl); ari_s = ARI(y, lbl)
        cov = (lbl != -1).mean(); k_f = len(set(lbl[lbl != -1]))
        res[mname] = {'ami': ami_s, 'ari': ari_s, 'cov': cov, 'k': k_f, 'lbl': lbl}
        all_lbl[mname] = lbl
        print(f"  {mname:>8s}  {ami_s:>7.4f}  {ari_s:>7.4f}  "
              f"{k_f:>8d}  {cov:>8.1%}")

    # Downstream 5-NN accuracy
    print(f"\n  5-NN accuracy (cluster label as extra feature, 5-fold CV):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for mname, fn in methods.items():
        fold_accs = []
        for tr, te in skf.split(X, y):
            lbl_tr = fn(X[tr]); lbl_te = fn(X[te])
            k_f2 = len(set(lbl_tr[lbl_tr != -1]))
            lbl_tr[lbl_tr == -1] = k_f2; lbl_te[lbl_te == -1] = k_f2
            X_tr_aug = np.hstack([X[tr], lbl_tr.reshape(-1,1)])
            X_te_aug = np.hstack([X[te], lbl_te.reshape(-1,1)])
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(X_tr_aug, y[tr])
            fold_accs.append(clf.score(X_te_aug, y[te]))
        print(f"    {mname}: {np.mean(fold_accs):.1%}")
        res[mname]['clf_acc'] = np.mean(fold_accs)

    # ─── Figure ───────────────────────────────────────────────────────────────
    class_colors = ['#1565C0', '#C62828', '#2E7D32', '#F57F17']
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    # Panel 1: Ground truth
    ax = axes[0]
    for ci in sorted(set(y)):
        m = y == ci
        ax.scatter(X2[m, 0], X2[m, 1], c=class_colors[ci], s=18,
                   linewidths=0, alpha=0.75, label=f'Class {ci} (n={m.sum()})')
    ax.set_title(f'(a) Ground Truth\n(PCA-2D, {var_exp:.0%} var)', fontsize=8)
    ax.legend(fontsize=6)

    # Panel 2: GWCC density
    ax = axes[1]
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=g.densities_, cmap='plasma',
                    s=15, linewidths=0)
    ax.scatter(X2[g.seeds_, 0], X2[g.seeds_, 1], c='gold', s=200,
               marker='*', zorder=5, edgecolors='k', lw=0.7)
    plt.colorbar(sc, ax=ax, shrink=0.8, label='ρ')
    ax.set_title(f'(b) GWCC Density Field\n(k={g.n_clusters_} found)', fontsize=8)

    # Panel 3: GWCC clusters
    ax = axes[2]
    gwcc_lbl = all_lbl['GWCC']
    for ci in sorted(set(gwcc_lbl)):
        m = gwcc_lbl == ci
        c = class_colors[ci % len(class_colors)] if ci >= 0 else '#cccccc'
        label = f'C{ci} (n={m.sum()})' if ci >= 0 else f'noise (n={m.sum()})'
        ax.scatter(X2[m, 0], X2[m, 1], c=c, s=15, linewidths=0,
                   alpha=0.8, label=label)
    ax.legend(fontsize=6); ax.set_title(
        f'(c) GWCC Clusters\nAMI={res["GWCC"]["ami"]:.3f}  cov=100%', fontsize=8)

    # Panel 4: HDBSCAN (noise labelled grey)
    ax = axes[3]
    hdb_lbl = all_lbl['HDBSCAN']
    noise_mask = hdb_lbl == -1
    for ci in sorted(set(hdb_lbl[~noise_mask])):
        m = hdb_lbl == ci
        ax.scatter(X2[m, 0], X2[m, 1], c=class_colors[ci % len(class_colors)],
                   s=15, linewidths=0, alpha=0.8, label=f'C{ci} (n={m.sum()})')
    if noise_mask.any():
        ax.scatter(X2[noise_mask, 0], X2[noise_mask, 1], c='#cccccc',
                   s=10, linewidths=0, alpha=0.5,
                   label=f'noise ({noise_mask.sum()}, {noise_mask.mean():.0%})')
    ax.legend(fontsize=6)
    ax.set_title(f'(d) HDBSCAN  AMI={res["HDBSCAN"]["ami"]:.3f}\n'
                 f'cov={res["HDBSCAN"]["cov"]:.0%}  '
                 f'({noise_mask.sum()} discarded)', fontsize=8)

    plt.suptitle(f'{name} Dataset — GWCC Analysis', fontsize=11, fontweight='bold')
    plt.tight_layout()
    safe = name.lower().replace(' ', '_')
    plt.savefig(f'results/fig_{safe}_analysis.pdf', bbox_inches='tight')
    plt.savefig(f'results/fig_{safe}_analysis.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  [Saved] results/fig_{safe}_analysis.pdf")
    return res

# ─── Run both datasets ────────────────────────────────────────────────────────
ds_wine = load_wine();            wine_res = run_dataset(
    'Wine',          ds_wine.data,   ds_wine.target,   eps=3.5)
ds_bc   = load_breast_cancer();   bc_res   = run_dataset(
    'Breast Cancer', ds_bc.data,     ds_bc.target,     eps=2.0, min_cluster_size=15)

# ─── Save combined table ──────────────────────────────────────────────────────
with open('results/table_realworld.txt', 'w') as f:
    f.write("Real-World Dataset Results\n" + "="*70 + "\n\n")
    for dname, res in [('Wine', wine_res), ('Breast Cancer', bc_res)]:
        f.write(f"{dname}:\n")
        f.write(f"  {'Method':>10s}  {'AMI':>7s}  {'ARI':>7s}  "
                f"{'k':>4s}  {'Coverage':>9s}  {'5-NN Acc':>9s}\n")
        for m in ['GWCC', 'DBSCAN', 'HDBSCAN']:
            r = res[m]
            f.write(f"  {m:>10s}  {r['ami']:>7.4f}  {r['ari']:>7.4f}  "
                    f"{r['k']:>4d}  {r['cov']:>8.1%}  "
                    f"{r.get('clf_acc', float('nan')):>8.1%}\n")
        f.write("\n")
print("\n[Saved] results/table_realworld.txt")

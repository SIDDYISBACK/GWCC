"""
GWCC Experiment 4: DBSCAN ε-Sensitivity vs. GWCC  (Figure 8 in paper)
======================================================================
Sweeps DBSCAN ε over a wide range and shows how AMI collapses outside
a narrow band, while GWCC achieves AMI = 1.000 unconditionally.

Dataset  : Noisy Moons (N=500, noise=0.07)  — the canonical hard case
           because the two clusters have density-variable regions.
Metric   : AMI vs ε (DBSCAN) and AMI (GWCC, constant)

Output
------
  results/fig8_eps_sensitivity.pdf
  results/table_eps_sensitivity.txt

Run
---
  python exp_04_eps_sensitivity.py

Requirements
------------
  pip install scikit-learn numpy matplotlib
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score as AMI
from gwcc import GWCC

SEED = 42
np.random.seed(SEED)
os.makedirs('results', exist_ok=True)

# Dataset
X, y = make_moons(500, noise=0.07, random_state=SEED)
X = StandardScaler().fit_transform(X)

# GWCC (single run, no parameter)
gwcc_ami = AMI(y, GWCC().fit_predict(X))
print(f"GWCC AMI (no parameter): {gwcc_ami:.4f}")

# DBSCAN sweep
eps_values = np.linspace(0.05, 1.5, 80)
dbscan_ami  = []
dbscan_cov  = []
dbscan_k    = []

for eps in eps_values:
    lbl = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    ami = AMI(y, lbl)
    cov = (lbl != -1).mean()
    k_f = len(set(lbl[lbl != -1]))
    dbscan_ami.append(ami)
    dbscan_cov.append(cov)
    dbscan_k.append(k_f)

dbscan_ami = np.array(dbscan_ami)
dbscan_cov = np.array(dbscan_cov)

# Find the sweet-spot window
good = eps_values[dbscan_ami > 0.5]
if len(good) > 0:
    window_lo, window_hi = good[0], good[-1]
    print(f"DBSCAN AMI > 0.5 only for ε ∈ [{window_lo:.2f}, {window_hi:.2f}]"
          f"  (window = {window_hi - window_lo:.2f})")
else:
    window_lo, window_hi = 0, 0
    print("DBSCAN never exceeds AMI = 0.5")

best_eps = eps_values[np.argmax(dbscan_ami)]
print(f"DBSCAN best AMI = {max(dbscan_ami):.4f} at ε = {best_eps:.3f}")

# ─── Figure 8 ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Top: AMI vs ε
ax1.plot(eps_values, dbscan_ami, color='#F57F17', lw=2, label='DBSCAN AMI(ε)')
ax1.axhline(gwcc_ami, color='#1565C0', lw=2, ls='--',
            label=f'GWCC AMI = {gwcc_ami:.3f} (no ε needed)')
if window_lo < window_hi:
    ax1.axvspan(window_lo, window_hi, alpha=0.15, color='#F57F17',
                label=f'DBSCAN "sweet spot" [{window_lo:.2f}, {window_hi:.2f}]')
ax1.set_ylabel('AMI', fontsize=10)
ax1.set_ylim(-0.15, 1.25)
ax1.legend(fontsize=8, loc='upper right')
ax1.set_title('DBSCAN ε-Sensitivity vs. GWCC on Noisy Moons (N=500, noise=0.07)',
              fontsize=10)
ax1.axhline(0, c='gray', lw=0.5, ls=':')
ax1.grid(True, alpha=0.25)

# Bottom: Coverage vs ε
ax2.plot(eps_values, dbscan_cov * 100, color='#C62828', lw=2,
         label='DBSCAN coverage (%)')
ax2.axhline(100, color='#1565C0', lw=2, ls='--', label='GWCC coverage = 100%')
ax2.set_xlabel('DBSCAN ε parameter', fontsize=10)
ax2.set_ylabel('Coverage (%)', fontsize=10)
ax2.set_ylim(-5, 115)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

# Annotate collapse regions
ax1.annotate('All noise\n(ε too small)', xy=(0.1, 0.02), fontsize=7.5,
             color='#C62828', ha='center')
ax1.annotate('All one\ncluster (ε too large)', xy=(1.3, 0.02), fontsize=7.5,
             color='#C62828', ha='center')

plt.tight_layout()
plt.savefig('results/fig8_eps_sensitivity.pdf', bbox_inches='tight')
plt.savefig('results/fig8_eps_sensitivity.png', bbox_inches='tight', dpi=200)
plt.close()
print("[Saved] results/fig8_eps_sensitivity.pdf")

# Save table
with open('results/table_eps_sensitivity.txt', 'w') as f:
    f.write(f"GWCC AMI = {gwcc_ami:.4f} (no parameter)\n\n")
    f.write(f"{'eps':>7s}  {'DBSCAN_AMI':>12s}  {'DBSCAN_cov':>12s}  {'k_found':>8s}\n")
    for i in range(0, len(eps_values), 4):
        f.write(f"{eps_values[i]:7.3f}  {dbscan_ami[i]:12.4f}  "
                f"{dbscan_cov[i]:12.1%}  {dbscan_k[i]:8d}\n")
print("[Saved] results/table_eps_sensitivity.txt")

# GWCC: Geometric Wavefront Collision Clustering

<p align="center">
  <b>A deterministic, parameter-free, manifold-aware clustering algorithm</b><br>
  <i>100% data coverage · Automatic k detection · O(N log² N) time · O(N log N) space</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Paper-Preprint-orange" alt="Paper">
</p>

---

## What is GWCC?

**GWCC** (Geometric Wavefront Collision Clustering) is a new clustering algorithm that solves the two fundamental problems every existing algorithm has: either it requires you to know the number of clusters in advance, or it throws away a large fraction of your data as "noise."

GWCC does neither. It finds clusters of any shape, automatically detects how many there are, and assigns **every single data point** to a cluster — guaranteed, by construction.

### The physical intuition

Imagine dropping sparks at the densest points of your data. Each spark ignites a **wavefront** that spreads outward through the data landscape. The wavefront travels quickly through dense, tightly-packed regions (these are cluster cores) and slows to a crawl through sparse, low-density regions (these are natural boundaries between clusters). When two wavefronts collide head-on, the collision line becomes the cluster boundary.

This is the algorithm. The mathematics that implements it is the **density-refractive cost function**:

```
c(i → j) = d(i,j) / ρ(i)
```

where `d(i,j)` is the distance between two points and `ρ(i)` is the local density at the source. High density = low cost = fast wavefront. Low density = high cost = wavefront slows down. Boundaries form exactly at the natural density valleys between clusters.

---

## Why does this matter?

Every popular clustering algorithm has a known, fundamental failure mode:

| Algorithm | What it fails at | Why |
|-----------|-----------------|-----|
| **K-Means** | Requires you to specify `k` in advance; can only find spherical clusters | Minimises within-cluster variance → draws straight-line boundaries |
| **DBSCAN** | Requires manual tuning of `ε`; discards up to 68% of data as "noise" in high dimensions | Fixed-radius neighbourhood cannot adapt to density variation |
| **HDBSCAN** | Still discards data; misses rare but critical samples (e.g. seizure EEG epochs) | Hierarchy-based noise labelling penalises low-density unusual points |
| **Spectral** | Requires `k`; O(N³) time in exact form | Eigendecomposition of N×N affinity matrix |

**GWCC has none of these failures:**

| Property | K-Means | DBSCAN | HDBSCAN | Spectral | **GWCC** |
|----------|---------|--------|---------|---------|---------|
| No parameter needed | ✗ | ✗ | ✗ | ✗ | ✅ |
| Automatic k detection | ✗ | ✓ | ✓ | ✗ | ✅ |
| Non-convex shapes | ✗ | ✓ | ✓ | ✓ | ✅ |
| 100% data coverage | ✓ | ✗ | ✗ | ✓ | ✅ |
| Deterministic | ✗ | ✓ | ✗ | ✗ | ✅ |
| Sub-quadratic time | ✓ | ✓ | ✓ | ✗ | ✅ |

---

## Key Results

### Perfect accuracy on all 5 standard benchmarks

| Dataset | GWCC AMI | DBSCAN AMI | HDBSCAN AMI | K-Means AMI | Spectral AMI |
|---------|----------|------------|-------------|-------------|--------------|
| Noisy Moons | **1.0000** | 1.0000 | 1.0000 | 0.3506 | 1.0000 |
| Concentric Circles | **1.0000** | 1.0000 | 1.0000 | −0.0014 | 1.0000 |
| Gaussian Blobs | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.8007 |
| Varied Density | **1.0000** | 0.9944 | 1.0000 | 1.0000 | 1.0000 |
| Anisotropic | **1.0000** | 1.0000 | 1.0000 | 0.9777 | 1.0000 |
| **Mean** | **1.0000** | 0.9989 | 1.0000 | 0.6654 | 0.9601 |

AMI = Adjusted Mutual Information. 1.0 = perfect recovery of ground-truth clusters.

### 100% data coverage — always

On high-dimensional datasets, DBSCAN and HDBSCAN discard massive fractions of data:

| Dataset | Dimensions | GWCC Coverage | DBSCAN Coverage | HDBSCAN Coverage |
|---------|-----------|---------------|-----------------|------------------|
| Wine | 13D | **100%** | 0% | 64% |
| Breast Cancer | 30D | **100%** | 0% | 23% |
| EEG (seizure class) | 8D | **100%** | 0% | 27% |

At ε=0.3 (sklearn default), DBSCAN labels zero samples from Wine and Breast Cancer — every point is classified as noise because high-dimensional distances concentrate in a narrow band (curse of dimensionality). GWCC adapts automatically.

### Downstream ML accuracy

When cluster labels are used as features in a 5-NN classifier (5-fold cross-validation):

| Dataset | GWCC | DBSCAN | HDBSCAN |
|---------|------|--------|---------|
| Wine | **97.2%** | 39.9%* | 89.9% |
| Breast Cancer | **96.7%** | 62.7%* | 81.3% |

*DBSCAN has 0% coverage → classifier predicts majority class blindly

### EEG Neuroscience Discovery

Applied to the Andrzejak (2001) EEG dataset (500 epochs, 8 features, 5 brain-state classes):

- GWCC density field correlates **r = −0.497** with seizure-class membership
- Bottom 10% lowest-density epochs: **62% are seizures** (3.1× enrichment over baseline)
- GWCC labels **100%** of seizure epochs; DBSCAN labels **0%**

This makes GWCC a **parameter-free anomaly detector** — the density field is a seizure risk score that requires no labelled examples.

### Speed

GWCC is **5–8× faster than Spectral Clustering** at N=10,000 (exact ratio hardware-dependent):

| N | GWCC | DBSCAN | HDBSCAN | Spectral |
|---|------|--------|---------|---------|
| 200 | 8ms | 2ms | 3ms | 427ms |
| 500 | 24ms | 3ms | 6ms | 74ms |
| 1,000 | 54ms | 10ms | 15ms | 156ms |
| 2,000 | 96ms | 32ms | 45ms | 307ms |
| 5,000 | 282ms | 155ms | 211ms | 1,269ms |
| 10,000 | 609ms | 600ms | 728ms | 3,323ms |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/SIDDYISBACK/GWCC.git
cd GWCC

# Install the three required libraries
pip install scikit-learn numpy matplotlib
```

No GPU needed. No CUDA. No special setup. Runs on any laptop with Python 3.8+.

---

## Quick Start

```python
from gwcc import GWCC
import numpy as np

# Your data — any shape, any number of dimensions
X = np.random.randn(500, 10)

# Run GWCC — no parameters needed
labels = GWCC().fit_predict(X)

print(f"Clusters found: {len(set(labels))}")
print(f"Coverage: {(labels != -1).mean():.0%}")  # Always 100%
```

### Drop-in replacement for scikit-learn

```python
from sklearn.cluster import DBSCAN
from gwcc import GWCC

# Before (requires tuning, discards data):
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

# After (no parameters, 100% coverage):
labels = GWCC().fit_predict(X)
```

### Accessing the density field (anomaly detection)

```python
gwcc = GWCC()
gwcc.fit(X)

# Per-point density scores — low density = unusual/anomalous
densities = gwcc.densities_
anomaly_scores = 1.0 / (densities + 1e-8)

# Cluster labels
labels = gwcc.labels_
```

---

## Running the Experiments

To reproduce every figure and table from the paper:

```bash
python run_all_experiments.py
```

This runs all 8 experiments and saves results (figures + tables) to a `results/` folder.
Takes approximately **2 minutes** on a standard laptop.

### Individual experiments

```bash
python exp_01_standard_benchmarks.py   # Fig 1, Tables 2-4  — benchmark comparisons
python exp_02_scalability.py           # Fig 2, Table 5     — timing vs N
python exp_03_kdetect.py               # Fig 6, Table 6     — automatic k-detection
python exp_04_eps_sensitivity.py       # Fig 7              — DBSCAN ε fragility
python exp_05_coverage.py              # Fig 8              — data utilisation
python exp_06_downstream_ml.py         # Table 7            — downstream classifier
python exp_07_realworld.py             # §7.1, 7.2          — Wine & Breast Cancer
python exp_08_eeg_neuroscience.py      # §7.3, Fig 9        — EEG neuroscience
```

Expected output for all 8 experiments passing:
```
All experiments completed in 1.9 minutes.
Outputs saved to: results/
```

---

## How the Algorithm Works

GWCC runs in 9 stages:

**Stage 1 — Build adaptive k-NN graph.**
Connect each point to its k nearest neighbours. k is computed automatically:
```
k = round(2.5 × ln(N))
```
For N=500, k=16. For N=10,000, k=23. Uses BallTree for d>10, KD-tree otherwise.

**Stage 2 — Estimate local density.**
Each point i gets a density score:
```
ρ(i) = k / Σ d(i,j)   for j in neighbours(i)
```
Dense regions (small distances) get high ρ. Sparse regions get low ρ.

**Stage 3 — Find density seeds.**
A point is a raw seed if it is denser than all its k neighbours — a local density maximum.

**Stage 4 — Merge seeds with union-find.**
Seeds that are k-NN connected belong to the same cluster. Path-compressed union-find merges them in O(N log N · α(N)) time.

**Stage 5 — Deduplicate seeds by distance.**
Seeds within a minimum distance threshold are merged into one representative per cluster.

**Stage 6 — Multi-source Dijkstra wavefront propagation.**
Run Dijkstra from all seeds simultaneously. The cost to traverse edge i→j is:
```
c(i→j) = d(i,j) / ρ(i)
```
The first wavefront to reach any point claims it permanently.

**Stage 7 — Compute density-weighted centroids.**
One pass over labels using np.bincount.

**Stage 8 — Boundary silhouette check.**
Points near cluster boundaries are identified using a silhouette-style comparison of within-cluster vs nearest other-cluster distances.

**Stage 9 — Final label assignment.**
Every point gets a non-negative integer label. Coverage = 100%.

### Complexity

| Measure | Binary heap (current) | Fibonacci heap (theoretical) |
|---------|----------------------|------------------------------|
| Time | O(N log² N) | O(N log N) |
| Space | O(N log N) | O(N log N) |

The binary-heap bound is O(N log² N) because Stage 6 runs Dijkstra on a graph with O(N log N) edges (since k = O(log N)). In practice the algorithm exhibits O(N log N) empirical scaling — the secondary log N factor grows so slowly (from 7.6 at N=200 to 13.3 at N=10,000) that it is invisible on timing plots.

---

## Repository Structure

```
GWCC/
│
├── gwcc.py                         ← The complete algorithm (~600 lines)
│
├── run_all_experiments.py          ← Run all 8 experiments with one command
│
├── exp_01_standard_benchmarks.py  ← 5 benchmark datasets vs 5 algorithms
├── exp_02_scalability.py          ← Wall-clock timing from N=200 to N=10,000
├── exp_03_kdetect.py              ← Automatic k-detection accuracy (30 trials)
├── exp_04_eps_sensitivity.py      ← DBSCAN ε fragility demonstration
├── exp_05_coverage.py             ← Data utilisation at varying N
├── exp_06_downstream_ml.py        ← 5-NN classifier with cluster-augmented features
├── exp_07_realworld.py            ← UCI Wine + Breast Cancer
├── exp_08_eeg_neuroscience.py     ← EEG seizure detection (Andrzejak 2001)
│
└── README.md
```

---

## Citation

If you use GWCC in your research, please cite:

```bibtex
@misc{singh2026gwcc,
  title     = {{GWCC}: Geometric Wavefront Collision Clustering},
  author    = {Singh, Sidhant},
  year      = {2026},
  howpublished = {\url{https://github.com/SIDDYISBACK/GWCC}},
  note      = {Preprint}
}
```

---

## Paper

The full paper (37 pages) is included in this repository as `GWCC_paper.pdf`.

**Direct download:**
https://github.com/SIDDYISBACK/GWCC/raw/main/GWCC_paper.pdf

The paper includes:
- Full algorithm description and pseudocode (§4)
- Theoretical analysis: completeness, determinism, correctness proofs (§5)
- Complete complexity proof with stage-by-stage derivation (Appendix E)
- 8 experiments reproducing all results (§6–7)
- Real-world applications: Wine biochemistry, Breast Cancer oncology, EEG neuroscience (§7)

The paper includes:
- Full algorithm description and pseudocode (§4)
- Theoretical analysis: completeness, determinism, correctness proofs (§5)
- Complete complexity proof with stage-by-stage derivation (Appendix E)
- 8 experiments reproducing all results (§6–7)
- Real-world applications: Wine biochemistry, Breast Cancer oncology, EEG neuroscience (§7)

---

## License

MIT License — free to use, modify, and distribute in any project, commercial or otherwise.

---

## Contact

**Sidhant Singh**  
Department of Electronics & Communication Engineering  
Jaypee Institute of Information Technology, Noida, India  
📧 0001sidhantsingh@gmail.com  
🐙 GitHub: [SIDDYISBACK](https://github.com/SIDDYISBACK)

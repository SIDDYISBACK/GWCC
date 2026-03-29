"""
Run ALL GWCC Experiments
========================
This script runs every experiment in order and saves all outputs
to the results/ directory.

Usage
-----
  python run_all_experiments.py

Or run individual experiments:
  python exp_01_standard_benchmarks.py
  python exp_02_scalability.py
  python exp_03_kdetect.py
  python exp_04_eps_sensitivity.py
  python exp_05_coverage.py
  python exp_06_downstream_ml.py
  python exp_07_realworld.py
  python exp_08_eeg_neuroscience.py

Requirements
------------
  pip install scikit-learn numpy matplotlib

Estimated runtime: ~5–10 minutes on a typical laptop (exp_01 and
exp_08 are the slowest due to ApEn computation and AMI evaluations).
"""

import subprocess, sys, os, time

experiments = [
    ('exp_01_standard_benchmarks.py', 'Standard Benchmarks (Fig 1, 3, 4)'),
    ('exp_02_scalability.py',         'Scalability (Fig 2)'),
    ('exp_03_kdetect.py',             'k-Detection Accuracy (Fig 9)'),
    ('exp_04_eps_sensitivity.py',     'DBSCAN ε-Sensitivity (Fig 8)'),
    ('exp_05_coverage.py',            'Data Utilisation/Coverage (Fig 10)'),
    ('exp_06_downstream_ml.py',       'Downstream ML (Table 6)'),
    ('exp_07_realworld.py',           'Real-World: Wine & Breast Cancer (§7.1, 7.2)'),
    ('exp_08_eeg_neuroscience.py',    'Neuroscience EEG (§7.3, Fig EEG)'),
]

os.makedirs('results', exist_ok=True)
root = os.path.dirname(os.path.abspath(__file__))
total_start = time.time()

print("=" * 60)
print("GWCC: Running All Experiments")
print("=" * 60)

for i, (script, description) in enumerate(experiments, 1):
    print(f"\n[{i}/{len(experiments)}] {description}")
    print(f"           Script: {script}")
    print("-" * 60)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(root, script)],
        cwd=root
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  → {status}  ({elapsed:.1f}s)")

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"All experiments completed in {total/60:.1f} minutes.")
print(f"Outputs saved to: {os.path.join(root, 'results/')}")
print("=" * 60)

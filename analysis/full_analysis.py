"""
full_analysis.py
AID323 — SIR Epidemic Simulation
Generates performance charts from ACTUAL measured results.
Run from: ~/SIR_Project/
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Setup directories and styling
os.makedirs('report', exist_ok=True)
plt.rcParams.update({
    'figure.dpi': 150,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.4,
})

# ─────────────────────────────────────────────
#  ACTUAL MEASURED DATA (2000×2000 grid)
# ─────────────────────────────────────────────
T_SEQ = 7.218   # Sequential baseline (seconds)

configs = [
    # label         cores   time     compute   comm     comm_pct
    ("Seq (1c)",       1,   7.218,   7.218,    0.000,    0.0),
    ("1r×1t",          1,   8.236,   8.225,    0.004,    0.1),
    ("1r×2t",          2,   4.715,   4.705,    0.004,    0.1),
    ("2r×2t",          4,   3.320,   3.115,    0.198,    6.0),
    ("4r×2t",          8,   3.596,   2.605,    0.969,    26.9),
    ("4r×4t",         16,   9.073,   4.192,    4.856,    53.5),
]

labels   = [c[0] for c in configs]
cores    = np.array([c[1] for c in configs])
times    = np.array([c[2] for c in configs])
computes = np.array([c[3] for c in configs])
comms    = np.array([c[4] for c in configs])
comm_pct = np.array([c[5] for c in configs])

speedup    = T_SEQ / times
efficiency = (speedup / cores) * 100

# Ideal speedup line for plotting
ideal_cores = np.array([1, 2, 4, 8, 16])
ideal_speedup = ideal_cores.astype(float)

# ─────────────────────────────────────────────
#  CHART 1 — Execution Time vs Cores
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cores, times, 'o-', color='tab:red', linewidth=2, markersize=8, label='Measured time')
ax.axhline(T_SEQ, color='gray', linestyle='--', linewidth=1.2, label=f'Sequential baseline ({T_SEQ}s)')
ax.set_xlabel('Number of Cores (MPI ranks × OMP threads)')
ax.set_ylabel('Wall-clock Time (s)')
ax.set_title('Chart 1 — Execution Time vs Cores\n(2000×2000 grid, 500 timesteps)')
ax.set_xticks(cores)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.legend()
fig.tight_layout()
fig.savefig('report/chart_01_time_linear.png')
plt.close()

# ─────────────────────────────────────────────
#  CHART 2 — Speedup vs Cores
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ideal_cores, ideal_speedup, '--', color='gray', linewidth=1.5, label='Ideal linear speedup')
ax.plot(cores, speedup, 'D-', color='tab:blue', linewidth=2, markersize=8, label='Measured speedup')
ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':')

for i, (c, s) in enumerate(zip(cores, speedup)):
    ax.annotate(f'{s:.2f}x', (c, s), textcoords='offset points', xytext=(6, 4), fontsize=9, color='tab:blue')

ax.set_xlabel('Number of Cores')
ax.set_ylabel('Speedup S(p) = T_seq / T_par')
ax.set_title(f'Chart 2 — Speedup vs Cores\n(T_seq = {T_SEQ}s)')
ax.set_xticks(cores)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.legend()
fig.tight_layout()
fig.savefig('report/chart_02_speedup.png')
plt.close()

# ─────────────────────────────────────────────
#  CHART 3 — Parallel Efficiency
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['green' if e >= 50 else 'orange' if e >= 25 else 'red' for e in efficiency]
bars = ax.bar(labels, efficiency, color=colors, edgecolor='white', alpha=0.85)
ax.axhline(100, color='gray', linestyle='--', label='100% Efficiency')
ax.axhline(50, color='orange', linestyle=':', label='50% Threshold')

for bar, val in zip(bars, efficiency):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Efficiency E(p) = S(p)/p (%)')
ax.set_title('Chart 3 — Parallel Efficiency vs Cores')
ax.set_ylim(0, 120)
ax.legend()
fig.tight_layout()
fig.savefig('report/chart_03_efficiency.png')
plt.close()

# ─────────────────────────────────────────────
#  CHART 4 — Compute vs Comm Breakdown
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, computes, label='Compute time', color='tab:blue', alpha=0.85)
ax.bar(labels, comms, bottom=computes, label='Comm time (MPI)', color='tab:orange', alpha=0.85)

for i, (c, t) in enumerate(zip(comm_pct, times)):
    if c > 0.5:
        ax.text(i, t + 0.1, f'{c:.0f}% comm', ha='center', va='bottom', fontsize=8.5, color='darkorange', fontweight='bold')

ax.set_ylabel('Time (s)')
ax.set_title('Chart 4 — Compute vs Communication Breakdown')
ax.legend()
fig.tight_layout()
fig.savefig('report/chart_04_comm_breakdown.png')
plt.close()

# ─────────────────────────────────────────────
#  CHART 5 — Communication Overhead %
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cores, comm_pct, 's-', color='tab:orange', linewidth=2, markersize=8, label='Comm overhead %')
ax.fill_between(cores, comm_pct, alpha=0.15, color='tab:orange')

for i, (c, p) in enumerate(zip(cores, comm_pct)):
    ax.annotate(f'{p:.1f}%', (c, p), textcoords='offset points', xytext=(5, 5), fontsize=9, color='darkorange')

ax.set_xlabel('Cores')
ax.set_ylabel('Communication Overhead (%)')
ax.set_title('Chart 5 — MPI Communication Overhead vs Cores')
ax.set_xticks(cores)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylim(-5, 65)
ax.legend()
fig.tight_layout()
fig.savefig('report/chart_05_comm_overhead.png')
plt.close()

# ─────────────────────────────────────────────
#  CHART 6 — Amdahl's Law (Peak Performance Logic)
# ─────────────────────────────────────────────
# Use index 3 (2r x 2t) which is your actual peak performance core count
best_s = speedup[3]   
best_p = 4
f_amdahl = (1/best_s - 1/best_p) / (1 - 1/best_p)

p_range = np.array([1, 2, 4, 8, 16, 32])
s_amdahl = 1.0 / (f_amdahl + (1 - f_amdahl) / p_range)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p_range, s_amdahl, '--', color='tab:red', linewidth=2, label=f"Amdahl's law (f={f_amdahl:.2f})")
ax.plot(p_range, p_range, ':', color='gray', linewidth=1.5, label='Ideal (linear)')
ax.plot(cores, speedup, 'D', color='tab:blue', markersize=9, label='Measured speedup', zorder=5)

ax.set_xlabel('Number of Cores (p)')
ax.set_ylabel('Speedup S(p)')
ax.set_title(f"Chart 6 — Amdahl's Law Analysis\nSerial fraction f ≈ {f_amdahl:.2f}")
ax.legend()
fig.tight_layout()
fig.savefig('report/chart_06_amdahl.png')
plt.close()

# ─────────────────────────────────────────────
#  CHART 9 — Throughput
# ─────────────────────────────────────────────
GRID_UPDATES = 2000 * 2000 * 500 
throughput = GRID_UPDATES / times / 1e6 

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, throughput, color='steelblue', alpha=0.85)
for bar, val in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}M', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Throughput (Million cell-updates / s)')
ax.set_title('Chart 9 — Simulation Throughput')
fig.tight_layout()
fig.savefig('report/chart_09_throughput.png')
plt.close()

# ─────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Config':<12} {'Cores':>5} {'Time(s)':>9} {'Comm%':>8} {'Speedup':>9} {'Effic%':>8}")
print("-"*75)
for i, c in enumerate(configs):
    print(f"{c[0]:<12} {c[1]:>5} {c[2]:>9.3f} {c[5]:>7.1f}% {speedup[i]:>8.2f}x {efficiency[i]:>7.1f}%")
print("="*75)
print(f"\nT_seq = {T_SEQ}s | Max Speedup = {max(speedup):.2f}x | Amdahl f = {f_amdahl:.2f}")
print("\n✓ All charts saved to report/")

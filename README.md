# Epidemic Spread Simulation — SIR Model

**Author:** Nour Ezz 

---

## Overview

A hybrid **MPI + OpenMP** parallel implementation of a spatial 2D SIR (Susceptible–Infected–Recovered) epidemic simulation. Each cell in a 2000 × 2000 grid represents one individual; infection spreads probabilistically to neighbouring cells each timestep.

| Parameter | Value |
|-----------|-------|
| Grid size | 2000 × 2000 (4 million cells) |
| Timesteps | 500 |
| β (transmission rate) | 0.30 |
| γ (recovery rate) | 0.10 |
| R₀ = β/γ | 3.00 |
| Initial infected | 5 random seeds |

---

## Repository Structure

```
SIR_Project/
├── seq/
│   ├── sir_seq.c          # Sequential implementation (v3.0)
│   └── Makefile
├── parallel/
│   ├── sir_hybrid.c       # Hybrid MPI + OpenMP (v3.0)
│   └── Makefile
├── results/
│   ├── scaling_results.csv
│   ├── sir_seq_results.csv
│   ├── seq_timing.txt
│   ├── hybrid_timing.txt
│   └── run_*.txt          # Raw output from each configuration
├── analysis/
│   └── full_analysis.py   # Generates all performance charts
├── report/
│   ├── SIR_Report.docx
│   └── chart_*.png        # All 9 performance charts
└── README.md
```

---

## Quick Start

### Requirements
```bash
sudo apt-get install gcc libopenmpi-dev openmpi-bin
pip install matplotlib numpy
```

### Build & Run Sequential
```bash
cd seq
make
./sir_seq
```

### Build & Run Hybrid (MPI + OpenMP)
```bash
cd parallel
make

# 1 rank, 1 thread
OMP_NUM_THREADS=1 mpirun -np 1 ./sir_hybrid

# 2 ranks × 2 threads (best speedup config)
OMP_NUM_THREADS=2 mpirun -np 2 ./sir_hybrid

# 4 ranks × 2 threads
OMP_NUM_THREADS=2 mpirun -np 4 ./sir_hybrid
```

### Generate Performance Charts
```bash
python3 analysis/full_analysis.py
```

---

## Implementation

### Sequential (`seq/sir_seq.c`)

**Pull-pattern update** — each cell reads its neighbours and writes only to itself:

```c
if (state == S) {
    // count infected neighbours
    int inf_nb = 0;
    if (grid[IDX(row-1,col)] == I) inf_nb++;
    if (grid[IDX(row+1,col)] == I) inf_nb++;
    if (col > 0 && grid[IDX(row,col-1)] == I) inf_nb++;
    if (col < GRID_SIZE-1 && grid[IDX(row,col+1)] == I) inf_nb++;

    // each infected neighbour gives one independent chance
    for (int n = 0; n < inf_nb; n++)
        if (rand_r(seed) < BETA_THRESH) { new_grid[idx] = I; break; }
}
if (state == I)
    if (rand_r(seed) < GAMMA_THRESH) new_grid[idx] = R;
```

**Key optimisations:**
- Integer threshold comparison (`BETA_THRESH`, `GAMMA_THRESH`) — no floating-point division per cell
- `rand_r(&seed)` — thread-safe RNG
- Pointer swap each step — no 16 MB memcpy
- Early exit when infected count = 0

### Parallel (`parallel/sir_hybrid.c`)

**Domain decomposition:** N × N grid split into horizontal row slabs, one per MPI rank.

```
Rank 0: [ghost_top | rows   0–499 | ghost_bot]
Rank 1: [ghost_top | rows 500–999 | ghost_bot]
Rank 2: [ghost_top | rows 1000–1499 | ghost_bot]
Rank 3: [ghost_top | rows 1500–1999 | ghost_bot]
```

**Halo exchange** (every step):
```c
MPI_Isend(first_row, GRID_SIZE, MPI_INT, top,    0, comm, &reqs[0]);
MPI_Irecv(ghost_top, GRID_SIZE, MPI_INT, top,    1, comm, &reqs[1]);
MPI_Isend(last_row,  GRID_SIZE, MPI_INT, bottom, 1, comm, &reqs[2]);
MPI_Irecv(ghost_bot, GRID_SIZE, MPI_INT, bottom, 0, comm, &reqs[3]);
MPI_Waitall(4, reqs, stats);
```

**OpenMP** (inside each rank):
```c
#pragma omp parallel for schedule(static) \
    reduction(+: s_acc, i_acc, r_acc)
for (int row = 1; row <= local_rows; row++) {
    unsigned int *seed = &thread_seeds[omp_get_thread_num()];
    // ... cell update
}
```

**MPI_Reduce** called every 10 steps only (not every step) to cut collective overhead by 10×.

---

## Results

### Performance Table

| Config | Cores | Time (s) | Speedup | Efficiency |
|--------|-------|----------|---------|------------|
| Sequential | 1 | 7.218 | 1.00× | 100% |
| 1 rank × 1 thread | 1 | 8.236 | 0.88× | 87.6% |
| 1 rank × 2 threads | 2 | 4.715 | 1.53× | 76.5% |
| 2 ranks × 2 threads | 4 | 3.320 | **2.17×** | 54.4% |
| 4 ranks × 2 threads | 8 | 3.596 | 2.01× | 25.1% |
| 4 ranks × 4 threads | 16 | 9.073 | 0.80× | 5.0% |

**T_seq = 7.218 s · Best speedup = 2.17× at 4 cores · Conservation check = 0 on every run**

### Key Findings

- **Best configuration:** 2 MPI ranks × 2 OpenMP threads (4 cores) — 2.17× speedup, 54.4% efficiency
- **Bottleneck:** Fixed-size halo exchange (16 KB/rank/step). As ranks increase, compute shrinks but communication stays constant
- **Amdahl serial fraction:** f ≈ 0.28 → theoretical maximum speedup ≈ 3.5×
- **4r×4t failure:** At 16 cores, communication (4.86 s) exceeds compute (4.19 s) → 26% slower than sequential

### Communication Overhead Growth

| Config | Comm time | Comm % |
|--------|-----------|--------|
| 1r×1t  | 0.004 s   | 0.1%   |
| 2r×2t  | 0.198 s   | 6.0%   |
| 4r×2t  | 0.969 s   | 26.9%  |
| 4r×4t  | 4.856 s   | 53.5%  |

---

## Correctness

Population conservation `S + I + R = N` verified on every run:
```
║  Conservation check (must be 0): 0               ║
```

All 6 configurations returned conservation = 0, confirming the pull-pattern update logic is correct.

---

## Proposed Optimisations

1. **Non-blocking halo overlap** — compute interior rows while ghost rows are in transit; call `MPI_Waitall` only before boundary rows. Hides communication latency completely.
2. **2D domain decomposition** — split into P×Q blocks; communication volume grows as O(N/√P) instead of O(N).
3. **Larger grid** — 4000×4000 or 8000×8000 improves compute-to-communication ratio at 8+ cores.
4. **Increase REDUCE_INTERVAL** — from 10 to 50 steps reduces collective calls from 150 to 30 per run.
5. **Single MPI_Allreduce** — combine S, I, R into one 3-element reduce instead of three separate calls.

---

## Files Description

| File | Description |
|------|-------------|
| `seq/sir_seq.c` | Sequential SIR, pull-pattern, integer thresholds, conservation check |
| `parallel/sir_hybrid.c` | Hybrid MPI+OpenMP, row slab decomp, non-blocking halo, per-thread RNG |
| `seq/Makefile` | `make` / `make run` / `make clean` |
| `parallel/Makefile` | `make` / `make run` / `make clean` |
| `analysis/full_analysis.py` | Generates 9 performance charts from measured data |
| `results/scaling_results.csv` | All timing measurements in CSV format |
| `report/SIR_Report.docx` | Full written report (8 sections, ~15 pages) |

---

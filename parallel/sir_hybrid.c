/*
 *   SIR EPIDEMIC SPREAD SIMULATION — HYBRID MPI + OpenMP
*/

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

/* ── Parameters ─────────────────────────────────────────────── */
#define GRID_SIZE        2000
#define TIMESTEPS        500
#define BETA             0.3
#define GAMMA            0.1
#define INIT_INF         5
#define REDUCE_INTERVAL  10

/* ── Cell state constants ───────────────────────────────────── */
#define S_STATE  0
#define I_STATE  1
#define R_STATE  2

/* ── Integer thresholds — avoid FP division per cell ────────── */
#define BETA_THRESH  ((int)(BETA  * (double)RAND_MAX))
#define GAMMA_THRESH ((int)(GAMMA * (double)RAND_MAX))

/*
 * Local grid layout (per rank):
 *   row 0              = TOP ghost row    (from rank-1)
 *   rows 1..local_rows = real data rows
 *   row local_rows+1   = BOTTOM ghost row (from rank+1)
 *
 * LIDX maps (row, col) to flat index in this layout.
 */
#define LIDX(row, col)  ((row) * GRID_SIZE + (col))

/* ── Prototypes ─────────────────────────────────────────────── */
void init_full_grid   (int *grid, unsigned int seed);
void distribute_grid  (int *full_grid, int *local_grid,
                       int local_rows, int rank, int num_ranks,
                       MPI_Comm comm);
void halo_exchange    (int *local_grid, int local_rows,
                       int rank, int num_ranks, MPI_Comm comm);
void step_and_count   (int * restrict local_grid,
                       int * restrict new_local_grid,
                       int local_rows,
                       unsigned int *thread_seeds,
                       long *ls, long *li, long *lr);
void gather_results   (int *local_grid, int *full_grid,
                       int local_rows, int rank, int num_ranks,
                       MPI_Comm comm);
void print_banner     (int rank, int num_ranks, int threads);
void print_summary    (long s, long i, long r,
                       double compute, double comm, double total);

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* Grid must divide evenly across ranks */
    if (GRID_SIZE % num_ranks != 0) {
        if (rank == 0)
            fprintf(stderr,
                "[ERROR] GRID_SIZE (%d) must be divisible by "
                "num_ranks (%d)\n", GRID_SIZE, num_ranks);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_rows  = GRID_SIZE / num_ranks;
    int num_threads = omp_get_max_threads();
    print_banner(rank, num_ranks, num_threads);

    /* ── Allocate local grid: real rows + 2 ghost rows ──────── */
    int local_total  = (local_rows + 2) * GRID_SIZE;
    int *local_grid  = (int *)calloc(local_total, sizeof(int));
    int *new_local   = (int *)calloc(local_total, sizeof(int));

    /* Rank 0 holds full grid for init + scatter/gather */
    int *full_grid = NULL;
    if (rank == 0) {
        full_grid = (int *)malloc(
            (size_t)GRID_SIZE * GRID_SIZE * sizeof(int));
        if (!full_grid) {
            fprintf(stderr, "[ERROR] Rank 0: full grid malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    if (!local_grid || !new_local) {
        fprintf(stderr, "[ERROR] Rank %d: local malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ── Per-thread RNG seeds (unique per rank and per thread) ── */
    unsigned int *thread_seeds = (unsigned int *)malloc(
        num_threads * sizeof(unsigned int));
    unsigned int base_seed =
        (unsigned int)time(NULL) ^ ((unsigned int)rank * 0x9E3779B9u);
    for (int t = 0; t < num_threads; t++)
        thread_seeds[t] = base_seed + (unsigned int)(t * 1234567u);

    /* ── Initialise and distribute grid ─────────────────────── */
    if (rank == 0)
        init_full_grid(full_grid, base_seed);

    distribute_grid(full_grid, local_grid, local_rows,
                    rank, num_ranks, MPI_COMM_WORLD);

    /* ── CSV log (rank 0 only) ──────────────────────────────── */
    FILE *fp = NULL;
    if (rank == 0) {
        fp = fopen("../results/sir_hybrid_results.csv", "w");
        if (fp)
            fprintf(fp,
                "timestep,susceptible,infected,"
                "recovered,compute_sec,comm_sec\n");
        else
            fprintf(stderr,
                "[WARN] Cannot open results/ — no CSV written.\n");
    }

    /* ── Initial global count ───────────────────────────────── */
    long ls = 0, li = 0, lr = 0; // local counts
    long gs = 0, gi = 0, gr = 0; // global counts
    for (int row = 1; row <= local_rows; row++)
        for (int col = 0; col < GRID_SIZE; col++) {
            int st = local_grid[LIDX(row, col)];
            if      (st == S_STATE) ls++;
            else if (st == I_STATE) li++;
            else                    lr++;
        }
    MPI_Reduce(&ls, &gs, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&li, &gi, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lr, &gr, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("[t=  0] S=%-8ld  I=%-8ld  R=%-8ld\n", gs, gi, gr);

    /* ── Timing accumulators ────────────────────────────────── */
    double total_compute = 0.0, total_comm = 0.0;
    double wall_start    = MPI_Wtime();

    /* ================================================================
     *  MAIN SIMULATION LOOP
     * ================================================================ */
    for (int t = 1; t <= TIMESTEPS; t++) {

        /* ── Step 1: Halo exchange (ghost rows) ──────────────── */
        double comm_t0 = MPI_Wtime();
        halo_exchange(local_grid, local_rows, rank, num_ranks,
                      MPI_COMM_WORLD);
        total_comm += MPI_Wtime() - comm_t0;

        /* ── Step 2: Compute one timestep (OpenMP inside) ───── */
        double comp_t0 = MPI_Wtime();
        step_and_count(local_grid, new_local, local_rows,
                       thread_seeds, &ls, &li, &lr);

        /* Swap grid pointers — no memcpy */
        int *tmp   = local_grid;
        local_grid = new_local;
        new_local  = tmp;

        total_compute += MPI_Wtime() - comp_t0;

        /* ── Step 3: Reduce global counts periodically ───────── */
        if (t % REDUCE_INTERVAL == 0 || t == 1 || t == TIMESTEPS) {

            double comm_r0 = MPI_Wtime();
            MPI_Reduce(&ls, &gs, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&li, &gi, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&lr, &gr, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            total_comm += MPI_Wtime() - comm_r0;

            if (rank == 0) {
                printf("[t=%3d] S=%-8ld  I=%-8ld  R=%-8ld\n",
                       t, gs, gi, gr);
                if (fp)
                    fprintf(fp, "%d,%ld,%ld,%ld,%.6f,%.6f\n",
                            t, gs, gi, gr, total_compute, total_comm);
            }

            /* ── Step 4: Early exit — broadcast decision ─────── */
            int done = (rank == 0 && gi == 0) ? 1 : 0;
            MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (done) {
                if (rank == 0)
                    printf("\n[INFO]  Epidemic ended at timestep %d\n", t);
                break;
            }
        }
    }

    double total_time = MPI_Wtime() - wall_start;

    /* ── Summary and timing file ─────────────────────────────── */
    if (rank == 0) {
        print_summary(gs, gi, gr, total_compute, total_comm, total_time);

        FILE *tp = fopen("../results/hybrid_timing.txt", "w");
        if (tp) {
            fprintf(tp, "PARALLEL_TIME_SECONDS=%.6f\n",  total_time);
            fprintf(tp, "COMPUTE_TIME_SECONDS=%.6f\n",   total_compute);
            fprintf(tp, "COMM_TIME_SECONDS=%.6f\n",      total_comm);
            fprintf(tp, "COMM_OVERHEAD_PCT=%.2f\n",
                    total_comm / total_time * 100.0);
            fprintf(tp, "MPI_RANKS=%d\n",   num_ranks);
            fprintf(tp, "OMP_THREADS=%d\n", num_threads);
            fprintf(tp, "GRID_SIZE=%d\n",   GRID_SIZE);
            fprintf(tp, "TIMESTEPS=%d\n",   TIMESTEPS);
            fclose(tp);
            printf("[INFO]  Timing saved → results/hybrid_timing.txt\n");
        }
        if (fp) fclose(fp);
        free(full_grid);
    }

    free(local_grid);
    free(new_local);
    free(thread_seeds);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

/* ================================================================
 *  init_full_grid  (rank 0 only)
 *  memset 0 is safe: S_STATE == 0, 0x00000000 fills int correctly.
 * ================================================================ */
void init_full_grid(int *grid, unsigned int seed)
{
    long total = (long)GRID_SIZE * GRID_SIZE;
    memset(grid, 0, (size_t)total * sizeof(int));

    srand(seed);
    int placed = 0;
    while (placed < INIT_INF) {
        int pos = rand() % (int)total;
        if (grid[pos] == S_STATE) {
            grid[pos] = I_STATE;
            placed++;
        }
    }
    printf("[INFO]  Grid initialised: %d infected seeds placed\n\n",
           INIT_INF);
}

/* ================================================================
 *  distribute_grid — rank 0 scatters row slabs to all ranks
 * ================================================================ */
void distribute_grid(int *full_grid, int *local_grid,
                     int local_rows, int rank, int num_ranks,
                     MPI_Comm comm)
{
    int slab = local_rows * GRID_SIZE;
    /* Scatter into row 1 of local_grid (skip ghost row 0) */
    MPI_Scatter(full_grid,
                slab, MPI_INT,
                local_grid + GRID_SIZE,   /* row 1 = first real row */
                slab, MPI_INT,
                0, comm);
    (void)rank; (void)num_ranks;
}

/* ================================================================
 *  halo_exchange — non-blocking ghost row exchange
 *
 *  Each rank sends its first real row up and last real row down,
 *  and receives ghost rows from its neighbours simultaneously.
 *  MPI_PROC_NULL silently discards sends/receives at boundaries.
 * ================================================================ */
void halo_exchange(int *local_grid, int local_rows,
                   int rank, int num_ranks, MPI_Comm comm)
{
    int top    = (rank > 0)             ? rank - 1 : MPI_PROC_NULL;
    int bottom = (rank < num_ranks - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Pointers into local_grid */
    int *first_real = local_grid + GRID_SIZE;              /* row 1 */
    int *last_real  = local_grid + local_rows * GRID_SIZE; /* row local_rows */
    int *ghost_top  = local_grid;                           /* row 0 */
    int *ghost_bot  = local_grid + (local_rows + 1) * GRID_SIZE; /* row local_rows+1 */

    MPI_Request reqs[4];
    MPI_Status  stats[4];

    /* Post all four operations simultaneously — overlap comm */
    MPI_Isend(first_real, GRID_SIZE, MPI_INT, top,    0, comm, &reqs[0]);
    MPI_Irecv(ghost_top,  GRID_SIZE, MPI_INT, top,    1, comm, &reqs[1]);
    MPI_Isend(last_real,  GRID_SIZE, MPI_INT, bottom, 1, comm, &reqs[2]);
    MPI_Irecv(ghost_bot,  GRID_SIZE, MPI_INT, bottom, 0, comm, &reqs[3]);

    MPI_Waitall(4, reqs, stats);
}

/* ================================================================
 *  step_and_count — PULL pattern (v3 fix)
 *
 *  For each real cell:
 *    S : counts infected neighbours (including ghost rows) → maybe I
 *    I : maybe recovers → R
 *    R : stays R
 *
 *  KEY PROPERTY: every cell writes ONLY to new_local_grid[its own idx].
 *    - Ghost rows are READ only → safe, no writes to neighbour memory.
 *    - OpenMP threads never write to the same location → no races.
 *    - Cross-boundary infections work: row=1 can see ghost row 0.
 *
 *  Returns accurate local S/I/R counts via ls/li/lr.
 *  No rescan needed — counts come directly from the write loop.
 * ================================================================ */
void step_and_count(int * restrict local_grid,
                    int * restrict new_local_grid,
                    int local_rows,
                    unsigned int *thread_seeds,
                    long *ls, long *li, long *lr)
{
    long s_acc = 0, i_acc = 0, r_acc = 0;

    #pragma omp parallel for schedule(static)              \
        reduction(+: s_acc, i_acc, r_acc)                 \
        shared(local_grid, new_local_grid, thread_seeds)   \
        default(none)                                       \
        firstprivate(local_rows)
    for (int row = 1; row <= local_rows; row++) {

        unsigned int *seed = &thread_seeds[omp_get_thread_num()];

        for (int col = 0; col < GRID_SIZE; col++) {

            int idx   = LIDX(row, col);
            int state = local_grid[idx];

            /* ── Recovered: stays recovered ──────────────────── */
            if (state == R_STATE) {
                new_local_grid[idx] = R_STATE;
                r_acc++;
                continue;
            }

            /* ── Susceptible: PULL infection from neighbours ─── */
            if (state == S_STATE) {
                /*
                 * Count infected neighbours.
                 * Row 0 is ghost (from rank above) — valid to READ.
                 * Row local_rows+1 is ghost (from rank below) — valid to READ.
                 * Ghost rows hold the real border rows of adjacent ranks,
                 * so cross-boundary infections are correctly captured.
                 */
                int inf_nb = 0;
                if (local_grid[LIDX(row - 1, col)] == I_STATE) inf_nb++;
                if (local_grid[LIDX(row + 1, col)] == I_STATE) inf_nb++;
                if (col > 0             && local_grid[LIDX(row, col-1)] == I_STATE) inf_nb++;
                if (col < GRID_SIZE - 1 && local_grid[LIDX(row, col+1)] == I_STATE) inf_nb++;

                /* Each infected neighbour gives one independent chance */
                int got_infected = 0;
                for (int n = 0; n < inf_nb; n++) {
                    if (rand_r(seed) < BETA_THRESH) {
                        got_infected = 1;
                        break;
                    }
                }

                /* Write only to self */
                if (got_infected) {
                    new_local_grid[idx] = I_STATE;
                    i_acc++;
                } else {
                    new_local_grid[idx] = S_STATE;
                    s_acc++;
                }
                continue;
            }

            /* ── Infected: try to recover ────────────────────── */
            if (rand_r(seed) < GAMMA_THRESH) {
                new_local_grid[idx] = R_STATE;
                r_acc++;
            } else {
                new_local_grid[idx] = I_STATE;
                i_acc++;
            }
        }
    }

    *ls = s_acc;
    *li = i_acc;
    *lr = r_acc;
}

/* ================================================================
 *  gather_results — collect full grid to rank 0 (optional)
 * ================================================================ */
void gather_results(int *local_grid, int *full_grid,
                    int local_rows, int rank, int num_ranks,
                    MPI_Comm comm)
{
    int slab = local_rows * GRID_SIZE;
    MPI_Gather(local_grid + GRID_SIZE, slab, MPI_INT,
               full_grid,              slab, MPI_INT,
               0, comm);
    (void)rank; (void)num_ranks;
}

/* ================================================================
 *  Pretty print
 * ================================================================ */
void print_banner(int rank, int num_ranks, int threads)
{
    if (rank != 0) return;
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  SIR EPIDEMIC SIMULATION — HYBRID MPI+OpenMP v3  ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  MPI ranks   : %-5d                             ║\n", num_ranks);
    printf("║  OMP threads : %-5d  (per rank)                 ║\n", threads);
    printf("║  Total cores : %-5d                             ║\n",
           num_ranks * threads);
    printf("║  Grid size   : %d x %d                      ║\n",
           GRID_SIZE, GRID_SIZE);
    printf("║  Timesteps   : %-5d                             ║\n", TIMESTEPS);
    printf("║  Beta (β)    : %.2f   Gamma (γ) : %.2f           ║\n",
           BETA, GAMMA);
    printf("║  R₀          : %.2f                              ║\n",
           BETA / GAMMA);
    printf("║  Reduce every: %-5d steps                       ║\n",
           REDUCE_INTERVAL);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

void print_summary(long s, long i, long r,
                   double compute, double comm, double total)
{
    long pop   = (long)GRID_SIZE * GRID_SIZE;
    long check = pop - s - i - r;   /* must be 0 */
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║           PARALLEL — FINAL RESULTS               ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Susceptible : %-8ld  (%.1f%%)                 ║\n",
           s, (double)s / pop * 100.0);
    printf("║  Infected    : %-8ld  (%.1f%%)                 ║\n",
           i, (double)i / pop * 100.0);
    printf("║  Recovered   : %-8ld  (%.1f%%)                 ║\n",
           r, (double)r / pop * 100.0);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Conservation check (must be 0): %-8ld        ║\n", check);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Total time  : %.4f s                           ║\n", total);
    printf("║  Compute     : %.4f s  (%.1f%%)                ║\n",
           compute, compute / total * 100.0);
    printf("║  Comm (MPI)  : %.4f s  (%.1f%%)                ║\n",
           comm, comm / total * 100.0);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}
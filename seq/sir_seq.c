/*
 * ============================================================
 *   SIR EPIDEMIC SPREAD SIMULATION — SEQUENTIAL VERSION
 * ============================================================
 *
 *   Model   : Spatial SIR on a 2D grid
 *
 *   Bug fixes over v2:
 *     - PULL pattern replaces PUSH pattern in step():
 *       Each susceptible cell checks its own neighbors.
 *       Each cell only ever writes to itself → no race conditions.
 *       (v2 had infected cells writing into neighbor memory — unsafe
 *        in parallel and logically inconsistent with SIR model)
 *     - memset(grid, S, ...) replaced with memset(grid, 0, ...)
 *       memset fills bytes not ints; only 0 is safe for int arrays.
 *     - Rescan loop removed: counts maintained accurately in single pass.
 *
 *   Optimisations kept from v2:
 *     - Integer threshold comparison (BETA_THRESH, GAMMA_THRESH)
 *     - clock_gettime(CLOCK_MONOTONIC) for accurate timing
 *     - rand_r() thread-safe RNG
 *     - Early epidemic-over detection
 *     - Per-step timing in CSV
 *     - Timing saved to results/seq_timing.txt
 *
 *   Cell states: 0=Susceptible  1=Infected  2=Recovered
 * ============================================================
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Simulation Parameters ─────────────────────────────────── */
#define GRID_SIZE   2000
#define TIMESTEPS   500
#define BETA        0.3
#define GAMMA       0.1
#define INIT_INF    5

/* ── Cell state constants ───────────────────────────────────── */
#define S  0
#define I  1
#define R  2

/* ── Pre-computed integer thresholds ───────────────────────── */
/* rand_r returns [0, RAND_MAX]; compare integer directly       */
#define BETA_THRESH  ((int)(BETA  * (double)RAND_MAX))
#define GAMMA_THRESH ((int)(GAMMA * (double)RAND_MAX))

/* ── Macros ─────────────────────────────────────────────────── */
#define IDX(row, col)  ((row) * GRID_SIZE + (col))
#define TOTAL          ((long)GRID_SIZE * GRID_SIZE)

/* ── Prototypes ─────────────────────────────────────────────── */
void  init_grid(int *grid, unsigned int seed); // Fills the world with healthy people and 5 sick ones
long  step(int * restrict grid, int * restrict new_grid,
           unsigned int *seed, long *i_out, long *r_out); // Runs one timestep, returns new S count, sets *i_out and *r_out
void  log_header(FILE *fp); // Writes CSV header line
void  log_step(FILE *fp, int t, long s, long iv, long rv, double elapsed);
void  print_banner(void);
void  print_summary(long s, long iv, long rv, double total_time);

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(void)
{
    print_banner();

    size_t bytes  = (size_t)GRID_SIZE * GRID_SIZE * sizeof(int); 
    int *grid     = (int *)malloc(bytes); // Allocate two grids: one for current state, one for next state (double buffering)
    int *new_grid = (int *)malloc(bytes); 

    if (!grid || !new_grid) {
        fprintf(stderr, "[ERROR] Malloc failed — need %.1f MB\n",
                (double)(bytes * 2) / (1024.0 * 1024.0));
        return EXIT_FAILURE;
    }

    printf("[INFO]  Grid        : %d x %d  (%ld cells)\n",
           GRID_SIZE, GRID_SIZE, TOTAL);
    printf("[INFO]  Timesteps   : %d\n",   TIMESTEPS);
    printf("[INFO]  Beta (β)    : %.2f   Gamma (γ) : %.2f\n", BETA, GAMMA);
    printf("[INFO]  R₀          : %.2f\n", BETA / GAMMA);
    printf("[INFO]  Memory      : %.1f MB\n\n",
           (double)(bytes * 2) / (1024.0 * 1024.0));

    /* ── CSV log ─────────────────────────────────────────────── */
    FILE *fp = fopen("../results/sir_seq_results.csv", "w");
    if (!fp)
        fprintf(stderr, "[WARN]  Cannot open results/ — no CSV written.\n");
    else
        log_header(fp);

    /* ── Initialise grid ─────────────────────────────────────── */
    unsigned int seed = (unsigned int)time(NULL);
    init_grid(grid, seed); 

    /* ── Count initial state ─────────────────────────────────── */
    long s_count = 0, i_count = 0, r_count = 0;
    for (long k = 0; k < TOTAL; k++) {
        if      (grid[k] == S) s_count++;
        else if (grid[k] == I) i_count++;
        else                   r_count++;
    }
    printf("[t=  0] S=%-8ld  I=%-8ld  R=%-8ld\n",
           s_count, i_count, r_count);
    if (fp) log_step(fp, 0, s_count, i_count, r_count, 0.0);

    /* ── Main simulation loop ────────────────────────────────── */
    unsigned int rng = seed ^ 0xDEADBEEF; //simulation run is truly unique and unpredictable.

    struct timespec t_start, t_end, step_s, step_e;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int t = 1; t <= TIMESTEPS; t++) {

        clock_gettime(CLOCK_MONOTONIC, &step_s);

        s_count = step(grid, new_grid, &rng, &i_count, &r_count);

        /* swap pointers — no memcpy needed */
        int *tmp  = grid;
        grid      = new_grid;
        new_grid  = tmp;

        clock_gettime(CLOCK_MONOTONIC, &step_e);

        double dt = (step_e.tv_sec  - step_s.tv_sec) +
                    (step_e.tv_nsec - step_s.tv_nsec) * 1e-9;

        if (t % 10 == 0 || t == 1 || t == TIMESTEPS)
            printf("[t=%3d] S=%-8ld  I=%-8ld  R=%-8ld  step=%.4fs\n",
                   t, s_count, i_count, r_count, dt);

        if (fp) log_step(fp, t, s_count, i_count, r_count, dt);

        /* Early exit if epidemic is over */
        if (i_count == 0) {
            printf("\n[INFO]  Epidemic ended at timestep %d\n", t);
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total = (t_end.tv_sec  - t_start.tv_sec) +
                   (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

    print_summary(s_count, i_count, r_count, total);

    /* ── Save timing for speedup calculations ───────────────── */
    FILE *tp = fopen("../results/seq_timing.txt", "w");
    if (tp) {
        fprintf(tp, "SEQUENTIAL_TIME_SECONDS=%.6f\n", total);
        fprintf(tp, "GRID_SIZE=%d\n",  GRID_SIZE);
        fprintf(tp, "TIMESTEPS=%d\n",  TIMESTEPS);
        fclose(tp);
        printf("[INFO]  Timing saved → results/seq_timing.txt\n");
    }

    if (fp) fclose(fp);
    free(grid);
    free(new_grid);
    return EXIT_SUCCESS;
}

/* ================================================================
 *  init_grid
 *  Sets all cells to Susceptible, then randomly places INIT_INF
 *  infected seeds. Uses memset(0) — safe for int arrays.
 * ================================================================ */
void init_grid(int *grid, unsigned int seed)
{
    /* memset with 0 is safe: S == 0, and 0x00000000 fills correctly */
    memset(grid, 0, (size_t)GRID_SIZE * GRID_SIZE * sizeof(int));

    srand(seed);
    int placed = 0;
    while (placed < INIT_INF) {
        int pos = rand() % (GRID_SIZE * GRID_SIZE);
        if (grid[pos] == S) {
            grid[pos] = I;
            placed++;
        }
    }
    printf("[INFO]  Grid initialised: %d infected seeds placed\n\n",
           INIT_INF);
}

/* ================================================================
 *  step  — PULL pattern (v3 fix)
 *
 *  For each cell:
 *    S cell : counts infected neighbours → maybe becomes I
 *    I cell : maybe recovers → becomes R
 *    R cell : stays R (immune)
 *
 *  KEY: every cell writes ONLY to new_grid[its own index].
 *       No cell touches a neighbour's slot → zero race conditions.
 *       This is correct in sequential AND safe in parallel.
 *
 *  Returns new S count. Sets *i_out and *r_out.
 * ================================================================ */
long step(int * restrict grid, int * restrict new_grid,
          unsigned int *seed, long *i_out, long *r_out)
{
    long new_s = 0, new_i = 0, new_r = 0;

    for (int row = 0; row < GRID_SIZE; row++) {
        for (int col = 0; col < GRID_SIZE; col++) {

            int idx   = IDX(row, col);
            int state = grid[idx];

            /* ── Recovered: stays recovered ──────────────────── */
            if (state == R) {
                new_grid[idx] = R;
                new_r++;
                continue;
            }

            /* ── Susceptible: check neighbours for infection ─── */
            if (state == S) {
                /* Count how many of the 4 neighbours are infected */
                int inf_neighbors = 0;
                if (row > 0             && grid[IDX(row-1, col)] == I) inf_neighbors++;
                if (row < GRID_SIZE - 1 && grid[IDX(row+1, col)] == I) inf_neighbors++;
                if (col > 0             && grid[IDX(row, col-1)] == I) inf_neighbors++;
                if (col < GRID_SIZE - 1 && grid[IDX(row, col+1)] == I) inf_neighbors++;

                /* Each infected neighbour gives one independent chance */
                int became_infected = 0;
                for (int n = 0; n < inf_neighbors; n++) {
                    if (rand_r(seed) < BETA_THRESH) {
                        became_infected = 1;
                        break;   /* one infection is enough */
                    }
                }

                if (became_infected) {
                    new_grid[idx] = I;
                    new_i++;
                } else {
                    new_grid[idx] = S;
                    new_s++;
                }
                continue;
            }

            /* ── Infected: try to recover ────────────────────── */
            /* (state == I) */
            if (rand_r(seed) < GAMMA_THRESH) {
                new_grid[idx] = R;
                new_r++;
            } else {
                new_grid[idx] = I;
                new_i++;
            }
        }
    }

    *i_out = new_i;
    *r_out = new_r;
    return new_s;
}

/* ================================================================
 *  Logging helpers
 * ================================================================ */
void log_header(FILE *fp)
{
    fprintf(fp, "timestep,susceptible,infected,recovered,step_time_sec\n");
}

void log_step(FILE *fp, int t, long s, long iv, long rv, double elapsed)
{
    fprintf(fp, "%d,%ld,%ld,%ld,%.6f\n", t, s, iv, rv, elapsed);
}

/* ================================================================
 *  print
 * ================================================================ */
void print_banner(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║     SIR EPIDEMIC SIMULATION — SEQUENTIAL  v3.0   ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

void print_summary(long s, long iv, long rv, double total_time)
{
    long total = TOTAL;
    /* Verify conservation — should always print 0 */
    long check = total - s - iv - rv;
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║                 FINAL RESULTS                    ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Susceptible : %-8ld  (%.1f%% of population)  ║\n",
           s,  (double)s  / total * 100.0);
    printf("║  Infected    : %-8ld  (%.1f%% of population)  ║\n",
           iv, (double)iv / total * 100.0);
    printf("║  Recovered   : %-8ld  (%.1f%% of population)  ║\n",
           rv, (double)rv / total * 100.0);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Conservation check (must be 0): %-8ld        ║\n", check);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Total time  : %-8.3f seconds                  ║\n", total_time);
    printf("║  Throughput  : %-8.0f cells/second             ║\n",
           (double)GRID_SIZE * GRID_SIZE * TIMESTEPS / total_time);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}
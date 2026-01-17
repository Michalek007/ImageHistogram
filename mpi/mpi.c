#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "image_data.h"
#include "my_timers.h"

#define HIST_SIZE 256
#define MAX_BAR_LENGTH 50

void print_histogram(const int* hist)
{
    int max_value = 0;
    for (int i = 0; i < HIST_SIZE; i++)
        if (hist[i] > max_value)
            max_value = hist[i];

    for (int i = 0; i < HIST_SIZE; i++) {
        int bar_length = (int)((double)hist[i] / max_value * MAX_BAR_LENGTH);
        printf("%3d | ", i);
        for (int j = 0; j < bar_length; j++)
            putchar('#');
        putchar('\n');
    }
}

int main(int argc, char** argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = IMG_HEIGHT / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1)
                  ? IMG_HEIGHT
                  : start_row + rows_per_proc;

    int local_hist[HIST_SIZE] = {0};

    start_time();

    /* Compute local histogram */
    for (int i = start_row; i < end_row; i++) {
        const uint8_t* row = &image[i][0];
        for (int j = 0; j < IMG_WIDTH; j++) {
            local_hist[row[j]]++;
        }
    }

    int global_hist[HIST_SIZE] = {0};

    /* Reduce all local histograms into global_hist on rank 0 */
    MPI_Reduce(
            local_hist,
            global_hist,
            HIST_SIZE,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
    );

    stop_time();

    if (rank == 0) {
        print_time("Elapsed:");
        print_histogram(global_hist);

        for (int i = 0; i < HIST_SIZE; i++) {
            assert(global_hist[i] == hist_gray[i]);
        }
    }

    MPI_Finalize();
    return 0;
}

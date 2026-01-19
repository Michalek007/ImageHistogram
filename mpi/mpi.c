#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "image_data.h"
#include "my_timers.h"

#define HIST_SIZE 256
#define MAX_BAR_LENGTH 100

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

    uint8_t* image;
    image = malloc(IMG_WIDTH * IMG_HEIGHT);
    if (image == NULL) {
        perror("Malloc failed");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        if (load_image(image) <= 0) {
            fprintf(stderr, "Failed to load image on rank 0\n");
            // Abort allows all ranks to exit cleanly if one fails
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(image, IMG_WIDTH * IMG_HEIGHT, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    start_time();
    int total_pixels = IMG_WIDTH * IMG_HEIGHT;
    int count_per_proc = total_pixels / size;
    int start_index = rank * count_per_proc;
    int end_index   = (rank == size - 1)
                       ? total_pixels
                       : start_index + count_per_proc;

    int local_hist[HIST_SIZE] = {0};

    /* Compute local histogram */
    for (long k = start_index; k < end_index; k++) {
        local_hist[image[k]]++;
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
        print_histogram(global_hist);

        for (int i = 0; i < HIST_SIZE; i++) {
            assert(global_hist[i] == hist_gray[i]);
        }

        printf("\n%-10s %s\n", "Tone", "Pixels");
        printf("--------------------\n");
        for (int i = 0; i < HIST_SIZE; i += 10) {
            printf("  %-8d %d\n", i, global_hist[i]);
        }
        printf("\n");

        print_time("Elapsed:");
    }

    free(image);
    MPI_Finalize();
    return 0;
}

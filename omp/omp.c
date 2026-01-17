#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>          // OpenMP
#include "my_timers.h"
#include "image_data.h"

#define HIST_SIZE 256
#define NUM_THREADS 4
#define MAX_BAR_LENGTH 50

// Print histogram in terminal
void print_histogram(const int* hist)
{
    int max_value = 0;
    for (int i = 0; i < HIST_SIZE; i++){
        if (hist[i] > max_value){
            max_value = hist[i];
        }
    }

    for (int i = 0; i < HIST_SIZE; i++) {
        int bar_length = (int)((double)hist[i] / max_value * MAX_BAR_LENGTH);
        printf("%3d | ", i);
        for (int j = 0; j < bar_length; j++)
            putchar('#');
        putchar('\n');
    }
}

int main(void)
{
    omp_set_num_threads(NUM_THREADS);
    int global_hist[HIST_SIZE] = {0};

    start_time();
    // Parallel region
    #pragma omp parallel
    {
        // Each thread has its own local histogram
        int local_hist[HIST_SIZE] = {0};

        #pragma omp for
        for (int i = 0; i < IMG_HEIGHT; i++) {
            const uint8_t* row = &image[i][0];
            for (int j = 0; j < IMG_WIDTH; j++) {
                local_hist[row[j]]++;
            }
        }

        // Merge local histograms into global
        #pragma omp critical
        {
            for (int k = 0; k < HIST_SIZE; k++) {
                global_hist[k] += local_hist[k];
            }
        }
    }
    stop_time();
    print_time("Elapsed:");

    // Print histogram
    print_histogram(global_hist);

    // Validate
    for (int i = 0; i < HIST_SIZE; i++) {
        assert(global_hist[i] == hist_gray[i]);
    }

    return 0;
}

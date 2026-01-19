#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>
#include "my_timers.h"
#include "image_data.h"

#define HIST_SIZE 256
#define NUM_THREADS 8
#define MAX_BAR_LENGTH 100

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
    uint8_t* image;
    image = malloc(IMG_WIDTH * IMG_HEIGHT);
    if (load_image(image) <= 0) {
        return -1;
    }

    omp_set_num_threads(NUM_THREADS);
    int global_hist[HIST_SIZE] = {0};

    start_time();
    // Parallel region
    #pragma omp parallel
    {
        // Each thread has its own local histogram
        int local_hist[HIST_SIZE] = {0};

        #pragma omp for
        for (int i = 0; i < IMG_HEIGHT * IMG_WIDTH; i++) {
            local_hist[image[i]]++;
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

    // Print histogram
    print_histogram(global_hist);

    // Validate
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
    free(image);
    return 0;
}

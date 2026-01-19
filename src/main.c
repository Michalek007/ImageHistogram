#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "image_data.h"
#include "my_timers.h"

#define HIST_SIZE 256
#define NUM_THREADS 4
#define MAX_BAR_LENGTH 100

// Example image
//#define IMG_HEIGHT 1000
//#define IMG_WIDTH  1000
//uint8_t image[IMG_HEIGHT][IMG_WIDTH];

// Structure to pass arguments to thread
typedef struct {
    const uint8_t* image; // pointer to first element of image
    int start_row;
    int end_row;
    int width;
    int local_hist[HIST_SIZE];
} ThreadData;

// Thread function
DWORD WINAPI compute_histogram(LPVOID lpParam) {
    ThreadData* data = (ThreadData*)lpParam;

    // Initialize local histogram
    for (int i = 0; i < HIST_SIZE; i++){
        data->local_hist[i] = 0;
    }

    for (int i = data->start_row; i < data->end_row; i++) {
        const uint8_t* row = data->image + i * data->width;
        for (int j = 0; j < data->width; j++) {
            data->local_hist[row[j]]++;
        }
    }
    return 0;
}

// Print histogram in terminal
void print_histogram(const int* hist) {
    int max_value = 0;
    for (int i = 0; i < HIST_SIZE; i++){
        if (hist[i] > max_value){
            max_value = hist[i];
        }
    }

    for (int i = 0; i < HIST_SIZE; i++) {
        int bar_length = (int)((double)hist[i] / max_value * MAX_BAR_LENGTH);
        printf("%3d | ", i);
        for (int j = 0; j < bar_length; j++){
            putchar('#');
        }
        putchar('\n');
    }
}


int main() {
    // Fill image with random values
//    for (int i = 0; i < IMG_HEIGHT; i++)
//        for (int j = 0; j < IMG_WIDTH; j++)
//            image[i][j] = rand() % 256;
//    uint8_t* image_array = &image[0][0];


    uint8_t* image;
    image = malloc(IMG_WIDTH * IMG_HEIGHT);
    if (load_image(image) <= 0){
        return -1;
    }

    // Thread data
    ThreadData thread_data[NUM_THREADS];
    HANDLE threads[NUM_THREADS];

    int rows_per_thread = IMG_HEIGHT / NUM_THREADS;

    start_time();
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].image = image;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == NUM_THREADS - 1) ? IMG_HEIGHT : (i + 1) * rows_per_thread;
        thread_data[i].width = IMG_WIDTH;

        threads[i] = CreateThread(
                NULL, 0, compute_histogram, &thread_data[i], 0, NULL
        );
    }

    // Wait for threads to finish
    WaitForMultipleObjects(NUM_THREADS, threads, TRUE, INFINITE);

    // Close thread handles
    for (int i = 0; i < NUM_THREADS; i++){
        CloseHandle(threads[i]);
    }

    // Merge histograms
    int global_hist[HIST_SIZE] = {0};
    for (int i = 0; i < NUM_THREADS; i++) {
        for (int j = 0; j < HIST_SIZE; j++) {
            global_hist[j] += thread_data[i].local_hist[j];
        }
    }
    stop_time();

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

    free(image);
    return 0;
}

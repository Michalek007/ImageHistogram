#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define IMG_WIDTH  3000
#define IMG_HEIGHT 2000

#define HIST_SIZE 256
#define MAX_BAR_LENGTH 100
#define BLOCK_SIZE 256

static double timestamp_start, timestamp_end;

void start_time(void)
{

    /* Get current time */
    struct timeval ts;
    gettimeofday(&ts, (struct timezone*)NULL);

    /* Store time-stamp in micro-seconds */
    timestamp_start = (double)ts.tv_sec * 1000000.0 + (double)ts.tv_usec;

}

void stop_time(void)
{

    /* Get current time */
    struct timeval ts;
    gettimeofday(&ts, (struct timezone*)NULL);

    /* Store time-stamp in microseconds */
    timestamp_end = (double)ts.tv_sec * 1000000.0 + (double)ts.tv_usec;

}

double elapsed_time(void)
{
    double time_diff;

    /* Compute difference */
    time_diff = timestamp_end - timestamp_start;
    if (time_diff <= 0.0)
    {
        fprintf(stdout,
                "Warning! The timer is not precise enough.\n");
        return 0.0;
    }

/* Return difference in milliseconds */
    return time_diff / 1000.0;
}

void print_time(char *message)
{
    double ms;

    ms = elapsed_time();
    printf("%s %.3f ms\n", message, ms);
}


const char* image_filename = "image_raw";

const uint32_t hist_gray[256] = {
        8963, 26673, 26155, 29717, 30684, 28041, 28992, 29730, 28653, 29055, 28476, 29506, 30332, 31596, 33365, 34834, 36621, 38065, 39956, 41464, 42715, 44529, 46358, 49122, 51598, 54353, 56209, 57750, 58646, 60487, 62839, 65153, 66969, 68262, 69643, 69217, 69894, 69978, 69974, 70304, 70723, 71004, 71988, 71889, 72935, 73565, 74319, 75191, 76165, 75779, 75879, 74225, 72677, 71167, 69090, 67667, 66492, 65070, 63850, 62187, 60264, 57952, 55979, 53740, 51380, 49531, 47996, 46794, 44969, 43909, 42514, 41083, 39856, 38926, 37289, 36608, 35122, 34050, 33066, 32221, 30767, 29867, 29097, 27874, 27342, 26263, 25835, 25121, 24328, 23455, 22790, 22495, 21683, 21183, 20554, 20125, 19695, 19073, 18735, 18473, 18051, 17640, 17332, 16972, 16921, 16329, 16259, 15661, 15237, 14945, 14716, 14336, 14196, 13869, 13851, 13603, 13186, 13399, 13080, 12612, 12758, 12314, 12316, 11966, 11771, 11676, 11297, 11032, 11054, 11199, 10995, 11002, 10633, 10831, 10823, 10557, 10854, 10704, 10841, 10812, 10835, 10674, 10781, 10693, 10769, 10823, 10782, 10684, 10558, 10636, 10467, 10581, 10700, 10311, 10485, 10193, 10449, 10339, 10662, 10429, 10209, 10204, 9999, 10060, 10041, 9924, 9978, 10009, 9764, 9753, 9925, 9741, 9666, 9825, 9663, 9760, 9379, 9635, 9339, 9603, 9578, 9523, 9369, 9310, 9262, 9307, 9359, 9349, 9433, 9283, 9389, 9405, 9123, 9233, 9195, 9260, 9134, 9131, 9037, 8961, 8895, 8923, 8970, 8902, 8693, 8835, 8480, 8223, 8231, 8087, 7887, 7716, 7562, 7265, 6987, 6769, 6438, 6008, 5676, 5460, 5294, 5278, 5316, 5175, 4718, 4538, 4104, 4029, 4166, 4111, 4211, 4384, 4412, 4076, 3657, 3514, 3529, 3650, 3984, 4174, 4210, 4187, 4353, 5183, 4109, 3187, 3113, 3313, 3127, 2283, 2154, 1901, 684, 365, 225, 936
};

int load_image(uint8_t* image){
    if (!image) {
        perror("image is NULL");
        return 0;
    }

    FILE *f = fopen(image_filename, "rb");
    if (!f) {
        perror("fopen");
        return 0;
    }
    size_t bytes_read = fread(image, 1, IMG_WIDTH * IMG_HEIGHT, f);
    printf("Image raw size: %llu\n", bytes_read);
    fclose(f);
    return bytes_read;
}

// macro to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)


__global__ void histogram_kernel(const uint8_t *input, int *histogram, int num_pixels) {
    // shared memory for all threads in block
    __shared__ int shared_hist[HIST_SIZE];

    // Calculate global thread index and stride for grid-stride loop
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // initialize shared memory histogram to zero
    // works even if threads < HIST_SIZE
    for (int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) {
        shared_hist[i] = 0;
    }

    // wait for all threads
    __syncthreads();

    // calculate histogram values
    for (int i = tid; i < num_pixels; i += stride) {
        atomicAdd(&shared_hist[input[i]], 1);
    }

    // wait for all threads
    __syncthreads();

    // merge histogram
    for (int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

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
    // --- HOST (CPU) SETUP ---
    uint8_t* h_image;
    const int num_pixels = IMG_WIDTH * IMG_HEIGHT;
    const size_t image_size_bytes = num_pixels * sizeof(*h_image);

    // allocate memory for image data
    h_image = (uint8_t*)malloc(image_size_bytes);
    if (!h_image) {
        fprintf(stderr, "failed to allocate host memory for image\n");
        return -1;
    }
    
    if (load_image(h_image) <= 0) {
        free(h_image);
        return -1;
    }

    // memory for result histogram
    int h_global_hist[HIST_SIZE] = {0};
    const size_t hist_size_bytes = sizeof(h_global_hist);

    // --- DEVICE (GPU) SETUP ---
    uint8_t *d_image = NULL;
    int *d_hist = NULL;

    // device memory for image and histogram
    CUDA_CHECK(cudaMalloc((void **)&d_image, image_size_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_hist, hist_size_bytes));

    // copy image data to device
    CUDA_CHECK(cudaMemcpy(d_image, h_image, image_size_bytes, cudaMemcpyHostToDevice));
    // initialize empty histogram 
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_size_bytes));

    // configure CUDA
    const int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;
    
    // // get device properties to optimize grid size
    // cudaDeviceProp deviceProp;
    // CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    // // Limit blocks to reasonable number based on SM count (avoid over-subscription)
    // const int maxBlocks = deviceProp.multiProcessorCount * 8;
    // if (blocksPerGrid > maxBlocks) {
    //     blocksPerGrid = maxBlocks;
    // }

    // printf("Device: %s (Compute Capability %d.%d, %d SMs)\n", 
    //     deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    printf("CUDA Configuration: Grid size = %d blocks, Block size = %d threads\n", 
           blocksPerGrid, threadsPerBlock);


    start_time();

    // --- KERNEL LAUNCH ---
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_hist, num_pixels);
    CUDA_CHECK(cudaGetLastError());

    // wait for device calculations
    CUDA_CHECK(cudaDeviceSynchronize());

    stop_time();

    // copy result from device to host memory
    CUDA_CHECK(cudaMemcpy(h_global_hist, d_hist, hist_size_bytes, cudaMemcpyDeviceToHost));

    // free device resources
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_hist));

    // print histogram
    print_histogram(h_global_hist);

    // check if calculations correct
    for (int i = 0; i < HIST_SIZE; i++) {
        assert(h_global_hist[i] == hist_gray[i]);
    }

    printf("\n%-10s %s\n", "Tone", "Pixels");
    printf("--------------------\n");
    for (int i = 0; i < HIST_SIZE; i += 10) {
        printf("  %-8d %d\n", i, h_global_hist[i]);
    }
    printf("\n");

    print_time("Elapsed:");
    free(h_image);

    cudaDeviceReset();

    return 0;
}
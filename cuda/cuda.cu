#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define IMG_WIDTH  10174
#define IMG_HEIGHT 10531

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




const char* image_filename = "verybig";

const uint32_t hist_gray[256] = {
    33, 616, 2092, 5646, 12541, 24359, 41574, 64085, 92950, 129091, 168007, 213376, 264646, 319593, 377411, 436810, 500091, 573922, 635723, 698048, 762047, 822016, 884353, 939474, 994427, 1040705, 1081692, 1120115, 1149878, 1167739, 1180304, 1173909, 1190606, 1206312, 1201515, 1188206, 1172997, 1160053, 1142311, 1128627, 1112183, 1090247, 1067025, 1040538, 1013411, 982260, 947734, 911006, 873472, 835275, 798009, 761730, 725870, 693490, 665041, 638464, 617576, 596321, 572069, 554073, 533653, 515849, 506227, 503106, 509183, 518703, 525645, 524607, 520767, 517409, 519461, 522039, 531989, 543364, 557797, 575192, 602627, 626988, 645164, 633469, 605937, 586239, 582472, 576978, 573206, 567109, 559118, 555970, 547673, 536946, 520427, 502401, 485671, 472245, 468436, 471235, 479798, 500333, 523809, 551092, 583982, 618888, 655655, 687242, 711007, 721833, 712580, 683394, 628837, 565285, 506939, 467417, 443862, 433244, 429932, 432235, 433952, 433869, 432165, 427808, 426288, 419243, 415149, 411279, 408162, 398152, 390347, 380555, 371570, 363362, 354850, 349612, 345976, 341308, 338952, 336390, 339368, 342736, 342865, 342183, 341168, 338007, 333673, 330690, 327981, 326390, 323619, 322089, 318253, 314240, 312708, 310146, 309381, 306291, 302955, 302970, 302982, 303293, 303485, 304283, 304840, 306538, 308839, 309163, 308915, 308805, 306749, 305332, 300995, 296820, 294705, 288614, 283342, 279222, 273723, 266462, 259426, 254239, 247357, 241844, 236066, 232187, 229373, 227394, 225947, 226291, 227862, 225545, 226031, 226939, 229473, 229722, 231756, 232619, 237286, 239384, 238842, 237132, 230820, 224195, 218862, 212962, 206830, 201780, 198498, 194272, 189877, 187070, 186539, 188455, 191514, 191874, 189887, 189956, 188218, 186748, 183045, 175758, 167819, 162212, 156966, 152146, 152291, 155242, 156711, 160072, 167902, 173845, 177256, 173875, 171635, 166169, 160450, 152496, 140682, 133661, 125412, 113260, 106317, 97367, 87438, 81030, 81535, 87763, 83081, 78731, 74041, 62727, 46925, 39350, 38752, 24411, 16786, 25429, 18512, 2053
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define IMG_WIDTH  6000
#define IMG_HEIGHT 4000

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
        3, 10, 16, 40, 139, 424, 933, 2290, 4653, 8905, 14112, 20428, 22001, 24903, 28539, 34385, 41907, 50988, 60758, 71130, 82477, 95358, 109752, 126101, 140466, 153915, 163895, 171544, 178669, 185734, 194829, 207168, 217693, 226354, 231816, 236101, 240104, 243779, 248300, 254212, 259272, 263313, 265228, 263066, 259801, 255230, 249056, 245222, 244666, 244516, 244379, 242427, 238364, 231504, 222266, 211680, 203380, 197163, 193105, 190001, 187063, 181903, 177406, 171589, 166703, 162702, 158784, 155762, 152609, 148315, 145968, 142737, 139856, 139782, 139957, 140561, 140994, 142068, 140414, 139616, 135958, 133092, 127471, 121768, 116319, 110091, 105265, 102469, 100785, 101323, 101839, 103864, 105108, 107433, 109019, 109175, 109034, 108119, 107383, 105712, 104103, 102081, 100532, 98746, 96177, 93244, 89086, 84600, 79816, 75208, 71980, 70319, 69032, 69403, 69172, 68820, 67661, 66138, 63834, 61439, 58708, 55524, 53072, 49993, 47866, 45988, 44611, 43001, 42030, 41437, 40835, 40546, 40138, 39705, 39686, 39659, 39180, 39440, 39404, 39683, 39443, 39952, 40492, 41089, 41313, 41735, 42356, 42806, 43542, 43575, 44462, 45096, 45093, 45556, 45930, 46340, 46697, 46367, 46372, 46561, 46084, 45569, 45424, 44668, 45171, 44986, 44704, 44640, 44409, 44811, 44673, 44569, 44555, 44720, 45572, 48543, 54086, 55901, 80460, 118990, 142254, 139015, 133662, 138526, 138127, 149127, 112129, 148435, 141514, 144728, 150405, 132365, 123587, 114101, 131435, 88385, 182209, 153996, 126012, 129761, 137593, 149790, 143062, 136779, 77601, 121324, 133223, 151795, 149187, 143642, 136433, 105739, 89623, 64768, 75520, 69762, 53017, 42720, 41447, 42198, 41031, 38481, 34876, 39189, 39852, 32977, 27520, 27535, 30134, 28699, 26830, 24201, 26327, 21528, 21109, 22522, 24548, 22694, 25403, 31095, 15235, 17787, 16801, 17477, 9560, 8979, 9071, 11159, 7831, 3813, 4338, 2212, 393, 56, 19, 27
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
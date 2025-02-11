#include <iostream>
#include <cuda_runtime.h>
#include <vector>


#define NUMBER_CLASS 84
#define NUMBER_ANCHOR 8400
#define BLOCK_SIZE 256

__global__ void addPaddingZeros(int *input, int *output)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = idx / 8448;
    int col = idx % 8448;

    
    if (col < NUMBER_ANCHOR)
    {
        output[row * 8448 + col] = input[row * NUMBER_ANCHOR + col];
    }
    else
    {
        output[row * 8448 + col] = 0;
    }
}

__global__ void addPaddingMIN(int *input, int *output)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = idx / 8448;
    int col = idx % 8448;

    
    if (col < NUMBER_ANCHOR)
    {
        output[row * 8448 + col] = input[row * NUMBER_ANCHOR + col];
    }
    else
    {
        output[row * 8448 + col] = INT_MIN;
    }
}

__global__ void addPaddingMAX(int *input, int *output)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = idx / 8448;
    int col = idx % 8448;

    
    if (col < NUMBER_ANCHOR)
    {
        output[row * 8448 + col] = input[row * NUMBER_ANCHOR + col];
    }
    else
    {
        output[row * 8448 + col] = INT_MAX;
    }
}

__global__ void reduceSum(int *input, int *output)
{
    __shared__ int sdata[256];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    

    //if (idx >= M * N) return;

    unsigned int idx_classes = idx / 8448;
    int *idata = input + blockIdx.x * blockDim.x;

    /*
    if((idx > 8300 && idx < 8500) || (idx > 16600 && idx < 19896)){
        printf("idx: %d - idata[threadIdx.x]: %d\n", idx, idata[threadIdx.x]);
    }
    */

    sdata[threadIdx.x] = idata[threadIdx.x];
    __syncthreads();

    if (blockDim.x >= 256 && threadIdx.x < 128)
        sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    __syncthreads();

    if (blockDim.x >= 128 && threadIdx.x < 64)
        sdata[threadIdx.x] += sdata[threadIdx.x + 64];
    __syncthreads();

    if (threadIdx.x < 32)
    {
        volatile int *vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0)
        atomicAdd(&output[idx_classes], sdata[0]);
}

__global__ void getMin(int *input, int *output)
{
    __shared__ int sdata[256];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int idx_classes = idx / 8448;
    int *idata = input + blockIdx.x * blockDim.x;

    sdata[threadIdx.x] = idata[threadIdx.x];
    __syncthreads();

    if (blockDim.x >= 256 && threadIdx.x < 128)
        sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + 128]);
    __syncthreads();

    if (blockDim.x >= 128 && threadIdx.x < 64)
        sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + 64]);
    __syncthreads();

    if (threadIdx.x < 32)
    {
        volatile int *vsmem = sdata;
        vsmem[threadIdx.x] = min(vsmem[threadIdx.x], vsmem[threadIdx.x + 32]);
        vsmem[threadIdx.x] = min(vsmem[threadIdx.x], vsmem[threadIdx.x + 16]);
        vsmem[threadIdx.x] = min(vsmem[threadIdx.x], vsmem[threadIdx.x + 8]);
        vsmem[threadIdx.x] = min(vsmem[threadIdx.x], vsmem[threadIdx.x + 4]);
        vsmem[threadIdx.x] = min(vsmem[threadIdx.x], vsmem[threadIdx.x + 2]);
        vsmem[threadIdx.x] = min(vsmem[threadIdx.x], vsmem[threadIdx.x + 1]);
    }

    if (threadIdx.x == 0)
        atomicMin(&output[idx_classes], sdata[0]);
}

__global__ void getMax(int *input, int *output)
{
    __shared__ int sdata[256];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int idx_classes = idx / 8448;
    int *idata = input + blockIdx.x * blockDim.x;

    sdata[threadIdx.x] = idata[threadIdx.x];
    __syncthreads();

    if (blockDim.x >= 256 && threadIdx.x < 128)
        sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + 128]);
    __syncthreads();

    if (blockDim.x >= 128 && threadIdx.x < 64)
        sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + 64]);
    __syncthreads();

    if (threadIdx.x < 32)
    {
        volatile int *vsmem = sdata;
        vsmem[threadIdx.x] = max(vsmem[threadIdx.x], vsmem[threadIdx.x + 32]);
        vsmem[threadIdx.x] = max(vsmem[threadIdx.x], vsmem[threadIdx.x + 16]);
        vsmem[threadIdx.x] = max(vsmem[threadIdx.x], vsmem[threadIdx.x + 8]);
        vsmem[threadIdx.x] = max(vsmem[threadIdx.x], vsmem[threadIdx.x + 4]);
        vsmem[threadIdx.x] = max(vsmem[threadIdx.x], vsmem[threadIdx.x + 2]);
        vsmem[threadIdx.x] = max(vsmem[threadIdx.x], vsmem[threadIdx.x + 1]);
    }

    if (threadIdx.x == 0)
        atomicMax(&output[idx_classes], sdata[0]);
}



int main()
{
    int number_anchor = static_cast<int>(ceil(static_cast<float>(NUMBER_ANCHOR) / BLOCK_SIZE)) * BLOCK_SIZE;
    
    int *input, *output, *output_padd_test;
    int *d_input, *d_output, *d_output_padd;

    size_t size_input = NUMBER_CLASS * NUMBER_ANCHOR;
    size_t byte_input = size_input * sizeof(int);

    size_t size_padding = NUMBER_CLASS * number_anchor;
    size_t byte_padding = size_padding * sizeof(int);

    size_t size_output = NUMBER_CLASS;
    size_t byte_output = size_output * sizeof(int);

    dim3 block_padding(BLOCK_SIZE);
    dim3 grid_padding((size_padding + block_padding.x - 1) / block_padding.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    printf("number_anchor: %d\n", number_anchor);
    printf("size_input: %d (%dx%d)\n", size_input, NUMBER_CLASS, NUMBER_ANCHOR);
    printf("byte_input: %d\n", byte_input);
    printf("size_padding: %d (%dx%d)\n", size_padding, NUMBER_CLASS, number_anchor);
    printf("byte_padding: %d\n", byte_padding);
    printf("size_output: %d\n", size_output);
    printf("byte_output: %d\n", byte_output);
    printf("block_padding: %d\n", block_padding.x);
    printf("grid_padding: %d\n\n\n", grid_padding.x);


    input = (int *)malloc(byte_input);
    output = (int *)malloc(byte_output);
    output_padd_test = (int *)malloc(byte_padding);

    cudaMalloc(&d_input, byte_input);
    cudaMalloc(&d_output, byte_output);
    cudaMalloc(&d_output_padd, byte_padding);

    for (int i = 0; i < NUMBER_CLASS * NUMBER_ANCHOR; i++)
    {
        input[i] = i;
    }


    cudaMemcpy(d_input, input, byte_input, cudaMemcpyHostToDevice);

    printf("grid_padding %d block_padding %d\n", grid_padding.x, block_padding.x);
    cudaMemset(d_output, 60000, byte_output);


    cudaEventRecord(start, 0);
    for(int i = 0; i < 1; i++)
    {
        //addPaddingZeros<<<grid_padding, block_padding>>>(d_input, d_output_padd);
        //reduceSum<<<grid_padding, block_padding>>>(d_output_padd, d_output);

        //addPaddingMIN<<<grid_padding, block_padding>>>(d_input, d_output_padd);
        //getMax<<<grid_padding, block_padding>>>(d_output_padd, d_output);

        addPaddingMAX<<<grid_padding, block_padding>>>(d_input, d_output_padd);
        getMin<<<grid_padding, block_padding>>>(d_output_padd, d_output);

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f ms\n", elapsedTime / 1);

    //cudaMemcpy(output_padd_test, d_output_padd, byte_padding, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, byte_output, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_padd_test, d_output_padd, byte_padding, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUMBER_CLASS; i++)
    {
        printf("output[%d]: %d\n", i, output[i]);
    }

    /*
    int batch = 83;
    int count = 50;
    for (int i = 0; i < count; i++)
    {
        printf("output_padd_test[%d]: %d\n", (batch * number_anchor) - i, output_padd_test[(batch * number_anchor) - i]);
        printf("output_padd_test[%d]: %d\n\n", (batch * number_anchor) + i, output_padd_test[(batch * number_anchor) + i]);
    }
    */

    /*
    for(size_t i = 701184; i < NUMBER_CLASS * number_anchor; i++)
    {
        printf("output_padd_test[%d]: %d\n", i, output_padd_test[i]);
    }
    */



}
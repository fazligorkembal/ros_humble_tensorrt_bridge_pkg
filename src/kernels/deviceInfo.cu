#include "ros_humble_tensorrt_bridge_pkg/kernels/deviceInfo.hpp"
#include "stdio.h"
#include "cuda_runtime.h"

void __global__ who_am_i()
{
    printf("ThreadIdx.x: %d, ThreadIdx.y: %d, ThreadIdx.z: %d, BlockIdx.x: %d, BlockIdx.y: %d, BlockIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}


void who_am_i_wrapper()
{
    dim3 threadsPerBlock(5, 1, 1);
    dim3 numBlocks(1, 1, 1);

    //who_am_i<<<numBlocks, threadsPerBlock>>>();
}
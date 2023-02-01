
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void idx_calc_tid(int* input)
{
    int tid = threadIdx.x;
    printf("[DEVICE] threadIdx.x: %d, data: %d\n\r", tid, input[tid]);
}

__global__ void idx_calc_gid(int* input)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;
    int gid = tid + block_offset;
    printf("[DEVICE] blockIdx.x: %d, threadIdx.x: %d, gId: %d, data: %d\n\r", blockIdx.x, tid, gid, input[gid]);
}

__global__ void idx_calc_2d(int* input)
{
    int tid = threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;

    int threads_per_row = gridDim.x * blockDim.x;
    int row_offset = blockIdx.y * threads_per_row;
    int gid = tid + row_offset + block_offset;
    printf("[DEVICE] gridDim.x: %d, blockIdx.x: %d, blockIdy.x: %d, threadIdx.x: %d, gId: %d, data: %d\n\r",
        gridDim.x, blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

__global__ void idx_calc_2d_2d(int* input)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * threads_per_block;

    int threads_per_row = gridDim.x * threads_per_block;
    int row_offset = blockIdx.y * threads_per_row;

    int gid = tid + row_offset + block_offset;
    printf("[DEVICE] gridDim.x: %d, blockIdx.x: %d, blockIdy.x: %d, threadidx.x: %d, gId: %d, data: %d\n\r",
        gridDim.x, blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

int main()
{
    const int vectorSize = 32; // 16 16 16 32
    int* dev_a;

    cudaMalloc((void**)&dev_a, vectorSize * sizeof(int));

    int* phost_a;

    phost_a = (int*)malloc(vectorSize * sizeof(int));

    for (int i = 0; i < vectorSize; i++) {
        phost_a[i] = i;
        printf("[HOST] data: %d\n\r", phost_a[i]);
    }

    cudaMemcpy(dev_a, phost_a, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(4, 2); //  (16) (8) (4) (4,2);
    dim3 gridDim(2, 2);  //  (1) (2) (2,2) (2,2);

    // idx_calc_tid << < gridDim, blockDim >> > (dev_a);
    // idx_calc_gid << < gridDim, blockDim >> > (dev_a);
    // idx_calc_2d << < gridDim, blockDim >> > (dev_a);
    idx_calc_2d_2d << < gridDim, blockDim >> > (dev_a);

    cudaDeviceSynchronize();

    cudaDeviceReset();
    cudaFree(dev_a);

    return 0;
}
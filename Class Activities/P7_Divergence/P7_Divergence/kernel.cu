
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void no_divergence() {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int a, b;
    int warp_gid = gid / 32;
    if (warp_gid % 2 == 0) {
        a = 2.5;
        b = 5.6;
    }
    else {
        a = 3.1415;
        b = 6.666;
    }
}

__global__ void divergence() {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int a, b;
    if (gid % 2 == 0) {
        a = 2.5;
        b = 5.6;
    }
    else {
        a = 3.1415;
        b = 6.666;
    }
}

int main()
{
    int size = 1 << 22;
    dim3 block(128);
    dim3 grid((size + block.x - 1) / block.x);

    no_divergence << <grid, block >> > ();
    cudaDeviceSynchronize();

    divergence << <grid, block >> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}

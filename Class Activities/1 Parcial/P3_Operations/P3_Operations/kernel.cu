
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

__global__ void addVectors(int* a, int* b, int* c, int size)
{
    int tid = blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    int bid = gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z) + blockIdx.x;
    int gid = tid + bid * threads_per_block;

    if (gid < size) {
        c[gid] = a[gid] + b[gid];
        printf("\n%d) %d + %d = %d", gid, a[gid], b[gid], c[gid]);
    }
}

__global__ void add3Vectors(int* a, int* b, int* c, int* d, int size)
{
    int tid = blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    int bid = gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z) + blockIdx.x;
    int gid = tid + bid * threads_per_block;

    if (gid < size) {
        d[gid] = a[gid] + b[gid] + c[gid];
        printf("\n%d) %d + %d + %d = %d", gid, a[gid], b[gid], c[gid], d[gid]);
    }
}

int main()
{
    const int vectorSize = 10000;
    const int size = vectorSize * sizeof(int);
    int* dev_a, * dev_b, * dev_c, * dev_d;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
    cudaMalloc((void**)&dev_d, size);

    int* phost_a, * phost_b, * phost_c, * phost_d, * phost_res;

    phost_a = (int*)malloc(size);
    phost_b = (int*)malloc(size);
    phost_c = (int*)malloc(size);
    phost_d = (int*)malloc(size);
    phost_res = (int*)malloc(size);

    for (int i = 0; i < vectorSize; i++) {
        phost_a[i] = rand() % 255;
        phost_b[i] = rand() % 255;
        phost_c[i] = rand() % 255;
        // phost_d[i] = phost_a[i] + phost_b[i];
        phost_d[i] = phost_a[i] + phost_b[i] + phost_c[i];
        // printf("\n%d) %d + %d + %d = %d", i, phost_a[i], phost_b[i], phost_c[i]);
    }

    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, phost_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, phost_c, size, cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4, 8);
    dim3 gridDim(4, 4, 5);

    printf("\n********** Add in kernel **********");

    clock_t gpu_start, gpu_stop;

    gpu_start = clock();
    // addVectors << < gridDim, blockDim >> > (dev_a, dev_b, dev_d, size);
    add3Vectors << < gridDim, blockDim >> > (dev_a, dev_b, dev_c, dev_d, size);
    cudaDeviceSynchronize();
    
    gpu_stop = clock();
    double cps_gpu = (double)((double)(gpu_stop - gpu_start) / CLOCKS_PER_SEC);
    printf("\n\nExecution Time [ET.GPU]: %4.6f\n\r", cps_gpu);

    cudaMemcpy(phost_res, dev_d, size, cudaMemcpyDeviceToHost);

    bool equal = true;
    for (int i = 0; i < vectorSize; i++) {
        if (phost_d[i] != phost_res[i]) {
            equal = false;
            printf("%d", i);
            break;
        }
    }
    printf("\n\n\nBoth are equal? %s\n\n", equal ? "True" : "False");

    cudaDeviceReset();
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    return 0;
}

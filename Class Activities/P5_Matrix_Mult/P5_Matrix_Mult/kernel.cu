
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

__global__ void dotProduct(int* a, int* b, int* c, int matSize)
{
    int tid = blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int bid = gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z) + blockIdx.x;
    int gid = tid + bid * threads_per_block;
    
    c[gid] = 0;

    int col = (int)(gid / matSize)* matSize;
    int row = (int)(gid % matSize);

    for (int i = 0; i < matSize; i++) {
        col += i;
        row += i * matSize;
        c[gid] += a[col] * b[row];
    }
}

void printMatrix(int* a, int matSize) {
    for (int i = 0; i < matSize * matSize; i++) {
        if (i % matSize == 0) {
            printf("\n");
        }
        printf("\t%d", a[i]);
    }
}

int main()
{
    const int vectorSize = 4;
    const int size = vectorSize * sizeof(int);
    int matSize = 2;
    int* dev_a, * dev_b, * dev_c;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    int* phost_a, * phost_b, * phost_c;

    phost_a = (int*)malloc(size);
    phost_b = (int*)malloc(size);
    phost_c = (int*)malloc(size);

    for (int i = 0; i < vectorSize; i++) {
        phost_a[i] = i + 1;
        phost_b[i] = i + 1 + vectorSize;
    }

    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, phost_b, size, cudaMemcpyHostToDevice);

    dim3 blockDim(2, 2);
    dim3 gridDim(1);

    clock_t gpu_start, gpu_stop;

    gpu_start = clock();
    dotProduct << < gridDim, blockDim >> > (dev_a, dev_b, dev_c, matSize);
    cudaDeviceSynchronize();

    gpu_stop = clock();
    double cps_gpu = (double)((double)(gpu_stop - gpu_start) / CLOCKS_PER_SEC);
    printf("\n\nExecution Time [ET.GPU]: %4.6f\n\r", cps_gpu);

    cudaMemcpy(phost_c, dev_c, size, cudaMemcpyDeviceToHost);

    printf("\n\n*****    MATRIX A    *****\n");
    printMatrix(phost_a, matSize);

    printf("\n\n*****    MATRIX B    *****\n");
    printMatrix(phost_b, matSize);

    printf("\n\n*****    MATRIX C    *****\n");
    printMatrix(phost_c, matSize);

    cudaDeviceReset();
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

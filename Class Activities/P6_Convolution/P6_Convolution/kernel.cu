
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

__global__ void convolution(int* a, int* k, int* newA, int xSize, int totSize)
{
    int tid = blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int bid = gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z) + blockIdx.x;
    int gid = tid + bid * threads_per_block;

    newA[gid] = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int aPos = gid + i + j * xSize;
            int side = gid % xSize == 0 ? -1 : (gid + 1) % xSize == 0 ? 1 : 0;

            if (aPos >= 0 && aPos < totSize && ( (side == -1 && (gid + 1 + i) % xSize != 0) || (side == 1 && (gid + i) % xSize != 0) || side == 0 )) {
                newA[gid] += a[aPos] * k[i + 1 + (j + 1) * 3];
                /*
                if (gid == 4) {
                    printf("\naPos %d\ta[aPos] %d\tk[x-i, j-i] %d\ti %d\tj %d", aPos, a[aPos], k[i + 1 + (j + 1) * 3], i, j);
                }*/
            }
        }
    }
}

void printMatrix(int* a, int xSize, int totSize) {
    for (int i = 0; i < totSize; i++) {
        if (i % xSize == 0) {
            printf("\n");
        }
        printf("  %d", a[i]);
    }
}

int main()
{
    const int xSize = 32;
    const int ySize = 32;
    const int totSize = xSize* ySize;
    const int size = totSize * sizeof(int);
    int* dev_a, * dev_k, * dev_new;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_new, size);
    cudaMalloc((void**)&dev_k, 9 * sizeof(int));

    int* phost_a, * phost_k, * phost_new;

    phost_a = (int*)malloc(size);
    phost_new = (int*)malloc(size);
    phost_k = (int*)malloc(9 * sizeof(int));

    for (int i = 0; i < totSize; i++) {
        phost_a[i] = rand() % 255;
    }

    // Filter matrix
    for (int i = 0; i < 9; i++) {
        phost_k[i] = 0;
    }
    phost_k[4] = 1;
    
    
    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k, phost_k, 9 * sizeof(int), cudaMemcpyHostToDevice);

    printf("\n\n*****    MATRIX k    *****\n");
    printMatrix(phost_k, 3, 9);

    dim3 blockDim(xSize, ySize);
    dim3 gridDim(1);

    clock_t gpu_start, gpu_stop;

    gpu_start = clock();
    convolution << < gridDim, blockDim >> > (dev_a, dev_k, dev_new, xSize, totSize);
    cudaDeviceSynchronize();

    gpu_stop = clock();
    double cps_gpu = (double)((double)(gpu_stop - gpu_start) / CLOCKS_PER_SEC);
    printf("\n\nExecution Time [ET.GPU]: %4.6f\n\r", cps_gpu);

    cudaMemcpy(phost_new, dev_new, size, cudaMemcpyDeviceToHost);

    printf("\n\n*****    MATRIX A    *****\n");
    printMatrix(phost_a, xSize, totSize);

    printf("\n\n*****   MATRIX NEW   *****\n");
    printMatrix(phost_new, xSize, totSize);

    /*cudaDeviceReset();*/
    cudaFree(dev_a);
    cudaFree(dev_k);
    cudaFree(dev_new);

    return 0;
}

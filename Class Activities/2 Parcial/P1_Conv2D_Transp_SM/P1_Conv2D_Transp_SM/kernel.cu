
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

__global__ void convolution(int* a, int* k, int* newA, int xSize, int totSize)
{
    __shared__ int aShared[64];

    int tid = blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int bid = gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z) + blockIdx.x;
    int gid = tid + bid * threads_per_block;

    int row = gid % xSize;
    int col = gid / xSize;

    int filterSize = 3;
    int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

    // Copy input matrix into shared memory
    if (gid < totSize) {
        aShared[sharedIndex] = a[col + row * xSize];
    }
    else {
        aShared[sharedIndex] = 0;
    }
    __syncthreads();

    // Convolution
    int sum = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int aPos = sharedIndex + i + j * blockDim.x;
            int filterIndex = (i + 1) * filterSize + j + 1;

            if (aPos >= 0 && aPos < blockDim.x* blockDim.y&& row + i >= 0 && row + i < xSize&& col + j >= 0 && col + j < xSize) {
                sum += aShared[aPos] * k[filterIndex];
            }
        }
    }

    if (gid < totSize) {
        newA[col + row * xSize] = sum;
    }
}

__global__ void transpose(int* a, int* b, int matSize)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < matSize && j < matSize) {
        int trans = i * matSize + j;
        int orign = j * matSize + i;
        b[trans] = a[orign];
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

int main() {
    const int xSize = 8;
    const int ySize = 8;
    const int totSize = xSize * ySize;
    const int size = totSize * sizeof(int);
    int* dev_a, * dev_k, * dev_trans, * dev_new;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_trans, size);
    cudaMalloc((void**)&dev_new, size);
    cudaMalloc((void**)&dev_k, 9 * sizeof(int));

    int* phost_a, * phost_k, * phost_trans, * phost_new;

    phost_a = (int*)malloc(size);
    phost_trans = (int*)malloc(size);
    phost_new = (int*)malloc(size);
    phost_k = (int*)malloc(9 * sizeof(int));

    for (int i = 0; i < totSize; i++) {
        phost_a[i] = i + 1; // rand() % 255;
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

    // Transpose
    transpose << < gridDim, blockDim >> > (dev_a, dev_trans, xSize);
    cudaDeviceSynchronize();

    cudaMemcpy(phost_trans, dev_trans, size, cudaMemcpyDeviceToHost);

    // Convolution
    convolution << < gridDim, blockDim >> > (dev_trans, dev_k, dev_new, xSize, totSize);
    cudaDeviceSynchronize();

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

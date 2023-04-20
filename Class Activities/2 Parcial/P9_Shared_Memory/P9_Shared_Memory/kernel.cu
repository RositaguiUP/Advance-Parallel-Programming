
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define BLOCK_SIZE 4

template<typename T>
__global__ void convolution(T* in, T* out, int matSize, T* kernel, int kernelSize)
{
    int col = threadIdx.x + (BLOCK_SIZE) * blockDim.x;
    int row = threadIdx.y + (BLOCK_SIZE) * blockDim.y;

    __shared__ T in_tile[BLOCK_SIZE][BLOCK_SIZE],
                    k_tile[9];

    if (col < matSize && row < matSize)
        in_tile[threadIdx.y][threadIdx.x] = in[col * matSize + row]; // save input

    if (threadIdx.x < kernelSize * kernelSize)
        k_tile[threadIdx.x] = kernel[threadIdx.x]; // save kernel

    __syncthreads();

    if (threadIdx.y < (BLOCK_SIZE - kernelSize + 1) && threadIdx.x < (BLOCK_SIZE - kernelSize + 1)) {
        T sum = 0;
        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                sum += (in_tile[threadIdx.y + i][threadIdx.x + j]) * k_tile[j * kernelSize + i]; // kernel transpose
        out[col * matSize + row] = 5;
        printf("Col =  %d, row = %d", col, row);
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
    const int xSize = 4;
    const int ySize = 4;
    const int totSize = xSize * ySize;
    const int size = totSize * sizeof(int);
    const int kSize = 9;
    int* dev_a, * dev_k, * dev_new;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_new, size);
    cudaMalloc((void**)&dev_k, kSize * sizeof(int));

    int* phost_a, * phost_k, * phost_new;

    phost_a = (int*)malloc(size);
    phost_new = (int*)malloc(size);
    phost_k = (int*)malloc(kSize * sizeof(int));

    for (int i = 0; i < totSize; i++) {
        phost_a[i] = i + 1; // rand() % 255;
    }

    // Filter matrix
    for (int i = 0; i < kSize; i++) {
        phost_k[i] = 0;
    }
    phost_k[4] = 1;


    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k, phost_k, kSize * sizeof(int), cudaMemcpyHostToDevice);

    printf("\n\n*****    MATRIX k    *****\n");
    printMatrix(phost_k, 3, kSize);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(1);

    convolution << < gridDim, blockDim >> > (dev_a, dev_new, 4, dev_k, 3);
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


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

using namespace std;


__global__ void dotProduct(int* a, int* b, int matSize)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < matSize && j < matSize) {
        int trans = i * matSize + j;
        int orign = j * matSize + i;
        b[trans] = a[orign];
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
    const int vectorSize = 16;
    const int size = vectorSize * sizeof(int);
    int matSize = 4;
    int* dev_a, * dev_b;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);

    int* phost_a, * phost_b;

    phost_a = (int*)malloc(size);
    phost_b = (int*)malloc(size);

    for (int i = 0; i < vectorSize; i++) {
        phost_a[i] = i + 1;
    }

    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(1);
    dotProduct << < gridDim, blockDim >> > (dev_a, dev_b, matSize);
    cudaDeviceSynchronize();

    cudaMemcpy(phost_b, dev_b, size, cudaMemcpyDeviceToHost);

    printf("\n\n*****    MATRIX A    *****\n");
    printMatrix(phost_a, matSize);

    printf("\n\n*****    MATRIX B    *****\n");
    printMatrix(phost_b, matSize);

    cudaDeviceReset();
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}

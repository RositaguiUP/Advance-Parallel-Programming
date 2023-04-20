
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void searchNum(int* dev_a, int num, int * pos, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        if (dev_a[tid] == num) {
            pos[0] = tid;
        }
    }
}

int main()
{
    const int vectorSize = 10;
    const int size = vectorSize * sizeof(int);
    int numToSearch = 2;
    int* dev_a, * dev_pos;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_pos, 1 * sizeof(int));

    int* phost_a, * phost_pos;

    phost_a = (int*)malloc(size);
    phost_pos = (int*)malloc(1);

    phost_pos[0] = -1;

    for (int i = 0; i < vectorSize; i++) {
        phost_a[i] = rand() % 10;
    }

    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pos, phost_pos, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(vectorSize);
    dim3 gridDim(1);

    // Search a number, get its position
    searchNum << < gridDim, blockDim >> > (dev_a, numToSearch, dev_pos, vectorSize);
    cudaDeviceSynchronize();

    cudaMemcpy(phost_pos, dev_pos, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n\n*****    VECTOR A    *****\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("  %d", phost_a[i]);
    }

    if (phost_pos[0] == -1) {
        printf("\n\nThe number %d is not on the vector\n", numToSearch);
    }
    else {
        printf("\n\nThe number %d is on the position %d\n", numToSearch, phost_pos[0]+1);

    }


    /*cudaDeviceReset();*/
    cudaFree(dev_a);
    cudaFree(dev_pos);

    return 0;
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void (int *a, int *b, int *c)
{multiplyVectors
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    c[threadId] = a[threadId] * b[threadId];
}

int main()
{
    const int vectoSize = 32;

    int* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**) & dev_a, vectoSize * sizeof(int));
    cudaMalloc((void**) & dev_b, vectoSize * sizeof(int));
    cudaMalloc((void**) & dev_c, vectoSize * sizeof(int));

    int* phost_a, * phost_b, *phost_c;
    phost_a = (int*)malloc(vectoSize*sizeof(int));
    phost_b = (int*)malloc(vectoSize*sizeof(int));
    phost_c = (int*)malloc(vectoSize*sizeof(int));

    for (int i = 0; i < vectoSize; i++) {
        phost_a[i] = i;
        phost_b[i] = i;
    }

    cudaMemcpy(dev_a, phost_a, vectoSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, phost_b, vectoSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(4);
    dim3 block(vectoSize);

    multiplyVectors <<< grid, block >> > (dev_a, dev_b, dev_c);

    cudaMemcpy(phost_c, dev_c, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < vectoSize; i++) {
        printf("%d\t", phost_c[i]);
    }

    return 0;
}
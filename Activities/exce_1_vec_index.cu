
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void idkernel(int *a, int *b, int *c)
{
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    a[threadId] = threadIdx.x;
    b[threadId] = blockIdx.x;
    c[threadId] = threadId;
}

int main()
{
    const int vectoSize = 64;

    int* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**) & dev_a, vectoSize * sizeof(int));
    cudaMalloc((void**) & dev_b, vectoSize * sizeof(int));
    cudaMalloc((void**) & dev_c, vectoSize * sizeof(int));

    int* phost_a, * phost_b, *phost_c;
    phost_a = (int*)malloc(vectoSize*sizeof(int));
    phost_b = (int*)malloc(vectoSize*sizeof(int));
    phost_c = (int*)malloc(vectoSize*sizeof(int));

    dim3 grid(1);
    dim3 block(64);

    idkernel <<< grid, block >>> (dev_a, dev_b, dev_c);

    cudaMemcpy(phost_a, dev_a, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_b, dev_b, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_c, dev_c, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);

    printf("1 bloque 64 hilos:\n\n");
    for (int i = 0; i < vectoSize; i++) {
        printf("Indice de hilo: %d\tIndice de bloque: %d\tIndice global: %d\n", phost_a[i], phost_b[i], phost_c[i]);
    }


    dim3 grid2(64);
    dim3 block2(1);

    idkernel <<< grid2, block2 >>> (dev_a, dev_b, dev_c);

    cudaMemcpy(phost_a, dev_a, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_b, dev_b, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_c, dev_c, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n\n64 bloques 1 hilo:\n\n");
    for (int i =0; i < vectoSize; i++) {
        printf("Indice de hilo: %d\tIndice de bloque: %d\tIndice global: %d\n", phost_a[i], phost_b[i], phost_c[i]);
    }


    dim3 grid3(4);
    dim3 block3(16);

    idkernel <<< grid3, block3 >>> (dev_a, dev_b, dev_c);

    cudaMemcpy(phost_a, dev_a, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_b, dev_b, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_c, dev_c, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n\n4 bloques 16 hilos:\n\n");
    for (int i = 0; i < vectoSize; i++) {
        printf("Indice de hilo: %d\tIndice de bloque: %d\tIndice global: %d\n", phost_a[i], phost_b[i], phost_c[i]);
    }



    return 0;
}








    int nx = 4;
    int ny = 4;
    int nz = 4;

    dim3 block(2, 2, 2);
    dim3 grid(nx / block.x, ny / block.y, nz / block.z);

    idkernel <<< grid, block >>> (dev_a, dev_b, dev_c);

    germa.pinedo@cinvestav.com
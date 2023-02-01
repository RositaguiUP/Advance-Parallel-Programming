
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void idkernel()
{
    printf("threadIdx %d %d %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
}

__global__ void multiplyVectors(int *a, int *b, int *c)
{
    int id = threadIdx.x;
    c[id] = a[id] * b[id];
}

int main()
{
    int nx = 4;
    int ny = 4;
    int nz = 4;

    dim3 blockDim(2, 2, 2);
    dim3 gridDim(nx / blockDim.x, ny / blockDim.y, nz / blockDim.z);

    idkernel << <gridDim, blockDim >> > ();


    const int vectoSize = 3;

    int* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**)&dev_a, vectoSize * sizeof(int));
    cudaMalloc((void**)&dev_b, vectoSize * sizeof(int));
    cudaMalloc((void**)&dev_c, vectoSize * sizeof(int));

    int* phost_a, * phost_b, * phost_c;
    phost_a = (int*)malloc(vectoSize * sizeof(int));
    phost_b = (int*)malloc(vectoSize * sizeof(int));
    phost_c = (int*)malloc(vectoSize * sizeof(int));

    for (int i = 0; i < vectoSize; i++) {
        phost_a[i] = i;
        phost_b[i] = i;
    }

    cudaMemcpy(dev_a, phost_a, vectoSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, phost_b, vectoSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(vectoSize);
    dim3 grid(1);

    multiplyVectors << < grid, block >> > (dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaMemcpy(phost_c, dev_c, vectoSize * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\n\nVector A + Vector B = Vector C");
    for (int i = 0; i < vectoSize; i++) {
        printf("\n     %d   +\t%d   =\t%d", phost_a[i], phost_b[i], phost_c[i]);
    }
    cudaDeviceReset();
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
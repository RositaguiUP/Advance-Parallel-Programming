
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void bubblesort(int* dev_a, int* dev_sort, int n)
{
    __shared__ int s_data[1024];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        s_data[threadIdx.x] = dev_a[i];
    }
    __syncthreads();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (s_data[j] > s_data[j + 1]) {
                int temp = s_data[j];
                s_data[j] = s_data[j + 1];
                s_data[j + 1] = temp;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        dev_sort[i] = s_data[threadIdx.x];
    }
}

int main()
{
    const int vectorSize = 10;
    const int size = vectorSize * sizeof(int);
    int* dev_a, * dev_sort;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_sort, size);

    int* phost_a, * phost_sort;

    phost_a = (int*)malloc(size);
    phost_sort = (int*)malloc(size);

    for (int i = 0; i < vectorSize; i++) {
        phost_a[i] = rand() % 10;
    }

    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);

    dim3 blockDim(vectorSize);
    dim3 gridDim(1);

    // Sort
    bubblesort << < gridDim, blockDim >> > (dev_a, dev_sort, vectorSize);
    cudaDeviceSynchronize();

    cudaMemcpy(phost_sort, dev_sort, size, cudaMemcpyDeviceToHost);

    printf("\n\n*****    VECTOR A    *****\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("  %d", phost_a[i]);
    }

    printf("\n\n*****    VECTOR SORTED   *****\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("  %d", phost_sort[i]);
    }


    /*cudaDeviceReset();*/
    cudaFree(dev_a);
    cudaFree(dev_sort);

    return 0;
}
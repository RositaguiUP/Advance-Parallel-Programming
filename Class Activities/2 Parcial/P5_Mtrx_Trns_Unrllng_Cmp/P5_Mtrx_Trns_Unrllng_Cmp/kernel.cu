
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32

__global__ void transpose(int* in, int* out, int rows, int cols) {
    __shared__ int tile[TILE_DIM][TILE_DIM + 1];

    int idx_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int idx_out = threadIdx.y * rows + blockIdx.y * TILE_DIM + threadIdx.x;

    for (int i = 0; i < TILE_DIM; i += blockDim.y) {
        if (idx_in < rows && threadIdx.y + i < TILE_DIM) {
            tile[threadIdx.y + i][threadIdx.x] = in[idx_in * cols + threadIdx.y + i];
        }
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += blockDim.y) {
        if (idx_out < cols * rows && threadIdx.y + i < TILE_DIM) {
            out[idx_out] = tile[threadIdx.x][threadIdx.y + i];
        }
        idx_out += rows;
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
    const int rows = 4;
    const int cols = 4;
    const int size = rows * cols * sizeof(int);
    int* dev_a, * dev_res;

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_res, size);
    
    int* phost_a, * phost_res;

    phost_a = (int*)malloc(size);
    phost_res = (int*)malloc(size);


    for (int i = 0; i < rows*cols; i++) {
        phost_a[i] = i + 1; // rand() % 255;
    }

    cudaMemcpy(dev_a, phost_a, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM, 1);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM, 1);

    transpose << <grid, block >> > (dev_a, dev_res, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(phost_res, dev_res, size, cudaMemcpyDeviceToHost);
    
    printf("\n\n*****    MATRIX A    *****\n");
    printMatrix(phost_a, rows, rows*cols);

    printf("\n\n*****   MATRIX NEW   *****\n");
    printMatrix(phost_res, rows, rows * cols);

    cudaFree(dev_a);
    cudaFree(dev_res);

    return 0;
}

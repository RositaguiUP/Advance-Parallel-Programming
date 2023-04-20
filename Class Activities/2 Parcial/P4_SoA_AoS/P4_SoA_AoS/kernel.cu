
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define LEN 8

struct coords {
    int x;
    int y;
};

struct coordsArray {
    int x[LEN];
    int y[LEN];
};


__global__ void soa(coordsArray* data, coordsArray* res, const int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        int a = data->x[gid];
        int b = data->y[gid];

        a += 1;
        b += 2;
        res->x[gid] = a;
        res->y[gid] = b;
    }
}

__global__ void aos(coords* data, coords* res, const int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        coords c = data[gid];

        c.x += 1;
        c.y += 2;
        res[gid] = c;
    }
}

int main()
{
    const int vectorSize = LEN;
    const int size_a = vectorSize * sizeof(coords);
    const int size_b = vectorSize * sizeof(coordsArray);
    coords* dev_a, * dev_a_res;
    coordsArray* dev_b, * dev_b_res;
    
    cudaMalloc(&dev_a, size_a);
    cudaMalloc(&dev_a_res, size_a);
    cudaMalloc(&dev_b, size_b);
    cudaMalloc(&dev_b_res, size_b);
    
    
    coords* phost_a, * phost_a_res;
    coordsArray* phost_b, * phost_b_res;

    phost_a = (coords*)malloc(size_b);
    phost_a_res = (coords*)malloc(size_b);
    phost_b = (coordsArray*)malloc(size_b);
    phost_b_res = (coordsArray*)malloc(size_b);
    
    
    for (int i = 0; i < vectorSize; i++) {
        phost_a[i].x = 1;
        phost_a[i].y = 2;
        
        phost_b->x[i] = 1;
        phost_b->y[i] = 2;
    }
    
    cudaMemcpy(dev_a, phost_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, phost_b, size_b, cudaMemcpyHostToDevice);
    
    
    dim3 block(32);
    dim3 grid((vectorSize + 32 - 1) / (block.x));

    // AoS
    /*
    aos << <grid, block >> > (dev_a, dev_a_res, vectorSize);
    cudaDeviceSynchronize();
    cudaMemcpy(phost_a_res, dev_a_res, size_a, cudaMemcpyDeviceToHost);

    printf("\n\n ***** DATA *****\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("\tx: %d y: %d\n", phost_a[i].x, phost_a[i].y);
    }

    printf("\n\n ***** AoS *****\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("\tx: %d y: %d\n", phost_a_res[i].x, phost_a_res[i].y);
    }
    */

    // SoA
    
    soa << <grid, block >> > (dev_b, dev_b_res, vectorSize);
    cudaDeviceSynchronize();
    cudaMemcpy(phost_b_res, dev_b_res, size_b, cudaMemcpyDeviceToHost);

    printf("\n\n ***** DATA *****\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("\tx: %d y: %d\n", phost_b->x[i], phost_b->y[i]);
    }

    printf("\n\n ***** SoA *****\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("\tx: %d y: %d\n", phost_b_res->x[i], phost_b_res->y[i]);
    }
   
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(phost_a_res);
    cudaFree(phost_b_res);
}

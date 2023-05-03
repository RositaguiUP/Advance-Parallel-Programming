
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void simple_kernel() {
    printf("Hello from kernel");
}

__global__ void stream_test(int* in, int* out, int size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        // ANY CALC
        for (int i = 0; i < 25; i++) {
            out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
        }
    }
}


int main()
{
    int size = 1 << 18;
    int byte_size = size * sizeof(int);

    // Initialize host pointer
    int* h_in, * h_ref, * h_in2, * h_ref2;

    cudaMallocHost((void**)&h_in, byte_size);
    cudaMallocHost((void**)&h_ref, byte_size);
    cudaMallocHost((void**)&h_in2, byte_size);
    cudaMallocHost((void**)&h_ref2, byte_size);

    srand((double)time(NULL));
    for (int i = 0; i < size; i++) {
        h_in[i] = rand();
        h_in2[i] = rand();
    }

    // allocate device pointers
    int* d_in, * d_out, * d_in2, * d_out2;
    cudaMalloc((void**)&d_in, byte_size);
    cudaMalloc((void**)&d_out, byte_size);
    cudaMalloc((void**)&d_in2, byte_size);
    cudaMalloc((void**)&d_out2, byte_size);

    // kernel launch
    dim3 block(128);
    dim3 grid(size / block.x);
    cudaStream_t str, str2;
    cudaStreamCreate(&str);
    cudaStreamCreate(&str2);

    // trasfer data from host to device
    cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice, str);
    stream_test <<< grid, block, 0, str >> > (d_in, d_out, size);
    cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost, str);

    cudaMemcpyAsync(d_in2, h_in2, byte_size, cudaMemcpyHostToDevice, str2);
    stream_test <<< grid, block, 0, str2 >> > (d_in, d_out, size);
    cudaMemcpyAsync(h_ref2, d_out2, byte_size, cudaMemcpyDeviceToHost, str2);

    cudaStreamSynchronize(str);
    cudaStreamDestroy(str);

    cudaStreamSynchronize(str2);
    cudaStreamDestroy(str2);

    cudaDeviceReset();
    /*int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (&deviceProp.concurrentKernels == 0) {
        printf("> GPU doesn not support concurrent kernel execution \n");
        printf("kernel executiion will be serialized \n");
    }

    cudaStream_t str1, str2, str3;

    cudaStreamCreate(&str1);
    cudaStreamCreate(&str2);
    cudaStreamCreate(&str3);

    simple_kernel <<< 1, 1, 0, str1 >> > ();
    simple_kernel <<< 1, 1, 0, str2 >> > ();
    simple_kernel <<< 1, 1, 0, str3 >> > ();

    cudaStreamDestroy(str1);
    cudaStreamDestroy(str2);
    cudaStreamDestroy(str3);

    cudaDeviceSynchronize();
    cudaDeviceReset();*/

    return 0;
}

/*
P1: SUma de arreglos
N = 8 streams

*/

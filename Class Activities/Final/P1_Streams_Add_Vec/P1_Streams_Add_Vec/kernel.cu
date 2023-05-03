
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void stream_add_vec(int* a, int* b, int* c, int size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

int main()
{
    int size = 1 << 18;
    int byte_size = size * sizeof(int);

    int const N_STREAMS = 8;
    int ELEMENTS_PER_STREAM = size / N_STREAMS;
    int BYTES_PER_STREAM = byte_size / N_STREAMS;

    // Initialize host pointer
    int* h_a, * h_b, * h_c;

    cudaMallocHost((void**)&h_a, byte_size);
    cudaMallocHost((void**)&h_b, byte_size);
    cudaMallocHost((void**)&h_c, byte_size);

    // srand((double)time(NULL));
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // allocate device pointers
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, byte_size);
    cudaMalloc((void**)&d_b, byte_size);
    cudaMalloc((void**)&d_c, byte_size);

    // kernel launch
    dim3 block(128);
    dim3 grid(ELEMENTS_PER_STREAM / block.x + 1);

    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int offset = 0;

    // trasfer data from host to device
    for (int i = 0; i < N_STREAMS; i++) {
        offset = i * ELEMENTS_PER_STREAM;
        cudaMemcpyAsync(d_a + offset, h_a + offset, BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b + offset, h_b + offset, BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
        stream_add_vec << <grid, block, 0, streams[i] >> > (d_a + offset, d_b + offset, d_c + offset, ELEMENTS_PER_STREAM);
        cudaMemcpyAsync(h_c + offset, d_c + offset, BYTES_PER_STREAM, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();

    /*for (int i = 0; i < size; i++) {
        printf("\n%d + %d = %d", h_a[i], h_b[i], h_c[i]);
    }*/
   
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

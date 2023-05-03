
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024

struct patients {
    int PatientID[N];
    int age[N];
    double glucosa[N];
    double heart_rate[N];
    double preassure_s[N];
    double preassure_d[N];
};

void fillData(patients * data) {
    for (int i = 0; i < N; i++) {
        data->PatientID[i] = i;
        data->age[i] = (rand() % (60 - 12 + 1)) + 12;
        data->glucosa[i] = (rand() % (350 - 100 + 1)) + 100;
        data->heart_rate[i] = (rand() % (170 - 90 + 1)) + 90;
        data->preassure_s[i] = (rand() % (150 - 100 + 1)) + 100;
        data->preassure_d[i] = (rand() % (90 - 70 + 1)) + 70;
    }
}

__global__ void getMeanStdDev(patients* data, double* res_mean, double* res_std_dev) {
    __shared__ patients s_data_m;
    __shared__ patients s_data_sd;

    int gid = (threadIdx.x + threadIdx.y * blockDim.x) + (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);
    int offset = N / 2;
    

    s_data_m = data[0];
    s_data_sd = data[0];

    __syncthreads();

    // Mean
    for (int i = offset; i > 0; i /= 2)
    {
        if (gid + i < N) {
            s_data_m.age[gid] += s_data_m.age[gid + i];
            s_data_m.glucosa[gid] += s_data_m.glucosa[gid + i];
            s_data_m.heart_rate[gid]  += s_data_m.heart_rate[gid + i];
            s_data_m.preassure_s[gid] += s_data_m.preassure_s[gid + i];
            s_data_m.preassure_d[gid] += s_data_m.preassure_d[gid + i];
        }
        __syncthreads();
    }

    res_mean[0] = s_data_m.age[0] / N;
    res_mean[1] = s_data_m.glucosa[0] / N;
    res_mean[2] = s_data_m.heart_rate[0] / N;
    res_mean[3] = s_data_m.preassure_s[0] / N;
    res_mean[4] = s_data_m.preassure_d[0] / N;

    // Std Dev

    __syncthreads();

    if (gid < N) {
        s_data_sd.age[gid] = pow(s_data_sd.age[gid] - res_mean[0], 2);
        s_data_sd.glucosa[gid] = pow(s_data_sd.glucosa[gid] - res_mean[1], 2);
        s_data_sd.heart_rate[gid]  = pow(s_data_sd.heart_rate[gid] - res_mean[2], 2);
        s_data_sd.preassure_s[gid] = pow(s_data_sd.preassure_s[gid] - res_mean[3], 2);
        s_data_sd.preassure_d[gid] = pow(s_data_sd.preassure_d[gid] - res_mean[4], 2);
    }
    
    __syncthreads();

    for (int i = offset; i > 0; i /= 2)
    {
        if (gid + i < N) {
            s_data_sd.age[gid] += s_data_sd.age[gid + i];
            s_data_sd.glucosa[gid] += s_data_sd.glucosa[gid + i];
            s_data_sd.heart_rate[gid]  += s_data_sd.heart_rate[gid + i];
            s_data_sd.preassure_s[gid] += s_data_sd.preassure_s[gid + i];
            s_data_sd.preassure_d[gid] += s_data_sd.preassure_d[gid + i];
        }
        __syncthreads();
    }
    
    res_std_dev[0] = s_data_sd.age[0] / (N - 2);
    res_std_dev[1] = s_data_sd.glucosa[0] / (N - 2);
    res_std_dev[2] = s_data_sd.heart_rate[0] / (N - 2);
    res_std_dev[3] = s_data_sd.preassure_s[0] / (N - 2);
    res_std_dev[4] = s_data_sd.preassure_d[0] / (N - 2);

}


void getMeanStdDevCPU(patients data, double* res_mean, double* res_std_dev) {
    // Mean
    res_mean[0] = 0;
    res_mean[1] = 0;
    res_mean[2] = 0;
    res_mean[3] = 0;
    res_mean[4] = 0;

    for (int i = 0; i < N; i++) {
        res_mean[0] += data.age[i];
        res_mean[1] += data.glucosa[i];
        res_mean[2] += data.heart_rate[i];
        res_mean[3] += data.preassure_s[i];
        res_mean[4] += data.preassure_d[i];
    }

    res_mean[0] /= N;
    res_mean[1] /= N;
    res_mean[2] /= N;
    res_mean[3] /= N;
    res_mean[4] /= N;

    // Std Dev
    res_std_dev[0] = 0;
    res_std_dev[1] = 0;
    res_std_dev[2] = 0;
    res_std_dev[3] = 0;
    res_std_dev[4] = 0;

    for (int i = 0; i < N; i++) {
        res_std_dev[0] += pow(data.age[i] - res_mean[0], 2);
        res_std_dev[1] += pow(data.glucosa[i] - res_mean[1], 2);
        res_std_dev[2] += pow(data.heart_rate[i] - res_mean[2], 2);
        res_std_dev[3] += pow(data.preassure_s[i] - res_mean[3], 2);
        res_std_dev[4] += pow(data.preassure_d[i] - res_mean[4], 2);
    }

    res_std_dev[0] = sqrt(res_std_dev[0] / (N-2));
    res_std_dev[1] = sqrt(res_std_dev[1] / (N-2));
    res_std_dev[2] = sqrt(res_std_dev[2] / (N-2));
    res_std_dev[3] = sqrt(res_std_dev[3] / (N-2));
    res_std_dev[4] = sqrt(res_std_dev[4] / (N-2));
}

void printResults(patients *data, double *res_mean, double *res_std_dev) {
    /*printf("\n\n ********** DATA **********\n");
    for (int i = 0; i < N; i++) {
        printf("\n\t %d = %d", i, data->age[i]);
    }*/

    printf("\n\n\t ********** MEAN RESULTS **********\n");
    printf("\n\t\tAge           = %f", res_mean[0]);
    printf("\n\t\tGlucosa       = %f", res_mean[1]);
    printf("\n\t\tHeart Rate    = %f", res_mean[2]);
    printf("\n\t\tPreassure S   = %f", res_mean[3]);
    printf("\n\t\tPreassure D   = %f", res_mean[4]);
                
    printf("\n\n\t ********** STD DEV RESULTS **********\n");
    printf("\n\t\tAge           = %f", res_std_dev[0]);
    printf("\n\t\tGlucosa       = %f", res_std_dev[1]);
    printf("\n\t\tHeart Rate    = %f", res_std_dev[2]);
    printf("\n\t\tPreassure S   = %f", res_std_dev[3]);
    printf("\n\t\tPreassure D   = %f\n\n", res_std_dev[4]);
}

int main()
{
    const int size_data = sizeof(patients);
    const int size_res  = 5 * sizeof(double);
    patients *dev_data;
    double   *dev_res_mean;
    double   *dev_res_std_dev;


    cudaMalloc(&dev_data, size_data);
    cudaMalloc(&dev_res_mean,  size_res);
    cudaMalloc(&dev_res_std_dev,  size_res);


    patients phost_data;
    double   *phost_res_mean;
    double   *phost_res_std_dev;
    
    
    phost_res_mean = (double*)malloc(size_res);
    phost_res_std_dev = (double*)malloc(size_res);

    fillData(&phost_data);

    cudaMemcpy(dev_data, &phost_data, size_data, cudaMemcpyHostToDevice);

    dim3 blockDim(32,32);
    dim3 gridDim(1);


    // Implemention with parallel techniques
    getMeanStdDev << < gridDim, blockDim >> >  (dev_data, dev_res_mean, dev_res_std_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(phost_res_mean, dev_res_mean, size_res, cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_res_std_dev, dev_res_std_dev, size_res, cudaMemcpyDeviceToHost);

    phost_res_std_dev[0] = sqrt(phost_res_std_dev[0]);
    phost_res_std_dev[1] = sqrt(phost_res_std_dev[1]);
    phost_res_std_dev[2] = sqrt(phost_res_std_dev[2]);
    phost_res_std_dev[3] = sqrt(phost_res_std_dev[3]);
    phost_res_std_dev[4] = sqrt(phost_res_std_dev[4]);
    
    printf("\n\n ******************** PARALLEL RESULTS ********************\n\n");
    printResults(&phost_data, phost_res_mean, phost_res_std_dev);

    // Secuencial calculus
    getMeanStdDevCPU(phost_data, phost_res_mean, phost_res_std_dev);
    
    printf("\n\n ******************** NORMAL RESULTS ********************\n\n");
    printResults(&phost_data, phost_res_mean, phost_res_std_dev);

    // cudaFree(dev_data);
    cudaFree(dev_res_mean);
    cudaFree(dev_res_std_dev);

    return 0;
}



#include<device_launch_parameters.h>
#include<cuda_runtime.h>
#include<algorithm>
#include<iostream>
#include<assert.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<math.h>
#include<vector>

// CUDA kernel for vector addition

__global__ void vecAdd(float *out, float *a, float *b, int n) {
    
    // Calculate global thread ID 
    
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    // Vector boundary gaurd
    
    if (i < n){

        // Each thread adds a single element
        
        out[i] = a[i] + b[i];
        }
}

// Check results to make sure they are correct
void verify_results(std :: vector<float> &out, std :: vector<float> &a, std :: vector<float> &b, int n) {
    for (int i = 0; i < n; i++) {
        assert(out[i] == a[i] + b[i]);
    }
}

int main() {
    
    // Set up device
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
    
    // Set up data size of vectors
    
    int n = 1 << 20;
    
    // Allocate host memory
    
    size_t size_a = n * sizeof(float);
    size_t size_b = n * sizeof(float);
    size_t size_out = n * sizeof(float);
    
    // Allocate device memory
    
    float *d_a, *d_b, *d_out;
    
    cudaMalloc((void **) &d_a, size_a);
    cudaMalloc((void **) &d_b, size_b);
    cudaMalloc((void **) &d_out, size_out);
    
    // Initialize host memory
    
    float *h_a = (float *)malloc(size_a);
    float *h_b = (float *)malloc(size_b);
    float *h_out = (float *)malloc(size_out);
    
    // Initialize host vectors
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Initialize device vectors
    
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    
    // Launch vector addition kernel
    vecAdd<<<1, n>>>(d_out, d_a, d_b, n);
    
    // Copy result from device to host
    
    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
    
    // Verify results
    
    std :: vector<float> out(n);
    std :: vector<float> a(n);
    std :: vector<float> b(n);
    
    cudaMemcpy(out.data(), d_out, size_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(a.data(), d_a, size_a, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, size_b, cudaMemcpyDeviceToHost);
    
    verify_results(out, a, b, n);
    
    // Free device memory
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    
    // Free host memory
    
    free(h_a);
    free(h_b);
    free(h_out);
    
    return 0;

}
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <cuda_runtime.h>
using namespace std;

constexpr int N = 1024;
constexpr size_t SIZE = N * sizeof(float);
constexpr int cnt = 1000000;

// Compute C = A + B
__global__ void element_wise_vector_add(float *A, float *B, float *C, int Length) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < Length)
        C[tid] = A[tid] + B[tid];
}

int64_t NotUseL2Cache() {
    float *host_A = (float *)malloc(SIZE);
    float *host_B = (float *)malloc(SIZE);
    float *host_C = (float *)malloc(SIZE);

    // Init
    for (int i = 0; i < N; ++i) {
        host_A[i] = i;
        host_B[i] = i;
        host_C[i] = 0;
    }

    // Allocate memory in device
    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, SIZE);
    cudaMalloc(&device_B, SIZE);
    cudaMalloc(&device_C, SIZE);

    // Move host_* array to device
    cudaMemcpy(device_A, host_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, SIZE, cudaMemcpyHostToDevice);

    // Repeat cnt - 5 times to get an average execution time,
    // The first 5 executions is a warmup operations.
    std::chrono::high_resolution_clock::time_point start;
    for (int i = 0; i < cnt; ++i) {
        // Skip the first 5 cycle
        if (i == 5)
            start = chrono::high_resolution_clock::now();

        // Invoke kernel
        element_wise_vector_add<<<N / 256, 256>>>(device_A, device_B, device_C, N);
    }
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    int64_t time_costs = chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return time_costs;
}

int64_t UseL2Cache() {
    float *host_A = (float *)malloc(SIZE);
    float *host_B = (float *)malloc(SIZE);
    float *host_C = (float *)malloc(SIZE);

    // Init
    for (int i = 0; i < N; ++i) {
        host_A[i] = i;
        host_B[i] = i;
        host_C[i] = 0;
    }

    // Allocate memory in device
    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, SIZE);
    cudaMalloc(&device_B, SIZE);
    cudaMalloc(&device_C, SIZE);

    // Move host_* array to device
    cudaMemcpy(device_A, host_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, SIZE, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Get device properties
    cudaDeviceProp prop;
    int device_id = 0;
    cudaGetDeviceProperties(&prop, device_id);

    // Set 75% of L2 to set-aside cache.
    size_t size = min((int)(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

    // Select minimum of user defined num_bytes and max window size.
    int num_bytes = 1024 * 16; // 16KB
    size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);

    cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void *>(device_A);
    stream_attribute.accessPolicyWindow.num_bytes = window_size; // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio = 0.6; // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Persistence Property
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; // Type of access property on cache miss

    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); // Set the attributes to a CUDA Stream

    // Repeat cnt - 5 times to get an average execution time,
    // The first 5 executions is a warmup operations.
    std::chrono::high_resolution_clock::time_point start;
    for (int i = 0; i < cnt; ++i) {
        // Skip the first 5 cycle
        if (i == 5)
            start = chrono::high_resolution_clock::now();

        // Invoke kernel
        element_wise_vector_add<<<N / 256, 256, 0, stream>>>(device_A, device_B, device_C, N);
    }
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    int64_t time_costs = chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return time_costs;
}

int main() {
    cout << "NotUseL2Cache: " << NotUseL2Cache() << " microsecond(s)" << endl;
    cout << "UseL2Cache: " << UseL2Cache() << " microsecond(s)" << endl;
}

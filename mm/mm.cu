#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <cuda_runtime.h>
using namespace std;

// #define DEBUG

#ifdef DEBUG
constexpr int HEIGHT = 3;
constexpr int WIDTH = 3;
constexpr int K = 3;
constexpr int BLOCK_SIZE = 16;
constexpr int RepeatTimes = 1;
constexpr int Skip = 0;
#else
constexpr int HEIGHT = 4096;
constexpr int WIDTH = 8192;
constexpr int K = 1024;
constexpr int BLOCK_SIZE = 16;
constexpr int RepeatTimes = 50;
constexpr int WarmUp = 5; //
#endif


namespace NAIVE_MM {
// Matrices are stored in row-major order.
// Matrix(row, col) = *(Matrix.element + row * Matrix.width + col);
struct Matrix {
    int height{0};
    int width{0};
    float *element{nullptr};
};

void InitMatrix(Matrix mat) {
    for (int i = 0; i < mat.height; ++i)
        for (int j = 0; j < mat.width; ++j)
            *(mat.element + mat.width * i + j) = i + j;
}

void PrintMatrix(const Matrix mat) {
    cout << "----------------------" << endl;
    for (int i = 0; i < mat.height; ++i) {
        for (int j = 0; j < mat.width; ++j)
            cout << *(mat.element + mat.width * i + j) << " ";
        cout << endl;
    }
    cout << endl;
}

__global__ void naive_mm(const Matrix A, const Matrix B, Matrix C) {
    // Each thread compute one element of C.
    float C_val = 0;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    for (int i = 0; i < A.width; ++i)
        C_val += A.element[row * A.width + i] * B.element[i * B.width + col];
    if (row < C.height && col < C.width)
        C.element[row * C.width + col] = C_val;
}

__global__ void mm(const Matrix A, const Matrix B, Matrix C) {
}

int64_t NotUseSM() {
    Matrix A, B, C;
    A.height = HEIGHT;
    A.width = K;
    B.height = K;
    B.width = WIDTH;
    C.height = HEIGHT;
    C.width = WIDTH;

    A.element = (float *)malloc(A.width * A.height * sizeof(float));
    B.element = (float *)malloc(B.width * B.height * sizeof(float));
    C.element = (float *)malloc(C.width * C.height * sizeof(float));

    NAIVE_MM::InitMatrix(A);
    NAIVE_MM::InitMatrix(B);


    // Allocate A, B, C in device memory.
    Matrix device_A, device_B, device_C;
    device_A.height = A.height;
    device_A.width = A.width;
    device_B.height = B.height;
    device_B.width = B.width;
    device_C.height = C.height;
    device_C.width = C.width;
    size_t size = A.height * A.width * sizeof(float);
    cudaMalloc(&device_A.element, size);
    cudaMemcpy(device_A.element, A.element, size, cudaMemcpyHostToDevice);

    size = B.height * B.width * sizeof(float);
    cudaMalloc(&device_B.element, size);
    cudaMemcpy(device_B.element, B.element, size, cudaMemcpyHostToDevice);

    size = C.height * C.width * sizeof(float);
    cudaMalloc(&device_C.element, size);

    // Invoke kernel func.
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE, (WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE);
    naive_mm<<<dim_grid, dim_block>>>(device_A, device_B, device_C);

    std::chrono::high_resolution_clock::time_point start;
    for (int i = 0; i < RepeatTimes; ++i) {
        // Skip the first `WarmUp` cycle
        if (i == WarmUp)
            start = chrono::high_resolution_clock::now();
        naive_mm<<<dim_grid, dim_block>>>(device_A, device_B, device_C);
    }
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    // Copy result C from device to host.
    cudaMemcpy(C.element, device_C.element, size, cudaMemcpyDeviceToHost);

    // Print result.
    // PrintMatrix(C);

    // Free device memory.
    cudaFree(device_A.element);
    cudaFree(device_B.element);
    cudaFree(device_C.element);

    int64_t time_costs = chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return time_costs;
}
}

namespace MM {
// Matrices are stored in row-major order.
// Matrix(row, col) = *(Matrix.element + row * Matrix.width + col);
struct Matrix {
    int height{0};
    int width{0};
    // The stride field refers to the row stride of the matrix.
    // In row-major order, a mat is stored like following format:
    // +-----+-----+-----+-----+-----+
    // | val | val | val | pad | pad |
    // +-----+-----+-----+-----+-----+
    // | val | val | val | pad | pad |
    // +-----+-----+-----+-----+-----+
    // In this format, stride=5 means that there are 5 elements
    // between two adjacent rows of elements.
    int stride{0};
    float *element{nullptr};
};

void PrintMatrix(const Matrix mat) {
    cout << "----------------------" << endl;
    for (int i = 0; i < mat.height; ++i) {
        for (int j = 0; j < mat.width; ++j)
            cout << *(mat.element + mat.width * i + j) << " ";
        cout << endl;
    }
    cout << endl;
}

void InitMatrix(Matrix mat) {
    for (int i = 0; i < mat.height; ++i)
        for (int j = 0; j < mat.width; ++j)
            mat.element[mat.stride * i + j] = i + j;
}

// Get a matrix element
__device__ float GetElement(const Matrix mat, int row, int col) {
    return mat.element[row * mat.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix mat, int row, int col, float val) {
    mat.element[row * mat.stride + col] = val;
}

// Get the BLOCK_SIZE*BLOCK_SIZE submatrix of `mat` that
// is located col sub-matrices to the right and row
// sub-matrices down from the upper-left corner of `mat`
__device__ Matrix GetSubMatrix(const Matrix mat, int row, int col) {
    Matrix submat;
    submat.width = BLOCK_SIZE;
    submat.height = BLOCK_SIZE;
    submat.stride = mat.stride;
    submat.element = &mat.element[mat.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return submat;
}

__global__ void matmul(const Matrix A, const Matrix B, Matrix C) {
    // Block row and col
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    // Each thread block computes one sub-matrix Csub of C.
    Matrix csub = GetSubMatrix(C, block_row, block_col);
    // Each thread computes one element of Csub
    float c_val = 0;

    // Thread row and col with Csub
    int row = threadIdx.x;
    int col = threadIdx.y;

    // Loop over all the sub-matrices of A and B that are required to compute Csub.
    // Multiply each pair of sub-matrices together and accumulate the results.
    for (int m = 0; m < ((A.width + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, block_row, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, block_col);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from global memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            c_val += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    if (blockIdx.x * blockDim.x + threadIdx.x >= C.height || blockIdx.y * blockDim.y + threadIdx.y >= C.width)
        return;

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(csub, row, col, c_val);
}


int64_t UseSM() {
    Matrix A, B, C;
    A.height = HEIGHT;
    A.width = K;
    A.stride = A.width;
    B.height = K;
    B.width = WIDTH;
    B.stride = B.width;
    C.height = HEIGHT;
    C.width = WIDTH;
    C.stride = C.width;

    A.element = (float *)malloc(A.width * A.height * sizeof(float));
    B.element = (float *)malloc(B.width * B.height * sizeof(float));
    C.element = (float *)malloc(C.width * C.height * sizeof(float));

    InitMatrix(A);
    InitMatrix(B);

    // Allocate A, B, C in device memory.
    Matrix device_A, device_B, device_C;
    device_A.height = A.height;
    device_A.width = A.width;
    device_A.stride = A.stride;
    device_B.height = B.height;
    device_B.width = B.width;
    device_B.stride = B.stride;
    device_C.height = C.height;
    device_C.width = C.width;
    device_C.stride = C.stride;
    size_t size = A.height * A.stride * sizeof(float);
    cudaMalloc(&device_A.element, size);
    cudaMemcpy(device_A.element, A.element, size, cudaMemcpyHostToDevice);

    size = B.height * B.stride * sizeof(float);
    cudaMalloc(&device_B.element, size);
    cudaMemcpy(device_B.element, B.element, size, cudaMemcpyHostToDevice);

    size = C.height * C.stride * sizeof(float);
    cudaMalloc(&device_C.element, size);

    // Invoke kernel func.
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE, (WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::chrono::high_resolution_clock::time_point start;
    for (int i = 0; i < RepeatTimes; ++i) {
        // Skip the first `WarmUp` cycle
        if (i == WarmUp)
            start = chrono::high_resolution_clock::now();
        matmul<<<dim_grid, dim_block>>>(device_A, device_B, device_C);
    }
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    // Copy result C from device to host.
    cudaMemcpy(C.element, device_C.element, size, cudaMemcpyDeviceToHost);

    // Print result.
    // PrintMatrix(C);

    // Free device memory.
    cudaFree(device_A.element);
    cudaFree(device_B.element);
    cudaFree(device_C.element);

    int64_t time_costs = chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return time_costs;
}
}


int main() {
    cout << "NotUseSM: " << NAIVE_MM::NotUseSM() << " microsecond(s)" << endl;
    cout << "UseSM: " << MM::UseSM() << " microsecond(s)" << endl;
}

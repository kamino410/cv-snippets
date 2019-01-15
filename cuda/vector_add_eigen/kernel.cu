#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <iostream>
#include "kernel.hpp"

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            std::cout << "Error: " << __FILE__ << ":" << __LINE__ << std::endl \
                      << cudaGetErrorString(error) << std::endl;               \
            exit(1);                                                           \
        }                                                                      \
    }

__global__ void add_vector_fixed_kernel(const MyVector *in1, const MyVector *in2, MyVector *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= VECLEN) return;
    out[i] = in1[i] + in2[i];
}

void add_vector_fixed(const MyVector &in1, const MyVector &in2, MyVector &out) {
    MyVector *d_in1, *d_in2, *d_out;
    size_t memsize = sizeof(MyVector);
    CHECK(cudaMalloc((void **)&d_in1, memsize));
    CHECK(cudaMalloc((void **)&d_in2, memsize));
    CHECK(cudaMalloc((void **)&d_out, memsize));

    CHECK(cudaMemcpy(d_in1, in1.data(), memsize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in2, in2.data(), memsize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (VECLEN + threadsPerBlock - 1) / threadsPerBlock;
    add_vector_fixed_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in1, d_in2, d_out);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(out.data(), d_out, memsize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_in2));
    CHECK(cudaFree(d_out));
}

__global__ void add_vector_kernel(const double *in1, const double *in2, double *out, const int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in1[i] + in2[i];
}

void add_vector(const Eigen::VectorXd &in1, const Eigen::VectorXd &in2, Eigen::VectorXd &out) {
    if(in1.rows() != in2.rows() || in1.rows() != out.rows()) return;

    int len = in1.rows();
    double *d_in1, *d_in2, *d_out;
    size_t memsize = sizeof(double) * len;
    CHECK(cudaMalloc((void **)&d_in1, memsize));
    CHECK(cudaMalloc((void **)&d_in2, memsize));
    CHECK(cudaMalloc((void **)&d_out, memsize));

    CHECK(cudaMemcpy(d_in1, in1.data(), memsize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in2, in2.data(), memsize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    add_vector_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in1, d_in2, d_out, len);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(out.data(), d_out, memsize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_in2));
    CHECK(cudaFree(d_out));
}

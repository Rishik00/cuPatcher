// kernels.h

#pragma once

// Define the kernels here
extern __global__ void dataCopy(float* A, float* B, int N)

extern __global__ void sigmoidKernel(float* A, int N)

// __global__ void sumKernel(float* A, int N, float sum) 

// __global__ void matMul(float* A, float* B, float* C, int N, int M);

// __global__ void RectMul(float* A, float* B, float* C, int N, int M, int K);


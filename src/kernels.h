// kernels.h

#pragma once


// Define the kernels here
__global__dataCopy(float* A, float* B, int N)

__global__ sigmoidKernel(float* A, int N)

__global__ sumKernel(float* A, int N, float sum) 

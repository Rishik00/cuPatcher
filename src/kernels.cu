#include <iostream>
#include <cuda_runtime.h>

__global__ void dataCopy(float* A, float* B, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N) {
		B[idx] = A[idx];
	}
}

__global__ void sigmoidKernel(float* A, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N) {
		A[idx] = 1 / (1 + expf(A[idx]);
	}
}


__global__ void sumKernel(float* A, float* sum, int N) {
	

}




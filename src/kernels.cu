#include <cuda_runtime.h>
#include "kernels.h"

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

__global__ void matMul (float* A, float* B, float* C, int N, int M) {

}

__global__ void RectMul (float* A, float* B, float* C, int N, int M, int K) {

}

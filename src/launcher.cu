#include <cuda_runtime.h>
#include "kernels.h"

float* sigmoidDispatcher(float* A_d, float* A_h, int N) {
	float* res_d = nullptr;

	cudaMalloc(&A_d, sizeof(float) * N);
	cudaMalloc(&res_d, sizeof(float) * N);

	cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

	dim3 blockDim(N);
	dim3 gridDim(1);

	sigmoidKernel<<<gridDim, blockDim>>>(A_d, res_d, N);	
	cudaDeviceSynchronize();

	cudaMemcpy(res_d, A_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaFree(A_d);

	return res_h;
}

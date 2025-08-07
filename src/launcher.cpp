#include <iostream>
#include <kernels.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

std::string LaunchSigmoid(float* A, int N) {
	float* A_d

	if (N > 256) {
		return "nope cant go beyond 1024";
	}

	cudaMalloc(&A_d, sizeof(float) * N);

	cudaMemcpy(A_d, A, sizeof(float) * N, cudaMemcpyHostToDevice);

	dim3 blockDim(N);
	dim3 gridDim(1)
	sigmoidKernel<<<gridDim, blockDim>>>(A_d, N);
	
	cudaDeviceSynchronize();

	cudaMemcpy(A, A_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

	cudaFree(A_d);

	delete[] A;
	return "Success"
}

void LaunchCopyData() {
}

PYBIND11_MODULE(cuPatcher, m) {
	m.doc() = "Basic CUDA dispatcher written by Rishik00";
	m.def("launch_sigmoid", &LaunchSigmoid, "A kernel that can do sigmoid using cuda");

	m.def("launch_copy", &LaunchCopyData, "A kernel that copies data");

}

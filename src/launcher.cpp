#include <iostream>
#include "kernels.h"
#include <cuda_runtime.h>

// Pybind11 imports
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::string LaunchSigmoid(py::array_t<float> A, int N) {
	py::buffer_info A_buff = A.request();
	
	if (A_buff.ndim > 1) {
		return "ndims cant be greater than 1. please ensure only vectors are passed";
	}

	if (A_buff.size > 256) {
		return "nope cant go beyond 256";
	}

	float* A_d = nullptr;
	float* A_h = static_cast<float*> (A_buff.ptr);

	cudaMalloc(&A_d, sizeof(float) * A_buff.size);
	cudaMemcpy(A_d, A_h, sizeof(float) * A_buff.size, cudaMemcpyHostToDevice);

	dim3 blockDim(N);
	dim3 gridDim(1);

	sigmoidKernel<<<gridDim, blockDim>>>(A_d, N);	
	cudaDeviceSynchronize();

	cudaMemcpy(A_h, A_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaFree(A_d);

	return "Success";
}

void LaunchCopyData(py::array_t<float> A, int N) {
}

PYBIND11_MODULE(cuPatcher, m) {
	m.doc() = "Basic CUDA dispatcher written by Rishik00";

	m.def("launch_sigmoid", &LaunchSigmoid, "A kernel that can do sigmoid using cuda");
	m.def("launch_copy", &LaunchCopyData, "A kernel that copies data");
}

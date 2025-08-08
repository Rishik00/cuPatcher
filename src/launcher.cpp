#include <iostream>
#include "launcher.h"

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

	float* A_h = static_cast<float*> (A_buff.ptr);
	float *res_d = sigmoidDispatcher(A_h, A_buff.size);

	if (res_d) {
		return "Success";
	} else {
		return "Failed";
	}
}

void LaunchCopyData(py::array_t<float> A, int N) {
}

PYBIND11_MODULE(cuPatcher, m) {
	m.doc() = "Basic CUDA dispatcher written by Rishik00";

	m.def("launch_sigmoid", &LaunchSigmoid, "A kernel that can do sigmoid using cuda");
	m.def("launch_copy", &LaunchCopyData, "A kernel that copies data");
}

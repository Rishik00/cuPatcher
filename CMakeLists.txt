cmake_minimum_required(VERSION 3.12)

project(cuPatcher LANGUAGES CXX CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


add_library(cuPatcher MODULE 
	src/launcher.cpp
	src/kernels.cu
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set_target_properties(cuPatcher PROPERTIES 
	CUDA_SEPARATION_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
	PREFIX ""
)


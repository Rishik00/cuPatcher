cmake_minimum_required(VERSION 3.12)
project(cuPatcher LANGUAGES CXX CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1  # latest stable
)
FetchContent_MakeAvailable(pybind11)

add_library(cuPatcher MODULE
    src/launcher.cpp
    src/launcher.cu
    src/kernels.cu
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_target_properties(cuPatcher PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
)

target_link_libraries(cuPatcher PRIVATE pybind11::module)


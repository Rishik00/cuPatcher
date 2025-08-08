
# apparently this is for checking and ensuring the version of CMAKE 3.12
cmake_minimum_required(VERSION 3.12)

# project() is the project name.
project(example)

# setting the c++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# adding the amazing executable
add_executable(hello hello.cpp)


# declare the minimum version for CMake
cmake_minimum_required(VERSION 3.16)

# declare the project name
project(TensorLibrary LANGUAGES CXX)

# set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the output directory for the compiled binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# for enable gtest or any other testing framework
enable_testing()

# to read other CMake files in the subdirectories
add_subdirectory(lib)

add_subdirectory(tests)

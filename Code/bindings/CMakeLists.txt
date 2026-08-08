cmake_minimum_required(VERSION 3.28)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(tinytorch
    losses/loss.cpp
    core/module.cpp
    core/tensor.cpp
    layer/layer.cpp
    data/data.cpp
    data/dataloader.cpp
    transforms/transforms.cpp
)

target_link_libraries(tinytorch
    PRIVATE
        Tensor
)


target_include_directories(tinytorch
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/bindings
)

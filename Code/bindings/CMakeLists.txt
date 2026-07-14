find_package(pybind11 REQUIRED)

pybind11_add_module(tinytorch
    bindings.cpp
)

target_link_libraries(tinytorch
    PRIVATE
        Tensor
)

target_include_directories(tinytorch
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

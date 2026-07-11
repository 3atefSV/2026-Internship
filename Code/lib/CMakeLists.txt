add_library(Tensor
    tensor_arithmetic.cpp
    tensor_access.cpp
    tensor_matrix.cpp
    tensor_broadcast.cpp
    tensor_properties.cpp
)

target_include_directories(Tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

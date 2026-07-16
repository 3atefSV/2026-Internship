add_library(Tensor
    tensor/tensor_arithmetic.cpp
    tensor/tensor_access.cpp
    tensor/tensor_matrix.cpp
    tensor/tensor_broadcast.cpp
    tensor/tensor_properties.cpp
    tensor/tensor_reduction.cpp
    tensor/tensor_shapeOps.cpp
    tensor/Tensor_ActivationOps.cpp
    activations/activations.cpp
    layer/linear.cpp
    layer/dropout.cpp
    layer/sequential.cpp
)
set_target_properties(Tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(Tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/tensor
        ${PROJECT_SOURCE_DIR}/include/activations
        ${PROJECT_SOURCE_DIR}/include/layer
)

set_target_properties(Tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
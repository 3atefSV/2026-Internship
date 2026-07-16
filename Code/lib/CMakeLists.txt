add_library(Tensor STATIC

    #================ Tensor =================

    tensor/tensor_access.cpp
    tensor/tensor_arithmetic.cpp
    tensor/tensor_broadcast.cpp
    tensor/tensor_matrix.cpp
    tensor/tensor_properties.cpp
    tensor/tensor_reduction.cpp
    tensor/tensor_shapeOps.cpp

    #================ Layers =================

    layer/linear.cpp
    layer/dropout.cpp
    layer/sequential.cpp
)

target_include_directories(Tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(Tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
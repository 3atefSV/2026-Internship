add_library(Tensor
    tensor_arithmetic.cpp
    tensor_access.cpp
    tensor_matrix.cpp
    tensor_broadcast.cpp
    tensor_properties.cpp
    tensor_reduction.cpp
    tensor_shapeOps.cpp
)
set_target_properties(Tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(Tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

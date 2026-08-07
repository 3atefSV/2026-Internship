cmake_minimum_required(VERSION 3.28)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(add_test TARGET_NAME SOURCE_FILE)

    add_executable(${TARGET_NAME}
        ${SOURCE_FILE}
    )

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            Tensor
            GTest::gtest_main
    )

    gtest_discover_tests(${TARGET_NAME})

endfunction()

# ----------------- Tensor Tests -----------------

add_test(test_tensor_log_softmax tensor/test_tensor_log_softmax.cpp)
add_test(test_tensor_properties tensor/test_tensor_properties.cpp)
add_test(test_tensor_arithmetic tensor/test_tensor_arithmetic.cpp)
add_test(test_tensor_matmul tensor/test_tensor_matmul.cpp)
add_test(test_tensor_broadcast tensor/test_tensor_broadcast.cpp)
add_test(test_tensor_reduction tensor/test_tensor_reduction.cpp)
add_test(test_tensor_shapeOps tensor/test_tensor_shapeOps.cpp)

add_test(test_tensor_activations activations/test_tensor_activations.cpp)

# ----------------- Loss Tests -----------------
add_test(test_loss_mse losses/test_mse_loss.cpp)
add_test(test_loss_cross_entropy losses/test_cross_entropy_loss.cpp)
add_test(test_loss_binary_cross_entropy losses/test_binary_cross_entropy_loss.cpp)
add_test(test_layer_linear layer/test_linear.cpp)
add_test(test_layer_dropout layer/test_dropout.cpp)
add_test(test_layer_sequential layer/test_sequential.cpp)

add_test(test_dataset data/test_dataset.cpp)
add_test(test_transform data/test_transforms.cpp)

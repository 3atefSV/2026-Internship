cmake_minimum_required(VERSION 3.28)
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(add_tensor_test TEST_NAME)
    add_executable(${TEST_NAME}
        ${TEST_NAME}.cpp
    )

    target_link_libraries(${TEST_NAME}
        PRIVATE
            Tensor
            GTest::gtest_main
    )

    gtest_discover_tests(${TEST_NAME})
endfunction()

add_tensor_test(test_tensor_properties)
add_tensor_test(test_tensor_arithmetic)
add_tensor_test(test_tensor_matmul)
add_tensor_test(test_tensor_broadcast)
add_tensor_test(test_tensor_reduction)
add_tensor_test(test_tensor_shapeOps)
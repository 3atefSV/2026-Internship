include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(test_tensor_properties
    test_tensor_properties.cpp
)

target_link_libraries(test_tensor_properties
    PRIVATE
        Tensor
        GTest::gtest_main
)

add_executable(test_tensor_arithmetic
    test_tensor_arithmetic.cpp
)

target_link_libraries(test_tensor_arithmetic
    PRIVATE
        Tensor
        GTest::gtest_main
)

add_executable(test_tensor_matmul
    test_tensor_matmul.cpp
)

target_link_libraries(test_tensor_matmul
    PRIVATE
        Tensor
        GTest::gtest_main
)


add_executable(test_tensor_broadcast
    test_tensor_broadcast.cpp
)

target_link_libraries(test_tensor_broadcast
    PRIVATE
        Tensor
        GTest::gtest_main
)


include(GoogleTest)

gtest_discover_tests(test_tensor_properties)
gtest_discover_tests(test_tensor_arithmetic)
gtest_discover_tests(test_tensor_matmul)
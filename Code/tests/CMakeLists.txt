include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(test_tensor
    test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        Tensor
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(test_tensor)
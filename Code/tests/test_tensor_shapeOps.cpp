#include "tensor.h"
#include <gtest/gtest.h>

bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-6f)
            return false;
    }
    return true;
}

TEST(TensorShapeOpsTest, ReshapeKeepsDataOrder) {
    Tensor tensor({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor reshaped = tensor.reShape({3, 2});

    EXPECT_EQ(reshaped.shape(), Tensor::Shape({3, 2}));
    EXPECT_TRUE(compare_vectors(reshaped.data(), {1, 2, 3, 4, 5, 6}));
}

TEST(TensorShapeOpsTest, ReshapeInvalidShapeThrows) {
    Tensor tensor({1, 2, 3, 4}, {2, 2});

    EXPECT_THROW(tensor.reShape({3, 1}), std::invalid_argument);
}

TEST(TensorShapeOpsTest, Transpose2DDefaultAxes) {
    Tensor tensor({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor transposed = tensor.Transpose({});

    EXPECT_EQ(transposed.shape(), Tensor::Shape({3, 2}));
    EXPECT_TRUE(compare_vectors(transposed.data(), {1, 4, 2, 5, 3, 6}));
}

TEST(TensorShapeOpsTest, Transpose2DExplicitAxes) {
    Tensor tensor({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor transposed = tensor.Transpose({1, 0});

    EXPECT_EQ(transposed.shape(), Tensor::Shape({3, 2}));
    EXPECT_TRUE(compare_vectors(transposed.data(), {1, 4, 2, 5, 3, 6}));
}

TEST(TensorShapeOpsTest, Transpose1DReturnsSame) {
    Tensor tensor({1, 2, 3}, {3});

    Tensor transposed = tensor.Transpose({});

    EXPECT_EQ(transposed.shape(), Tensor::Shape({3}));
    EXPECT_TRUE(compare_vectors(transposed.data(), {1, 2, 3}));
}

TEST(TensorShapeOpsTest, Transpose3DDefaultAxes) {
    Tensor tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});

    Tensor transposed = tensor.Transpose({});

    EXPECT_EQ(transposed.shape(), Tensor::Shape({3, 2, 2}));
    EXPECT_TRUE(compare_vectors(transposed.data(), {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}));
}

TEST(TensorShapeOpsTest, Transpose3DExplicitAxes) {
    Tensor tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});

    Tensor transposed = tensor.Transpose({2, 0, 1});

    EXPECT_EQ(transposed.shape(), Tensor::Shape({3, 2, 2}));
    EXPECT_TRUE(compare_vectors(transposed.data(), {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12}));
}

TEST(TensorShapeOpsTest, TransposeInvalidAxesThrows) {
    Tensor tensor({1, 2, 3, 4}, {2, 2});

    EXPECT_THROW(tensor.Transpose({0, 1, 2}), std::invalid_argument);
}

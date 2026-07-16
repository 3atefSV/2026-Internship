#include "tensor.h"
#include <cmath>
#include <gtest/gtest.h>

TEST(TensorActivationOpsTest, Relu) {
    Tensor tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});

    Tensor result = tensor.relu();

    Tensor expected({0.0f, 0.0f, 0.0f, 1.0f, 2.0f}, {5});
    EXPECT_TRUE(result == expected);
}

TEST(TensorActivationOpsTest, Sigmoid) {
    Tensor tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});

    Tensor result = tensor.sigmoid();

    EXPECT_FLOAT_EQ(result.data()[0], 1.0f / (1.0f + std::exp(2.0f)));
    EXPECT_FLOAT_EQ(result.data()[1], 1.0f / (1.0f + std::exp(1.0f)));
    EXPECT_FLOAT_EQ(result.data()[2], 0.5f);
    EXPECT_FLOAT_EQ(result.data()[3], 1.0f / (1.0f + std::exp(-1.0f)));
    EXPECT_FLOAT_EQ(result.data()[4], 1.0f / (1.0f + std::exp(-2.0f)));
}

TEST(TensorActivationOpsTest, Tanh) {
    Tensor tensor({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}, {5});

    Tensor result = tensor.tanh();

    EXPECT_FLOAT_EQ(result.data()[0], std::tanh(-1.0f));
    EXPECT_FLOAT_EQ(result.data()[1], std::tanh(-0.5f));
    EXPECT_FLOAT_EQ(result.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(result.data()[3], std::tanh(0.5f));
    EXPECT_FLOAT_EQ(result.data()[4], std::tanh(1.0f));
}

TEST(TensorActivationOpsTest, Gelu) {
    Tensor tensor({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}, {5});

    Tensor result = tensor.gelu();

    const auto sigmoid = [](float x) {
        if (x >= 0.0f) {
            const float exp_neg = std::exp(-x);
            return 1.0f / (1.0f + exp_neg);
        }
        const float exp_pos = std::exp(x);
        return exp_pos / (1.0f + exp_pos);
    };

    EXPECT_NEAR(result.data()[0], -1.0f * sigmoid(1.702f * -1.0f), 1e-6f);
    EXPECT_NEAR(result.data()[1], -0.5f * sigmoid(1.702f * -0.5f), 1e-6f);
    EXPECT_FLOAT_EQ(result.data()[2], 0.0f);
    EXPECT_NEAR(result.data()[3], 0.5f * sigmoid(1.702f * 0.5f), 1e-6f);
    EXPECT_NEAR(result.data()[4], 1.0f * sigmoid(1.702f * 1.0f), 1e-6f);
}

TEST(TensorActivationOpsTest, SoftmaxStable) {
    Tensor tensor({1000.0f, 1001.0f}, {2});

    Tensor result = tensor.softmax();

    EXPECT_NEAR(result.data()[0], 0.268941f, 1e-6f);
    EXPECT_NEAR(result.data()[1], 0.731059f, 1e-6f);
}

TEST(TensorActivationOpsTest, SoftmaxAlongDimension) {
    Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    Tensor result = tensor.softmax(1);

    EXPECT_NEAR(result.data()[0], 0.268941f, 1e-6f);
    EXPECT_NEAR(result.data()[1], 0.731059f, 1e-6f);
    EXPECT_NEAR(result.data()[2], 0.268941f, 1e-6f);
    EXPECT_NEAR(result.data()[3], 0.731059f, 1e-6f);
}

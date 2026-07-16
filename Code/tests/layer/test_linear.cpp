#include "layer/linear.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

TEST(LinearLayerTest, InitializationAndShapes) {
    // 3 input features, 2 output features, with bias
    Linear layer(3, 2, true);

    EXPECT_EQ(layer.in_features(), 3);
    EXPECT_EQ(layer.out_features(), 2);
    EXPECT_TRUE(layer.has_bias());

    auto params = layer.parameters();
    // Should have 2 parameters: Weight and Bias
    ASSERT_EQ(params.size(), 2);
    EXPECT_EQ(params[0]->shape().size(), 2);
    EXPECT_EQ(params[0]->shape()[0], 3);
    EXPECT_EQ(params[0]->shape()[1], 2);

    EXPECT_EQ(params[1]->shape().size(), 1);
    EXPECT_EQ(params[1]->shape()[0], 2);
}

TEST(LinearLayerTest, ForwardPassShape) {
    // Fixed seed for deterministic behavior if needed
    Linear layer(4, 3, false, 42);

    // Batch size 2, 4 features
    Tensor input({1, 1, 1, 1, 2, 2, 2, 2}, {2, 4});
    Tensor output = layer.forward(input);

    // Output should be (Batch Size, Out Features) -> (2, 3)
    ASSERT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
}

TEST(LinearLayerTest, NoBiasParametersCount) {
    Linear layer(5, 5, false); // No bias
    auto params = layer.parameters();

    ASSERT_EQ(params.size(), 1);
}

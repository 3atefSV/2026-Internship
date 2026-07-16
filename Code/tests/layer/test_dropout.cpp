#include "layer/dropout.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(DropoutLayerTest, InvalidProbabilityThrows) {
    EXPECT_THROW(Dropout(1.5f), std::invalid_argument);
    EXPECT_THROW(Dropout(-0.1f), std::invalid_argument);
}

TEST(DropoutLayerTest, InferenceModePassesThrough) {
    Dropout layer(0.5f);
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor output = layer.forward(input, false);
    EXPECT_FLOAT_EQ(output.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 4.0f);
}

TEST(DropoutLayerTest, TrainingModeExtremeDrop) {
    Dropout layer(0.999f);
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor output = layer.forward(input, true);
    EXPECT_TRUE(output.data()[0] == 0.0f || output.data()[0] > 1.0f);
}
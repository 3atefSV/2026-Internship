#include "layer/dropout.h"
#include "layer/linear.h"
#include "layer/sequential.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>

TEST(SequentialLayerTest, ForwardPassAndParameters) {
    Sequential model;
    model.add(std::make_shared<Linear>(4, 8, true));
    model.add(std::make_shared<Dropout>(0.5f));
    model.add(std::make_shared<Linear>(8, 2, false));

    EXPECT_EQ(model.size(), 3);

    // Test parameters aggregation
    auto params = model.parameters();
    // 2 (from first Linear) + 0 (Dropout) + 1 (from second Linear) = 3
    EXPECT_EQ(params.size(), 3);

    // Test Forward Pass Chain
    Tensor input({1, 1, 1, 1, 2, 2, 2, 2}, {2, 4});
    Tensor output = model.forward(input, false);

    // Output should be (Batch Size, Final Out Features) -> (2, 2)
    ASSERT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 2);
}
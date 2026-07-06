#include "tensor.h"
#include <gtest/gtest.h>

TEST(TensorTest, DefaultConstructor) {
  Tensor tensor;

  EXPECT_TRUE(tensor.isEmpty());
  EXPECT_EQ(tensor.getSize(), 0);
  EXPECT_EQ(tensor.getNumDimensions(), 0);
  EXPECT_TRUE(tensor.getShape().empty());
  EXPECT_TRUE(tensor.getData().empty());
}

TEST(TensorTest, Constructor) {
  Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

  EXPECT_FALSE(tensor.isEmpty());

  EXPECT_EQ(tensor.getSize(), 4);
  EXPECT_EQ(tensor.getNumDimensions(), 2);

  EXPECT_EQ(tensor.getShape()[0], 2);
  EXPECT_EQ(tensor.getShape()[1], 2);

  EXPECT_EQ(tensor.getData()[0], 1.0f);
  EXPECT_EQ(tensor.getData()[1], 2.0f);
  EXPECT_EQ(tensor.getData()[2], 3.0f);
  EXPECT_EQ(tensor.getData()[3], 4.0f);

  EXPECT_EQ(tensor.dtype(), "float");
}
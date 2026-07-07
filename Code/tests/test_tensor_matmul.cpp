#include "tensor.h"
#include <gtest/gtest.h>

TEST(TensorMatMulTest, SquareMatrix) {
  Tensor a({1, 2, 3, 4}, {2, 2});

  Tensor b({5, 6, 7, 8}, {2, 2});

  Tensor c = a.matmul(b);

  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 2);
  EXPECT_EQ(c.size(), 4);

  EXPECT_FLOAT_EQ(c(0, 0), 19.0f);
  EXPECT_FLOAT_EQ(c(0, 1), 22.0f);
  EXPECT_FLOAT_EQ(c(1, 0), 43.0f);
  EXPECT_FLOAT_EQ(c(1, 1), 50.0f);
}

TEST(TensorMatMulTest, RectangularMatrix) {
  Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});

  Tensor b({7, 8, 9, 10, 11, 12}, {3, 2});

  Tensor c = a.matmul(b);

  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 2);

  EXPECT_FLOAT_EQ(c(0, 0), 58.0f);
  EXPECT_FLOAT_EQ(c(0, 1), 64.0f);
  EXPECT_FLOAT_EQ(c(1, 0), 139.0f);
  EXPECT_FLOAT_EQ(c(1, 1), 154.0f);
}

TEST(TensorMatMulTest, IdentityMatrix) {
  Tensor identity({1, 0, 0, 1}, {2, 2});

  Tensor a({5, 6, 7, 8}, {2, 2});

  Tensor c = identity.matmul(a);

  EXPECT_FLOAT_EQ(c(0, 0), 5.0f);
  EXPECT_FLOAT_EQ(c(0, 1), 6.0f);
  EXPECT_FLOAT_EQ(c(1, 0), 7.0f);
  EXPECT_FLOAT_EQ(c(1, 1), 8.0f);
}

TEST(TensorMatMulTest, IncompatibleShapes) {
  Tensor a({1, 2, 3, 4}, {2, 2});

  Tensor b({1, 2, 3, 4, 5, 6}, {3, 2});

  EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST(TensorMatMulTest, OneDimensionalTensor) {
  Tensor a({1, 2, 3}, {3});

  Tensor b({4, 5, 6}, {3});

  EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST(TensorMatMulTest, OutOfBoundsAccess) {
  Tensor a({1, 2, 3, 4}, {2, 2});

  EXPECT_THROW(a(2, 0), std::out_of_range);

  EXPECT_THROW(a(0, 2), std::out_of_range);
}
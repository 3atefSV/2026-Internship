#include "tensor.h"

#include <gtest/gtest.h>
#include <stdexcept>

//
// Tensor + Tensor
//
TEST(TensorArithmeticTest, AddTwoTensors) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({5, 6, 7, 8}, {2, 2});

    Tensor c = a + b;

    EXPECT_EQ(c.size(), 4);
    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    EXPECT_FLOAT_EQ(c.data()[0], 6);
    EXPECT_FLOAT_EQ(c.data()[1], 8);
    EXPECT_FLOAT_EQ(c.data()[2], 10);
    EXPECT_FLOAT_EQ(c.data()[3], 12);
}

TEST(TensorArithmeticTest, AddZeroTensor) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor zero({0, 0, 0, 0}, {2, 2});

    Tensor c = a + zero;

    EXPECT_FLOAT_EQ(c.data()[0], 1);
    EXPECT_FLOAT_EQ(c.data()[1], 2);
    EXPECT_FLOAT_EQ(c.data()[2], 3);
    EXPECT_FLOAT_EQ(c.data()[3], 4);
}

//
// Tensor + Scalar
//
TEST(TensorArithmeticTest, AddScalar) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    Tensor c = a + 5.0f;

    EXPECT_FLOAT_EQ(c.data()[0], 6);
    EXPECT_FLOAT_EQ(c.data()[1], 7);
    EXPECT_FLOAT_EQ(c.data()[2], 8);
    EXPECT_FLOAT_EQ(c.data()[3], 9);
}

TEST(TensorArithmeticTest, ScalarPlusTensor) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    Tensor c = 5.0f + a;

    EXPECT_FLOAT_EQ(c.data()[0], 6);
    EXPECT_FLOAT_EQ(c.data()[1], 7);
    EXPECT_FLOAT_EQ(c.data()[2], 8);
    EXPECT_FLOAT_EQ(c.data()[3], 9);
}

TEST(TensorArithmeticTest, AddScalarToEmptyTensor) {
    Tensor empty;

    Tensor c = empty + 5.0f;

    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0);
}

//
// +=
//
TEST(TensorArithmeticTest, PlusEqualTensor) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({5, 6, 7, 8}, {2, 2});

    a += b;

    EXPECT_FLOAT_EQ(a.data()[0], 6);
    EXPECT_FLOAT_EQ(a.data()[1], 8);
    EXPECT_FLOAT_EQ(a.data()[2], 10);
    EXPECT_FLOAT_EQ(a.data()[3], 12);
}

TEST(TensorArithmeticTest, PlusEqualScalar) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    a += 5.0f;

    EXPECT_FLOAT_EQ(a.data()[0], 6);
    EXPECT_FLOAT_EQ(a.data()[1], 7);
    EXPECT_FLOAT_EQ(a.data()[2], 8);
    EXPECT_FLOAT_EQ(a.data()[3], 9);
}

//
// Tensor - Tensor
//
TEST(TensorArithmeticTest, SubtractTwoTensors) {
    Tensor a({5, 6, 7, 8}, {2, 2});
    Tensor b({1, 2, 3, 4}, {2, 2});

    Tensor c = a - b;

    EXPECT_FLOAT_EQ(c.data()[0], 4);
    EXPECT_FLOAT_EQ(c.data()[1], 4);
    EXPECT_FLOAT_EQ(c.data()[2], 4);
    EXPECT_FLOAT_EQ(c.data()[3], 4);
}

TEST(TensorArithmeticTest, SubtractScalar) {
    Tensor a({5, 6, 7, 8}, {2, 2});

    Tensor c = a - 2.0f;

    EXPECT_FLOAT_EQ(c.data()[0], 3);
    EXPECT_FLOAT_EQ(c.data()[1], 4);
    EXPECT_FLOAT_EQ(c.data()[2], 5);
    EXPECT_FLOAT_EQ(c.data()[3], 6);
}

//
// -=
//
TEST(TensorArithmeticTest, MinusEqualTensor) {
    Tensor a({5, 6, 7, 8}, {2, 2});
    Tensor b({1, 2, 3, 4}, {2, 2});

    a -= b;

    EXPECT_FLOAT_EQ(a.data()[0], 4);
    EXPECT_FLOAT_EQ(a.data()[1], 4);
    EXPECT_FLOAT_EQ(a.data()[2], 4);
    EXPECT_FLOAT_EQ(a.data()[3], 4);
}

TEST(TensorArithmeticTest, MinusEqualScalar) {
    Tensor a({5, 6, 7, 8}, {2, 2});

    a -= 2.0f;

    EXPECT_FLOAT_EQ(a.data()[0], 3);
    EXPECT_FLOAT_EQ(a.data()[1], 4);
    EXPECT_FLOAT_EQ(a.data()[2], 5);
    EXPECT_FLOAT_EQ(a.data()[3], 6);
}

//
// Tensor * Tensor
//
TEST(TensorArithmeticTest, MultiplyTwoTensors) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({5, 6, 7, 8}, {2, 2});

    Tensor c = a * b;

    EXPECT_FLOAT_EQ(c.data()[0], 5);
    EXPECT_FLOAT_EQ(c.data()[1], 12);
    EXPECT_FLOAT_EQ(c.data()[2], 21);
    EXPECT_FLOAT_EQ(c.data()[3], 32);
}

TEST(TensorArithmeticTest, MultiplyScalar) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    Tensor c = a * 3.0f;

    EXPECT_FLOAT_EQ(c.data()[0], 3);
    EXPECT_FLOAT_EQ(c.data()[1], 6);
    EXPECT_FLOAT_EQ(c.data()[2], 9);
    EXPECT_FLOAT_EQ(c.data()[3], 12);
}

TEST(TensorArithmeticTest, ScalarMultiplyTensor) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    Tensor c = 3.0f * a;

    EXPECT_FLOAT_EQ(c.data()[0], 3);
    EXPECT_FLOAT_EQ(c.data()[1], 6);
    EXPECT_FLOAT_EQ(c.data()[2], 9);
    EXPECT_FLOAT_EQ(c.data()[3], 12);
}

//
// *=
//
TEST(TensorArithmeticTest, MultiplyEqualTensor) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({5, 6, 7, 8}, {2, 2});

    a *= b;

    EXPECT_FLOAT_EQ(a.data()[0], 5);
    EXPECT_FLOAT_EQ(a.data()[1], 12);
    EXPECT_FLOAT_EQ(a.data()[2], 21);
    EXPECT_FLOAT_EQ(a.data()[3], 32);
}

TEST(TensorArithmeticTest, MultiplyEqualScalar) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    a *= 3.0f;

    EXPECT_FLOAT_EQ(a.data()[0], 3);
    EXPECT_FLOAT_EQ(a.data()[1], 6);
    EXPECT_FLOAT_EQ(a.data()[2], 9);
    EXPECT_FLOAT_EQ(a.data()[3], 12);
}

//
// Tensor / Tensor
//
TEST(TensorArithmeticTest, DivideTwoTensors) {
    Tensor a({10, 20, 30, 40}, {2, 2});
    Tensor b({2, 4, 5, 8}, {2, 2});

    Tensor c = a / b;

    EXPECT_FLOAT_EQ(c.data()[0], 5);
    EXPECT_FLOAT_EQ(c.data()[1], 5);
    EXPECT_FLOAT_EQ(c.data()[2], 6);
    EXPECT_FLOAT_EQ(c.data()[3], 5);
}

TEST(TensorArithmeticTest, DivideTensorByZero) {
    Tensor a({10, 20, 30, 40}, {2, 2});
    Tensor b({1, 2, 0, 4}, {2, 2});

    EXPECT_THROW(a / b, std::invalid_argument);
}

//
// Tensor / Scalar
//
TEST(TensorArithmeticTest, DivideScalar) {
    Tensor a({10, 20, 30, 40}, {2, 2});

    Tensor c = a / 10.0f;

    EXPECT_FLOAT_EQ(c.data()[0], 1);
    EXPECT_FLOAT_EQ(c.data()[1], 2);
    EXPECT_FLOAT_EQ(c.data()[2], 3);
    EXPECT_FLOAT_EQ(c.data()[3], 4);
}

TEST(TensorArithmeticTest, DivideByZeroScalar) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    EXPECT_THROW(a / 0.0f, std::invalid_argument);
}

//
// /=
//
TEST(TensorArithmeticTest, DivideEqualTensor) {
    Tensor a({10, 20, 30, 40}, {2, 2});
    Tensor b({2, 4, 5, 8}, {2, 2});

    a /= b;

    EXPECT_FLOAT_EQ(a.data()[0], 5);
    EXPECT_FLOAT_EQ(a.data()[1], 5);
    EXPECT_FLOAT_EQ(a.data()[2], 6);
    EXPECT_FLOAT_EQ(a.data()[3], 5);
}

TEST(TensorArithmeticTest, DivideEqualScalar) {
    Tensor a({10, 20, 30, 40}, {2, 2});

    a /= 10.0f;

    EXPECT_FLOAT_EQ(a.data()[0], 1);
    EXPECT_FLOAT_EQ(a.data()[1], 2);
    EXPECT_FLOAT_EQ(a.data()[2], 3);
    EXPECT_FLOAT_EQ(a.data()[3], 4);
}

//
// Scalar - Tensor
//
TEST(TensorArithmeticTest, ScalarMinusTensor) {
    Tensor a({1, 2, 3, 4}, {2, 2});

    Tensor c = 10.0f - a;

    EXPECT_FLOAT_EQ(c.data()[0], 9);
    EXPECT_FLOAT_EQ(c.data()[1], 8);
    EXPECT_FLOAT_EQ(c.data()[2], 7);
    EXPECT_FLOAT_EQ(c.data()[3], 6);
}

//
// Scalar / Tensor
//
TEST(TensorArithmeticTest, ScalarDivideTensor) {
    Tensor a({2, 4, 5, 10}, {2, 2});

    Tensor c = 20.0f / a;

    EXPECT_FLOAT_EQ(c.data()[0], 10);
    EXPECT_FLOAT_EQ(c.data()[1], 5);
    EXPECT_FLOAT_EQ(c.data()[2], 4);
    EXPECT_FLOAT_EQ(c.data()[3], 2);
}

TEST(TensorArithmeticTest, ScalarDivideTensorByZero) {
    Tensor a({1, 2, 0, 4}, {2, 2});

    EXPECT_THROW(10.0f / a, std::invalid_argument);
}
TEST(TensorArithmeticTest, ScalarMinusEmptyTensor) {
    Tensor empty;

    Tensor c = 5.0f - empty;

    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0);
}

TEST(TensorArithmeticTest, ScalarMultiplyEmptyTensor) {
    Tensor empty;

    Tensor c = 5.0f * empty;

    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0);
}

TEST(TensorArithmeticTest, ScalarDivideEmptyTensor) {
    Tensor empty;

    Tensor c = 5.0f / empty;

    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0);
}

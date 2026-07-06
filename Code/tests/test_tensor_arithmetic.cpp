#include <gtest/gtest.h>
#include "tensor.h"

//
// Tensor + Tensor
//
TEST(TensorArithmeticTest, AddTwoTensors)
{
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({5,6,7,8}, {2,2});

    Tensor c = a + b;

    EXPECT_EQ(c.getSize(), 4);
    EXPECT_EQ(c.getShape()[0], 2);
    EXPECT_EQ(c.getShape()[1], 2);

    EXPECT_FLOAT_EQ(c.getData()[0], 6);
    EXPECT_FLOAT_EQ(c.getData()[1], 8);
    EXPECT_FLOAT_EQ(c.getData()[2], 10);
    EXPECT_FLOAT_EQ(c.getData()[3], 12);
}

TEST(TensorArithmeticTest, AddShapeMismatch)
{
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({1,2}, {2});

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TensorArithmeticTest, AddZeroTensor)
{
    Tensor a({1,2,3,4}, {2,2});
    Tensor zero({0,0,0,0}, {2,2});

    Tensor c = a + zero;

    EXPECT_FLOAT_EQ(c.getData()[0], 1);
    EXPECT_FLOAT_EQ(c.getData()[1], 2);
    EXPECT_FLOAT_EQ(c.getData()[2], 3);
    EXPECT_FLOAT_EQ(c.getData()[3], 4);
}

//
// Tensor + Scalar
//
TEST(TensorArithmeticTest, AddScalar)
{
    Tensor a({1,2,3,4}, {2,2});

    Tensor c = a + 5.0f;

    EXPECT_FLOAT_EQ(c.getData()[0], 6);
    EXPECT_FLOAT_EQ(c.getData()[1], 7);
    EXPECT_FLOAT_EQ(c.getData()[2], 8);
    EXPECT_FLOAT_EQ(c.getData()[3], 9);
}

TEST(TensorArithmeticTest, AddScalarToEmptyTensor)
{
    Tensor empty;

    Tensor c = empty + 5.0f;

    EXPECT_TRUE(c.isEmpty());
    EXPECT_EQ(c.getSize(), 0);
}

//
// Tensor - Tensor
//
TEST(TensorArithmeticTest, SubtractTwoTensors)
{
    Tensor a({5,6,7,8}, {2,2});
    Tensor b({1,2,3,4}, {2,2});

    Tensor c = a - b;

    EXPECT_FLOAT_EQ(c.getData()[0], 4);
    EXPECT_FLOAT_EQ(c.getData()[1], 4);
    EXPECT_FLOAT_EQ(c.getData()[2], 4);
    EXPECT_FLOAT_EQ(c.getData()[3], 4);
}

TEST(TensorArithmeticTest, SubtractShapeMismatch)
{
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({1,2}, {2});

    EXPECT_THROW(a - b, std::invalid_argument);
}

//
// Tensor - Scalar
//
TEST(TensorArithmeticTest, SubtractScalar)
{
    Tensor a({5,6,7,8}, {2,2});

    Tensor c = a - 2.0f;

    EXPECT_FLOAT_EQ(c.getData()[0], 3);
    EXPECT_FLOAT_EQ(c.getData()[1], 4);
    EXPECT_FLOAT_EQ(c.getData()[2], 5);
    EXPECT_FLOAT_EQ(c.getData()[3], 6);
}

//
// Tensor * Tensor
//
TEST(TensorArithmeticTest, MultiplyTwoTensors)
{
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({5,6,7,8}, {2,2});

    Tensor c = a * b;

    EXPECT_FLOAT_EQ(c.getData()[0], 5);
    EXPECT_FLOAT_EQ(c.getData()[1], 12);
    EXPECT_FLOAT_EQ(c.getData()[2], 21);
    EXPECT_FLOAT_EQ(c.getData()[3], 32);
}

TEST(TensorArithmeticTest, MultiplyShapeMismatch)
{
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({1,2}, {2});

    EXPECT_THROW(a * b, std::invalid_argument);
}

//
// Tensor * Scalar
//
TEST(TensorArithmeticTest, MultiplyScalar)
{
    Tensor a({1,2,3,4}, {2,2});

    Tensor c = a * 3.0f;

    EXPECT_FLOAT_EQ(c.getData()[0], 3);
    EXPECT_FLOAT_EQ(c.getData()[1], 6);
    EXPECT_FLOAT_EQ(c.getData()[2], 9);
    EXPECT_FLOAT_EQ(c.getData()[3], 12);
}

//
// Tensor / Tensor
//
TEST(TensorArithmeticTest, DivideTwoTensors)
{
    Tensor a({10,20,30,40}, {2,2});
    Tensor b({2,4,5,8}, {2,2});

    Tensor c = a / b;

    EXPECT_FLOAT_EQ(c.getData()[0], 5);
    EXPECT_FLOAT_EQ(c.getData()[1], 5);
    EXPECT_FLOAT_EQ(c.getData()[2], 6);
    EXPECT_FLOAT_EQ(c.getData()[3], 5);
}

TEST(TensorArithmeticTest, DivideShapeMismatch)
{
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({1,2}, {2});

    EXPECT_THROW(a / b, std::invalid_argument);
}

TEST(TensorArithmeticTest, DivideTensorByZero)
{
    Tensor a({10,20,30,40}, {2,2});
    Tensor b({1,2,0,4}, {2,2});

    EXPECT_THROW(a / b, std::invalid_argument);
}

//
// Tensor / Scalar
//
TEST(TensorArithmeticTest, DivideScalar)
{
    Tensor a({10,20,30,40}, {2,2});

    Tensor c = a / 10.0f;

    EXPECT_FLOAT_EQ(c.getData()[0], 1);
    EXPECT_FLOAT_EQ(c.getData()[1], 2);
    EXPECT_FLOAT_EQ(c.getData()[2], 3);
    EXPECT_FLOAT_EQ(c.getData()[3], 4);
}

TEST(TensorArithmeticTest, DivideByZeroScalar)
{
    Tensor a({1,2,3,4}, {2,2});

    EXPECT_THROW(a / 0.0f, std::invalid_argument);
}
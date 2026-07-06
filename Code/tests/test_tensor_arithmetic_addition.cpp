#include <gtest/gtest.h>
#include "tensor.h"

TEST(TensorArithmeticTest, AddTwoTensors)
{
    Tensor a(
        {1,2,3,4},
        {2,2}
    );

    Tensor b(
        {5,6,7,8},
        {2,2}
    );

    Tensor c = a + b;

    EXPECT_EQ(c.getShape()[0], 2);
    EXPECT_EQ(c.getShape()[1], 2);

    EXPECT_EQ(c.getSize(), 4);

    EXPECT_FLOAT_EQ(c.getData()[0], 6.0f);
    EXPECT_FLOAT_EQ(c.getData()[1], 8.0f);
    EXPECT_FLOAT_EQ(c.getData()[2], 10.0f);
    EXPECT_FLOAT_EQ(c.getData()[3], 12.0f);
}

TEST(TensorArithmeticTest, ShapeMismatch)
{
    Tensor a(
        {1,2,3,4},
        {2,2}
    );

    Tensor b(
        {1,2},
        {2}
    );

    EXPECT_THROW(
        a + b,
        std::invalid_argument
    );
}

TEST(TensorArithmeticTest, AddZeroTensor)
{
    Tensor a(
        {1,2,3,4},
        {2,2}
    );

    Tensor zero(
        {0,0,0,0},
        {2,2}
    );

    Tensor c = a + zero;

    EXPECT_FLOAT_EQ(c.getData()[0], 1.0f);
    EXPECT_FLOAT_EQ(c.getData()[1], 2.0f);
    EXPECT_FLOAT_EQ(c.getData()[2], 3.0f);
    EXPECT_FLOAT_EQ(c.getData()[3], 4.0f);
}

TEST(TensorArithmeticTest, AddScalar)
{
    Tensor a(
        {1,2,3,4},
        {2,2}
    );

    float scalar = 5.0f;

    Tensor c = a + scalar;

    EXPECT_FLOAT_EQ(c.getData()[0], 6.0f);
    EXPECT_FLOAT_EQ(c.getData()[1], 7.0f);
    EXPECT_FLOAT_EQ(c.getData()[2], 8.0f);
    EXPECT_FLOAT_EQ(c.getData()[3], 9.0f);
}
TEST(TensorArithmeticTest, AddScalarToEmptyTensor)
{
    Tensor empty;

    float scalar = 5.0f;

    Tensor c = empty + scalar;

    EXPECT_TRUE(c.isEmpty());
    EXPECT_EQ(c.getSize(), 0);
    EXPECT_EQ(c.getNumDimensions(), 0);
    EXPECT_TRUE(c.getShape().empty());
    EXPECT_TRUE(c.getData().empty());
}
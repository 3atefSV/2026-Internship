#include "tensor.h"
#include <gtest/gtest.h>

TEST(TensorBroadcastTest, Broadcast1DTo2D) {
    Tensor A({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor B({10, 20, 30}, {3});

    Tensor C = A + B;

    EXPECT_EQ(C.shape(), Tensor::Shape({2, 3}));

    EXPECT_FLOAT_EQ(C.data()[0], 11);
    EXPECT_FLOAT_EQ(C.data()[1], 22);
    EXPECT_FLOAT_EQ(C.data()[2], 33);

    EXPECT_FLOAT_EQ(C.data()[3], 14);
    EXPECT_FLOAT_EQ(C.data()[4], 25);
    EXPECT_FLOAT_EQ(C.data()[5], 36);
}

TEST(TensorBroadcastTest, BroadcastColumnVector) {
    Tensor A({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor B({10, 20}, {2, 1});

    Tensor C = A + B;

    EXPECT_EQ(C.shape(), Tensor::Shape({2, 3}));

    EXPECT_FLOAT_EQ(C.data()[0], 11);
    EXPECT_FLOAT_EQ(C.data()[1], 12);
    EXPECT_FLOAT_EQ(C.data()[2], 13);

    EXPECT_FLOAT_EQ(C.data()[3], 24);
    EXPECT_FLOAT_EQ(C.data()[4], 25);
    EXPECT_FLOAT_EQ(C.data()[5], 26);
}

TEST(TensorBroadcastTest, BroadcastScalarLikeTensor) {
    Tensor A({1, 2, 3, 4}, {2, 2});

    Tensor B({5}, {1});

    Tensor C = A + B;

    EXPECT_EQ(C.shape(), Tensor::Shape({2, 2}));

    EXPECT_FLOAT_EQ(C.data()[0], 6);
    EXPECT_FLOAT_EQ(C.data()[1], 7);
    EXPECT_FLOAT_EQ(C.data()[2], 8);
    EXPECT_FLOAT_EQ(C.data()[3], 9);
}

TEST(TensorBroadcastTest, SameShapeStillWorks) {
    Tensor A({1, 2, 3, 4}, {2, 2});

    Tensor B({5, 6, 7, 8}, {2, 2});

    Tensor C = A + B;

    EXPECT_EQ(C.shape(), Tensor::Shape({2, 2}));

    EXPECT_FLOAT_EQ(C.data()[0], 6);
    EXPECT_FLOAT_EQ(C.data()[1], 8);
    EXPECT_FLOAT_EQ(C.data()[2], 10);
    EXPECT_FLOAT_EQ(C.data()[3], 12);
}

TEST(TensorBroadcastTest, InvalidBroadcastThrows) {
    Tensor A({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor B({1, 2, 3, 4}, {2, 2});

    EXPECT_THROW(A + B, std::invalid_argument);
}

TEST(TensorBroadcastTest, BroadcastMultiply) {
    Tensor A({1, 2, 3, 4, 5, 6}, {2, 3});

    Tensor B({10, 20, 30}, {3});

    Tensor C = A * B;

    EXPECT_FLOAT_EQ(C.data()[0], 10);
    EXPECT_FLOAT_EQ(C.data()[1], 40);
    EXPECT_FLOAT_EQ(C.data()[2], 90);

    EXPECT_FLOAT_EQ(C.data()[3], 40);
    EXPECT_FLOAT_EQ(C.data()[4], 100);
    EXPECT_FLOAT_EQ(C.data()[5], 180);
}

TEST(TensorBroadcastTest, BroadcastSubtract) {
    Tensor A({10, 20, 30, 40, 50, 60}, {2, 3});

    Tensor B({1, 2, 3}, {3});

    Tensor C = A - B;

    EXPECT_FLOAT_EQ(C.data()[0], 9);
    EXPECT_FLOAT_EQ(C.data()[1], 18);
    EXPECT_FLOAT_EQ(C.data()[2], 27);

    EXPECT_FLOAT_EQ(C.data()[3], 39);
    EXPECT_FLOAT_EQ(C.data()[4], 48);
    EXPECT_FLOAT_EQ(C.data()[5], 57);
}

TEST(TensorBroadcastTest, BroadcastDivide) {
    Tensor A({10, 20, 30, 40, 50, 60}, {2, 3});

    Tensor B({10, 10, 10}, {3});

    Tensor C = A / B;

    EXPECT_FLOAT_EQ(C.data()[0], 1);
    EXPECT_FLOAT_EQ(C.data()[1], 2);
    EXPECT_FLOAT_EQ(C.data()[2], 3);

    EXPECT_FLOAT_EQ(C.data()[3], 4);
    EXPECT_FLOAT_EQ(C.data()[4], 5);
    EXPECT_FLOAT_EQ(C.data()[5], 6);
}
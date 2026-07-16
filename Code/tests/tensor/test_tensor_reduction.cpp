#include "tensor/tensor.h"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-5f)
            return false;
    }
    return true;
}

TEST(TensorOpsTest, ReductionSumAll) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t(data, {2, 2});

    float total_sum = t.sum();

    EXPECT_FLOAT_EQ(total_sum, 10.0f);
}

TEST(TensorOpsTest, ReductionSumAxis0) {

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t(data, {2, 3});

    Tensor sum_axis0 = t.sum(0);

    std::vector<float> expected = {5.0f, 7.0f, 9.0f};

    EXPECT_EQ(sum_axis0.ndim(), 1);
    EXPECT_EQ(sum_axis0.shape()[0], 3);
    EXPECT_TRUE(compare_vectors(sum_axis0.data(), expected));
}

TEST(TensorOpsTest, ReductionMeanAxis1) {

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t(data, {2, 3});

    Tensor mean_axis1 = t.mean(1);

    std::vector<float> expected = {2.0f, 5.0f};

    EXPECT_EQ(mean_axis1.ndim(), 1);
    EXPECT_EQ(mean_axis1.shape()[0], 2);
    EXPECT_TRUE(compare_vectors(mean_axis1.data(), expected));
}
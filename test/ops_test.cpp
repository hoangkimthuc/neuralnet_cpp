#include "include/ops.h"
#include <gtest/gtest.h>

TEST(Linear_init, Initialization) {
    // Set the weights, bias
    Linear linear(2, 2);
    EXPECT_EQ(linear.weights.numel(), 4);
    EXPECT_EQ(linear.bias.numel(), 2);
    EXPECT_EQ(linear.grad_weights.numel(), 4);
    EXPECT_EQ(linear.grad_bias.numel(), 2);
    EXPECT_EQ(linear.grad_fn, "Linear_backward");
}

TEST(Linear_forward, shape_checking_fail)
{
    // Set the weights, bias
    Linear linear(2, 2);
    Tensor input(std::vector<int>{1,3}, std::vector<float>{1.0, 1.0, 1.0}, false);
    EXPECT_THROW(linear.forward(input), std::invalid_argument);
}

TEST(Linear_forward, shape_checking_pass)
{
    Linear linear(2, 2);
    Tensor input(std::vector<int>{2,2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}, false);
    Tensor output = linear.forward(input);
    std::vector<int> expected_shape = {2, 2};
    EXPECT_EQ(output.shape, expected_shape);
}
    
#include "include/ops.h"
#include <gtest/gtest.h>

TEST(Linear_forward, Initialization) {
    // Set the weights, bias
    Linear linear(2, 2);
    EXPECT_EQ(linear.weights.numel(), 4);
    EXPECT_EQ(linear.bias.numel(), 2);
    EXPECT_EQ(linear.grad_weights.numel(), 4);
    EXPECT_EQ(linear.grad_bias.numel(), 2);
    EXPECT_EQ(linear.grad_fn, "Linear_backward");
}

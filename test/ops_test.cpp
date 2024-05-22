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

TEST(Linear_forward, output_values)
{
    Linear linear(2, 2);
    linear.weights.data = std::vector<float>{1, 2, 3, 4};
    linear.bias.data = std::vector<float>{1, 1};
    Tensor input(std::vector<int>{2,2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}, false);
    Tensor output = linear.forward(input);
    std::vector<float> expected_output = std::vector<float>{5, 7, 5, 7};
    EXPECT_EQ(output.data, expected_output);
}

TEST(Linear_backward, grad_values)
{
    Linear linear(2, 3);
    Tensor input(std::vector<int>{3,2}, std::vector<float>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, true);
    Tensor linear_weights(std::vector<int>{2,3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    linear.weights = linear_weights;
    Tensor linear_bias(std::vector<int>{3}, std::vector<float>{1.0, 1.0, 1.0}, true);
    linear.bias = linear_bias;
    Tensor output = linear.forward(input);
    Tensor grad_output(std::vector<int>{3,3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, false);
    linear.backward(grad_output);
    std::vector<float> expected_weights_grad_data = std::vector<float>{120.0, 147.0, 174.0, 132.0, 162.0, 192.0};
    std::vector<int> expected_weights_grad_shape = std::vector<int>{2, 3};
    Tensor expected_grad_weights(expected_weights_grad_shape, expected_weights_grad_data, false);
    EXPECT_EQ(linear.grad_weights, expected_grad_weights);
    EXPECT_EQ(linear.grad_bias, grad_output);

    std::vector<float> expected_input_grad_data = std::vector<float>{14, 32, 32, 77, 50, 122};
    std::vector<int> expected_input_grad_shape = std::vector<int>{3, 2};
    Tensor expected_input_grad(expected_input_grad_shape, expected_input_grad_data, false);
    EXPECT_EQ(linear.grad_input, expected_input_grad);
    
}

TEST(Sum_forward, forward_pass)
{
    Sum sum;
    Tensor input(std::vector<int>{2,2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}, false);
    Tensor output = sum.forward(input);
    Tensor expected_output(std::vector<int>{1}, std::vector<float>{4}, false);
    EXPECT_EQ(output, expected_output);
}

TEST(Sum_backward, backward_pass)
{
    Sum sum;
    Tensor input(std::vector<int>{2,2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}, true);
    Tensor grad_output(std::vector<int>{1}, std::vector<float>{1.0}, false);
    sum.forward(input);
    sum.backward(grad_output);
    Tensor expected_grad_input(std::vector<int>{2,2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}, false);
    EXPECT_EQ(sum.grad_input, expected_grad_input);
}
#include "include/tensor.h"
#include "include/ops.h"
#include "include/autograd.h"

#include <vector>
#include <gtest/gtest.h>
#include <memory>
#include <iostream>

TEST(ComputeGraph, backward_sum) {
    ComputeGraph graph;
    Sum sum;
    Tensor input(std::vector<int>{2,2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}, true);
    Tensor output = sum.forward(input);
    // std::cout << sum.grad_fn << std::endl;
    graph.add(&sum);
    graph.backward();
    std::vector<float> expected_grad = std::vector<float>{1.0, 1.0, 1.0, 1.0};
    EXPECT_EQ(sum.input.grad, expected_grad);
}

TEST(ComputeGraph, backward_linear_sum) {
    ComputeGraph graph;
    Linear linear(2, 3);
    Tensor input(std::vector<int>{3,2}, std::vector<float>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, true);
    Tensor linear_weights(std::vector<int>{2,3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    linear.weights = linear_weights;
    Tensor linear_bias(std::vector<int>{3}, std::vector<float>{1.0, 1.0, 1.0}, true);
    linear.bias = linear_bias;
    Tensor output = linear.forward(input);
    for (int i = 0; i < output.numel(); i++) {
        std::cout << output.data[i] << std::endl;
    }
    graph.add(&linear);

    Sum sum;
    Tensor loss = sum.forward(output);
    std::cout << loss.data[0] << std::endl;
    graph.add(&sum);
    
    graph.backward();    
    std::vector<float> expected_weights_grad_data = std::vector<float>{27, 27, 27, 30, 30, 30};
    EXPECT_EQ(linear.weights.grad, expected_weights_grad_data);

    std::vector<float> expected_bias_grad_data = std::vector<float>{3, 3, 3};
    EXPECT_EQ(linear.bias.grad, expected_bias_grad_data);
    
    std::vector<float> expected_input_grad_data = std::vector<float>{6, 15, 6, 15, 6, 15};
    EXPECT_EQ(linear.input.grad, expected_input_grad_data);
}
#include "neuralnet.h"
#include <gtest/gtest.h>

TEST(Tensor, non_zero_initialization) {
    std::initializer_list<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    bool require_grad = true;
    Tensor tensor(shape, data, require_grad);
    EXPECT_EQ(tensor.shape.size(), 3);
    EXPECT_EQ(tensor.shape[0], 3);
    EXPECT_EQ(tensor.shape[1], 2);
    EXPECT_EQ(tensor.shape[2], 4);
    EXPECT_EQ(tensor.data.size(), 24);
    EXPECT_FLOAT_EQ(tensor.data[0], 1);
    EXPECT_FLOAT_EQ(tensor.data[23], 24);
    EXPECT_EQ(tensor.require_grad, true);
}

TEST(Tensor, zero_initialization) {
    std::initializer_list<int> shape = {3, 2, 4};
    std::vector<float> data = {};
    bool require_grad = true;
    Tensor tensor(shape, data, require_grad);
    EXPECT_EQ(tensor.shape.size(), 3);
    EXPECT_EQ(tensor.shape[0], 3);
    EXPECT_EQ(tensor.shape[1], 2);
    EXPECT_EQ(tensor.shape[2], 4);
    EXPECT_EQ(tensor.data.size(), 24);
    EXPECT_FLOAT_EQ(tensor.data[0], 0);
    EXPECT_FLOAT_EQ(tensor.data[23], 0);
    EXPECT_EQ(tensor.require_grad, true);
}

TEST(Tensor, invalid_data_size) {
    std::initializer_list<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    bool require_grad = true;
    EXPECT_THROW(Tensor tensor(shape, data, require_grad), std::invalid_argument);
}

TEST(Linear, forward) {
    // Set the weights and bias
    Linear linear(2, 2);
    linear.weights = {{1, 2}, {3, 4}};
    linear.bias = {1, 1};
    std::vector<std::vector<float>> x = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> y = linear.forward(x);
    EXPECT_EQ(y.size(), 2);
    EXPECT_EQ(y[0].size(), 2);
    EXPECT_EQ(y[1].size(), 2);
    EXPECT_FLOAT_EQ(y[0][0], 8);
    EXPECT_FLOAT_EQ(y[0][1], 11);
    EXPECT_FLOAT_EQ(y[1][0], 16);
    EXPECT_FLOAT_EQ(y[1][1], 23);
}

// TEST(Linear, backward) {
//     // Set the weights and bias
//     Linear linear(2, 2);
//     linear.weights = {{1, 2}, {3, 4}};
//     linear.bias = {1, 1};
//     std::vector<std::vector<float>> x = {{1, 2}, {3, 4}};
//     std::vector<std::vector<float>> y = linear.forward(x);
//     std::vector<std::vector<float>> dy = {{1, 1}, {1, 1}};
//     std::vector<std::vector<float>> dx = linear.backward(dy);
//     EXPECT_EQ(dx.size(), 2);
//     EXPECT_EQ(dx[0].size(), 2);
//     EXPECT_EQ(dx[1].size(), 2);
//     EXPECT_FLOAT_EQ(dx[0][0], 3);
//     EXPECT_FLOAT_EQ(dx[0][1], 7);
//     EXPECT_FLOAT_EQ(dx[1][0], 3);
//     EXPECT_FLOAT_EQ(dx[1][1], 7);
//     EXPECT_EQ(linear.grad_weights.size(), 2);
//     EXPECT_EQ(linear.grad_weights[0].size(), 2);
//     EXPECT_EQ(linear.grad_weights[1].size(), 2);
//     EXPECT_FLOAT_EQ(linear.grad_weights[0][0], 4);
//     EXPECT_FLOAT_EQ(linear.grad_weights[0][1], 6);
//     EXPECT_FLOAT_EQ(linear.grad_weights[1][0], 4);
//     EXPECT_FLOAT_EQ(linear.grad_weights[1][1], 6);
//     EXPECT_EQ(linear.grad_bias.size(), 2);
//     EXPECT_FLOAT_EQ(linear.grad_bias[0], 2);
//     EXPECT_FLOAT_EQ(linear.grad_bias[1], 2);
// }
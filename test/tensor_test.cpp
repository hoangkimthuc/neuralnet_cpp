#include "include/tensor.h"
#include <gtest/gtest.h>

TEST(Tensor_Init, non_zero_initialization) {
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

TEST(Tensor_Init, zero_initialization) {
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

TEST(Tensor_Init, invalid_data_size) {
    std::initializer_list<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    bool require_grad = true;
    EXPECT_THROW(Tensor tensor(shape, data, require_grad), std::invalid_argument);
}

TEST(Tensor_number_of_element, numel) {
    std::initializer_list<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    bool require_grad = true;
    Tensor tensor(shape, data, require_grad);
    EXPECT_EQ(tensor.numel(), 24);
}

TEST(Tensor_equality, equal) {
    std::initializer_list<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    bool require_grad = true;
    Tensor tensor1(shape, data, require_grad);
    Tensor tensor2(shape, data, require_grad);
    EXPECT_TRUE(tensor1==tensor2);
}

TEST(Tensor_equality, not_equal) {
    std::initializer_list<int> shape = {3, 2, 4};
    std::vector<float> data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    std::vector<float> data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25};
    bool require_grad = true;
    Tensor tensor1(shape, data1, require_grad);
    Tensor tensor2(shape, data2, require_grad);
    EXPECT_TRUE(tensor1!=tensor2);
}

// TEST(Tensor_indexing, indexing) {
//     std::initializer_list<int> shape = {3, 2, 4};
//     std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
//                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
//     Tensor tensor(shape, data);
//     // first element in the tensor
//     std::initializer_list<int> shape_first_elem = {0};
//     std::vector<float> data_first_elem = {1};
//     EXPECT_EQ(tensor[0][0][0], Tensor(shape_first_elem, data_first_elem));
//     // last element in the tensor
//     std::initializer_list<int> shape_last_elem = {0};
//     std::vector<float> data_last_elem = {24};
//     EXPECT_EQ(tensor[2][1][3], Tensor(shape_last_elem, data_last_elem));
//     // First matrix in the tensor
//     std::initializer_list<int> shape_first_matrix = {2, 4};
//     std::vector<float> data_first_matrix = {1, 2, 3, 4, 5, 6, 7, 8};
//     EXPECT_EQ(tensor[0], Tensor(shape_first_matrix, data_first_matrix));
//     // Last matrix in the tensor
//     std::initializer_list<int> shape_last_matrix = {2, 4};
//     std::vector<float> data_last_matrix = {17, 18, 19, 20, 21, 22, 23, 24};
//     EXPECT_EQ(tensor[2], Tensor(shape_last_matrix, data_last_matrix));
//     // First row in the first matrix
//     std::initializer_list<int> shape_first_row = {4};
//     std::vector<float> data_first_row = {1, 2, 3, 4};
//     EXPECT_EQ(tensor[0][0], Tensor(shape_first_row, data_first_row));
//     // Last row in the last matrix
//     std::initializer_list<int> shape_last_row = {4};
//     std::vector<float> data_last_row = {21, 22, 23, 24};
// }

// TEST(Linear_forward, batch) {
//     // Set the weights, bias
//     Linear linear(2, 2);
//     std::initializer_list<int> shape_w = {2, 2};
//     std::vector<float> data_w = {1, 2, 3, 4};
//     linear.weights = Tensor(shape_w, data_w);

//     // Set input x
//     Tensor x = Tensor(shape_w, data_w);

//     std::initializer_list<int> shape_b = {2};
//     std::vector<float> data_b = {1, 1};
//     linear.bias = Tensor(shape_b, data_b);
//     Tensor y = linear.forward(x);
//     EXPECT_EQ(y.data.size(), 2);
//     EXPECT_EQ(y.shape[0], 2);
//     EXPECT_EQ(y.shape[1], 2);
//     EXPECT_FLOAT_EQ(y.data[0], 9);
//     EXPECT_FLOAT_EQ(y.data[1], 12);
//     EXPECT_FLOAT_EQ(y.data[2], 19);
//     EXPECT_FLOAT_EQ(y.data[3], 26);
// }

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
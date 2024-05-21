#include "include/tensor.h"
#include <gtest/gtest.h>

TEST(Tensor_Init, non_zero_initialization) {
    std::vector<int> shape = {3, 2, 4};
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
    std::vector<int> shape = {3, 2, 4};
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
    std::vector<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    bool require_grad = true;
    EXPECT_THROW(Tensor tensor(shape, data, require_grad), std::invalid_argument);
}

TEST(Tensor_Init, zero_shape) {
    std::vector<int> shape = {};
    std::vector<float> data = {1};
    bool require_grad = true;
    Tensor tensor(shape, data, require_grad);
    EXPECT_EQ(tensor.numel(), 1);
}

TEST(Tensor_Init, non_zero_shape) {
    std::vector<int> shape = {1, 2};
    std::vector<float> data = {10, 20};
    bool require_grad = true;
    Tensor tensor(shape, data, require_grad);
    EXPECT_EQ(tensor.numel(), 2);
}

TEST(Tensor_number_of_element, numel) {
    std::vector<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    bool require_grad = true;
    Tensor tensor(shape, data, require_grad);
    EXPECT_EQ(tensor.numel(), 24);
}

TEST(Tensor_equality, equal) {
    std::vector<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    bool require_grad = true;
    Tensor tensor1(shape, data, require_grad);
    Tensor tensor2(shape, data, require_grad);
    EXPECT_TRUE(tensor1==tensor2);
}

TEST(Tensor_equality, not_equal) {
    std::vector<int> shape = {3, 2, 4};
    std::vector<float> data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    std::vector<float> data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25};
    bool require_grad = true;
    Tensor tensor1(shape, data1, require_grad);
    Tensor tensor2(shape, data2, require_grad);
    EXPECT_TRUE(tensor1!=tensor2);
}

TEST(Tensor_indexing, zero_dim) {
    std::vector<int> shape = {};
    std::vector<float> data = {1};
    Tensor tensor(shape, data);
    EXPECT_THROW(tensor[0], std::invalid_argument);
}

TEST(Tensor_indexing, indexing) {
    std::vector<int> shape = {3, 2, 4};
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    Tensor tensor(shape, data);
    // first element in the tensor
    std::vector<int> shape_first_elem = {};
    std::vector<float> data_first_elem = {1};
    EXPECT_EQ(tensor[0][0][0], Tensor(shape_first_elem, data_first_elem));
    // last element in the tensor
    std::vector<int> shape_last_elem = {};
    std::vector<float> data_last_elem = {24};
    EXPECT_EQ(tensor[2][1][3], Tensor(shape_last_elem, data_last_elem));
    // First matrix in the tensor
    std::vector<int> shape_first_matrix = {2, 4};
    std::vector<float> data_first_matrix = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(tensor[0], Tensor(shape_first_matrix, data_first_matrix));
    // Last matrix in the tensor
    std::vector<int> shape_last_matrix = {2, 4};
    std::vector<float> data_last_matrix = {17, 18, 19, 20, 21, 22, 23, 24};
    EXPECT_EQ(tensor[2], Tensor(shape_last_matrix, data_last_matrix));
    // First row in the first matrix
    std::vector<int> shape_first_row = {4};
    std::vector<float> data_first_row = {1, 2, 3, 4};
    EXPECT_EQ(tensor[0][0], Tensor(shape_first_row, data_first_row));
    // Last row in the last matrix
    std::vector<int> shape_last_row = {4};
    std::vector<float> data_last_row = {21, 22, 23, 24};
}

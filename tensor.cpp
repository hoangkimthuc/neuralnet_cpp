#include "include/tensor.h"

Tensor::Tensor() {
    shape = std::vector<int>();
    data = std::vector<float>();
    require_grad = false;
}

Tensor::Tensor(std::vector<int> shape, std::vector<float> data, bool require_grad) 
: shape(shape), require_grad(require_grad) {
    // Calculate the total size of the tensor
    int size = 1;
    // If the shape is empty, size is 1
    if (shape.size()==0) {
        size = 1;
    }
    else {
        for (int dim : shape) {
            size *= dim;
        }
    }
    // If the data is not provided, initialize it with zeros
    if (data.size() == 0) {
        this->data = std::vector<float>(size, 0);
    }
    // If the data is provided, check if the size matches the tensor size
    else if (data.size() != size) {
        throw std::invalid_argument("Data size does not match the tensor size. Data size: " + std::to_string(data.size()) + ", Tensor size: " + std::to_string(size));
    }
    else {
        this->data = data;
    }    
}

int Tensor::numel() const {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    return size;
}

bool operator==(Tensor tensor1, Tensor tensor2) {
    if (tensor1.shape != tensor2.shape) {
        return false;
    }
    if (tensor1.data != tensor2.data) {
        return false;
    }
    return true;
}

bool operator!=(Tensor tensor1, Tensor tensor2) {
    if (tensor1.shape != tensor2.shape) {
        return true;
    }
    if (tensor1.data != tensor2.data) {
        return true;
    }
    return false;
}

Tensor Tensor::operator[](int index) {
    // Calculate the new shape of the return tensor
    if (shape.size() == 0) {
        throw std::invalid_argument("Cannot index a zero-dim tensor");
    }
    // New shape is the last n-1 dimensions of the current shape
    std::vector<int> new_shape(shape.begin() + 1, shape.end());
    // Calculate the size of the new tensor
    int new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    // Calculate the start and end index of the data
    int start = index * new_size;
    int end = start + new_size;
    
    return Tensor(new_shape, std::vector<float>(data.begin() + start, data.begin() + end), require_grad);
}
#include "neuralnet.h"

Tensor::Tensor(std::initializer_list<int> shape, std::vector<float> data, bool require_grad) {
    this->shape.assign(shape.begin(), shape.end());
    this->require_grad = require_grad;
    // Calculate the total size of the tensor
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    // If the data is not provided, initialize it with zeros
    if (data.size() == 0) {
        this->data = std::vector<float>(size, 0);
    }
    // If the data is provided, check if the size matches the tensor size
    else if (data.size() != size) {
        throw std::invalid_argument("Data size does not match the tensor size");
    }
    else {
        this->data = data;
    }    
}


// Initialize the Linear Layer
Linear::Linear(int in_features, int out_features) {
    weights = std::vector<std::vector<float>>(in_features, std::vector<float>(out_features, 0));
    bias = std::vector<float>(out_features, 0);
    grad_weights = std::vector<std::vector<float>>(in_features, std::vector<float>(out_features, 0));
    grad_bias = std::vector<float>(out_features, 0);
}

// Implement the forward pass of the Linear layer
std::vector<std::vector<float>> Linear::forward(std::vector<std::vector<float>> x) {
    input = x;
    std::vector<std::vector<float>> y;
    for (int i = 0; i < x.size(); i++) {
        std::vector<float> y_i;
        for (int j = 0; j < weights[0].size(); j++) {
            float y_ij = 0;
            for (int k = 0; k < x[0].size(); k++) {
                y_ij += x[i][k] * weights[k][j];
            }
            y_ij += bias[j];
            y_i.push_back(y_ij);
        }
        y.push_back(y_i);
    }
    return y;
}

// Implement the backward pass of the Linear layer
std::vector<std::vector<float>> Linear::backward(std::vector<std::vector<float>> grad_output) {
    int batch_size = grad_output.size();
    int out_features = grad_output[0].size();
    int in_features = input[0].size();
    
    // Compute gradients w.r.t. weights and biases
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < out_features; j++) {
            grad_weights[i][j] = 0;
            for (int b = 0; b < batch_size; b++) {
                grad_weights[i][j] += input[b][i] * grad_output[b][j];
            }
            grad_weights[i][j] /= batch_size;
        }
    }
    
    for (int j = 0; j < out_features; j++) {
        grad_bias[j] = 0;
        for (int b = 0; b < batch_size; b++) {
            grad_bias[j] += grad_output[b][j];
        }
        grad_bias[j] /= batch_size;
    }
    
    // Compute gradients w.r.t. input
    std::vector<std::vector<float>> grad_input;
    for (int b = 0; b < batch_size; b++) {
        std::vector<float> grad_input_b;
        for (int i = 0; i < in_features; i++) {
            float sum = 0;
            for (int j = 0; j < out_features; j++) {
                sum += grad_output[b][j] * weights[i][j];
            }
            grad_input_b.push_back(sum);
        }
        grad_input.push_back(grad_input_b);
    }
    
    return grad_input;
}
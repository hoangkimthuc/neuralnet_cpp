#include "include/ops.h"
#include <random>
#include <iostream>

// Initialize the Linear Layer
Linear::Linear(int in_features, int out_features)
    {
    std::vector<int> shape_w = {in_features, out_features};
    std::vector<float> data_w;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1);
    // Initialize weights and biases with random values from a normal distribution with mean 0 and standard deviation 1
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < out_features; j++) {
            data_w.push_back(distribution(generator));
        }
    }
    weights = Tensor(shape_w, data_w, true);

    std::vector<int> shape_b = {out_features};
    std::vector<float> data_b;
    for (int i = 0; i < out_features; i++) {        
        data_b.push_back(distribution(generator));
    }
    bias = Tensor(shape_b, data_b, true);

    // Initialize the gradients with zeros
    weights.grad = std::vector<float>(data_w.size(), 0);
    bias.grad = std::vector<float>(data_b.size(), 0);

    // Fix the grad_fn name
    grad_fn = "Linear_backward";
}

// Implement the forward pass of the Linear layer
Tensor Linear::forward(const Tensor& x) {
    // Set the input tensor when the forward pass is called for later grad computation
    this->input = x;

    int batch_size = x.shape[0];
    int in_features = x.shape[1];
    int out_features = weights.shape[1];
    
    // First check if the shape is correct
    if (in_features != weights.shape[0]) {
        throw std::invalid_argument("The number of input features should be equal to the number of weights. Got in features:" + std::to_string(in_features) + " and number of weights: " + std::to_string(weights.shape[0]) + " instead.");
    }
    bool output_require_grad = x.require_grad || weights.require_grad || bias.require_grad;
    // Initialize the output tensor
    std::vector<int> output_shape = {batch_size, out_features};
    std::vector<float> output_data(batch_size * out_features, 0);
    Tensor output(output_shape, output_data, output_require_grad);
    
    // Perform the matrix multiplication
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_features; o++) {
            for (int i = 0; i < in_features; i++) {
                output.data[b * out_features + o] += x.data[b * in_features + i] * weights.data[i * out_features + o];
            }
            output.data[b * out_features + o] += bias.data[o];
        }
    }
    
    return output;
}

// Implement the backward pass of the Linear layer
void Linear::backward(const Tensor& grad_output) {
    // Check the shape of grad_output
    std::cout << "grad_output shape: " << grad_output.shape[0] << " " << grad_output.shape[1] << std::endl;
    if (grad_output.shape[1] != weights.shape[1]) {
        throw std::invalid_argument("The number of output features should be equal to the number of weights. Got out features:" + std::to_string(grad_output.shape[1]) + " and number of weights: " + std::to_string(weights.shape[1]) + " instead.");
    }
    if (grad_output.shape[0] != input.shape[0]) {
        throw std::invalid_argument("The batch size of the input and the gradient of the output should be the same. Got batch size of input:" + std::to_string(input.shape[0]) + " and batch size of grad_output: " + std::to_string(grad_output.shape[0]) + " instead.");
    }
    
    // Compute gradients with respect to weights and biases
    int batch_size = grad_output.shape[0];
    int in_features = weights.shape[0];
    int out_features = weights.shape[1];

    // Initialize the gradients with zeros    
    input.grad = std::vector<float>(input.numel(), 0);
    weights.grad = std::vector<float>(weights.numel(), 0);
    bias.grad = std::vector<float>(bias.numel(), 0);
    
    // Compute the gradients for the input if require_grad is set to true
    if (input.require_grad) {
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < in_features; i++) {
                for (int o = 0; o < out_features; o++) {
                    input.grad[b * in_features + i] += grad_output.data[b * out_features + o] * weights.data[i * out_features + o];
                }
            }
        }
    }
    // Compute the gradients for weights if require_grad is set to true
    if (weights.require_grad) {
        for (int i = 0; i < in_features; i++) {
            for (int o = 0; o < out_features; o++) {
                for (int b = 0; b < batch_size; b++) {
                    weights.grad[i * out_features + o] += input.data[b * in_features + i] * grad_output.data[b * out_features + o];
                }
            }
        }
    }
    // Compute the gradients for bias if require_grad is set to true
    if (bias.require_grad) {
        bias.grad = grad_output.data;
    }
}

// Implement the forward pass of the Sum layer
Tensor Sum::forward(const Tensor& x) {
    // Set the input tensor when the forward pass is called for later grad computation
    this->input = x;
    bool output_require_grad = x.require_grad;

    // Initialize the output tensor
    std::vector<int> output_shape = {1};
    std::vector<float> output_data(1, 0);
    Tensor output(output_shape, output_data, output_require_grad);
    
    // Perform the sum operation
    for (int i = 0; i < x.numel(); i++) {
        output.data[0] += x.data[i];
    }
    // Set the grad_fn name
    this->grad_fn = "Sum_backward";
    return output;
}

// Implement the backward pass of the Sum layer
void Sum::backward(const Tensor& grad_output) {
    // Check if the shape of the grad_output is correct
    if (grad_output.numel() != 1) {
        throw std::invalid_argument("The gradient of the output should be a scalar. Got shape:" + std::to_string(grad_output.numel()) + " instead.");
    }
    
    // Initialize the gradients with zeros
    input.grad = std::vector<float>(input.numel(), 0);
    // Compute the gradients for the input if require_grad is set to true
    if (input.require_grad) {
        input.grad = std::vector<float>(input.numel(), grad_output.data[0]);
    }
}
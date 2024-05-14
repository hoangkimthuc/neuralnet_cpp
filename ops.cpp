#include "include/ops.h"
#include <random>

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
    grad_weights = Tensor(shape_w, std::vector<float>(data_w.size(), 0), false);
    grad_bias = Tensor(shape_b, std::vector<float>(data_b.size(), 0), false);

    // Fix the grad_fn name
    grad_fn = "Linear_backward";
}

// Implement the forward pass of the Linear layer
Tensor Linear::forward(const Tensor& x) {
    int batch_size = x.shape[0];
    int in_features = x.shape[1];
    int out_features = bias.numel();
    
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
        for (int j = 0; j < out_features; j++) {
            for (int i = 0; i < in_features; i++) {
                output.data[b * out_features + j] += x.data[b * in_features + i] * weights.data[i * out_features + j];
            }
            output.data[b * out_features + j] += bias.data[j];
        }
    }
    
    return output;
}


#include "include/ops.h"
#include <random>

// Initialize the Linear Layer
Linear::Linear(int in_features, int out_features) {
    std::initializer_list<int> shape_w = {in_features, out_features};
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

    std::initializer_list<int> shape_b = {out_features};
    std::vector<float> data_b;
    for (int i = 0; i < out_features; i++) {        
        data_w.push_back(distribution(generator));
    }
    bias = Tensor(shape_b, data_b, true);

    // Initialize the gradients with zeros
    grad_weights = Tensor(shape_w, std::vector<float>(shape_w.size(), 0), false);
    grad_bias = Tensor(shape_b, std::vector<float>(shape_b.size(), 0), false);

    // Fix the grad_fn name
    grad_fn = "Linear_backward";
}
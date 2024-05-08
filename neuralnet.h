#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <initializer_list>

// Define the neural Linear layer
class Linear {
public:
    Linear(int in_features, int out_features);
    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x);
    std::vector<std::vector<float>> backward(std::vector<std::vector<float>> grad);
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    std::vector<std::vector<float>> input;
    std::vector<std::vector<float>> grad_weights;
    std::vector<float> grad_bias;
};

class Tensor {
public:
    Tensor(std::initializer_list<int> shape, std::vector<float> data = std::vector<float>(), bool require_grad = false);
    
    std::vector<float> data;
    std::vector<int> shape;
    bool require_grad;
    char* grad_fn;
};
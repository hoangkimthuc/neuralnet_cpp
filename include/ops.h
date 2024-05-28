#pragma once

#include "tensor.h"
#include <string>

// Define the neural Linear layer
class Linear {
public:
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output);

    Tensor input;
    Tensor weights;
    Tensor bias;
    std::string grad_fn;
    
};

class Sum {
public:
    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output);
    Tensor input;
    std::string grad_fn;
};
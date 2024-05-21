#pragma once

#include "tensor.h"
// Define the neural Linear layer
class Linear {
public:
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output);

    Tensor input;
    Tensor weights;
    Tensor bias;
    Tensor grad_input;
    Tensor grad_weights;
    Tensor grad_bias;
    std::string grad_fn;
    
};

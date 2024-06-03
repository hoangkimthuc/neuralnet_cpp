#pragma once

#include "tensor.h"
#include <string>
#include <memory>

// Base class for all operations
class Operation {
public:
    virtual Tensor forward() = 0;
    virtual void backward(const Tensor& grad_output) = 0;
    virtual ~Operation() = default;
};

// Define the neural Linear layer
class Linear : public Operation {
public:
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output) override;

    Tensor input;
    Tensor weights;
    Tensor bias;
    std::string grad_fn;

    Tensor forward() override {
        return forward(input);
    } 
    
};

class Sum : public Operation{
public:
    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output) override;
    Tensor input;
    std::string grad_fn;

    Tensor forward() override {
        return forward(input);
    }
};

class ReLU : public Operation{
public:
    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output) override;
    Tensor input;
    std::string grad_fn;

    Tensor forward() override {
        return forward(input);
    }
};

class CrossEntropy : public Operation{
public:
    Tensor forward(const Tensor& input, const Tensor& target);
    void backward(const Tensor& grad_output) override;
    Tensor input;
    Tensor target;
    std::string grad_fn;

    Tensor forward() override {
        return forward(input, target);
    }
};
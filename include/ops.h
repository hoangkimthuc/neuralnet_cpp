
#include "tensor.h"
// Define the neural Linear layer
class Linear {
public:
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    Tensor weights;
    Tensor bias;
    Tensor grad_weights;
    Tensor grad_bias;
    std::string grad_fn;
    
};

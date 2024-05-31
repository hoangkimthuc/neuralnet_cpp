#include "include/autograd.h"
#include "include/tensor.h"
#include "include/ops.h"

#include <vector>
#include <memory>
#include <string>
#include <iostream>

// Implement the default constructor for the GraphNode class
void ComputeGraph::add(Operation* operation) {
    operations.push_back(operation);
}

void ComputeGraph::backward() {
    // Set the gradient of the last operation to 1
    Tensor output_grad(std::vector<int>{1}, std::vector<float>{1.0}, false);
    // Traverse the graph in reverse order to perform backpropagation
    for (int i = operations.size() - 1; i >= 0; i--) {
        if (auto* sum = dynamic_cast<Sum*>(operations[i])) {
            std::cout << "Operation index: " << i << std::endl;
            std::cout << "Operation grad_fn: " << sum->grad_fn << std::endl;
            sum->backward(output_grad);
            output_grad = Tensor(sum->input.shape, sum->input.grad, false);
        }
        else if (auto* linear = dynamic_cast<Linear*>(operations[i])) {
            std::cout << "Operation index: " << i << std::endl;
            std::cout << "Operation grad_fn: " << linear->grad_fn << std::endl;
            linear->backward(output_grad);
            output_grad = Tensor(linear->input.shape, linear->input.grad, false);
        }
    }   
}

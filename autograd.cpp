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
    // Print the number of operations in the graph
    std::cout << "Number of operations in the graph: " << operations.size() << std::endl;
    // Set the gradient of the last operation to 1
    Tensor output_grad(std::vector<int>{1}, std::vector<float>{1.0}, false);
    // Traverse the graph in reverse order to perform backpropagation
    for (int i = operations.size() - 1; i >= 0; i--) {
        if (auto* sum = dynamic_cast<Sum*>(operations[i])) {
            sum->backward(output_grad);
            output_grad = Tensor(sum->input.shape, sum->input.grad, false);
        }
        else if (auto* linear = dynamic_cast<Linear*>(operations[i])) {
            linear->backward(output_grad);
            output_grad = Tensor(linear->input.shape, linear->input.grad, false);
        }
        else if (auto* relu = dynamic_cast<ReLU*>(operations[i])) {
            relu->backward(output_grad);
            output_grad = Tensor(relu->input.shape, relu->input.grad, false);
        }
        else if (auto* cross_entropy = dynamic_cast<CrossEntropy*>(operations[i])) {
            cross_entropy->backward(output_grad);
            output_grad = Tensor(cross_entropy->input.shape, cross_entropy->input.grad, false);
        }
        else {
            throw std::invalid_argument("Operation not supported");
        }
    }   
}

void ComputeGraph::sgd_step(float lr) {
    for (int i = 0; i < operations.size(); i++) {
        if (auto* linear = dynamic_cast<Linear*>(operations[i])) {
            if (linear->weights.require_grad) {
            for (int j = 0; j < linear->weights.numel(); j++) {
                linear->weights.data[j] -= lr * linear->weights.grad[j];
            }
            }
            if (linear->bias.require_grad) {
            for (int j = 0; j < linear->bias.numel(); j++) {
                linear->bias.data[j] -= lr * linear->bias.grad[j];
            }
            }
        }
    }
}
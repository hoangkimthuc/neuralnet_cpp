#include "include/tensor.h"
#include "include/ops.h"
#include "include/autograd.h"

int main() {
    // Initialize a simple dataset
    std::vector<float> input_data = {0.5, 1.5, -0.5, -1.5, 1.0, 2.0, -1.0, -2.0};
    std::vector<float> target_data = {1, 0, 1, 0};
    Tensor input({4, 2}, input_data, true);
    Tensor target({4}, target_data, true);

    // Initialize model and loss
    Linear linear1(2, 2);
    linear1.weights.data = {1, 1, 1, 1};
    linear1.bias.data = {0, 0};
    ReLU relu;
    Linear linear2(2, 2);
    linear2.weights.data = {1, 1, 1, 1};
    linear2.bias.data = {0, 0};
    CrossEntropy loss;

    // Create compute graph
    ComputeGraph graph;

    // Add operations to graph
    graph.add(&linear1);
    graph.add(&relu);
    graph.add(&linear2);
    graph.add(&loss);
    
    // Number of epochs
    int epochs = 10;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Tensor output = linear1.forward(input);
        output = relu.forward(output);
        output = linear2.forward(output);
        Tensor loss_value = loss.forward(output, target);

        // Backward pass
        graph.backward();

        // Print loss
        std::cout << "Epoch " << epoch << ", Loss: " << loss_value.data[0] << std::endl;
        
        float learning_rate = 0.01;
        graph.sgd_step(learning_rate);
        // print weights and bias gradients
        std::cout << "Linear1 weights grad: " << linear1.weights.grad[0] << ", " << linear1.weights.grad[1] << ", " << linear1.weights.grad[2] << ", " << linear1.weights.grad[3] << std::endl;
        std::cout << "Linear1 bias grad: " << linear1.bias.grad[0] << ", " << linear1.bias.grad[1] << std::endl;
    }

    return 0;
}
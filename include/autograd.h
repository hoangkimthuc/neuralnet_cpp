#pragma once

#include "tensor.h"
#include "ops.h"

#include <vector>
#include <string>
#include <memory>

// The ComputeGraph class
class ComputeGraph {
public:
    void add(Operation* operation);
    void backward();
    void sgd_step(float lr);

private:
    std::vector<Operation*> operations;
};

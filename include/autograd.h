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

private:
    std::vector<Operation*> operations;
};

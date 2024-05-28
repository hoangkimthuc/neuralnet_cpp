#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>


class Tensor {
public:
    Tensor();
    Tensor(std::vector<int> shape, std::vector<float> data = std::vector<float>(), bool require_grad = false);
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<float> grad;
    bool require_grad;
    std::string grad_fn;
    int numel() const;
    Tensor operator[](int index);
    void backward();
};

bool operator==(Tensor t1, Tensor t2);
bool operator!=(Tensor t1, Tensor t2);
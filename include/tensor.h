#pragma once

#include <iostream>
#include <vector>
#include <cmath>


class Tensor {
public:
    Tensor();
    Tensor(std::vector<int> shape, std::vector<float> data = std::vector<float>(), bool require_grad = false);
    std::vector<float> data;
    std::vector<int> shape;
    bool require_grad;
    char* grad_fn;

    int numel() const;
    Tensor operator[](int index);
};

bool operator==(Tensor t1, Tensor t2);
bool operator!=(Tensor t1, Tensor t2);

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <initializer_list>


class Tensor {
public:
    Tensor(std::initializer_list<int> shape, std::vector<float> data = std::vector<float>(), bool require_grad = false);
    
    std::vector<float> data;
    std::vector<int> shape;
    bool require_grad;
    char* grad_fn;

    int numel() const;
    Tensor operator[](int index);
};

bool operator==(Tensor t1, Tensor t2);
bool operator!=(Tensor t1, Tensor t2);

# Implement the entire neural net in C++.

### Things included:

- Tensor class

- Forward and backward for operations (Linear, ReLU, Sum, CrossEntropy are included so far)

- A minimal computation graph and an autograd engine that automatically calculate the gradient

- SGD optimizer

### How to build

```bash
mkdir build
cd build
cmake ..
cmake --build .
```
### How to train model

```bash
cd build
./train
```
### Compare with Pytorch's implementation

To ensure the correctness of my implementation, I ported my training code in `train.cpp` to `train.py`, which is an equivalent implementation in pytorch. To run pytorch code:

```bash
conda create -n cpp_neuralnet
conda activate cpp_neuralnet
pip install -r requirements.txt
python train.py
```




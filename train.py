import torch
import torch.nn as nn
import torch.optim as optim

# Initialize a simple dataset
input_data = torch.tensor([[0.5, 1.5], [-0.5, -1.5], [1.0, 2.0], [-1.0, -2.0]], dtype=torch.float32)
target_data = torch.tensor([1, 0, 1, 0], dtype=torch.long)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights to 1s and biases to 0s
        nn.init.constant_(self.linear1.weight, 1.0)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.weight, 1.0)
        nn.init.constant_(self.linear2.bias, 0.0)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleNN()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Number of epochs
epochs = 10

# Training loop
for epoch in range(epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, target_data)
    
    # Backward pass
    loss.backward()
    
    # Optimize
    optimizer.step()
    
    # Print loss
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    print(f"Gradients for linear1 weights: {model.linear1.weight.grad}")
    print(f"Gradients for linear1 bias: {model.linear1.bias.grad}")

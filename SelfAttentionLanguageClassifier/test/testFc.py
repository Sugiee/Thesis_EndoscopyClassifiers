import torch
import pytest
import torch.nn as nn

# Class to be mocked
class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)     # First fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)    # Output layer

    def forward(self, x):
        x = self.flatten(x)           # Flatten the input
        x = self.fc1(x)               # First linear layer
        x = self.relu(x)              # Apply ReLU activation
        logits = self.fc2(x)          # Second linear layer (output)
        return logits
    
# Sample input data
input_dim = 16
hidden_dim = 32
output_dim = 10 # number of output labels

# Initialise model
model = FullyConnectedClassifier(input_dim = input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# -------------------------------------------------------
# Test 1: Flattening operation
# -------------------------------------------------------
def testFlatten():
    print("\n\nFully connected classifier model initiliased:") 
    print(f"input_dim = {input_dim}, hidden_dim = {hidden_dim}, output_dim = {output_dim}\n")
     
    print("Running test: Flattening operation")
    x = torch.randn(1, 1, 4, 4)  # [batch, channel, row, column] 4x4 image
    output = model.flatten(x)
    print("Input tensor shape:", x.shape)
    print("Flattened output shape:", output.shape)
    
    assert output.shape == (1, 16), "Flattening failed: expected shape (1,16)"
    print("Flattening test passed!\n")
    
# -------------------------------------------------------
# Test 2: First fully connected layer (FC1)
# -------------------------------------------------------
def testFc1():
    print("Running test: First linear layer (FC1)")
    x = torch.randn(1, 16) # Sample input
    output = model.fc1(x)
    
    print("Input shape:", x.shape)
    print("FC1 output shape:", output.shape)
    assert output.shape == (1, hidden_dim), f"FC1 failed: expected shape (1,{hidden_dim})"
    print("FC1 Test Passed!\n")
    
# -------------------------------------------------------
# Test 3: ReLU activation function
# -------------------------------------------------------
def testRelu():
    print("Running test: ReLU activation function")
    x = torch.tensor([[-1.0, 0.0, 1.0]])  # Sample input
    output = model.relu(x)
    
    print("Input values:\n", x)
    print("ReLU output values:\n", output)
    assert torch.equal(output, torch.tensor([[0.0, 0.0, 1.0]])), "ReLU activation function incorrect"
    print("ReLU activation function test passed!\n")
     
# -------------------------------------------------------
# Test 4: Second fully connected layer (FC2)
# -------------------------------------------------------
def testFc2():
    print("Running test: Second linear layer (FC2)")
    x = torch.randn(1, hidden_dim)
    output = model.fc2(x)
    
    print("Input shape:", x.shape)
    print("FC2 output shape:", output.shape)
    assert output.shape == (1, output_dim), f"FC2 failed: Expected shape (1,{output_dim})"
    print("FC2 test passed!\n")
    
# -------------------------------------------------------
# Test 5: Forward pass 
# -------------------------------------------------------
def testForwardPass():
    print("Running test: Forward Pass")
    x = torch.randn(1, 16)
    logits = model(x)
    
    print("Input shape:", x.shape)
    print("Forward pass output shape:", logits.shape)
    assert logits.shape == (1, output_dim), f"Forward pass failed: Expected shape (1,{output_dim})"
    print("Forward pass test passed!\n")
    
if __name__ == "__main__":
    pytest.main()
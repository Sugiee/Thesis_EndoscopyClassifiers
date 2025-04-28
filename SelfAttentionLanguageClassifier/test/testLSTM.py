import torch
import pytest
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass tokenizedEmbeddings into LSTM
        _, (hiddenLastLayer, _) = self.lstm(x)  # x.shape (batch_size, seq_length, input_dim)
        logits = self.fc(hiddenLastLayer[-1])   # logits stores raw unnormalised scores
        return logits

# Sample input parameters
embedding_dim = 50
hidden_dim = 64
output_dim = 3
num_layers = 2
seq_length = 10
batch_size = 4

# Initialise the model
model = LSTMClassifier(embedding_dim, hidden_dim, output_dim, num_layers)

# -------------------------------------------------------
# Test 1: LSTM layer 
# -------------------------------------------------------
def testLSTM():
    print("\nRunning test: LSTM layer")
    
    assert isinstance(model.lstm, nn.LSTM), "lstm layer not initialised correctly"
    
    print("LSTM initialisation test passed")

# -------------------------------------------------------
# Test 2: Fully connected (fc) layer
# -------------------------------------------------------
def testFc():
    print("\nRunning test: fully connected layer initialisation")
    
    assert isinstance(model.fc, nn.Linear), "fc layer not initialised correctly"
    assert model.fc.in_features == hidden_dim, "fc layer input dimension is incorrect"
    assert model.fc.out_features == output_dim, "fc layer output dimension is incorrect"
    
    print("Fully connected layer initialisation test passed")

# -------------------------------------------------------
# Test 3: Softmax activation
# -------------------------------------------------------
def testSoftmax():
    print("\nRunning test: Softmax activation function")
    
    x = torch.randn(3, 5)  # 3 sample input tensor
    
    output = model.softmax(x) # apply softmax

    print("Input values:\n", x)
    print("Softmax output values:\n", output)
    print("Sum of softmax outputs along dim=1:\n", output.sum(dim=1))

    # check if outputs between 0 and 1
    assert torch.all(output >= 0) and torch.all(output <= 1), \
        "Softmax outputs are not in valid probability range (0 to 1)"

    # check if probabilities sum to 1 along dim=1
    assert torch.allclose(output.sum(dim=1), torch.ones(output.shape[0])), \
        "Softmax outputs do not sum to 1"

    print("Softmax activation function test passed")

# -------------------------------------------------------
# Test 4: LSTM forward pass
# -------------------------------------------------------
def testLSTMForward():
    print("\nRunning test: LSTM forward pass")
    x = torch.randn(batch_size, seq_length, embedding_dim)  # mock input embeddings
    _, (hiddenLastLayer, _) = model.lstm(x)
    
    print("Input tensor shape:", x.shape)
    print("Hidden last layer shape:", hiddenLastLayer.shape)
    
    assert hiddenLastLayer.shape == (num_layers, batch_size, hidden_dim), \
        f"LSTM hidden state shape incorrect: expected ({num_layers}, {batch_size}, {hidden_dim})"
    
    print("LSTM forward pass test passed")

# -------------------------------------------------------
# Test 5: Fully connected (fc) layer forward pass
# -------------------------------------------------------
def testFCForward():
    print("\nRunning test: Fully connected layer forward pass")
    
    x = torch.randn(num_layers, batch_size, hidden_dim)
    logits = model.fc(x[-1])  # take last layer of hidden state
    
    print("Input to fc layer shape:", x[-1].shape)
    print("Output logits shape:", logits.shape)
    
    assert logits.shape == (batch_size, output_dim), \
        f"Fc forward pass failed: expected shape ({batch_size}, {output_dim})"
    
    print("Fully connected layer forward pass test passed")

# -------------------------------------------------------
# Test 6: End-to-end forward pass
# -------------------------------------------------------
def testEndToEnd():
    print("\nRunning test: End-to-end forward pass")
    x = torch.randn(batch_size, seq_length, embedding_dim)  # mock input embeddings
    logits = model(x)
    
    print("Input tensor shape:", x.shape)
    print("Output logits shape:", logits.shape)
    
    assert logits.shape == (batch_size, output_dim), \
        f"Forward pass failed: expected shape ({batch_size}, {output_dim})"
    assert not torch.isnan(logits).any(), "Forward pass failed: output contains nan values"
    
    print("End-to-end forward pass test passed")

if __name__ == "__main__":
    pytest.main()

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

# mock input parameters
embedding_dim = 64
hidden_dim = 128
output_dim = 5
dropout_rate = 0.1
batch_size = 4
tokens_max_length = 10

class TunableSelfAttentionClassifier(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate):
            super(TunableSelfAttentionClassifier, self).__init__()
            # Define Linear transformations for Query, Key, Value
            self.query = nn.Linear(embedding_dim, hidden_dim)
            self.key = nn.Linear(embedding_dim, hidden_dim)
            self.value = nn.Linear(embedding_dim, hidden_dim)
            # Define Dropout & Classification layer
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(hidden_dim, output_dim)

        def forward(self, x): # Shape: (batch_size, sequence_length, embedding_dim).
            # Compute Q, K, V matrices
            Q = self.query(x)   # Shape: (batch_size, sequence_length, hidden_dim)
            K = self.key(x)     # Shape: (batch_size, sequence_length, hidden_dim)
            V = self.value(x)   # Shape: (batch_size, sequence_length, hidden_dim)
            
            # Compute attention scores: scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
            
            # Apply softmax to normalise scores along the sequence_length dimension
            attention = F.softmax(scores, dim=-1) # Shape: (batch_size, sequence_length, sequence_length)
            
            # Compute weighted sum of Value using attention scores
            hidden = torch.matmul(attention, V) # Shape: (batch_size, sequence_length, hidden_dim)
            
            # Pool the hidden representations across the sequence length (mean pooling)
            pooled = hidden.mean(dim=1) # Shape: (batch_size, hidden_dim)
            
            # Apply dropout for regularisation
            pooled = self.dropout(pooled)
            
            # Pass the pooled representation through the classifier to get logits
            return self.classifier(pooled) # Shape: (batch_size, output_dim)

# Initialise the model
model = TunableSelfAttentionClassifier(embedding_dim, hidden_dim, output_dim, dropout_rate)

# -------------------------------------------------------
# Test 1: Scaled dot-product attention calculation
# -------------------------------------------------------
def testAttentionCalculation():
    print("\nRunning test: Attention calculation")
    x = torch.randn(batch_size, tokens_max_length, embedding_dim)  # mock input embeddings
    Q = model.query(x)  
    K = model.key(x)  
    V = model.value(x)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
    attention = F.softmax(scores, dim=-1)
    hidden = torch.matmul(attention, V)

    print("Input shape:", x.shape)
    print("Query shape:", Q.shape, "\n| Key shape:", K.shape, "\n| Value shape:", V.shape)
    print("Attention scores shape:", scores.shape)
    print("Hidden output shape:", hidden.shape)

    assert hidden.shape == (batch_size, tokens_max_length, hidden_dim), \
        "Attention calculation failed: incorrect shape"
    
    print("Attention calculation test passed")

# -------------------------------------------------------
# Test 2: Mean pooling operation
# -------------------------------------------------------
def testMeanPooling():
    print("\nRunning test: Mean pooling operation")
    x = torch.randn(batch_size, tokens_max_length, hidden_dim)  # mock attention output
    pooled = x.mean(dim=1)

    print("Input shape:", x.shape)
    print("Pooled output shape:", pooled.shape)

    assert pooled.shape == (batch_size, hidden_dim), \
        "Mean pooling failed: expected shape (batch_size, hidden_dim)"
        
    print("Mean pooling test passed")

# -------------------------------------------------------
# Test 3: Dropout operation
# -------------------------------------------------------
def testDropoutForward():
    print("\nRunning test: Dropout forward pass")
    x = torch.randn(batch_size, hidden_dim)  # mock pooled representations
    output = model.dropout(x)

    print("Input shape:", x.shape)
    print("Output shape after dropout:", output.shape)

    assert output.shape == x.shape, "Dropout forward pass failed: incorrect output shape"
    
    print("Dropout forward test passed")

# -------------------------------------------------------
# Test 4: Fully connected classification layer
# -------------------------------------------------------
def testFcClassification():
    print("\nRunning test: Fully connected classification")
    x = torch.randn(batch_size, hidden_dim)  # mock pooled embeddings
    logits = model.classifier(x)

    print("Input to classifier shape:", x.shape)
    print("Logits output shape:", logits.shape)

    assert logits.shape == (batch_size, output_dim), \
        f"FC forward pass failed: expected shape ({batch_size}, {output_dim})"
        
    print("Fully connected classification test passed")

# -------------------------------------------------------
# Test 5: End-to-end forward pass
# -------------------------------------------------------
def testEndToEnd():
    print("\nRunning test: End-to-end forward pass")
    x = torch.randn(batch_size, tokens_max_length, embedding_dim)  # simulate input embeddings
    logits = model(x)  # forward pass through entire model

    print("Input shape:", x.shape)
    print("Logits output shape:", logits.shape)

    assert logits.shape == (batch_size, output_dim), \
        f"End-To-End failed: expected shape ({batch_size}, {output_dim})"
    assert not torch.isnan(logits).any(), "Forward pass failed: output contains NaN values"
    
    print("End-to-end forward pass test passed")

if __name__ == "__main__":
    pytest.main()
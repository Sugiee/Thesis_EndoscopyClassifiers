import torch
import pytest
import torch.nn as nn

class VisualGeometryGroup(nn.Module):
    def __init__(self, output_dim, resize_size):
        super(VisualGeometryGroup, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * int(resize_size // 32) * int(resize_size // 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Mock setup
output_dim = 10
resize_size = 224
model = VisualGeometryGroup(output_dim=output_dim, resize_size=resize_size)

# -------------------------------------------------------
# Test 1: Feature extractor
# -------------------------------------------------------
def testFeatureExtractor():
    print("\n\nVisualGeometryGroup model initialised:")
    print(f"resize_size = {resize_size}, output_dim = {output_dim}\n")

    print("Running test: Feature extractor")
    x = torch.randn(1, 3, resize_size, resize_size)
    features = model.features(x)
    print("Input shape:", x.shape)
    print("Feature output shape:", features.shape)

    assert features.shape == (1, 512, 7, 7), "Feature extractor output shape incorrect"
    print("Feature extractor test passed!\n")

# -------------------------------------------------------
# Test 2: Flatten operation
# -------------------------------------------------------
def testFlatten():
    print("Running test: Flatten operation")
    x = torch.randn(1, 3, resize_size, resize_size)
    features = model.features(x)
    flattened = features.view(features.size(0), -1)

    print("Feature output shape:", features.shape)
    print("Flattened shape:", flattened.shape)
    
    assert flattened.shape == (1, 512 * 7 * 7), f"Flatten failed: expected shape (1, 512 * 7 * 7)"
    
    print("Flatten test passed!\n")

# -------------------------------------------------------
# Test 3: Classifier initilisation
# -------------------------------------------------------
def testClassifierLayers():
    print("Running test: Classifier layers present")
    layer_types = [type(layer).__name__ for layer in model.classifier]
    print("Classifier contains layers:", layer_types)

    assert "ReLU" in layer_types, "Missing ReLU layer"
    assert "Dropout" in layer_types, "Missing Dropout layer"
    assert "Linear" in layer_types, "Missing Linear layer"
    
    print("Classifier layer test passed!\n")

# -------------------------------------------------------
# Test 4: Classifier layer
# -------------------------------------------------------
def testClassifier():
    print("Running test: Classifier layer")
    x = torch.randn(1, 3, resize_size, resize_size)
    logits = model(x)

    print("Input shape:", x.shape)
    print("Output (logits) shape:", logits.shape)
    assert logits.shape == (1, output_dim), f"Classifier failed: expected shape (1, {output_dim})"
    print("Classifier test passed!\n")

if __name__ == "__main__":
    pytest.main()
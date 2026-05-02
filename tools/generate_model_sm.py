import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # PyTorch natively uses (Channels, Height, Width)
        # padding=1 with kernel=3 keeps spatial dimensions the same (like 'same' in TF)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        # 8 channels * 4x4 spatial size = 128
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = CNN()

# Create a dummy input tensor with shape (Batch, Channels, Height, Width) -> (1, 1, 32, 32)
dummy_input = torch.randn(1, 1, 32, 32)

# Export natively to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output']
)

print("model.onnx generated cleanly in native NCHW format. No Transpose nodes!")
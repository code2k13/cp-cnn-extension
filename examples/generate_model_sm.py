import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
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

dummy_input = torch.randn(1, 1, 32, 32)

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
print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
print(f"Weight bytes:    {sum(p.numel() for p in model.parameters()) * 4} bytes")

# --- Inline validation: generate fixed input, run ONNX inference, save test_input.bin ---
import numpy as np
import onnxruntime as ort

SEED = 42
INPUT_SHAPE = (1, 1, 32, 32)  # NCHW

rng = np.random.default_rng(SEED)
inp = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

inp.flatten().tofile("test_input.bin")
print(f"\nSaved test_input.bin ({inp.size} floats, {inp.nbytes} bytes)")

sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: inp})[0].flatten()

print(f"\nONNX output ({len(out)} values):")
for i, v in enumerate(out):
    print(f"  [{i}] {v:.6f}")
print(f"\nArgmax (predicted class): {int(np.argmax(out))}")
print("\nCopy model.bin and test_input.bin to CIRCUITPY and compare argmax with Pico output.")
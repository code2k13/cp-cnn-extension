import torch
import torch.nn as nn

class TinyDense(nn.Module):
    def __init__(self):
        super(TinyDense, self).__init__()
        # 16 -> 16 -> 16 -> 4
        # Weights: (16*16 + 16*16 + 16*4) * 4 bytes = (256+256+64)*4 = 2304 bytes
        # Biases:  (16+16+4) * 4 bytes = 144 bytes
        # Total weights+biases: ~2.4 KB
        # Largest activation buffer: 16 floats = 64 bytes
        # Total RAM: well under 10 KB
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TinyDense()

# Input: flat vector of 16 floats, batch size 1
dummy_input = torch.randn(1, 16)

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

print("model.onnx generated — 16->16->16->4 dense network")
print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
print(f"Weight bytes:    {sum(p.numel() for p in model.parameters()) * 4} bytes")

# --- Inline validation: generate fixed input, run ONNX inference, save test_input.bin ---
import numpy as np
import onnxruntime as ort

SEED = 42
INPUT_SHAPE = (1, 16)

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
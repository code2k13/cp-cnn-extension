"""
tally_pc.py - Run inference on PC using onnxruntime.
Generates a fixed random input, saves it as test_input.bin,
and prints the output logits.

Usage:
    python3 tally_pc.py

Requires: onnxruntime, numpy
"""

import numpy as np
import onnxruntime as ort

# Fixed seed -> same input every time -> predictable output to compare with Pico
SEED = 42
INPUT_SHAPE = (1, 1, 32, 32)  # NCHW, matches model

rng = np.random.default_rng(SEED)
inp = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

# Save flat float32 bytes for the Pico
inp.flatten().tofile("test_input.bin")
print(f"Saved test_input.bin ({inp.size} floats, {inp.nbytes} bytes)")

# Run ONNX inference
sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: inp})[0].flatten()

print(f"\nONNX output ({len(out)} values):")
for i, v in enumerate(out):
    print(f"  [{i}] {v:.6f}")

print(f"\nArgmax (predicted class): {int(np.argmax(out))}")
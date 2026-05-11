import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import struct
import os
import onnxruntime as ort

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
        self.fc = nn.Linear(72, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

SIZE = 30
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

model.eval()
model.cpu()
dummy_input = torch.randn(1, 1, SIZE, SIZE)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    export_params=True, opset_version=18,
    do_constant_folding=True,
    input_names=['input'], output_names=['output']
)

OUT_DIR = "mnist_samples"
os.makedirs(OUT_DIR, exist_ok=True)
sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name

samples_per_digit = {}
selected = []
for img_tensor, label in test_data:
    d = int(label)
    if d not in samples_per_digit: samples_per_digit[d] = 0
    if samples_per_digit[d] < 2:
        selected.append((img_tensor.numpy()[0], d))
        samples_per_digit[d] += 1
    if len(selected) == 20: break

for idx, (img_f32, digit) in enumerate(selected):
    bin_path = os.path.join(OUT_DIR, f"sample_{idx:02d}.bin")
    img_f32.flatten().astype(np.float32).tofile(bin_path)
    
    rgb_path = os.path.join(OUT_DIR, f"sample_{idx:02d}.rgb")
    with open(rgb_path, "wb") as f:
        for y in range(SIZE):
            for x in range(SIZE):
                v = int(img_f32[y, x] * 255)
                rgb565 = ((v >> 3) << 11) | (((v >> 2)) << 5) | (v >> 3)
                f.write(struct.pack(">H", rgb565))

print("\n--- VALIDATION STEP: Running ONNX with generated .bin files ---")
for idx in range(20):
    bin_path = os.path.join(OUT_DIR, f"sample_{idx:02d}.bin")
    loaded_data = np.fromfile(bin_path, dtype=np.float32).reshape(1, 1, SIZE, SIZE)
    
    ort_out = sess.run(None, {input_name: loaded_data})[0].flatten()
    predicted = int(np.argmax(ort_out))
    actual = selected[idx][1]
    status = "PASS" if predicted == actual else "FAIL"
    print(f"Sample {idx:02d}: Target={actual}, Predicted={predicted} [{status}]")
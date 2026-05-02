Here is a cleaned-up, professional version of your README. It uses consistent formatting, improved hierarchy, and callouts to make it more readable for developers.

---

# `cnn_helper`: Native CircuitPython CNN Extension (RP2040)

This repository provides a native CircuitPython C extension (`cnn_helper`) that wraps the `tiny_inference` CNN engine[cite: 1]. It enables **hardware-accelerated neural network inference** directly on microcontrollers like the Raspberry Pi Pico[cite: 1].

## 📂 Repository Structure

To maintain a clean build environment, source code is separated from utility scripts and configuration patches:

| Directory | Description |
| :--- | :--- |
| **`src/`** | Core C code: `shared-bindings`, `shared-module`, and `lib/tiny_inference`[cite: 1]. |
| **`patches/`** | Reference configs (`Makefile`, `mpconfigport.mk`) to enable the module[cite: 1]. |
| **`tools/`** | Python utilities for model conversion (e.g., ONNX to `.bin`)[cite: 1]. |
| **`test_data/`** | Sample `.bin` models and input vectors for verification[cite: 1]. |

---

## 🛠️ Build Instructions

### 1. Prepare the Environment
Ensure you have the **ARM GNU Toolchain** installed and the [CircuitPython build environment configured](https://learn.adafruit.com/building-circuitpython/).

### 2. Clone Repositories
Clone the main CircuitPython source and this extension into an `extensions` subfolder:

```bash
# Clone CircuitPython
git clone https://github.com/adafruit/circuitpython.git
cd circuitpython

# Navigate specifically to the Raspberry Pi port
cd ports/raspberrypi

# Fetch ONLY the submodules needed for the RP2040
make fetch-port-submodules

# Go back to the root to clone your extension
cd ../../
git clone https://github.com/YOUR_USERNAME/cnn-helper-extension.git extensions/cnn_helper
```

### 3. Link Source Files
Symlink the extension code into the standard CircuitPython directory structure[cite: 1]:

```bash
# Run from the 'circuitpython' root directory
ln -s ../../extensions/cnn_helper/src/lib/tiny_inference lib/tiny_inference
ln -s ../../extensions/cnn_helper/src/shared-bindings/cnn_helper shared-bindings/cnn_helper
ln -s ../../extensions/cnn_helper/src/shared-module/cnn_helper shared-module/cnn_helper
```

### 4. Enable the Module (Patching)
You must register the new C files with the build system[cite: 1]. You can use the files in `patches/` as a reference or manually edit:

*   **Define the Module**: In `py/circuitpy_mpconfig.h`, add:
    ```c
    #define CIRCUITPY_CNN_HELPER (1)
    ```
*   **Update the Makefile**: In `ports/raspberrypi/Makefile`, append the source path[cite: 1]:
    ```makefile
    SRC_C += lib/tiny_inference/tiny_inference.c
    ```

### 5. Compile for RP2040
Build the firmware for the Raspberry Pi Pico[cite: 1]:

```bash
cd ports/raspberrypi
make BOARD=raspberry_pi_pico -j$(nproc)
```
Flash the resulting `build-raspberry_pi_pico/firmware.uf2` to your board[cite: 1].

---

## 💻 Example Usage

Once flashed, copy your `model.bin` and `test_input.bin` to the `CIRCUITPY` drive[cite: 1].

```python
import array
import cnn_helper

# 1. Load the model into RAM
cnn_helper.load_model("model.bin")

# 2. Prepare the input tensor (float32 array)
inp = array.array("f")
with open("test_input.bin", "rb") as f:
    inp.frombytes(f.read())

# 3. Run inference
out = cnn_helper.perform_inference(inp)
print("Inference Output:", out)

# 4. Clean up
cnn_helper.unload_model()
```

---

## 🔧 Utilities & Tools

Use the conversion scripts in `tools/` to prepare your custom models[cite: 1].

**Convert ONNX to Blob:**
```bash
python3 tools/onnx_to_blob.py my_model.onnx
```
This generates a `model.bin` compatible with the `cnn_helper.load_model()` function[cite: 1].
```
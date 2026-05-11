# Building CircuitPython with cnn_helper Extension for Raspberry Pi Pico

This guide walks through compiling CircuitPython from source and integrating the [`cnn_helper`](https://github.com/code2k13/cp-cnn-extension) native C extension, which enables CNN inference directly on the RP2040 microcontroller.

---

## Prerequisites

You will need a Linux machine (Ubuntu 22.04 or 24.04 recommended), internet access, and around 2–3 GB of disk space. All commands assume a standard user account with `sudo` privileges.

---

## Part 1 — Set Up the Build Environment

Install the required system packages. `build-essential` provides `make` and `gcc`, `git-lfs` is needed for CircuitPython's large binary assets, and `python3-venv` isolates the Python dependencies from your system Python.

```bash
sudo apt update && sudo apt install -y build-essential git git-lfs gettext cmake python3-venv software-properties-common
```

Create and activate a Python virtual environment. All subsequent `pip` installs will go here, keeping your system clean.

```bash
python3 -m venv ~/.py
source ~/.py/bin/activate
```

Clone the CircuitPython source and install its Python build dependencies.

```bash
git clone https://github.com/adafruit/circuitpython.git
cd circuitpython
pip3 install --upgrade -r requirements-dev.txt -r requirements-doc.txt
```

---

## Part 2 — Install the ARM Cross-Compiler

CircuitPython for RP2040 must be cross-compiled using the ARM GNU Toolchain. Download and extract it, then add it to your PATH for the current session. If you want this permanent, add the `export` line to your `~/.bashrc`.

```bash
wget https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz
mkdir -p ~/arm-gnu-toolchain
tar -xf arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz -C ~/arm-gnu-toolchain --strip-components=1
export PATH="$HOME/arm-gnu-toolchain/bin:$PATH"
```

---

## Part 3 — Build the Base Firmware

First build `mpy-cross`, the MicroPython bytecode cross-compiler that CircuitPython depends on. Then fetch the RP2040-specific submodules (Pico SDK, TinyUSB, etc.) and do an initial build to confirm the base setup is working before adding the extension.

```bash
# From ~/circuitpython
make -C mpy-cross
cd ports/raspberrypi
make fetch-port-submodules
make BOARD=raspberry_pi_pico
cd ../..
```

A successful build produces `ports/raspberrypi/build-raspberry_pi_pico/firmware.uf2`. If this step fails, fix it before proceeding — the extension build will inherit any underlying issues.

---

## Part 4 — Clone the cnn_helper Extension

Clone the extension repository into a separate directory outside the CircuitPython tree. This keeps the two repos independent and makes updates easier.

```bash
git clone https://github.com/code2k13/cp-cnn-extension ~/extensions/cnn_helper
```

The repo structure relevant to the build is:

```
extensions/cnn_helper/
├── src/
│   ├── lib/tiny_inference/          # The CNN engine (tiny_inference.c/.h)
│   ├── shared-bindings/cnn_helper/  # CircuitPython Python-facing API
│   └── shared-module/cnn_helper/    # CircuitPython C implementation
└── patches/                         # Reference versions of the 3 files to edit
    ├── ports/raspberrypi/Makefile
    ├── ports/raspberrypi/mpconfigport.mk
    └── py/circuitpy_mpconfig.h
```

---

## Part 5 — Symlink the Extension into CircuitPython

CircuitPython's build system expects source files in specific directories. Rather than copying files (which makes updates painful), symlink the extension's source folders into the CircuitPython tree. Run these from the `~/circuitpython` root.

```bash
cd ~/circuitpython
ln -s ~/extensions/cnn_helper/src/lib/tiny_inference lib/tiny_inference
ln -s ~/extensions/cnn_helper/src/shared-bindings/cnn_helper shared-bindings/cnn_helper
ln -s ~/extensions/cnn_helper/src/shared-module/cnn_helper shared-module/cnn_helper
```

Verify the symlinks resolve correctly:

```bash
ls lib/tiny_inference/
ls shared-bindings/cnn_helper/
ls shared-module/cnn_helper/
```

Each should list actual files, not an error. If you see `No such file or directory`, the path in the extension repo has changed — re-check with `find ~/extensions/cnn_helper/src -type f`.

---

## Part 6 — Patch the CircuitPython Build System

Three files need to be edited to register the module with the build system. The `patches/` folder in the extension repo contains reference versions of each file — use `diff` to see the exact changes if you prefer to apply them manually.

### `py/circuitpy_mpconfig.h`

This file controls which modules are compiled in. Add the following block to enable `cnn_helper`:

```c
#ifndef CIRCUITPY_CNN_HELPER
#define CIRCUITPY_CNN_HELPER (1)
#endif
```

### `ports/raspberrypi/Makefile`

The `INC` block tells the compiler where to find header files. Add the `tiny_inference` include path:

```makefile
INC += -I$(TOP)/lib/tiny_inference
```

### `ports/raspberrypi/mpconfigport.mk`

This file controls RP2040-specific build configuration. Append the following to register all three C source files with the build. Unlike some other CircuitPython ports, the RP2040 port does not auto-scan `shared-bindings` and `shared-module` — all three files must be listed explicitly.

```makefile
CIRCUITPY_CNN_HELPER ?= 1
CFLAGS += -DCIRCUITPY_CNN_HELPER=$(CIRCUITPY_CNN_HELPER)
SRC_C += lib/tiny_inference/tiny_inference.c \
         shared-bindings/cnn_helper/__init__.c \
         shared-module/cnn_helper/__init__.c
```

---

## Part 7 — Final Build

With all patches applied, rebuild the firmware. The `-j$(nproc)` flag parallelises the build across all CPU cores.

```bash
cd ~/circuitpython/ports/raspberrypi
make BOARD=raspberry_pi_pico -j$(nproc)
```

The output firmware is at:

```
build-raspberry_pi_pico/firmware.uf2
```

Flash it to your Pico by holding BOOTSEL while plugging in USB, then copying the `.uf2` file to the mounted drive.

---

## Part 8 — Preparing a Model with the Tools

The `tools/` folder in the extension repo provides a complete pipeline: generate a sample model → convert it to the binary blob format → validate it on x86 before deploying to the Pico.

Install the required Python dependencies first (do this in the same venv used for the build):

```bash
pip install torch onnx onnxruntime numpy
```

---

### Step 1 — `generate_model_sm.py` — Create a sample ONNX model

This script defines and exports a small CNN in PyTorch ONNX format. The architecture is a 3-layer conv+pool network followed by a fully connected layer, designed to accept a `(1, 1, 32, 32)` input (grayscale 32×32 image) and output 10 class logits. It is a starting point — replace it with your own model once you have the pipeline working.

The key design constraint is that the model must use **NCHW format** (channels first). This script exports cleanly in that format with no extra Transpose nodes, which matters because `onnx_to_blob.py` does not support Transpose ops.

```bash
cd ~/extensions/cnn_helper/examples
python3 generate_model_sm.py
# Output: model.onnx
```

Supported layer types are: `Conv2d`, `MaxPool2d`, `ReLU`, `Flatten`, `Linear`. Anything outside this set will cause `onnx_to_blob.py` to abort with a `CRITICAL: Unsupported ONNX node` error.

Models must use NCHW format (channels first). Export from PyTorch using opset_version=11.

---

### Step 2 — `onnx_to_blob.py` — Convert ONNX to Pico binary format

This script reads `model.onnx`, extracts all weights and layer metadata, and packs everything into a compact binary blob (`model.bin`) that matches the C struct layout expected by `tiny_inference.c` on the Pico.

The blob format is: a 4-byte magic header, followed by a fixed 64-byte metadata record per layer, followed by raw float32 weight and bias data. Offsets into the weight block are embedded in each layer's metadata so the C engine can index weights directly without parsing.

```bash
cd ~/extensions/cnn_helper/tools
cp ../examples/model.onnx .
cp ../examples/test_input.bin .
python3 onnx_to_blob.py
# Reads:  model.onnx
# Output: model.bin
```

The script prints a layer-by-layer summary showing input/output channels, spatial dimensions, and weight offsets — useful for debugging if inference results look wrong.

Constraints to be aware of: `Conv2d` must use `stride=1`, `MaxPool2d` must use `padding=0`. Violations are caught at conversion time with a clear error message.

---

### Step 3 — Deploy and run on the Pico

No separate validation script is needed. The `./examples/generate_model_sm.py` and `./examples/generate_model_dense.py` scripts already handle validation inline — after exporting `model.onnx` they automatically generate a fixed random input (seeded at 42), save it as `test_input.bin`, run ONNX inference via `onnxruntime`, and print the output logits and predicted class.

Note the argmax printed at the end of the generate script — this is your ground truth to compare against the Pico output.

Copy `./tools/model.bin` and `./tools/test_input.bin` to the `CIRCUITPY` drive, then run:

```python
import array
import cnn_helper

cnn_helper.load_model("model.bin")

inp = array.array("f")
with open("test_input.bin", "rb") as f:
    inp.frombytes(f.read())

out = cnn_helper.perform_inference(inp)
print("Output:", out)

cnn_helper.unload_model()
```

The argmax of the Pico output should be identical to what the generate script printed. If the logit values differ slightly that is normal due to float32 rounding; if the argmax differs, there is likely a weight layout issue in the blob conversion.


### Attribution

#### MNIST Dataset

This project uses the **MNIST database of handwritten digits**, created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.

- Original source: http://yann.lecun.com/exdb/mnist/

- @article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]},
  volume={2},
  year={2010},
  url={http://yann.lecun.com/exdb/mnist}
}

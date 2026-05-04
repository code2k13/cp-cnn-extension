import struct
import onnx
from onnx import numpy_helper
import numpy as np
import sys

# Magic number and layer type constants
MAGIC = 0x434E4E01
LAYER_CONV2D = 1
LAYER_MAXPOOL = 2
LAYER_RELU = 3
LAYER_FLATTEN = 4
LAYER_LINEAR = 5
LAYER_SOFTMAX = 6

# Fixed size to match C struct alignment expectations
LAYER_META_SIZE = 64

def align_to(value, alignment):
    """Aligns a value to the specified byte boundary."""
    return ((value + alignment - 1) // alignment) * alignment

def get_attribute(node, attr_name, default=None):
    """Extracts an attribute from an ONNX node."""
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
    return default

def get_onnx_input_shape(graph):
    """Extracts the input shape dynamically from the ONNX graph."""
    onnx_input = graph.input[0]
    shape = []
    for dim in onnx_input.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(1)
    return tuple(shape)

def parse_onnx_model(onnx_path):
    """Loads the ONNX model and extracts initializers (weights/biases)."""
    try:
        model = onnx.load(onnx_path)
        initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        return model.graph, initializers
    except Exception as e:
        print(f"Error loading ONNX file: {e}")
        sys.exit(1)

def extract_layers(graph, initializers):
    """Iterates through ONNX nodes and extracts compatible layers and weights."""
    layers = []
    weight_data = bytearray()
    current_offset = 0

    for node in graph.node:
        layer = {"op_type": node.op_type}

        if node.op_type == "Conv":
            stride = get_attribute(node, "strides", [1, 1])[0]
            if stride != 1:
                raise ValueError(f"Unsupported Conv stride: {stride}. The C engine only supports stride 1.")

            weights = initializers[node.input[1]]
            layer.update({
                "type": LAYER_CONV2D,
                "in_channels": weights.shape[1],
                "out_channels": weights.shape[0],
                "kernel_size": weights.shape[2],
                "stride": 1,
                "padding": get_attribute(node, "pads", [0, 0, 0, 0])[0]
            })

            # Process Weights
            weight_bytes = weights.astype(np.float32).tobytes()
            weight_offset = align_to(current_offset, 4)
            weight_data.extend(b"\x00" * (weight_offset - current_offset))
            weight_data.extend(weight_bytes)
            layer["weight_offset"] = weight_offset
            current_offset = len(weight_data)

            # Process Bias
            if len(node.input) >= 3:
                bias_bytes = initializers[node.input[2]].astype(np.float32).tobytes()
                bias_offset = align_to(current_offset, 4)
                weight_data.extend(b"\x00" * (bias_offset - current_offset))
                weight_data.extend(bias_bytes)
                layer["bias_offset"] = bias_offset
                current_offset = len(weight_data)
            else:
                layer["bias_offset"] = 0

        elif node.op_type == "MaxPool":
            padding = get_attribute(node, "pads", [0, 0, 0, 0])[0]
            if padding != 0:
                raise ValueError(f"Unsupported MaxPool padding: {padding}. The C engine ignores padding, so only 0 is supported.")

            layer.update({
                "type": LAYER_MAXPOOL,
                "kernel_size": get_attribute(node, "kernel_shape", [2, 2])[0],
                "stride": get_attribute(node, "strides", [2, 2])[0],
                "padding": 0
            })

        elif node.op_type == "Relu":
            layer["type"] = LAYER_RELU

        elif node.op_type == "Flatten" or node.op_type == "Reshape":
            layer["type"] = LAYER_FLATTEN

        elif node.op_type == "Gemm" or node.op_type == "MatMul":
            w = initializers[node.input[1]]
            transB = get_attribute(node, "transB", 0)
            layer.update({
                "type": LAYER_LINEAR,
                "in_features": w.shape[1] if len(w.shape) > 1 else w.shape[0],
                "out_features": w.shape[0] if len(w.shape) > 1 else 1
            })

            weight_bytes = w.astype(np.float32).tobytes()
            weight_offset = align_to(current_offset, 4)
            weight_data.extend(b"\x00" * (weight_offset - current_offset))
            weight_data.extend(weight_bytes)
            layer["weight_offset"] = weight_offset
            current_offset = len(weight_data)

            if len(node.input) >= 3:
                bias_bytes = initializers[node.input[2]].astype(np.float32).tobytes()
                bias_offset = align_to(current_offset, 4)
                weight_data.extend(b"\x00" * (bias_offset - current_offset))
                weight_data.extend(bias_bytes)
                layer["bias_offset"] = bias_offset
                current_offset = len(weight_data)
            else:
                layer["bias_offset"] = 0

        elif node.op_type == "Softmax":
            layer["type"] = LAYER_SOFTMAX

        else:
            raise ValueError(f"CRITICAL: Unsupported ONNX node '{node.op_type}' found. The C engine cannot execute this.")

        layers.append(layer)

    return layers, bytes(weight_data)

def infer_dimensions(layers, input_shape):
    """Calculates the spatial dimensions of the tensors passing through the network.
    Supports both 4D NCHW inputs (conv models) and 2D flat inputs (dense-only models).
    """
    if len(input_shape) == 4:
        # NCHW: conv/pool model
        curr_c = input_shape[1]
        curr_h = input_shape[2]
        curr_w = input_shape[3]
    else:
        # Flat input: dense-only model (N, features)
        curr_c = input_shape[1]
        curr_h = 1
        curr_w = 1

    for layer in layers:
        layer["in_channels"] = curr_c
        layer["in_height"] = curr_h
        layer["in_width"] = curr_w

        if layer["type"] == LAYER_CONV2D:
            k, s, p = layer["kernel_size"], layer["stride"], layer["padding"]
            curr_h = (curr_h + 2 * p - k) // s + 1
            curr_w = (curr_w + 2 * p - k) // s + 1
            curr_c = layer["out_channels"]
        elif layer["type"] == LAYER_MAXPOOL:
            k, s = layer["kernel_size"], layer["stride"]
            curr_h = (curr_h - k) // s + 1
            curr_w = (curr_w - k) // s + 1
        elif layer["type"] == LAYER_FLATTEN:
            curr_c = curr_c * curr_h * curr_w
            curr_h, curr_w = 1, 1
        elif layer["type"] == LAYER_LINEAR:
            curr_c = layer["out_features"]
            curr_h, curr_w = 1, 1

        layer["out_channels"] = curr_c
        layer["out_height"] = curr_h
        layer["out_width"] = curr_w


def create_blob(layers, weight_data):
    """Packs the layers into the final binary blob matching the C struct exactly.

    LayerMeta is __attribute__((packed)) in C:
      uint8_t  type, in_channels, out_channels, kernel_size, stride, padding  -> 6 bytes
      uint16_t in_height, in_width, out_height, out_width                     -> 8 bytes
      uint32_t weight_offset, bias_offset                                      -> 8 bytes
      Total: 22 bytes packed, NO padding.
    """

    header = struct.pack("<II", MAGIC, len(layers))
    meta_table = bytearray()

    for l in layers:
        # 22 bytes: matches LayerMeta __attribute__((packed)) exactly
        meta = struct.pack("<BBBBBBHHHHII",
            l["type"], l["in_channels"], l["out_channels"],
            l.get("kernel_size", 0), l.get("stride", 1), l.get("padding", 0),
            l["in_height"], l["in_width"], l["out_height"], l["out_width"],
            l.get("weight_offset", 0), l.get("bias_offset", 0)
        )
        assert len(meta) == 22, f"LayerMeta pack size mismatch: {len(meta)}"
        meta_table.extend(meta)

    # get_weight_ptr in C jumps over METADATA_SIZE(64) bytes per layer,
    # so pad the meta block up to 64 * num_layers before appending weight data.
    expected_meta_block_size = LAYER_META_SIZE * len(layers)
    padding = b"\x00" * (expected_meta_block_size - len(meta_table))

    return header + meta_table + padding + weight_data

def main():
    print("Parsing model.onnx...")
    graph, inits = parse_onnx_model("model.onnx")

    input_shape = get_onnx_input_shape(graph)
    print(f"Detected ONNX input shape: {input_shape}")

    try:
        layers, weights = extract_layers(graph, inits)
        infer_dimensions(layers, input_shape)

        blob = create_blob(layers, weights)
        with open("model.bin", "wb") as f:
            f.write(blob)
        print(f"Success! model.bin generated with {len(layers)} layers.")
    except ValueError as e:
        print(f"CONFIGURATION ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
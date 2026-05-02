#ifndef TINY_INFERENCE_H
#define TINY_INFERENCE_H

#include <stdint.h>

#define LAYER_CONV2D  1
#define LAYER_MAXPOOL 2
#define LAYER_RELU    3
#define LAYER_FLATTEN 4
#define LAYER_LINEAR  5
#define LAYER_SOFTMAX 6

typedef struct __attribute__((packed)) {
    uint8_t  type;
    uint8_t  in_channels;
    uint8_t  out_channels;
    uint8_t  kernel_size;
    uint8_t  stride;
    uint8_t  padding;
    uint16_t in_height;
    uint16_t in_width;
    uint16_t out_height;
    uint16_t out_width;
    uint32_t weight_offset;
    uint32_t bias_offset;
} LayerMeta;

typedef struct {
    uint32_t magic;
    uint32_t num_layers;
} ModelHeader;

void tiny_predict(
    const uint8_t* blob_ptr,
    const float* input_ptr,
    float* scratch_a,
    float* scratch_b
);

#endif

#include "tiny_inference.h"
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define METADATA_SIZE 64

static const float* get_weight_ptr(const uint8_t* blob_ptr, uint32_t offset) {
    return (const float*)(blob_ptr + 8 +
           METADATA_SIZE * ((const ModelHeader*)blob_ptr)->num_layers + offset);
}

static const float* get_bias_ptr(const uint8_t* blob_ptr, uint32_t offset) {
    return (const float*)(blob_ptr + 8 +
           METADATA_SIZE * ((const ModelHeader*)blob_ptr)->num_layers + offset);
}

static void conv2d(
    const float* input,
    const uint8_t* blob_ptr,
    const LayerMeta* meta,
    float* output
) {
    const float* weights = get_weight_ptr(blob_ptr, meta->weight_offset);
    const float* bias = get_bias_ptr(blob_ptr, meta->bias_offset);

    uint8_t in_ch = meta->in_channels;
    uint8_t out_ch = meta->out_channels;
    uint8_t k = meta->kernel_size;
    uint8_t s = meta->stride;
    uint8_t p = meta->padding;
    uint16_t ih = meta->in_height;
    uint16_t iw = meta->in_width;
    uint16_t oh = meta->out_height;
    uint16_t ow = meta->out_width;

    for (uint8_t oc = 0; oc < out_ch; oc++) {
        for (uint16_t oy = 0; oy < oh; oy++) {
            for (uint16_t ox = 0; ox < ow; ox++) {
                float sum = bias[oc];

                for (uint8_t ic = 0; ic < in_ch; ic++) {
                    for (uint8_t ky = 0; ky < k; ky++) {
                        for (uint8_t kx = 0; kx < k; kx++) {
                            int16_t ix = (int16_t)(ox * s + kx) - p;
                            int16_t iy = (int16_t)(oy * s + ky) - p;

                            if (ix >= 0 && ix < (int16_t)iw && iy >= 0 && iy < (int16_t)ih) {
                                float in_val = input[ic * ih * iw + iy * iw + ix];
                                float w_val = weights[oc * in_ch * k * k +
                                                     ic * k * k +
                                                     ky * k + kx];
                                sum += in_val * w_val;
                            }
                        }
                    }
                }

                output[oc * oh * ow + oy * ow + ox] = sum;
            }
        }
    }
}

static void maxpool2d(
    const float* input,
    const LayerMeta* meta,
    float* output
) {
    uint8_t k = meta->kernel_size;
    uint8_t s = meta->stride;
    uint16_t ih = meta->in_height;
    uint16_t iw = meta->in_width;
    uint16_t oh = meta->out_height;
    uint16_t ow = meta->out_width;
    uint8_t channels = meta->in_channels;

    for (uint8_t c = 0; c < channels; c++) {
        for (uint16_t oy = 0; oy < oh; oy++) {
            for (uint16_t ox = 0; ox < ow; ox++) {
                float max_val = -1e30f;

                for (uint8_t ky = 0; ky < k; ky++) {
                    for (uint8_t kx = 0; kx < k; kx++) {
                        int16_t ix = (int16_t)(ox * s + kx);
                        int16_t iy = (int16_t)(oy * s + ky);

                        if (ix >= 0 && ix < (int16_t)iw && iy >= 0 && iy < (int16_t)ih) {
                            float val = input[c * ih * iw + iy * iw + ix];
                            if (val > max_val) max_val = val;
                        }
                    }
                }

                output[c * oh * ow + oy * ow + ox] = max_val;
            }
        }
    }
}

static void relu(
    const float* input,
    const LayerMeta* meta,
    float* output
) {
    uint16_t total = meta->out_height * meta->out_width * meta->out_channels;

    for (uint16_t i = 0; i < total; i++) {
        output[i] = MAX(input[i], 0.0f);
    }
}

static void flatten(
    const float* input,
    const LayerMeta* meta,
    float* output
) {
    uint32_t total = meta->in_height * meta->in_width * meta->in_channels;

    for (uint32_t i = 0; i < total; i++) {
        output[i] = input[i];
    }
}

static void linear(
    const float* input,
    const uint8_t* blob_ptr,
    const LayerMeta* meta,
    float* output
) {
    const float* weights = get_weight_ptr(blob_ptr, meta->weight_offset);
    const float* bias = get_bias_ptr(blob_ptr, meta->bias_offset);

    uint16_t in_features = meta->in_height * meta->in_width * meta->in_channels;
    uint16_t out_features = meta->out_height * meta->out_width * meta->out_channels;

    for (uint16_t oc = 0; oc < out_features; oc++) {
        float sum = bias[oc];

        for (uint16_t ic = 0; ic < in_features; ic++) {
            sum += input[ic] * weights[oc * in_features + ic];
        }

        output[oc] = sum;
    }
}

static void softmax(
    const float* input,
    const LayerMeta* meta,
    float* output
) {
    uint16_t count = meta->out_height * meta->out_width * meta->out_channels;

    float max_val = input[0];
    for (uint16_t i = 1; i < count; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (uint16_t i = 0; i < count; i++) {
        output[i] /= sum;
    }
}

void tiny_predict(
    const uint8_t* blob_ptr,
    const float* input_ptr,
    float* scratch_a,
    float* scratch_b
) {
    const ModelHeader* header = (const ModelHeader*)blob_ptr;
    const LayerMeta* layers = (const LayerMeta*)(blob_ptr + sizeof(ModelHeader));

    float* prev = (float*)input_ptr;
    float* curr = scratch_a;
    float* temp;

    (void)scratch_b;

    for (uint32_t i = 0; i < header->num_layers; i++) {
        const LayerMeta* meta = &layers[i];

        switch (meta->type) {
            case LAYER_CONV2D:
                conv2d(prev, blob_ptr, meta, curr);
                break;
            case LAYER_MAXPOOL:
                maxpool2d(prev, meta, curr);
                break;
            case LAYER_RELU:
                relu(prev, meta, curr);
                break;
            case LAYER_FLATTEN:
                flatten(prev, meta, curr);
                break;
            case LAYER_LINEAR:
                linear(prev, blob_ptr, meta, curr);
                break;
            case LAYER_SOFTMAX:
                softmax(prev, meta, curr);
                break;
        }

        temp = prev;
        prev = curr;
        curr = temp;
    }
}

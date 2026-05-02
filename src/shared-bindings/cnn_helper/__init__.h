// Minimal wrapper around a tiny CNN inference engine.
#pragma once

#include <stddef.h>

void shared_modules_cnn_helper_load_model(const char *filename);
const float *shared_modules_cnn_helper_perform_inference(const void *input_buf, size_t input_len_bytes, size_t *out_len);
void shared_modules_cnn_helper_unload_model(void);

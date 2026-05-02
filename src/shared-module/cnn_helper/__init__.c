// Core implementation for shared-bindings/cnn_helper.

#include <string.h>

#include "py/misc.h"
#include "py/runtime.h"
#include "py/stream.h"

#include "extmod/vfs.h"

#include "shared-bindings/cnn_helper/__init__.h"

#include "tiny_inference.h"

static uint8_t *model_blob = NULL;
static float *scratch_a = NULL;
static float *scratch_b = NULL;
static size_t scratch_elems = 0;

static size_t expected_input_elems = 0;
static size_t output_elems = 0;
static uint32_t num_layers = 0;

static void cnn_helper_reset_state(void) {
    model_blob = NULL;
    scratch_a = NULL;
    scratch_b = NULL;
    scratch_elems = 0;
    expected_input_elems = 0;
    output_elems = 0;
    num_layers = 0;
}

static void cnn_helper_free_state(void) {
    if (model_blob != NULL) {
        m_free(model_blob);
    }
    if (scratch_a != NULL) {
        m_free(scratch_a);
    }
    if (scratch_b != NULL) {
        m_free(scratch_b);
    }
    cnn_helper_reset_state();
}

static mp_obj_t open_file_rb(const char *filename) {
    mp_obj_t args[2] = {
        mp_obj_new_str(filename, strlen(filename)),
        MP_OBJ_NEW_QSTR(MP_QSTR_rb),
    };
    return mp_vfs_open(MP_ARRAY_SIZE(args), &args[0], (mp_map_t *)&mp_const_empty_map);
}

static size_t read_entire_file(mp_obj_t file, uint8_t **out_buf) {
    int errcode = 0;

    mp_off_t end = mp_stream_seek(file, 0, MP_SEEK_END, &errcode);
    if (errcode != 0) {
        mp_raise_OSError(errcode);
    }
    if (end <= 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("empty model file"));
    }

    mp_stream_seek(file, 0, MP_SEEK_SET, &errcode);
    if (errcode != 0) {
        mp_raise_OSError(errcode);
    }

    size_t len = (size_t)end;
    uint8_t *buf = (uint8_t *)m_malloc_without_collect(len);

    size_t remaining = len;
    uint8_t *p = buf;
    while (remaining > 0) {
        mp_uint_t n = mp_stream_rw(file, p, remaining, &errcode, MP_STREAM_RW_READ);
        if (errcode != 0) {
            m_free(buf);
            mp_raise_OSError(errcode);
        }
        if (n == 0) {
            m_free(buf);
            mp_raise_ValueError(MP_ERROR_TEXT("short read"));
        }
        p += n;
        remaining -= n;
    }

    *out_buf = buf;
    return len;
}

void shared_modules_cnn_helper_load_model(const char *filename) {
    // Reloading is allowed: unload the previous model first.
    cnn_helper_free_state();

    mp_obj_t file = open_file_rb(filename);
    uint8_t *blob = NULL;
    size_t blob_len = read_entire_file(file, &blob);
    mp_stream_close(file);

    if (blob_len < sizeof(ModelHeader) + sizeof(LayerMeta)) {
        m_free(blob);
        mp_raise_ValueError(MP_ERROR_TEXT("invalid model"));
    }

    const ModelHeader *header = (const ModelHeader *)blob;
    const LayerMeta *layers = (const LayerMeta *)(blob + sizeof(ModelHeader));

    if (header->num_layers == 0) {
        m_free(blob);
        mp_raise_ValueError(MP_ERROR_TEXT("invalid model"));
    }

    size_t max_layer_elems = 0;
    for (uint32_t i = 0; i < header->num_layers; i++) {
        const LayerMeta *m = &layers[i];
        size_t elems = (size_t)m->out_height * (size_t)m->out_width * (size_t)m->out_channels;
        if (elems > max_layer_elems) {
            max_layer_elems = elems;
        }
    }

    expected_input_elems =
        (size_t)layers[0].in_height * (size_t)layers[0].in_width * (size_t)layers[0].in_channels;
    output_elems =
        (size_t)layers[header->num_layers - 1].out_height *
        (size_t)layers[header->num_layers - 1].out_width *
        (size_t)layers[header->num_layers - 1].out_channels;

    if (max_layer_elems == 0 || expected_input_elems == 0 || output_elems == 0) {
        m_free(blob);
        mp_raise_ValueError(MP_ERROR_TEXT("invalid model"));
    }

    // tiny_predict swaps buffers each layer; allocate two equally-sized float buffers.
    scratch_a = (float *)m_malloc_without_collect(max_layer_elems * sizeof(float));
    scratch_b = (float *)m_malloc_without_collect(max_layer_elems * sizeof(float));

    model_blob = blob;
    scratch_elems = max_layer_elems;
    num_layers = header->num_layers;
}

const float *shared_modules_cnn_helper_perform_inference(const void *input_buf, size_t input_len_bytes, size_t *out_len) {
    if (model_blob == NULL || scratch_a == NULL || scratch_b == NULL) {
        mp_raise_RuntimeError(MP_ERROR_TEXT("model not loaded"));
    }

    if (input_len_bytes != expected_input_elems * sizeof(float)) {
        mp_raise_ValueError(MP_ERROR_TEXT("wrong input size"));
    }
    if (expected_input_elems > scratch_elems) {
        mp_raise_RuntimeError(MP_ERROR_TEXT("model scratch too small"));
    }

    // Copy input bytes into scratch_b to avoid modifying the caller's buffer and
    // to ensure alignment for float reads.
    memcpy(scratch_b, input_buf, input_len_bytes);

    tiny_predict(model_blob, scratch_b, scratch_a, scratch_b);

    // Output buffer depends on layer parity (see tiny_predict's swap logic).
    const float *out = (num_layers & 1) ? scratch_a : scratch_b;
    *out_len = output_elems;
    return out;
}

void shared_modules_cnn_helper_unload_model(void) {
    cnn_helper_free_state();
}

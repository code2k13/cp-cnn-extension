// CircuitPython native module: cnn_helper

#include "py/misc.h"
#include "py/obj.h"
#include "py/runtime.h"

#include "shared-bindings/cnn_helper/__init__.h"

//| """Tiny CNN helper
//|
//| Load a binary model from the filesystem and run inference on float inputs.
//| """
//|

//| def load_model(filename: str) -> None:
//|     """Load a ``.bin`` model from the filesystem into RAM."""
//|     ...
//|
static mp_obj_t cnn_helper_load_model(mp_obj_t filename_obj) {
    const char *filename = mp_obj_str_get_str(filename_obj);
    shared_modules_cnn_helper_load_model(filename);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(cnn_helper_load_model_obj, cnn_helper_load_model);

//| def perform_inference(input_array: object) -> list[float]:
//|     """Run inference on a float buffer (for example an ``array.array('f')``).
//|
//|     Returns a list of floats for the output layer.
//|     """
//|     ...
//|
static mp_obj_t cnn_helper_perform_inference(mp_obj_t input_obj) {
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(input_obj, &bufinfo, MP_BUFFER_READ);

    size_t out_len = 0;
    const float *out = shared_modules_cnn_helper_perform_inference(bufinfo.buf, bufinfo.len, &out_len);

    mp_obj_t *items = m_new(mp_obj_t, out_len);
    for (size_t i = 0; i < out_len; i++) {
        items[i] = mp_obj_new_float(out[i]);
    }
    return mp_obj_new_list(out_len, items);
}
static MP_DEFINE_CONST_FUN_OBJ_1(cnn_helper_perform_inference_obj, cnn_helper_perform_inference);

//| def unload_model() -> None:
//|     """Free model weights and reset the inference engine state."""
//|     ...
//|
static mp_obj_t cnn_helper_unload_model(void) {
    shared_modules_cnn_helper_unload_model();
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(cnn_helper_unload_model_obj, cnn_helper_unload_model);

static const mp_rom_map_elem_t cnn_helper_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_cnn_helper) },
    { MP_ROM_QSTR(MP_QSTR_load_model), MP_ROM_PTR(&cnn_helper_load_model_obj) },
    { MP_ROM_QSTR(MP_QSTR_perform_inference), MP_ROM_PTR(&cnn_helper_perform_inference_obj) },
    { MP_ROM_QSTR(MP_QSTR_unload_model), MP_ROM_PTR(&cnn_helper_unload_model_obj) },
};

static MP_DEFINE_CONST_DICT(cnn_helper_module_globals, cnn_helper_module_globals_table);

const mp_obj_module_t cnn_helper_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&cnn_helper_module_globals,
};

#if CIRCUITPY_CNN_HELPER
MP_REGISTER_MODULE(MP_QSTR_cnn_helper, cnn_helper_module);
#endif

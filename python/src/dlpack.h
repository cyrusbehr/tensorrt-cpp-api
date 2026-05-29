#pragma once

// Minimal vendored subset of the DLPack ABI (https://github.com/dmlc/dlpack), the stable
// cross-framework tensor exchange used by PyTorch/CuPy/JAX. We ship only the structures the
// bindings need to PRODUCE and CONSUME the classic "dltensor" capsule; the ABI is stable, so
// vendoring avoids a build dependency on the dlpack headers.

#include <cstdint>

extern "C" {

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
} DLDeviceType;

typedef struct {
    int32_t device_type;
    int32_t device_id;
} DLDevice;

typedef enum {
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
    kDLBfloat = 4U,
    kDLBool = 6U,
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides; // nullptr => compact row-major (C-contiguous)
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;

} // extern "C"

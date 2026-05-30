#pragma once

// Umbrella header: includes the whole core public API in one shot. The optional sublibrary
// headers (preproc.h, opencv_interop.h) are intentionally NOT pulled in here -- include
// them explicitly when you build/link those modules.

#include "tensorrt_cpp_api/build_config.h"

#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/layout.h"
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/status.h"
#include "tensorrt_cpp_api/tensor.h"
#include "tensorrt_cpp_api/version.h"

#include "tensorrt_cpp_api/logger.h"

#include "tensorrt_cpp_api/allocator.h"
#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"

#include "tensorrt_cpp_api/build_options.h"
#include "tensorrt_cpp_api/calibrator.h" // self-gates out on TensorRT >= 11
#include "tensorrt_cpp_api/engine.h"
#include "tensorrt_cpp_api/engine_builder.h"
#include "tensorrt_cpp_api/engine_pool.h"
#include "tensorrt_cpp_api/quant.h"

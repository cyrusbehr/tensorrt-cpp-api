#!/usr/bin/env python3
"""Generate the tiny ONNX fixtures used by the engine tests.

These are deliberately trivial (a single Relu) so a build + inference roundtrip is fast
and the expected output is obvious (max(x, 0)). The generated .onnx files are committed
(see the tests/models/.gitignore exception) so CI does not need onnx installed.

Regenerate with:  python3 tests/models/gen_models.py
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
OPSET = 17


def relu_model(path, batch):
    """input[batch,3,8,8] -> Relu -> output[batch,3,8,8]. batch may be an int or 'N'."""
    shape = [batch, 3, 8, 8]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "relu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9  # broad runtime compatibility
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(HERE, path))
    print("wrote", path)


def conv_model(path):
    """input[1,3,8,8] -> Conv(3->4, 3x3, pad 1) -> output[1,4,8,8]. Has a quantizable
    weighted layer, so it is suitable for INT8 calibration (a lone Relu is folded away)."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])
    rng = np.random.RandomState(0)
    weight = numpy_helper.from_array(rng.randn(4, 3, 3, 3).astype(np.float32), "W")
    bias = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "B")
    node = helper.make_node("Conv", ["input", "W", "B"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "conv", [x], [y], initializer=[weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(HERE, path))
    print("wrote", path)


if __name__ == "__main__":
    relu_model("relu_1x3x8x8.onnx", 1)
    relu_model("relu_dynamic_batch.onnx", "N")
    conv_model("conv_1x3x8x8.onnx")

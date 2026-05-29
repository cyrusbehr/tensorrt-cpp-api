#!/usr/bin/env python3
"""Generate the tiny ONNX fixtures used by the engine tests.

These are deliberately trivial (a single Relu) so a build + inference roundtrip is fast
and the expected output is obvious (max(x, 0)). The generated .onnx files are committed
(see the tests/models/.gitignore exception) so CI does not need onnx installed.

Regenerate with:  python3 tests/models/gen_models.py
"""
import os

import onnx
from onnx import TensorProto, helper

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


if __name__ == "__main__":
    relu_model("relu_1x3x8x8.onnx", 1)
    relu_model("relu_dynamic_batch.onnx", "N")

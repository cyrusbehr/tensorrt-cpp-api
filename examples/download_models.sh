#!/usr/bin/env bash
# Fetch/export the ONNX models the reference examples use, into ../models/.
#   classification : MobileNetV2 (ONNX Model Zoo)              -> mobilenetv2-7.onnx
#   detection      : YOLOv8n     (ultralytics export)          -> yolov8n.onnx
#   segmentation   : DeepLabV3-MobileNetV3 (torchvision export)-> deeplabv3_mobilenetv3.onnx
#
# Classification needs only curl. Detection/segmentation need Python with `ultralytics` and
# `torch`/`torchvision` respectively (CPU is fine):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install ultralytics onnx onnxscript
set -euo pipefail
cd "$(dirname "$0")/../models"

echo "[1/3] classification: mobilenetv2-7.onnx"
if [ ! -f mobilenetv2-7.onnx ]; then
    curl -fL -o mobilenetv2-7.onnx \
        "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
fi

echo "[2/3] detection: yolov8n.onnx"
if [ ! -f yolov8n.onnx ]; then
    if command -v yolo >/dev/null 2>&1; then
        yolo export model=yolov8n.pt format=onnx opset=12 imgsz=640
    else
        echo "  ! 'yolo' (ultralytics) not found. Install it, then:"
        echo "      yolo export model=yolov8n.pt format=onnx opset=12 imgsz=640"
    fi
fi

echo "[3/3] segmentation: deeplabv3_mobilenetv3.onnx"
if [ ! -f deeplabv3_mobilenetv3.onnx ]; then
    python3 - <<'PY' || echo "  ! torchvision export failed; install torch/torchvision/onnx/onnxscript"
import torch, torchvision
m = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT").eval()
class Wrap(torch.nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x): return self.inner(x)["out"]
torch.onnx.export(Wrap(m), torch.randn(1, 3, 520, 520), "deeplabv3_mobilenetv3.onnx",
                  opset_version=12, input_names=["input"], output_names=["out"], dynamo=False)
print("  exported deeplabv3_mobilenetv3.onnx")
PY
fi

echo "done. models in $(pwd):"
ls -1 *.onnx 2>/dev/null || true

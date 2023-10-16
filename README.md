[![Stargazers][stars-shield]][stars-url]
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/cyrusbehr/tensorrt-cpp-api">
    <img width="40%" src="images/logo.png" alt="logo">
  </a>

  <h3 align="center">TensorRT C++ API Tutorial</h3>

  <p align="center">
    <b>
    How to use TensorRT C++ API for high performance GPU machine-learning inference.
    </b>
    <br />
    Supports models with single / multiple inputs and single / multiple outputs with batching.
    <br />
    <br />
    <a href="https://www.youtube.com/watch?v=kPJ9uDduxOs">Project Overview Video</a>
    .
    <a href="https://youtu.be/Z0n5aLmcRHQ">Code Deep-Dive Video</a>
  </p>
</p>

# TensorRT C++ Tutorial
*I read all the NVIDIA TensorRT docs so that you don't have to!*

This project demonstrates how to use the TensorRT C++ API for high performance GPU inference on image data. It covers how to do the following:
- How to install TensorRT 8 on Ubuntu 20.04.
- How to generate a TensorRT engine file optimized for your GPU.
- How to specify a simple optimization profile.
- How to run FP32, FP16, or INT8 precision inference. 
- How to read / write data from / into GPU memory and work with GPU images.
- How to use cuda stream to run async inference and later synchronize. 
- How to work with models with static and dynamic batch sizes.
- How to work with models with single or multiple output tensors.
- How to work with models with multiple inputs.
- Includes a [Video walkthrough](https://youtu.be/Z0n5aLmcRHQ) where I explain every line of code.
- The code can be used as a base for any model which takes a fixed size image / images as input, including [Insightface](https://github.com/deepinsight/insightface) [ArcFace](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface), [YoloV8](https://github.com/ultralytics/ultralytics), [SCRFD](https://insightface.ai/scrfd) face detection.
  - You will just need to implement the appropriate post-processing code.
- TODO: Add support for models with dynamic input shapes.
- TODO: Add support for Windows

## Getting Started
The following instructions assume you are using Ubuntu 20.04.
You will need to supply your own onnx model for this sample code or you can download the sample model (see Sanity Check section below). 

### Prerequisites
- Tested and working on Ubuntu 20.04
- Install CUDA, instructions [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).
  - Recommended >= 11.8 
- Install cudnn, instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download).
  - Recommended >= 8
- `sudo apt install build-essential`
- `sudo snap install cmake --classic`
- Install OpenCV with cuda support. To compile OpenCV from source, run the `build_opencv.sh` script provided in `./scripts/`.
  - If you use the provided script and you have installed cuDNN to a non-standard location, you must modify the `CUDNN_INCLUDE_DIR` and `CUDNN_LIBRARY` variables in the script.  
  - Recommended >= 4.8
- Download TensorRT 8 from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
  - Recommended >= 8.6
  - Required >= 8.6 
- Navigate to the `CMakeLists.txt` file and replace the `TODO` with the path to your TensorRT installation.

### Building the Library
- `mkdir build`
- `cd build`
- `cmake ..`
- `make -j$(nproc)`

### Running the Executable
- Navigate to the build directory
- Run the executable and provide the path to your onnx model.
- ex. `./run_inference_benchmark ../models/yolov8n.onnx`
  - Note: See sanity check section below for instructions on how to obtain the yolov8n model.  
- The first time you run the executable for a given model and options, a TensorRT engine file will be built from your onnx model. This process is fairly slow and can take 5+ minutes for some models (ex. yolo models). 

### Sanity Check
- To perform a sanity check, download the `YOLOv8n` model from [here](https://github.com/ultralytics/ultralytics#models).
- Next, convert it from pytorch to onnx using the following script:
  - You will need to run `pip3 install ultralytics` first.
     
```python
from ultralytics import YOLO
model = YOLO("./yolov8n.pt")
model.fuse()
model.info(verbose=False)  # Print model information
model.export(format="onnx", opset=12) # Export the model to onnx using opset 12
```

- Place the resulting onnx model, `yolov8n.onnx`, in the `./models/` directory. 
- Running inference using said model and the image located in `./inputs/team.jpg` should produce the following feature vector:
  - Note: The feature vector will not be identical (but very similar) as [TensorRT is not deterministic](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#determinism). 
```text
3.41113 16.5312 20.8828 29.8984 43.7266 54.9609 62.0625 65.8594 70.0312 72.9531 ...
```

### INT8 Inference
Enabling INT8 precision can further speed up inference at the cost of accuracy reduction due to reduced dynamic range. 
For INT8 precision, the user must supply calibration data which is representative of real data the model will see. 
It is advised to use 1K+ calibration images. To enable INT8 inference with the YoloV8 sanity check model, the following steps must be taken:
-  Change `options.precision = Precision::FP16;` to `options.precision = Precision::INT8;` in `main.cpp`
- `options.calibrationDataDirectoryPath = "";` must be changed in `main.cpp` to specify path containing calibration data. 
  - If using the YoloV8 model, it is advised to used the COCO validation dataset, which can be downloaded with `wget http://images.cocodataset.org/zips/val2017.zip`
- Make sure the resizing code in the `Int8EntropyCalibrator2::getBatch` method in `engine.cpp` (see `TODO`) is correct for your model.
  - If using the YoloV8 model, the preprocessing code is correct and does not need to be changed.
- Recompile, run the executable. 
- The calibration cache will be written to disk (`.calibration` extension) so that on subsequent model optimizations it can be reused. If you'd like to regenerate the calibration data, you must delete this cache file.  
- If you get an "out of memory in function allocate" error, then you must reduce `Options.calibrationBatchSize` so that the entire batch can fit in your GPU memory. 

### Benchmarks
Benchmarks run on RTX 3050 Ti Laptop GPU, 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz.

| Model   | Precision | Batch Size | Avg Inference Time |
|---------|-----------|------------|--------------------|
| yolov8n | FP32      | 1          | 4.732 ms           |
| yolov8n | FP16      | 1          | 2.493 ms           |
| yolov8n | INT8      | 1          | 2.009 ms           |
| yolov8x | FP32      | 1          | 76.63 ms           |
| yolov8x | FP16      | 1          | 25.08 ms           |
| yolov8x | INT8      | 1          | 11.62 ms           |

### Sample Integration
Wondering how to integrate this library into your project? Or perhaps how to read the outputs of the YoloV8 model to extract meaningful information? 
If so, check out my newest project, [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP), which demonstrates how to use the TensorRT C++ API to run YoloV8 inference (supports object detection, semantic segmentation, and body pose estimation). It makes use of this project in the backend!

### Understanding the Code
- The bulk of the implementation is in `src/engine.cpp`. I have written lots of comments all throughout the code which should make it easy to understand what is going on. 
- You can also check out my [deep-dive video](https://youtu.be/Z0n5aLmcRHQ) in which I explain every line of code.

### How to Debug
- If you have issues creating the TensorRT engine file from the onnx model, navigate to `src/engine.cpp` and change the log level by changing the severity level to `kVERBOSE` and rebuild and rerun. This should give you more information on where exactly the build process is failing.

### Show your Appreciation
If this project was helpful to you, I would appreciate if you could give it a star. That will encourage me to ensure it's up to date and solve issues quickly. I also do consulting work if you require more specific help. Connect with me on [LinkedIn](https://www.linkedin.com/in/cyrus-behroozi/). 

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://ltetrel.github.io/"><img src="https://avatars.githubusercontent.com/u/37963074?v=4?s=100" width="100px;" alt="Loic Tetrel"/><br /><sub><b>Loic Tetrel</b></sub></a><br /><a href="https://github.com/cyrusbehr/tensorrt-cpp-api/commits?author=ltetrel" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

### Changelog

**V4.0**

- Added support for INT8 precision.


**V3.0**

- Implementation has been updated to use TensorRT 8.6 API (ex. `IExecutionContext::enqueueV3()`). 
- Executable has renamed from `driver` to `run_inference_benchmark` and now must be passed path to onnx model as command line argument. 
- Removed `Options.doesSupportDynamicBatchSize`. Implementation now auto-detects supported batch sizes.
- Removed `Options.maxWorkspaceSize`. Implementation now does not limit GPU memory during model constructions, allowing implementation to use as much of memory pool as is available for intermediate layers.

**v2.2**

- Serialize model name as part of engine file. 

**V2.1**

- Added support for models with multiple inputs. Implementation now supports models with single inputs, multiple inputs, single outputs, multiple outputs, and batching. 

**V2.0**

- Requires OpenCV cuda to be installed. To install, follow instructions [here](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7).
- `Options.optBatchSizes` has been removed, replaced by `Options.optBatchSize`.
- Support models with more than a single output (ex. SCRFD).  
- Added support for models which do not support batch inference (first input dimension is fixed).
- More error checking.
- Fixed a bunch of common issues people were running into with the original V1.0 version.
- Remove whitespace from GPU device name 

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[stars-shield]: https://img.shields.io/github/stars/cyrusbehr/tensorrt-cpp-api.svg?style=flat-square
[stars-url]: https://github.com/cyrusbehr/tensorrt-cpp-api/stargazers
[issues-shield]: https://img.shields.io/github/issues/cyrusbehr/tensorrt-cpp-api.svg?style=flat-square
[issues-url]: https://github.com/cyrusbehr/tensorrt-cpp-api/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/cyrus-behroozi/

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
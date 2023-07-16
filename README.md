[![Stargazers][stars-shield]][stars-url]
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
**I read all the NVIDIA TensorRT docs so that you don't have to!**

This project demonstrates how to use the TensorRT C++ API for high performance GPU inference on image data. It covers how to do the following:
- How to install TensorRT 8 on Ubuntu 20.04.
- How to generate a TRT engine file optimized for your GPU.
- How to specify a simple optimization profile.
- How to read / write data from / into GPU memory and work with GPU images.
- How to use cuda stream to run async inference and later synchronize. 
- How to work with models with static and dynamic batch sizes.
- **New:** Supports models with multiple output tensors (and even works with batching).
- **New:** Supports models with multiple inputs.
- **New:** New [video walkthrough](https://youtu.be/Z0n5aLmcRHQ) where I explain every line of code.
- The code can be used as a base for many models, including [Insightface](https://github.com/deepinsight/insightface) [ArcFace](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface), [YoloV7](https://github.com/WongKinYiu/yolov7), [SCRFD](https://insightface.ai/scrfd) face detection, and many other single / multiple input - single / multiple output models. You will just need to implement the appropriate post-processing code.
- TODO: Add support for models with dynamic input shapes.

## Getting Started
The following instructions assume you are using Ubuntu 20.04.
You will need to supply your own onnx model for this sample code, or you can download the sample model (see Sanity Check section below). 

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
- ex. `./run_inference_benchmark ../models/arcfaceresnet100-8.onnx`
  - Note: See sanity check section below for instructions on how to obtain the arcface model.  

### Sanity Check
- To perform a sanity check, download the following [ArcFace model](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface) from [here](https://github.com/onnx/models/blob/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx) and place it in the `./models/` directory.
- Running inference using said model and the image located in `./inputs/face_chip.jpg` should produce the following feature vector:
  - Note: The feature vector will not be identical (but very similar) as [TensorRT is not deterministic](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#determinism). 
```text
-0.0548096 -0.0994873 0.176514 0.161377 0.226807 0.215942 -0.296143 -0.0601807 0.240112 -0.18457 ...
```

### Sample Integration
Wondering how to integrate this library into your project? Or perhaps how to read the outputs to extract meaningful information? 
If so, check out my newest project, [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP), which demonstrates how to use the TensorRT C++ API to run YoloV8 inference (supports segmentation). It makes use of this project in the backend!

### Understanding the Code
- The bulk of the implementation is in `src/engine.cpp`. I have written lots of comments all throughout the code which should make it easy to understand what is going on. 
- You can also check out my [deep-dive video](https://youtu.be/Z0n5aLmcRHQ) in which I explain every line of code.

### How to Debug
- If you have issues creating the TensorRT engine file from the onnx model, navigate to `src/engine.cpp` and change the log level by changing the severity level to `kVERBOSE` and rebuild and rerun. This should give you more information on where exactly the build process is failing.

### Show your Appreciation
If this project was helpful to you, I would appreciate if you could give it a star. That will encourage me to ensure it's up to date and solve issues quickly. I also do consulting work if you require more specific help. Connect with me on [LinkedIn](https://www.linkedin.com/in/cyrus-behroozi/). 

### Changelog

**V3.0**

- Implementation has been updated to use TensorRT 8.6 API. 
- Implementation has been updated to use `IExecutionContext::enqueueV3()` instead of now deprecated `IExecutionContext::enqueueV2()`.
- Executable has renamed from `driver` to `run_inference_benchmark` and now must be passed path to onnx model as command line argument. 
- TODO Cyrus: Update this. 

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

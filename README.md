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
    How to use TensorRT C++ API for high performance GPU inference.
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
This project demonstrates how to use the TensorRT C++ API for high performance GPU inference. It covers how to do the following:
- How to install TensorRT 8 on Ubuntu 20.04
- How to generate a TRT engine file optimized for your GPU
- How to specify a simple optimization profile
- How to read / write data from / into GPU memory and work with GPU images.
- How to use cuda stream to run async inference and later synchronize. 
- How to work with models with static and dynamic batch sizes.
- **New:** Supports models with multiple outputs (and even works with batching!).
- **New:** Supports models with multiple inputs.
- **New:** New [video walkthrough](https://youtu.be/Z0n5aLmcRHQ) where I explain every line of code.
- The code can be used as a base for many models, including [Insightface](https://github.com/deepinsight/insightface) [ArcFace](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface), [YoloV7](https://github.com/WongKinYiu/yolov7), [SCRFD](https://insightface.ai/scrfd) face detection, and many other single / multiple input - single / multiple output models. You will just need to implement the appropriate post-processing code.
- TODO: Add support for models with dynamic input shapes.

## Getting Started
The following instructions assume you are using Ubuntu 20.04.
You will need to supply your own onnx model for this sample code, or you can download the sample model (see Sanity Check section below). Ensure to specify a dynamic batch size when exporting the onnx model if you would like to use batching. If not, you will need to set `Options.doesSupportDynamicBatchSize` to false.

### Prerequisites
- `sudo apt install build-essential`
- `sudo apt install python3-pip`
- `pip3 install cmake`
- Install OpenCV with cuda support. Instructions can be found [here](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7).
- Download TensorRT 8 from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
- Extract, and then navigate to the `CMakeLists.txt` file and replace the `TODO` with the path to your TensorRT installation.

### Building the library
- `mkdir build && cd build`
- `cmake ..`
- `make -j$(nproc)`

### Sanity check
- To perform a sanity check, download the following [ArcFace model](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface) from [here](https://github.com/onnx/models/blob/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx) and place it in the `./models` directory.
- Make sure `Options.doesSupportDynamicBatchSize` is set to `false` before passing it the `Options` to the `Engine` constructor on [this](https://github.com/cyrusbehr/tensorrt-cpp-api/blob/003b72ba032d40afee241adeb7ebe7ca1ea685ca/src/main.cpp#L12) line.
- Uncomment the code for printing out the feature vector at the bottom of `./src/main.cpp`.
- Running inference using said model and the image located in `inputs/face_chip.jpg` should produce the following feature vector:
```text
-0.0548096 -0.0994873 0.176514 0.161377 0.226807 0.215942 -0.296143 -0.0601807 0.240112 -0.18457 ...
```

### Understanding the code
- The bulk of the implementation is in `src/engine.cpp`. I have written lots of comments all throughout the code which should make it easy to understand what is going on. 

### How to debug
- If you have having issues creating the TensorRT engine file from the onnx model, I would advise using the `trtexec` command line tool (comes packaged in the TensorRT download bundle in the `/bin` directory). It will provide you with more debug information.

### Show your appreciation
If this project was helpful to you, I would appreicate if you could give it a star. That will encourage me to ensure it's up to date and solve issues quickly. 

### Changelog

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

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
    How to use TensorRT C++ API for high performance GPU inference.
    <br />
    A Venice Computer Vision Presentation
    <br />
    <br />
    <a href="https://www.youtube.com/watch?v=kPJ9uDduxOs">Video Presentation</a>
    .
    <a href="https://docs.google.com/presentation/d/1vOw4fCzCbD-jJZCE3cjsOUq83UGlLA4g/edit?usp=share_link&ouid=110822293658782092853&rtpof=true&sd=true">Presentation Slides</a>
    <!-- <a href="https://social.trueface.ai/34gcD2q">Blog Post</a> -->
    ·
    <a href="https://venicecomputervision.com/">Venice Computer Vision</a>
  </p>
</p>

# TensorRT C++ Tutorial
This project demonstrates how to use the TensorRT C++ API for high performance GPU inference. It covers how to do the following:
- How to install TensorRT on Ubuntu 20.04
- How to generate a TRT engine file optimized for your GPU
- How to specify a simple optimization profile
- How to read / write data from / into GPU memory and work with GPU images.
- How to use cuda stream to run async inference and later synchronize. 
- How to work with models with static and dynamic batch sizes.
- **New:** Supports models with multiple outputs (and even works with batching!).
- **New:** Supports models with multiple inputs.
- The code can be used as a base for many models, including [Insightface](https://github.com/deepinsight/insightface) [ArcFace](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface), [YoloV7](https://github.com/WongKinYiu/yolov7), [SCRFD](https://insightface.ai/scrfd) face detection, and many other single / multiple input - single / multiple output models. You will just need to implement the appropriate post-processing code.

## Getting Started
The following instructions assume you are using Ubuntu 20.04.
You will need to supply your own onnx model for this sample code, or you can download the sample model (see Sanity Check section below). Ensure to specify a dynamic batch size when exporting the onnx model if you would like to use batching. If not, you will need to set `Options.doesSupportDynamicBatchSize` to false.

### Prerequisites
- Install OpenCV with cuda support. Instructions can be found [here](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7).
- `sudo apt install build-essential`
- `sudo apt install python3-pip`
- `pip3 install cmake`
- Download TensorRT from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
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

### How to debug
- If you have having issues creating the TensorRT engine file from the onnx model, I would advise using the `trtexec` command line tool (comes packaged in the TensorRT download bundle in the `/bin` directory). It will provide you with more debug information.

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

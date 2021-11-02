# tensorrt-cpp-api
A tutorial project demonstrating how to use the TensorRT C++ API

- Explain that the model must have a dynamic batch size when exported from onnx.
- Explain motiviation for this project is shitty docs. 
- They need to provide their own model sadly. 


[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/cyrusbehr/sdk_design">
    <img width="30%" src="images/logo.svg" alt="logo">
  </a>

  <h3 align="center">SDK Design</h3>

  <p align="center">
    How to Design a Language Agnostic SDK for Cross Platform Deployment and Maximum Extensibility
    <br />
    A Venice Computer Vision Presentation
    <br />
    <br />
    <a href="https://www.youtube.com/watch?v=R4KH2V5pTLI&feature=youtu.be">Video Presentation</a>
  ·
  <a href="https://docs.google.com/presentation/d/1hluqipWiqk3ACReqXf_7tpwtTfCvFcvSwOj4sf3Su6A/edit?usp=sharing">Presentation Slides</a>
    ·
    <a href="https://social.trueface.ai/34gcD2q">Blog Post</a>
    ·
    <a href="https://venicecomputervision.com/">Venice Computer Vision</a>
  </p>
</p>

# SDK Design
This project demonstrates how to build a language agnostic SDK for cross platform deployment and maximum extensibility. It covers how to do the following:
- Build a basic face detection computer vision library in C++
- Compile / cross compile the library for amd64, arm64, arm32
- Package that library and its dependencies as a single static library
- Add unit tests
- Set up a CI pipeline
- Write python bindings for our library
- Generate documentation directly from our API

Please refer to the [blog post](https://social.trueface.ai/34gcD2q) for a detailed tutorial and explanation of all the components of this project.

![alt text](./images/face_detection.jpeg)

## Getting Started
The following instructions assume you are using Ubuntu 18.04

### Prerequisites
- `sudo apt install build-essential`
- `sudo apt-get install g++-aarch64-linux-gnu`
- `sudo apt-get install gcc-arm-linux-gnueabihf binutils-arm-linux-gnueabihf g++-arm-linux-gnueabihf`
- `sudo apt install python3.8`
- `sudo apt install python3-pip`
- `pip3 install cmake`
- `sudo apt-get install doxygen`
- `sudo apt-get install wget`
- `sudo apt-get install zip`

### Install 3rd Party Libraries
Navigate to `3rdparty` then run the following:
- `./build_catch.sh`
- `./build_pybind11.sh`
- `./build_ncnn.sh`
- `./build_opencv.sh`

### Building the library
- `mkdir build && cd build`
- `cmake ..`
- `make -j$(nproc)`
- `make install`

The outputs will be copied to `dist`

### Cross comping for arm32
- `mkdir build && cd build`
- `cmake -D BUILD_ARM32=ON ..`
- `make -j$(nproc)`
- `make install`

### Cross compiling for arm64
- `mkdir build && cd build`
- `cmake -D BUILD_ARM64=ON ..`
- `make -j$(nproc)`
- `make install`


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[stars-shield]: https://img.shields.io/github/stars/cyrusbehr/sdk_design.svg?style=flat-square
[stars-url]: https://github.com/cyrusbehr/sdk_design/stargazers
[issues-shield]: https://img.shields.io/github/issues/cyrusbehr/sdk_design.svg?style=flat-square
[issues-url]: https://github.com/cyrusbehr/sdk_design/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/cyrus-behroozi/

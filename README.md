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
    <a href="https://docs.google.com/presentation/d/1pUnB2zvz2THyUaRuNmyoslBD2ESQijxKcx2iM8GMvw4/edit?usp=sharing">Presentation Slides</a>
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
- How to read / write data from / into GPU memory
- How to run synchronous inference
- How to work with models with dynamic batch sizes


## Getting Started
The following instructions assume you are using Ubuntu 20.04.
You will need to supply your own onnx model for this sample code. Ensure to specify a dynamic batch size when exporting the onnx model. 

### Prerequisites
- `sudo apt install build-essential`
- `sudo apt install python3-pip`
- `pip3 install cmake`
- Download TensorRT from here: https://developer.nvidia.com/nvidia-tensorrt-8x-download
- Extract, and then navigate to the `CMakeLists.txt` file and replace the `TODO` with the path to your TensorRT installation

### Building the library
- `mkdir build && cd build`
- `cmake ..`
- `make -j$(nproc)`

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[stars-shield]: https://img.shields.io/github/stars/cyrusbehr/tensorrt-cpp-api.svg?style=flat-square
[stars-url]: https://github.com/cyrusbehr/tensorrt-cpp-api/stargazers
[issues-shield]: https://img.shields.io/github/issues/cyrusbehr/tensorrt-cpp-api.svg?style=flat-square
[issues-url]: https://github.com/cyrusbehr/tensorrt-cpp-api/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/cyrus-behroozi/

# ApraPipes
A pipeline framework for developing video and image processing applications. Supports multiple GPUs and Machine Learning tooklits

# Build and Run Tests
Tested on Ubuntu 18.04 and Jetson Boards

## Ubuntu 18.04 x64

### Prerequisites
* Install [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804)
* Download [Nvidia Video Codec SDK v10](https://developer.nvidia.com/designworks/video_codec_sdk/downloads/v10) and extract to `thirdparty` directory. Make sure `thirdparty/Video_Codec_SDK_10.0.26/Interface` and `thirdparty/Video_Codec_SDK_10.0.26/Lib` exist
* CMake minimum version 3.14 - Follow [this article](https://anglehit.com/how-to-install-the-latest-version-of-cmake-via-command-line/) to update cmake

### Build
* `chmod +x build_linux_x64.sh`
* `./build_linux_x64.sh`

Build can take ~2 hours depending on the machine configuration.
This project uses [hunter package manager](https://github.com/cpp-pm/hunter).

## Jetson boards - Nano, TX2, NX, AGX
* Setup the board with Jetpack 4.4 
* `chmod +x build_jetson.sh`
* `./build_jetson.sh`

## Run Tests
* run all tests  `_build/aprapipesut`
* run one test `_build/aprapipesut --run_test=filenamestrategy_tests/boostdirectorystrategy`

This project uses boost tests for unit tests.

## Documentation
* Open `docs/build/html/index.html` using Google Chrome
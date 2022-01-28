
# ApraPipes
A pipeline framework for developing video and image processing applications. Supports multiple GPUs and Machine Learning tooklits

# Build and Run Tests
Tested on Ubuntu 18.04 and Jetson Boards

## Setup
* Clone with submodules
```
git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
```

## Prerequisites
* Run ```sudo apt-get update && sudo apt-get install build-essential```  to get latest build tools
* CMake minimum version 3.14 - Follow [this article](https://anglehit.com/how-to-install-the-latest-version-of-cmake-via-command-line/) to update cmake
* ffmpeg
```
sudo apt install yasm -y
cd thirdparty/ffmpeg
./configure --enable-pic
make -j"$(($(nproc) - 1))"
```
* zxing
```
cd thirdparty/zxing-cpp
chmod +x build.sh
./build.sh
```

## Ubuntu 18.04 x64

### Prerequisites
* Install [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804)
* Download [Nvidia Video Codec SDK v10](https://developer.nvidia.com/designworks/video_codec_sdk/downloads/v10) and extract to `thirdparty` directory. Make sure `thirdparty/Video_Codec_SDK_10.0.26/Interface` and `thirdparty/Video_Codec_SDK_10.0.26/Lib` exist

### Build

* `chmod +x build_linux_x64.sh`
* `./build_linux_x64.sh`

Build can take ~2 hours depending on the machine configuration.
This project uses [hunter package manager](https://github.com/cpp-pm/hunter).

## Jetson boards - Nano, TX2, NX, AGX

### Prerequisites
* Setup the board with [Jetpack 4.4](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)
* sudo apt-get install libncurses5-dev

### Build
* `chmod +x build_jetson.sh`
* `./build_jetson.sh`

Build can take ~12 hours on Jetson Nano. 

## Run Tests
* list all tests `_build/aprapipesut --list_content`
* run all tests  `_build/aprapipesut`
* run one test `_build/aprapipesut --run_test=filenamestrategy_tests/boostdirectorystrategy`
* run one test with arguments `_build/aprapipesut --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera`
  * Look at the unit_tests/params_test to check for sample usage of parameters in test code

This project uses boost tests for unit tests.

## Update Submodules
```
git submodule update --init --recursive
```

## Documentation
* Open `docs/build/html/index.html` using Google Chrome

### To regenerate documentation
```
To build docs
apt-install get python-sphinx 
pip install sphinx-rtd-theme
cd docs
make html
```

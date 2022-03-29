
# ApraPipes
A pipeline framework for developing video and image processing applications. Supports multiple GPUs and Machine Learning toolkits. More details can be found here https://apra-labs.github.io/ApraPipes.

# Build and Run Tests
Tested on Ubuntu 18.04, Jetson Boards and Windows 11 x64 Visual Studio 2017 Community No Cuda

## Setup
* Clone with submodules
```
git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
```

### Prerequisites for CUDA 
* Install [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804)
* Download [Cudnn](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-102) and extract where cuda is installed.
* Rename the file base/vcpk.json to base/vcpkg.json.bkp and base/vcpkg.cuda.json to base/vcpkg.json



## Prerequisites Windows
* Install Visual Studio 2017 Community 
  * Install Desktop development C++
  * .NET Desktop development
  * Universal Windows Development Platform
* Install CMake 3.22.1
* Download [Nvidia Video Codec SDK v10](https://developer.nvidia.com/designworks/video_codec_sdk/downloads/v10) and extract to `thirdparty` directory. Make sure `thirdparty/Video_Codec_SDK_10.0.26/Interface` and `thirdparty/Video_Codec_SDK_10.0.26/Lib` exist
* Run bootstrap-vcpkg.bat in the vcpkg/ directory
* Run 
```
vcpkg.exe integrate install
```

## Build windows

### Without Cuda
```
build_windows.bat
```

### With Cuda
```
build_windows_cuda.bat
```


### Run Tests
* list all tests
```
_build/BUILD_TYPE/aprapipesut.exe --list_content
```
* run all tests  
```
_build/BUILD_TYPE/aprapipesut.exe
```
* run one test 
```
_build/BUILD_TYPE/aprapipesut.exe --run_test=filenamestrategy_tests/boostdirectorystrategy
```
* run one test with arguments 
```
_build/BUILD_TYPE/aprapipesut.exe --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera
```
  * Look at the unit_tests/params_test to check for sample usage of parameters in test code


## Ubuntu 18.04 x64
###  Prerequisites 
* Run the following to get latest build tools
```
sudo apt-get update && sudo apt-get -y install   autoconf   automake  autopoint  build-essential  git-core   libass-dev   libfreetype6-dev   libgnutls28-dev   libmp3lame-dev   libsdl2-dev   libtool   libva-dev   libvdpau-dev   libvorbis-dev   libxcb1-dev   libxcb-shm0-dev   libxcb-xfixes0-dev   meson   ninja-build   pkg-config   texinfo   wget   yasm   zlib1g-dev   nasm   gperf  bison curl zip unzip tar
```  
* CMake minimum version 3.22 - Follow [this article](https://anglehit.com/how-to-install-the-latest-version-of-cmake-via-command-line/) to update cmake
* Run `./bootstrap-vcpkg.sh` in vcpkg/ directory
* Run `./vcpkg integrate install`


### Build Linux

* `chmod +x build_linux_x64.sh` or `chmod +x build_linux_no_cuda.sh`
* `./build_linux_x64.sh` or `./build_linux_no_cuda.sh` depending on previous step. No Cuda as the name suggests will not build the Nvidia Cuda GPU Modules

Build can take ~2 hours depending on the machine configuration.

## Jetson boards - Nano, TX2, NX, AGX

### Prerequisites
* Setup the board with [Jetpack 4.4](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)
* run the following 
```
sudo apt-get update && sudo apt-get -y install libncurses5-dev ninja-build nasm curl libudev-dev && sudo snap install cmake --classic
```
* append following lines to ~/.bashrc
```
export VCPKG_FORCE_SYSTEM_BINARIES=1
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
* reload ~/.bashrc:
```
source ~/.bashrc:
```
* Run `./bootstrap-vcpkg.sh` in vcpkg/ directory
* Run `./vcpkg integrate install`
* Use the correct vcpkg for Jetson:
```
mv base/vcpkg.json base/vcpkg.json.bkp && mv base/vcpkg.jetson.json base/vcpkg.json
```

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

## Update Documentation
If any changes are made in the documentation i.e. in /docs/source folder, the docs must be regenerated again follwing the steps given below. New contents from the /docs/build directory should be committed.

### To regenerate documentation
```
To build docs
apt-install get python-sphinx 
pip install sphinx-rtd-theme
cd docs
make html
```

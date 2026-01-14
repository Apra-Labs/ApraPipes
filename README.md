<p align="center">
  <img src="./data/ReadMe Images/ApraPipes.png" alt="Header Image">
</p>

# ApraPipes
A pipeline framework for developing video and image processing applications. Supports multiple GPUs and Machine Learning toolkits.
Learn more about ApraPipes [here](https://deepwiki.com/Apra-Labs/ApraPipes).

## Declarative Pipelines & Node.js Support

Build video processing pipelines using **JSON configuration** or **JavaScript**:

```json
{
  "modules": {
    "source": { "type": "FileReaderModule", "props": { "path": "video.mp4" } },
    "decoder": { "type": "Mp4ReaderSource" },
    "sink": { "type": "FileSinkModule" }
  },
  "connections": [
    { "from": "source", "to": "decoder" },
    { "from": "decoder", "to": "sink" }
  ]
}
```

**Features:**
- 50+ registered modules (sources, transforms, encoders, AI inference)
- GPU acceleration with automatic CPU↔GPU memory bridging
- Node.js addon for JavaScript/TypeScript applications
- CLI tool for running and validating pipelines

See [examples/](./examples/) and [Pipeline Author Guide](./docs/declarative-pipeline/PIPELINE_AUTHOR_GUIDE.md) to get started.

## Build Status
ApraPipes is automatically built and tested on Windows, Linux (x64 and ARM64), and macOS.

|Platform|OS|Version|Test Results|Workflow|
|--------|--|-------|------------|--------|
|x86_64|Windows|2022|<div align="right">NOCUDA: [![NOCUDA](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows.svg)<br>CUDA: [![CUDA](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows-CUDA.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows-CUDA.svg)</div>|<div align="right">[![CI-Windows](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Windows.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Windows.yml)</div>|
|x86_64|Linux|22.04|<div align="right">NOCUDA: [![NOCUDA](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux.svg)<br>CUDA: [![CUDA](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux-CUDA.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux-CUDA.svg)<br>Docker: [![Docker](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux-Docker.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux-Docker.svg)</div>|<div align="right">[![CI-Linux](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux.yml)</div>|
|x86_64|macOS|15.0+|<div align="right">NOCUDA: [![NOCUDA](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_MacOSX.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_MacOSX.svg)</div>|<div align="right">[![CI-MacOSX](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-MacOSX-NoCUDA.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-MacOSX-NoCUDA.yml)</div>|
|ARM64|Linux (Jetson)|20.04|<div align="right">CUDA: [![CUDA](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux_ARM64.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux_ARM64.svg)</div>|<div align="right">[![CI-Linux-ARM64](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-ARM64.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-ARM64.yml)</div>|

## Getting Started with ApraPipes

<details>
  <summary>Please select your Operating System</summary>
  <ul>
    <li><a href="#windows">Windows</a></li>
    <li><a href="#linux">Linux</a></li>
    <li><a href="#macos">macOS</a></li>
    <li><a href="#jetson">Jetson</a></li>
    <li><a href="#docker">Docker</a></li>
  </ul>
</details>

 * Note :  Make sure to clone using recursive flag
    ```
    git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
    ```

<h2 id="windows">Windows (Version ≥ 10)</h2>  
<img src="./data/ReadMe Images/windows.png" alt="Windows Logo" align="right" height = "100" width="100">
<details>
  <summary>Requirements</summary>

  ###  Prerequisites
  
  ### Visual Studio
  * Install Visual Studio 2019 Community 
    * Install Desktop development C++
    * .NET Desktop development
    * Universal Windows Development Platform

  ### Cuda
  * Create an account on developer.nvidia.com if you're not already a member. Note : Otherwise the next step will show HTTP 404/403 error.
  * Windows 10/11 : [Cuda Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)  or  [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows).

  ### Cudnn
  * Download [Cudnn](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-102) and extract files where cuda is installed. Note: Please be aware that this process requires some effort. Here are the necessary steps:
    * Download the correct zip file matching your cuda version. _Do not download the exe/installer/deb package._
    * Windows: 
      * Download [this file](https://developer.nvidia.com/compute/cudnn/secure/8.3.2/local_installers/10.2/cudnn-windows-x86_64-8.3.2.44_cuda10.2-archive.zip).
    
  * Clone with submodules and LFS. 
    ```
    git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
    ```

</details>

<details>
  <summary>Build</summary>

  Open PowerShell as an administrator and execute the following commands

  ### Build Without Cuda
  If your windows system does not have an NVIDIA GPU use this script
  ```
  build_windows_no_cuda.bat
  ```
  ### Build With Cuda
  ```
  build_windows_cuda.bat
  ```
  ### To Build With Documentation
  ```
  build_windows_cuda.bat --build-doc
  ```
</details>

<details>
  <summary>Test</summary>

  ### Run Tests
  * list all tests
    ```
    _build/BUILD_TYPE/aprapipesut.exe --list_content
    ```
  * run all tests  
    ```
    _build/BUILD_TYPE/aprapipesut.exe
    ```
  * run all tests disabling memory leak dumps and better progress logging
    ```
    _build/BUILD_TYPE/aprapipesut.exe -p -l all --detect_memory_leaks=0
    ```
  * run one test 
    ```
    _build/BUILD_TYPE/aprapipesut.exe --run_test=filenamestrategy_tests/boostdirectorystrategy
    ```
  * run one test with arguments 
    ```
    _build/BUILD_TYPE/aprapipesut.exe --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera
    ```
    * Look at the unit_tests/params_test to check for sample usage of parameters in test code.
  

</details>

<h2 id="linux">Linux (Ubuntu ≥ 20.04)</h2>
<img src="./data/ReadMe Images/Linux.png" alt="Linux Logo" align="right" height = "100" width="100">
<details>
  <summary>Requirements</summary>

  ### Prerequisites

  ### Cuda
  * Create an account on developer.nvidia.com if you're not already a member. Note : Otherwise the next step will show HTTP 404/403 error.
  * Ubuntu 20.04/22.04:
    * 20.04 - [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04)
    * 22.04 - [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04)

  ### Cudnn
  * Download [cuDNN](https://developer.nvidia.com/cudnn) matching your CUDA version
  * Linux:
      * For CUDA 11.8: [cuDNN 8.x](https://developer.nvidia.com/rdp/cudnn-archive)
      * For CUDA 12.x: [cuDNN 9.x](https://developer.nvidia.com/cudnn)

  * Clone with submodules and LFS.
    ```
    git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
    ```

</details>

<details>
  <summary>Build</summary>
  
  * Run this command to make the script file executable.   
  ```
  chmod +x build_linux_*.sh
  ```
  ### Build Without Cuda
  If your windows system does not have an NVIDIA GPU use this script
  ```
  ./build_linux_no_cuda.sh
  ```
  ### Build With Cuda
  ```
  ./build_linux_cuda.sh
  ```
  ### To Build With Documentation
  ```
  ./build_linux_cuda.sh --build-doc
  ```

  Build can take ~2 hours depending on the machine configuration.
</details>

<details>
  <summary>Test</summary>

  ### Run Tests
  * list all tests
    ```
    ./_build/aprapipesut --list_content
    ```
  * run all tests  
    ```
    ./_build/aprapipesut
    ```
  * run all tests disabling memory leak dumps and better progress logging
    ```
    ./_build/aprapipesut -p -l all --detect_memory_leaks=0
    ```
  * run one test 
    ```
    ./_build/aprapipesut --run_test=filenamestrategy_tests/boostdirectorystrategy
    ```
  * run one test with arguments 
    ```
    ./_buildaprapipesut --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera
    ```
    * Look at the unit_tests/params_test to check for sample usage of parameters in test code.
</details>

<h2 id="macos">macOS (Version ≥ 15.0)</h2>
<img src="./data/ReadMe Images/macos.png" alt="macOS Logo" align="right" height = "100" width="100">
<details>
  <summary>Requirements</summary>

  ### Prerequisites

  ### Homebrew
  * Install [Homebrew](https://brew.sh/) package manager if not already installed:
    ```
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

  ### Build Tools
  * Install nasm (required for FFmpeg):
    ```
    brew install nasm
    ```

  * Clone with submodules and LFS.
    ```
    git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
    ```

</details>

<details>
  <summary>Build</summary>

  * Run this command to make the script file executable.
  ```
  chmod +x build_macos.sh
  ```
  ### Build Without Cuda (Only NoCUDA Build)
  macOS builds do not support CUDA. Use the following script:
  ```
  ./build_macos.sh
  ```

  Build can take ~2 hours depending on the machine configuration.
</details>

<details>
  <summary>Test</summary>

  ### Run Tests
  * list all tests
    ```
    ./build/aprapipesut --list_content
    ```
  * run all tests
    ```
    ./build/aprapipesut
    ```
  * run all tests disabling memory leak dumps and better progress logging
    ```
    ./build/aprapipesut -p -l all --detect_memory_leaks=0
    ```
  * run one test
    ```
    ./build/aprapipesut --run_test=filenamestrategy_tests/boostdirectorystrategy
    ```
  * run one test with arguments
    ```
    ./build/aprapipesut --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera
    ```
    * Look at the unit_tests/params_test to check for sample usage of parameters in test code.
</details>

<h2 id="jetson">Jetson Boards - Nano, TX2, NX, AGX, Orin (JetPack ≥ 5.0)</h2>
<img src="./data/ReadMe Images/nvidia.png" alt="Nvidia Logo" align="right" height = "100" width="100">
<details>
  <summary >Requirements</summary>

  ###  Prerequisites
  * Setup the board with [JetPack 5.0+](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html):
    * JetPack 5.x: Ubuntu 20.04, CUDA 11.4
    * JetPack 6.x: Ubuntu 22.04, CUDA 12.x
  
  * Clone with submodules and LFS. 
    ```
    git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
    ```
</details>

<details>
  <summary>Build</summary>

### Build for Jetson (Only Cuda Build)
  * Run this command to make the script file executable.
  ```
  chmod +x build_jetson.sh
  ```
  * ApraPipes builds CUDA version on Jetson Boards.
  ```
  ./build_jetson.sh
  ```
  * To Build With Documentation
  ```
  ./build_jetson.sh --build-doc
  ```
  Build can take ~12 hours on Jetson Nano.
  Note: Jetson build can also be done using Ubuntu 20.04+ x86_64 Laptop via cross compilation.
</details>

<details>
  <summary>Cross Compilation using qemu</summary>

### Cross compilation using qemu
  Conceptual steps adapted from [here](https://github.com/zhj-buffer/Cross-Compile-Jetson):

  * On any Intel Ubuntu 20.04+ computer (physical or virtual including wsl) mount a Jetson SD Card Image as described above
  * Copy relevant files from mounted image to created a rootfs 
  * Install qemu on ubuntu host
  * chroot into emulated aarm64 environment using script provided in the github link above
  * install extra tools and build aprapipes and aprapipesut
  * the built aprapipesut can be copied to a Jetson board and run. 

  This approach can use all 12-16 cores of a laptop and hence builds faster.
</details>

<details>
  <summary>Test</summary>

### Run Tests
  * list all tests `./_build/aprapipesut --list_content`
  * run all tests  `./_build/aprapipesut`
  * run one test `./_build/aprapipesut --run_test=filenamestrategy_tests/boostdirectorystrategy`
  * run one test with arguments `./_build/aprapipesut --run_test=unit_tests/params_test -- -ip 10.102.10.121 -data ArgusCamera`
  * Look at the unit_tests/params_test to check for sample usage of parameters in test code
</details>

<h2 id="docker">Docker</h2>  
<img src="./data/ReadMe Images/Docker.png" alt="Nvidia Logo" align="right" height = "100" width="100">
<details>
  <summary>Requirements</summary>

###  Prerequisites
  * Ensure virtualization is enabled in both the BIOS settings of your computer and the Windows virtualization feature -Refer [this article](https://support.microsoft.com/en-us/windows/enable-virtualization-on-windows-11-pcs-c5578302-6e43-4b4b-a449-8ced115f58e1#:~:text=Virtualization%20lets%20your%20PC%20emulate,will%20help%20you%20enable%20virtualization) to enable them
  * Install WSL 2 on your system:
    ```
    wsl --install
    ```
  * Set WSL 2 as the default version using the command line:
    ```
    wsl --set-default-version 2
    ```
  * Install Ubuntu 22.04 from [Microsoft Store](https://apps.microsoft.com/detail/9pn20msr04dw) or run `wsl --install -d Ubuntu-22.04`
  * Install Docker Desktop on Windows from [here](https://docs.docker.com/desktop/install/windows-install/)
  * Enable Docker integration with WSL 2 (Docker Desktop settings → Resources → WSL integration → Enable Ubuntu-22.04 → Apply & Restart)
  * Install nvidia-container-toolkit in WSL Ubuntu for GPU access - Follow [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  * Note:"Follow the exact instructions outlined in the document to ensure the correct and successful installation of the NVIDIA Container Toolkit"
</details>

<details>
  <summary>Build</summary>

### Build for Docker
  * Use this [docker image](https://github.com/users/kumaakh/packages/container/package/aprapipes-build-x86-ubutu18.04-cuda) with all dependencies pre-installed:
  ```
  docker pull ghcr.io/kumaakh/aprapipes-build-x86-ubutu18.04-cuda:last-good
  ```
  > **Note:** This image is based on Ubuntu 18.04 with pre-cached vcpkg dependencies for fast builds (~10 min). The WSL host can be Ubuntu 22.04.
* Mount an external volume as a build area, and then use the Windows command line to create a Docker container using the above image with the following command:  
  ```
  docker run -dit --gpus all -v "</path/to/external_volume>":"/mnt/b/" --name <give-container-name> a799cc26f4b7
  ```
  ..your command should look like this [where D:\ws\docker-pipes->local_folder_path , pipes->container_name ]
  ```
  docker run -dit --gpus all -v "D:\ws\docker-pipes":"/mnt/b/" --name pipes a799cc26f4b7
  ```
* After creating the container, execute the following command to access its command line interface
  ```
  docker exec -it <container-name> /bin/bash
  ```
* Note:"When inside the container, build all contents within the mounted external folder"
* clone the repository with submodules and LFS as described above
* build using build_linux_\*.sh scripts as described [above](#build-for-linux)

This build will be fairly fast (~10 mins) as entire vcpkg cache comes down with the docker image
</details>

## Update Submodules
```
git submodule update --init --recursive
```
## Update Documentation
To update documentation, refer to Documentation Guidelines in the [Contribution-Guidelines](https://github.com/Apra-Labs/ApraPipes/wiki/Contribution-Guidelines).

### To regenerate documentation
Run,
```
./build_documentation.sh
```

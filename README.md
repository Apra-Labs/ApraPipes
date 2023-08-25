<p align="center">
  <img src="./data/ReadMe Images/ApraPipes.png" alt="Header Image">
</p>

# ApraPipes
A pipeline framework for developing video and image processing applications. Supports multiple GPUs and Machine Learning toolkits.  
Learn more about ApraPipes here https://apra-labs.github.io/ApraPipes.

## Build status
Aprapipes is automatically built and tested on Ubuntu (18.04 and 20.04), Jetson Boards (Jetpack 4.4) and Windows (11) x64 Visual Studio 2017 Community.
|OS|Version|With Cuda|Tests|Status|
|--|-------|---------|------|------|
|Windows|2019|No|[![Test Results](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows.svg)|[![CI-Win-NoCUDA](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Win-NoCUDA.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Win-NoCUDA.yml)|
|Windows|2019|Yes|[![Test Results](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows-cuda.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Windows-cuda.svg)|[![CI-Win-CUDA](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Win-CUDA.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Win-CUDA.yml)|
|Ubuntu x64_86|20.04|No|[![Test Results](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux.svg)|[![CI-Linux-NoCUDA](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-NoCUDA.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-NoCUDA.yml)|
|Ubuntu x64_86|18.04|Yes|[![Test Results](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux-CudaT.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux-CudaT.svg)|[![CI-Linux-CUDA](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-CUDA.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-CUDA.yml)|
|Ubuntu ARM64 (Jetsons)|18.04|Yes|[![Test Results](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux_ARM64.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_Linux_ARM64.svg)|[![CI-Linux-ARM64](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-ARM64.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-ARM64.yml)|
|Ubuntu x64_86-WSL|20.04|Yes|[![Test Results](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_WSL.svg)](https://gist.githubusercontent.com/kumaakh/f80af234a4aabedc69af3ee197f66944/raw/badge_WSL.svg)|[![CI-Linux-CUDA-wsl](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-CUDA-wsl.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-CUDA-wsl.yml)|
|Ubuntu x64_86-docker|18.04|Yes|No|[![CI-Linux-CUDA-Docker](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-CUDA-Docker.yml/badge.svg)](https://github.com/Apra-Labs/ApraPipes/actions/workflows/CI-Linux-CUDA-Docker.yml)|

## Getting Started with ApraPipes

<details>
  <summary>Please select your Operating System</summary>
  <ul>
    <li><a href="#windows">Windows</a></li>
    <li><a href="#linux">Linux</a></li>
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

  ### Cuda
  * Create an account on developer.nvidia.com if you're not already a member. Note : Otherwise the next step will show HTTP 404/403 error.
  * Windows 10/11 : [Cuda Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)  or  [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64).

  ### Cudnn
  * Download [Cudnn](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-102) and extract files where cuda is installed. Note: Please be aware that this process requires some effort. Here are the necessary steps:
   * Download the correct tar/zip file matching your cuda version. _Do not download the exe/installer/deb package._
   * Windows: 
     * Download [this file](https://developer.nvidia.com/compute/cudnn/secure/8.3.2/local_installers/10.2/cudnn-windows-x86_64-8.3.2.44_cuda10.2-archive.zip). 
     * Extract the downloaded file and copy files to ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2``` using an __administrative command prompt__ as follows
       ```
       cd .\extracted_folder
       cd include
       copy *.h "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include\"
       cd ..\lib
       copy *.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64\"
       cd ..\bin
       copy *.dll "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\"
       ```
  ###  Prerequisites 
  * Install Visual Studio 2019 Community 
    * Install Desktop development C++
    * .NET Desktop development
    * Universal Windows Development Platform
  * Install choco:
    Open Windows PowerShell as Administrator and run:
    ```
    Set-ExecutionPolicy AllSigned
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    ``` 
  * Install build dependencies using choco: 
    ```
    choco feature enable -n allowEmptyChecksums && choco install 7zip git python3 cmake pkgconfiglite -y && pip3 install ninja && pip3 install meson
    ```
  * Clone with submodules and LFS. 
    ```
    git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
    ```

</details>

<details>
  <summary>Build</summary>

  ### Build Without Cuda
  If your windows system does not have an NVIDIA GPU use this script
  ```
  build_windows_no_cuda.bat
  ```
  ### Build With Cuda
  ```
  build_windows_cuda.bat
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

<h2 id="linux">Linux (Ubuntu ≥ 18.04)</h2>  
<img src="./data/ReadMe Images/Linux.png" alt="Linux Logo" align="right" height = "100" width="100">
<details>
  <summary>Requirements</summary>
  
  ### Cuda
  * Create an account on developer.nvidia.com if you're not already a member. Note : Otherwise the next step will show HTTP 404/403 error.
  * Ubuntu 18.04/20.04:   
    18.04 - [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)  
    20.04 - [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04)

  ### Cudnn  
  * Download [Cudnn](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-102) and extract files where cuda is installed. Note: Please be aware that this process requires some effort. Here are the necessary steps:
  * Linux:
      * Download [this file](https://developer.nvidia.com/compute/cudnn/secure/8.3.2/local_installers/10.2/cudnn-linux-x86_64-8.3.2.44_cuda10.2-archive.tar.xz)
      * extract the files
        ``` 
        xz -d cudnn-linux-x86_64-8.3.2.44_cuda10.2-archive.tar.xz
        tar xvf cudnn-linux-x86_64-8.3.2.44_cuda10.2-archive.tar
        ```
      * copy files retaining the links
        ```
        cd ./cudnn-linux-x86_64-8.3.2.44_cuda10.2-archive
        sudo cp -P include/* /usr/local/cuda/include/
        sudo cp -P lib/* /usr/local/cuda/lib64/
        ```
  * To add Cuda to path append following lines to ~/.bashrc. 
    ```
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```
  * Reload ~/.bashrc:
    ```
    source ~/.bashrc:
    ```
  * Verify by checking nvcc --version and nvidia-smi both must point to correct versions of cuda and nvidia driver in the system.
  ### Prerequisites
  * Run the following to get latest build tools
    ```
    sudo apt-get update && sudo apt-get -y install   autoconf   automake  autopoint  build-essential  git-core  git-lfs libass-dev   libfreetype6-dev  libgnutls28-dev   libmp3lame-dev libsdl2-dev  libssl-dev libtool libsoup-gnome2.4-dev libncurses5-dev libva-dev   libvdpau-dev   libvorbis-dev   libxdamage-dev   libxcursor-dev   libxinerama-dev   libx11-dev   libgles2-mesa-dev   libxcb1-dev   libxcb-shm0-dev   libxcb-xfixes0-dev  ninja-build   pkg-config   texinfo   wget   yasm   zlib1g-dev   nasm   gperf bison curl zip unzip tar python3-pip flex && pip3 install meson
    ```  
  * Note: start a new terminal as pip3 settings do not get effective on the same shell
  * CMake minimum version 3.24 - Follow [this article](https://anglehit.com/how-to-install-the-latest-version-of-cmake-via-command-line/) to update cmake
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

<h2 id="jetson">Jetson Boards - Nano, TX2, NX, AGX (Jetpack ≥ 4.4)</h2>  
<img src="./data/ReadMe Images/nvidia.png" alt="Nvidia Logo" align="right" height = "100" width="100">
<details>
  <summary >Requirements</summary>
  
  ###  Prerequisites
  * Setup the board with [Jetpack 4.4](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html) or higher as supported.
  * Run the following commands to build required libraries. 
    ```
    sudo apt-get update && sudo apt-get -y install git-lfs libncurses5-dev ninja-build nasm curl libudev-dev libssl-dev && sudo snap install cmake --classic
    ```
  * Append following lines to ~/.bashrc. 
    ```
    export VCPKG_FORCE_SYSTEM_BINARIES=1
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```
  * Reload ~/.bashrc:
    ```
    source ~/.bashrc:
    ```
  * Clone with submodules and LFS. 
    ```
    git clone --recursive https://github.com/Apra-Labs/ApraPipes.git
    ```
  * Run `./bootstrap-vcpkg.sh` in vcpkg/ directory
  * Run `./vcpkg integrate install`
</details>

<details>
  <summary>Build</summary>

### Build for Jetson (Only Cuda Build)
  * Run this command to make the script file executable.
  ```
  chmod +x build_jetson.sh
  ```
  * ApraPipes builds CUDA version on Jerson Boads.
  ```
  ./build_jetson.sh
  ```

  Build can take ~12 hours on Jetson Nano. 
  Note: Jetson build can also be done using Ubuntu 18.04 x86_64 Laptop via cross compilation.
</details>

<details>
  <summary>Cross Compilation using qemu</summary>

### Cross compilation using qemu
  Conceptual steps adapted from [here](https://github.com/zhj-buffer/Cross-Compile-Jetson):

  * On any Intel Ubuntu 18.04 computer (physical or virtual including wsl ) mount a Jetson SD Card Image as described above
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
  * Install Ubuntu-18.04 from [Microsoft store](https://apps.microsoft.com/store/detail/ubuntu-1804-on-windows/9N9TNGVNDL3Q?hl=en-in&gl=in&rtc=1) , Refer [this article](https://learn.microsoft.com/en-us/windows/wsl/install-manual) for any issues regarding installation 
  * Install Docker Desktop on Windows -from [here](https://docs.docker.com/desktop/install/windows-install/)
  * Enable Docker integration with WSL 2 (in Docker Desktop settings -> Resources -> WSL integration -> Enable Ubuntu-18.04 -> Apply&restart)
  * Install nvida-container-toolkit using (WSL Ubuntu-18.04) for docker to access Host-system GPU -Follow [this document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install nvidia-container-toolkit
  * Note:"Follow the exact instructions outlined in the document to ensure the correct and successful installation of the NVIDIA Container Toolkit"
</details>

<details>
  <summary>Build</summary>

### Build for Docker
  * Use this [docker image](https://github.com/users/kumaakh/packages/container/package/aprapipes-build-x86-ubutu18.04-cuda) with all the software setup.
  ```
  docker pull ghcr.io/kumaakh/aprapipes-build-x86-ubutu18.04-cuda:last-good
  ```
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
 After making changes to the documentation located in the /docs/source folder, it's essential to regenerate the documentation by following the provided steps. Once regenerated, commit the new content to ensure the latest documentation is up-to-date.

### To regenerate documentation
```
To build docs
apt-install get python-sphinx 
pip install sphinx-rtd-theme
cd docs
make html
```
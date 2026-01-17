# Custom triplet for Ubuntu 24.04 with CUDA support
# Use with: -DVCPKG_OVERLAY_TRIPLETS=../thirdparty/triplets

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

# Use GCC-11 for all builds - required for CUDA 11.8 compatibility on Ubuntu 24.04
# Ubuntu 24.04 ships with GCC 13 which is not supported by CUDA 11.8
if(EXISTS "/usr/bin/gcc-11" AND EXISTS "/usr/bin/g++-11")
    set(VCPKG_C_COMPILER "/usr/bin/gcc-11")
    set(VCPKG_CXX_COMPILER "/usr/bin/g++-11")
endif()

# Pass CUDA environment variables into vcpkg builds
set(VCPKG_ENV_PASSTHROUGH CUDAHOSTCXX CUDA_PATH CUDAToolkit_ROOT)

# CUDA configuration for OpenCV and other CUDA-enabled packages
# Only applies when CUDA is installed at /usr/local/cuda
if(EXISTS "/usr/local/cuda/bin/nvcc")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
        "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11"
        "-DCUDAToolkit_ROOT=/usr/local/cuda"
        "-DCUDA_PATH=/usr/local/cuda"
    )
endif()

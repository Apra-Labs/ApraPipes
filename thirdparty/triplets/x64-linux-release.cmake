# Custom triplet for Ubuntu 24.04 with CUDA support
# Use with: -DVCPKG_OVERLAY_TRIPLETS=../thirdparty/triplets

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

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

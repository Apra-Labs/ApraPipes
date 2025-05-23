# VCPKG Integration of LLM Libraries

## Overview
This document describes the integration of llama.cpp and whisper.cpp into the ApraPipes project using vcpkg package manager. The integration enables efficient deployment and management of these LLM libraries across different platforms.

## Portfile System

### Portfile Structure
A portfile in vcpkg consists of several key components:
1. **Source Acquisition**
   - Repository URL and commit/tag
   - SHA512 hash verification
   - Custom patches
2. **Build Configuration**
   - Feature flags
   - CMake options
   - Platform-specific settings
3. **Installation Rules**
   - Library installation
   - Header file placement
   - CMake configuration files
   - License and documentation

### Portfile Workflow
1. **Source Download**
   ```cmake
   vcpkg_from_github(
       OUT_SOURCE_PATH SOURCE_PATH
       REPO Apra-Labs/llama.cpp
       REF e5bd6e1abb146b38649236429c22ed6b4db0f3da
       SHA512 f36a0731e7b5044b1d75297fdd806cf19206a439bc9996bba1ee36b0b2e692e4482d5fac9b7dcd111c7d69bbd900b99ed38b301c572c450a48ad6fd484b3322f
       HEAD_REF kj/vcpkg-port
   )
   ```
   - Downloads source from specified repository
   - Verifies integrity using SHA512
   - Extracts to temporary build directory

2. **Feature Configuration**
   ```cmake
   vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
    "cuda" LLAMA_CUBLAS
   )
   ```
   - Defines available features
   - Maps features to CMake options
   - Handles feature dependencies

3. **Build Configuration**
   ```cmake
   vcpkg_cmake_configure(
       SOURCE_PATH "${SOURCE_PATH}"
       OPTIONS
           ${FEATURE_OPTIONS}
           -DLLAMA_CUBLAS=${LLAMA_CUBLAS}
       DISABLE_PARALLEL_CONFIGURE
   )
   ```
   - Configures CMake build
   - Applies feature options
   - Sets platform-specific flags

4. **Installation**
   ```cmake
   vcpkg_cmake_install()
   vcpkg_cmake_config_fixup(
       CONFIG_PATH lib/cmake/Llama
       PACKAGE_NAME Llama
   )
   ```
   - Builds and installs libraries
   - Fixes CMake configuration files
   - Installs headers and documentation

## Llama.cpp Integration

### Port Configuration Details
- **Source Management**
  - Custom fork in Apra-Labs repository
  - Specific commit for stability
  - Custom patches for vcpkg compatibility

- **Build Options**
  ```cmake
  set(LLAMA_CUBLAS OFF)
  if("cuda" IN_LIST FEATURES)
    set(LLAMA_CUBLAS ON)
  endif()
  ```
  - CUDA support through feature flags
  - Platform-specific optimizations
  - Static/dynamic linking control

- **Installation Rules**
  ```cmake
  file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
  file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
  ```
  - Proper header file organization
  - License and documentation installation
  - Debug/release separation

### Build Process
1. **Dependency Resolution**
   - vcpkg resolves all dependencies
   - Handles platform-specific requirements
   - Manages version conflicts

2. **Configuration Phase**
   ```cmake
   # CMake configuration
   find_package(Llama CONFIG REQUIRED)
   target_link_libraries(aprapipesut PRIVATE llama)
   ```
   - CMake package configuration
   - Library discovery
   - Linkage settings

3. **Build Phase**
   - Compilation with platform-specific flags
   - Feature-based conditional compilation
   - Optimization settings

## Whisper.cpp Integration

### Port Configuration Details
- **Source Management**
  - Custom fork with vcpkg support
  - ARM64 compatibility patches
  - CUDA integration

- **ARM64 Support**
  ```patch
  # fix-for-arm64.patch
  diff --git a/ggml-cuda.cu b/ggml-cuda.cu
  - Custom ARM NEON implementations
  - CUDA header order fixes
  - Vector type handling
  ```

- **Build Options**
  ```cmake
  set(WHISPER_CUBLAS OFF)
  if("cuda" IN_LIST FEATURES)
    set(WHISPER_CUBLAS ON)
  endif()
  ```
  - CUDA feature control
  - Platform-specific settings
  - Optimization flags

### Platform-Specific Handling
1. **Windows**
   ```cmake
   IF(ENABLE_WINDOWS)
     set(VCPKG_TARGET_TRIPLET "x64-windows")
     add_compile_definitions(WINDOWS)
   ENDIF()
   ```
   - Windows-specific triplet
   - MSVC compiler settings
   - DLL/static library handling

2. **Linux**
   ```cmake
   IF(ENABLE_LINUX)
     add_compile_definitions(LINUX)
     target_include_directories(aprapipesut PRIVATE ${GTK3_INCLUDE_DIRS})
   ENDIF()
   ```
   - Linux-specific settings
   - System library integration
   - Platform-specific dependencies

3. **ARM64**
   ```cmake
   IF(ENABLE_ARM64)
     add_compile_definitions(ARM64)
     set(VCPKG_OVERLAY_TRIPLETS ../vcpkg/triplets/community/arm64-linux.cmake)
     set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
   ENDIF()
   ```
   - ARM64-specific triplet
   - CUDA compiler settings
   - NEON optimizations

## Build System Integration

### CMake Configuration
```cmake
# Main CMakeLists.txt
cmake_minimum_required(VERSION 3.29)
set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/custom-overlay")
set(VCPKG_INSTALL_OPTIONS "--clean-after-build")

# Library discovery
find_package(whisper CONFIG REQUIRED)
find_package(Llama CONFIG REQUIRED)

# Library linkage
target_link_libraries(aprapipesut
  PRIVATE
  llama
  whisper::whisper
  ${COMMON_LIB}
  ${LLAVA_LIB}
)
```

### Feature Management
1. **Feature Control**
   ```bash
   # fix-vcpkg-json.sh
   if $removeCUDA; then
       # Remove CUDA features
       v=$(echo "$v" | jq ".dependencies[$index].features |= map(select(. != \"cuda\"))")
   fi
   ```
   - Script-based feature management
   - Platform-specific feature control
   - Dependency resolution

2. **Platform Configuration**
   - Windows/Linux/ARM64 support
   - CUDA integration control
   - Build type management

## Dependencies and Requirements

### System Requirements
- CMake 3.29 or higher
- vcpkg package manager
- Platform-specific build tools
- CUDA toolkit (optional)

### Library Dependencies
- CUDA (optional)
- OpenCV (optional)
- Platform-specific libraries
- System dependencies

### Build Tools
- C++ compiler (MSVC/GCC/Clang)
- CMake
- vcpkg
- Platform-specific tools

## Troubleshooting

### Common Issues
1. **Build Failures**
   - Check feature compatibility
   - Verify platform support
   - Review dependency versions

2. **Runtime Issues**
   - Verify library linkage
   - Check CUDA compatibility
   - Validate platform support

3. **Feature Problems**
   - Review feature flags
   - Check dependency resolution
   - Verify platform compatibility 
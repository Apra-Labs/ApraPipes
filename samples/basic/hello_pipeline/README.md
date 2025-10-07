# Hello Pipeline Sample

## Overview
This is the most basic ApraPipes sample - a "Hello World" for the pipeline framework. It demonstrates how to create, configure, and run a simple pipeline.

## What This Sample Demonstrates
- Basic pipeline creation
- Module instantiation and connection
- Pipeline initialization and execution
- Proper resource cleanup

## Prerequisites
- ApraPipes library built successfully
- Basic understanding of C++ and pipeline concepts

## Building This Sample

### From repository root:
```bash
# Windows with Visual Studio 2019 + CUDA
powershell -ExecutionPolicy Bypass -File build_windows_cuda_vs19.ps1 -BuildSamples

# Linux with CUDA
./build_linux_cuda.sh --with-samples

# Or manually with CMake:
cmake -B _build -S . -DBUILD_SAMPLES=ON -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build _build --target sample_hello_pipeline
```

## Running the Sample

### Windows:
```bash
_build\samples\sample_hello_pipeline.exe
```

### Linux:
```bash
./_build/samples/sample_hello_pipeline
```

## Expected Output
The sample will print pipeline initialization messages and demonstrate successful execution.

## Next Steps
After understanding this sample, explore:
- `samples/video/` - Video capture and processing
- `samples/image/` - Image transformation examples
- `samples/advanced/` - Complex multi-module pipelines

## Code Structure
```cpp
main() {
    // 1. Initialize logger
    // 2. Create pipeline
    // 3. Add and connect modules
    // 4. Initialize and run
    // 5. Cleanup
}
```

## Common Issues
- **Link errors**: Ensure aprapipes library is built first
- **Missing DLLs** (Windows): Copy runtime DLLs from vcpkg to executable directory
- **CUDA errors**: Check CUDA is properly installed if using GPU features

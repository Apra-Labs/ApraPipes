# CUDA Pipeline Examples

These examples demonstrate GPU-accelerated video processing using NVIDIA CUDA and NPP libraries.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Build with `-DENABLE_CUDA=ON`

## Examples

| Example | Description | Output |
|---------|-------------|--------|
| 01_gaussian_blur_demo | GPU-accelerated Gaussian blur | `cuda_blur_????.jpg` |
| 02_effects_demo | Brightness, contrast, saturation | `cuda_effects_????.jpg` |
| 03_resize_demo | GPU image resizing (640x480 → 320x240) | `cuda_resize_????.jpg` |
| 04_rotate_demo | GPU image rotation (45 degrees) | `cuda_rotate_????.jpg` |
| 05_processing_chain_demo | Multi-stage GPU processing | `cuda_chain_????.jpg` |
| 06_nvjpeg_encoder_demo | GPU-accelerated JPEG encoding | `cuda_nvjpeg_????.jpg` |

## Pipeline Pattern

All CUDA examples follow this pattern:

```
TestSignalGenerator → ColorConversion → CudaMemCopy(H→D) → [GPU Processing] → CudaMemCopy(D→H) → ImageEncoder → FileWriter
```

1. **TestSignalGenerator**: Creates test frames (YUV420 planar)
2. **ColorConversion**: Converts YUV420 to RGB on CPU
3. **CudaMemCopy (HostToDevice)**: Uploads to GPU memory
4. **GPU Processing**: One or more CUDA modules (blur, resize, effects, etc.)
5. **CudaMemCopy (DeviceToHost)**: Downloads back to CPU (or use nvJPEG to encode on GPU)
6. **ImageEncoderCV/JPEGEncoderNVJPEG**: Encodes to JPEG
7. **FileWriterModule**: Writes to disk

## Running Examples

```bash
# Validate a pipeline
./build/aprapipes_cli validate docs/declarative-pipeline/examples/cuda/01_gaussian_blur_demo.json

# Run a pipeline (2 second duration)
timeout 2 ./build/aprapipes_cli run docs/declarative-pipeline/examples/cuda/01_gaussian_blur_demo.json

# Check output
ls -la data/testOutput/cuda_blur_*.jpg
```

## CUDA Modules Used

| Module | Description | Key Properties |
|--------|-------------|----------------|
| CudaMemCopy | Host/Device memory transfer | `kind`: HostToDevice/DeviceToHost |
| GaussianBlur | NPP Gaussian blur filter | `kernelSize`: 3-31 (odd numbers) |
| ResizeNPPI | NPP image resizing | `width`, `height` |
| RotateNPPI | NPP image rotation | `angle` (degrees) |
| EffectsNPPI | Brightness/contrast/saturation | `brightness`, `contrast`, `saturation`, `hue` |
| CCNPPI | NPP color conversion | `imageType`: RGB/BGR/RGBA/etc. |
| JPEGEncoderNVJPEG | nvJPEG hardware encoder | `quality`: 1-100 |

## Building with CUDA

On Ubuntu 24.04 with CUDA 11.8:

```bash
export CUDA_PATH=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_OVERLAY_PORTS=thirdparty/custom-overlay \
  -DVCPKG_TARGET_TRIPLET=x64-linux-release \
  -DENABLE_CUDA=ON \
  base
```

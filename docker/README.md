# ApraPipes Docker Build Environment

This directory contains Docker configurations for building ApraPipes in both CUDA and NoCUDA flavors.

## Directory Structure

```
docker/
├── Dockerfile.nocuda     # Dockerfile for building without CUDA support
├── Dockerfile.cuda       # Dockerfile for building with CUDA support
├── build.sh             # Script to build Docker images
├── run.sh               # Script to run Docker containers
├── build-inside.sh      # Script to build ApraPipes inside container
└── README.md            # This file
```

## Quick Start

### 1. Build the Docker Image

```bash
# For NoCUDA build
cd docker
./build.sh nocuda

# For CUDA build
./build.sh cuda
```

### 2. Run the Container

```bash
# For NoCUDA build
./run.sh nocuda

# For CUDA build
./run.sh cuda
```

This will start an interactive bash session inside the container with your ApraPipes workspace mounted at `/workspace`.

### 3. Build ApraPipes Inside the Container

Once inside the container:

```bash
# For NoCUDA build
/workspace/docker/build-inside.sh nocuda

# For CUDA build
/workspace/docker/build-inside.sh cuda
```

## Manual Build Steps

If you prefer to run the build steps manually inside the container:

```bash
# Bootstrap vcpkg
cd /workspace
./vcpkg/bootstrap-vcpkg.sh

# For NoCUDA: Remove CUDA dependencies
cd base
pwsh ./fix-vcpkg-json.ps1 -removeCUDA
cd ..

# Configure CMake
mkdir -p build && cd build

# For NoCUDA
cmake -DENABLE_WINDOWS=OFF \
      -DENABLE_LINUX=ON \
      -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DENABLE_CUDA=OFF \
      ../base

# For CUDA
cmake -DENABLE_WINDOWS=OFF \
      -DENABLE_LINUX=ON \
      -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DENABLE_CUDA=ON \
      ../base

# Build
cmake --build . -j $(nproc)
```

## Important Notes

### Workspace Mounting

The scripts use volume mounting (`-v`) to mount your local ApraPipes directory into the container at `/workspace`. This means:

- No code is copied into the Docker image
- Changes made inside the container are reflected on your host system
- The Docker image stays lightweight
- Build artifacts (in `build/` directory) persist on your host

### Build Time

The first build inside the container will take 60-90 minutes because vcpkg needs to build all dependencies from source. Subsequent builds will be faster as vcpkg caches compiled packages.

### CUDA Requirements

For CUDA builds:
- You must have an NVIDIA GPU
- You must have the NVIDIA Docker runtime installed
- The container will use `--gpus all` flag to access GPU

### CMake Version

The Docker images use CMake 3.29.6 to match the CI environment. This is installed via pip3 and takes precedence over any system CMake.

### PowerShell

PowerShell is installed in the Docker images to run the `fix-vcpkg-json.ps1` script for removing CUDA dependencies in NoCUDA builds.

## Troubleshooting

### Docker image doesn't have latest changes

Rebuild the Docker image:
```bash
./build.sh nocuda --no-cache
```

### Permission issues with mounted files

The container runs as root by default. Files created inside the container will be owned by root. To fix ownership after exiting the container:
```bash
sudo chown -R $(whoami):$(whoami) build/
```

### Out of disk space

Docker images and build artifacts can consume significant disk space. To clean up:
```bash
# Remove old containers
docker container prune

# Remove old images
docker image prune

# Clean up build directory on host
rm -rf build/
```

# ApraPipes CUDA build images

Two multi-stage Dockerfiles that build ApraPipes **entirely with the CUDA
toolkit baked into the image** — your host's CUDA/driver is never used at build
time. Each produces a **slim runtime image** (compilers and the toolkit stay in
the throwaway builder stage).

| Target | Dockerfile | Base (CUDA in-image) | Code path |
|--------|-----------|----------------------|-----------|
| Linux x86_64 | `Dockerfile.x64-cuda` | `nvidia/cuda:11.8.0-cudnn8-*-ubuntu20.04` | `ENABLE_CUDA=ON` |
| Jetson ARM64 | `Dockerfile.jetson-cuda` | `nvcr.io/nvidia/l4t-jetpack:r35.4.1` (CUDA 11.4) | `ENABLE_ARM64=ON` |

## Build

Always build **from the repo root** (the context must include `base/`,
`vcpkg/`, `thirdparty/`). A `.dockerignore` keeps the context small by excluding
`data/`, `_build/`, `_debugbuild/` and vcpkg's build artifacts.

```bash
# x86_64
docker build -f docker/Dockerfile.x64-cuda -t aprapipes:x64-cuda11.8 .

# Jetson / ARM64 (native on a Jetson, or via qemu on x86 — slow)
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.jetson-cuda -t aprapipes:jetson-cuda --load .

# or just:
./docker/build-images.sh x64      # | jetson | all
```

## Run (GPU required)

The runtime images rely on the **NVIDIA Container Toolkit** to inject the driver
(and NVDEC/NVENC) at run time:

```bash
# x86_64
docker run --rm --gpus all aprapipes:x64-cuda11.8 aprapipesut --list_content

# Jetson
docker run --rm --runtime nvidia aprapipes:jetson-cuda aprapipesut --list_content
```

Each image ships the built static lib + headers + cmake config under
`/opt/aprapipes`, plus the `aprapipesut` test executable on `PATH`.

## Knobs (`--build-arg`)

- `CUDA_ARCH` — SM archs to generate. x64 default `52;60;70;75` (add `80;86;89`
  for Ampere/Ada); Jetson default `72;87` (Xavier+Orin). Fewer arches → smaller,
  faster builds.
- `BUILD_TYPE` — `RelWithDebInfo` (default), `Release`, `Debug`.
- Jetson `BASE_IMAGE` — set to the L4T tag matching your board's flashed JetPack.

## Important notes

- **x86_64 driver-lib stubs.** ApraPipes links `libnvcuvid.so` /
  `libnvidia-encode.so`, which come from the *driver*, not the toolkit, so they
  are absent in a build container. The builder stage synthesises link-time stubs
  (symbols pulled from the vendored Video Codec SDK headers); the real libraries
  are provided by the NVIDIA runtime when you `--gpus all`.
- **Jetson CUDA version.** On Jetson the CUDA version is fixed by the L4T image.
  JetPack 5.1.2 (r35.4.1) bundles **CUDA 11.4** — the closest in-image CUDA to
  11.8 that exists for Jetson. There is no L4T image with 11.8. Match
  `BASE_IMAGE` to the JetPack on your board.
- **CUDA-accelerated OpenCV/whisper.** These are enabled via `cuda`/`cudnn`
  features in `base/vcpkg.json`. Your current working tree has those features
  removed — restore them (e.g. `git checkout base/vcpkg.json`) before building
  if you want GPU-accelerated OpenCV/whisper inside the image.
- **Runtime shared-lib list** in each `runtime` stage is a sensible default for
  the dynamic tail (most deps are static via vcpkg). If `ldd aprapipesut` shows
  a `not found` after a first run, add that package and rebuild the runtime stage.
- **First build is long** (vcpkg compiles OpenCV, FFmpeg, Boost, whisper… from
  source). ARM64 under qemu emulation can take many hours — build on a Jetson
  when you can.
```

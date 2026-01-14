# ApraPipes Declarative Pipeline Examples

This directory contains example pipelines demonstrating the declarative JSON-based pipeline configuration for ApraPipes.

## Directory Structure

```
examples/
├── basic/              # Simple working examples (CPU-only)
├── cuda/               # GPU-accelerated examples (requires NVIDIA GPU)
├── advanced/           # Complex pipelines and templates
├── node/               # Node.js addon examples
├── needs-investigation/ # Examples that need fixes or have known issues
├── test_all_examples.sh
├── test_cuda_examples.sh
└── test_declarative_pipelines.sh
```

## Quick Start

### Prerequisites

1. Build ApraPipes:
   ```bash
   cmake -B _build -S . -DENABLE_CUDA=ON  # Use OFF if no NVIDIA GPU
   cmake --build _build --target aprapipes_cli
   ```

2. Install to bin/ directory:
   ```bash
   ./scripts/install_to_bin.sh
   ```

3. For Node.js examples, also build the addon:
   ```bash
   cmake --build _build --target aprapipes_node
   ./scripts/install_to_bin.sh  # Re-run to copy addon
   ```

### Running Examples

**Using CLI:**
```bash
cd bin
./aprapipes_cli run ../examples/basic/simple_source_sink.json
```

**Using Node.js:**
```bash
cd examples/node
node basic_pipeline.js
```

### Running All Tests
```bash
./examples/test_all_examples.sh
```

---

## Basic Examples (CPU)

| Example | Description |
|---------|-------------|
| `simple_source_sink.json` | Minimal pipeline: test signal generator to statistics sink |
| `three_module_chain.json` | Three-module chain with color conversion |
| `split_pipeline.json` | Demonstrates splitting a stream to multiple outputs |
| `ptz_with_conversion.json` | Pan-Tilt-Zoom with color conversion |
| `transform_ptz_with_conversion.json` | PTZ transform pipeline |
| `bmp_converter_pipeline.json` | BMP format conversion |
| `affine_transform_chain.json` | Affine transformation chain |
| `affine_transform_demo.json` | Affine transformation demonstration |

---

## CUDA Examples (GPU)

Requires NVIDIA GPU and CUDA toolkit.

| Example | Description |
|---------|-------------|
| `gaussian_blur.json` | GPU-accelerated Gaussian blur using NPP |
| `auto_bridge.json` | Auto-bridging: automatic CPU/GPU memory transfers |
| `effects.json` | GPU image effects (brightness, contrast) |
| `resize.json` | GPU-accelerated image resizing |
| `rotate.json` | GPU-accelerated 90-degree rotation |
| `processing_chain.json` | Multi-stage GPU processing pipeline |
| `nvjpeg_encoder.json` | GPU-accelerated JPEG encoding |

**Output:** Files are written to `bin/data/testOutput/cuda_*.jpg`

---

## Advanced Examples

| Example | Description |
|---------|-------------|
| `file_reader_writer.json` | Read and write image files |
| `mp4_reader_writer.json` | MP4 video processing (MJPEG) |
| `motion_vector_pipeline.json` | Motion vector extraction |
| `affine_transform_pipeline.json` | Complex affine transformations |

---

## Node.js Examples

Located in `node/` subdirectory. Requires the Node.js addon (`bin/aprapipes.node`).

| Example | Description |
|---------|-------------|
| `basic_pipeline.js` | Basic pipeline construction |
| `image_processing.js` | Image processing operations |
| `face_detection_demo.js` | Face detection with Node.js |
| `ptz_control.js` | PTZ camera control |
| `event_handling.js` | Pipeline event handling |
| `rtsp_pusher_demo.js` | RTSP stream pushing |
| `archive_space_demo.js` | Archive space management |

See `node/README.md` for detailed Node.js documentation.

---

## Test Pattern Types

The `TestSignalGenerator` module supports these patterns:

- `GRADIENT` - Horizontal gradient bands (default)
- `CHECKERBOARD` - Black/white checkerboard
- `COLOR_BARS` - Vertical color bars (recommended for testing)
- `GRID` - Numbered grid cells

Example:
```json
{
  "type": "TestSignalGenerator",
  "props": {
    "width": 640,
    "height": 480,
    "pattern": "COLOR_BARS",
    "maxFrames": 100
  }
}
```

---

## Creating New Examples

1. Create a JSON file following the schema:
   ```json
   {
     "pipeline": {
       "name": "My Pipeline",
       "description": "Description of what this pipeline does"
     },
     "modules": {
       "module_name": {
         "type": "ModuleType",
         "props": { ... }
       }
     },
     "connections": [
       { "from": "source_module", "to": "dest_module" }
     ]
   }
   ```

2. Validate without running:
   ```bash
   ./bin/aprapipes_cli validate your_pipeline.json
   ```

3. Run with verbose output:
   ```bash
   ./bin/aprapipes_cli run your_pipeline.json --verbose
   ```

---

## Troubleshooting

**Pipeline fails to start:**
- Check module types are registered (use `./bin/aprapipes_cli list-modules`)
- Verify connection compatibility (frame types must match)

**CLI not found:**
- Run `./scripts/install_to_bin.sh` to install build outputs to bin/

**CUDA examples fail:**
- Ensure CUDA is enabled: rebuild with `-DENABLE_CUDA=ON`
- Check GPU is available: `nvidia-smi`

**Node.js addon not found:**
- Build with: `cmake --build _build --target aprapipes_node`
- Run: `./scripts/install_to_bin.sh`
- Addon will be at `bin/aprapipes.node`

---

## Related Documentation

- [Pipeline Author Guide](../docs/declarative-pipeline/PIPELINE_AUTHOR_GUIDE.md)
- [Developer Guide](../docs/declarative-pipeline/DEVELOPER_GUIDE.md)
- [Progress & Module List](../docs/declarative-pipeline/PROGRESS.md)
- [Build Instructions](../README.md)

# ApraPipes Node.js Examples

This directory contains example applications demonstrating the ApraPipes Node.js addon API.

## Prerequisites

1. Build ApraPipes with Node.js addon support:
   ```bash
   # Standard build (Linux/macOS/Windows)
   cmake -B build -S . -DBUILD_NODE_ADDON=ON
   cmake --build build --parallel

   # Jetson build (ARM64 + CUDA)
   cmake -B build -S . -DBUILD_NODE_ADDON=ON -DENABLE_ARM64=ON -DENABLE_CUDA=ON
   cmake --build build -j4
   ```

2. The addon file `aprapipes.node` should be in the `bin/` directory:
   ```bash
   mkdir -p bin
   ln -s ../build/aprapipes.node bin/aprapipes.node
   ```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x64 | ✅ | Full support |
| Linux x64 CUDA | ✅ | GPU modules available |
| macOS | ✅ | Full support |
| Windows | ✅ | Full support |
| **Jetson ARM64** | ✅ | L4TM hardware acceleration |

## Examples

### 1. Basic Pipeline (`basic_pipeline.js`)

A simple "Hello World" example showing how to:
- Load the addon
- Create a pipeline from JSON configuration
- Get module handles
- Set up event handlers

```bash
node examples/node/basic_pipeline.js
```

### 2. PTZ Control (`ptz_control.js`)

Demonstrates runtime property control using the VirtualPTZ module:
- Check for dynamic property support
- Read and write properties at runtime
- Simulate pan/tilt/zoom operations

```bash
node examples/node/ptz_control.js
```

### 3. Event Handling (`event_handling.js`)

Shows how to use the event system:
- Register health and error event handlers
- Method chaining for event registration
- Remove event handlers
- Event data structure

```bash
node examples/node/event_handling.js
```

### 4. Image Processing (`image_processing.js`)

A more complex pipeline with multiple processing stages:
- Color conversion
- VirtualPTZ cropping
- Real-time parameter adjustment

```bash
node examples/node/image_processing.js
```

### 5. Jetson L4TM Demo (`jetson_l4tm_demo.js`) ⚡

**Jetson-specific** example using hardware-accelerated JPEG codec:
- Uses `JPEGDecoderL4TM` and `JPEGEncoderL4TM`
- Leverages NVIDIA L4T Multimedia API
- Achieves 60+ FPS with hardware acceleration

```bash
# Only works on Jetson devices
node examples/node/jetson_l4tm_demo.js
```

**Output:**
```
=== Jetson L4TM Hardware JPEG Demo ===

Pipeline:
  FileReaderModule (JPEG file)
       |
       v
  JPEGDecoderL4TM (HW decode)
       |
       v
  JPEGEncoderL4TM (HW encode @ quality=90)
       |
       v
  FileWriterModule -> ./data/testOutput/

NvMMLiteBlockCreate : Block : BlockType = 256
[JPEG Decode] BeginSequence Display WidthxHeight 1920x454

Generated 181 JPEG files in ./data/testOutput/
Throughput: 60.0 frames/sec (hardware accelerated)
```

## API Quick Reference

### Creating Pipelines

```javascript
const ap = require('./aprapipes.node');

const config = {
    name: "MyPipeline",
    modules: {
        source: { type: "TestSignalGenerator", props: { width: 640, height: 480 } },
        sink: { type: "StatSink" }
    },
    connections: [
        { from: "source", to: "sink" }
    ]
};

const pipeline = ap.createPipeline(config);
```

### Getting Module Handles

```javascript
const module = pipeline.getModule('source');
console.log(module.type);  // "TestSignalGenerator"
console.log(module.id);    // Unique instance ID
```

### Event Handling

```javascript
pipeline
    .on('health', (event) => {
        console.log(`Health: ${event.moduleId} - ${event.message}`);
    })
    .on('error', (event) => {
        console.error(`Error: ${event.moduleId} - ${event.message}`);
    });

// Remove specific handler
pipeline.off('health', myHandler);

// Remove all handlers
pipeline.removeAllListeners('health');
```

### Dynamic Properties (Runtime Control)

```javascript
const ptz = pipeline.getModule('ptz');

// Check if module supports dynamic properties
if (ptz.hasDynamicProperties()) {
    // Get available property names
    const props = ptz.getDynamicPropertyNames();
    // ['roiX', 'roiY', 'roiWidth', 'roiHeight']

    // Read property
    const x = ptz.getProperty('roiX');

    // Write property
    ptz.setProperty('roiX', 0.5);
}
```

### Pipeline Lifecycle

```javascript
// Initialize (validates connections, allocates buffers)
await pipeline.init();

// Start processing
pipeline.run();

// Stop gracefully
await pipeline.stop();
```

## Available Modules

Run the CLI to see all registered modules:

```bash
./bin/aprapipes_cli list-modules
```

Or get details about a specific module:

```bash
./bin/aprapipes_cli describe VirtualPTZ
```

### Jetson-Specific Modules

On Jetson devices, these additional modules are available:

| Module | Description |
|--------|-------------|
| `JPEGDecoderL4TM` | Hardware JPEG decoder (L4T Multimedia) |
| `JPEGEncoderL4TM` | Hardware JPEG encoder (L4T Multimedia) |
| `NvArgusCamera` | CSI camera via Argus API |
| `NvV4L2Camera` | USB camera via V4L2 |
| `NvTransform` | GPU resize/crop/transform |
| `EglRenderer` | EGL display output |

## Testing

Run all Node.js examples:

```bash
# Standard platforms
for f in examples/node/*.js; do
    echo "Testing $f..."
    node "$f" || echo "Failed: $f"
done

# Jetson (includes L4TM tests)
./examples/test_jetson_examples.sh --node
```

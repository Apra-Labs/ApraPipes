# ApraPipes SDK

ApraPipes is a high-performance multimedia pipeline framework with declarative JSON configuration support.

## SDK Contents

```
aprapipes-sdk-{platform}/
├── bin/
│   ├── aprapipes_cli          # Command-line tool for running pipelines
│   ├── aprapipesut            # Unit test executable
│   ├── aprapipes.node         # Node.js addon
│   └── *.so / *.dll / *.dylib # Shared libraries
├── lib/
│   └── *.a / *.lib            # Static libraries
├── include/
│   └── *.h                    # Header files for C++ development
├── examples/
│   ├── basic/                 # Basic JSON pipeline examples
│   ├── cuda/                  # CUDA examples (CUDA builds only)
│   ├── jetson/                # Jetson examples (ARM64 only)
│   └── node/                  # Node.js examples
├── data/
│   ├── frame.jpg              # Sample input image
│   └── faces.jpg              # Sample face image
├── VERSION                    # SDK version string
└── README.md                  # This file
```

## Quick Start

### Using the CLI

```bash
# List available modules
./bin/aprapipes_cli list-modules

# Describe a specific module
./bin/aprapipes_cli describe-module FileReaderModule

# Run a pipeline from JSON
./bin/aprapipes_cli run examples/basic/simple_source_sink.json
```

### Using Node.js

```javascript
const aprapipes = require('./bin/aprapipes.node');

// List available modules
console.log(aprapipes.listModules());

// Create and run a pipeline
const pipeline = aprapipes.createPipeline({
  modules: {
    source: {
      type: "FileReaderModule",
      props: { path: "./data/frame.jpg" }
    }
  }
});

pipeline.start();
```

### Using C++ Library

```cpp
#include "Module.h"
#include "declarative/ModuleFactory.h"

// Create modules from registry
auto factory = ModuleFactory::instance();
auto module = factory.create("FileReaderModule", props);
```

## Platform-Specific Notes

### Windows

- Requires Visual C++ Redistributable 2019 or later
- CUDA DLLs are delay-loaded (CUDA runtime optional for non-GPU operations)

### Linux

- Built with GCC 11+ (x64) or GCC 9.4 (ARM64)
- Shared libraries in `bin/` directory

### macOS

- Built with Apple Clang
- Universal binary support (Intel/ARM)

### Jetson (ARM64)

- Requires JetPack 5.0+
- Includes Jetson-specific examples for:
  - CSI cameras (NvArgusCamera)
  - USB cameras (NvV4L2Camera)
  - Hardware JPEG encode/decode (L4TM)
  - EGL display output

## Examples

### Basic Pipeline (JSON)

```json
{
  "modules": {
    "reader": {
      "type": "FileReaderModule",
      "props": {
        "path": "./data/frame.jpg"
      }
    },
    "encoder": {
      "type": "JPEGEncoderCV",
      "props": {
        "quality": 90
      }
    },
    "writer": {
      "type": "FileWriterModule",
      "props": {
        "path": "./output.jpg",
        "append": false
      }
    }
  },
  "connections": [
    ["reader", "encoder"],
    ["encoder", "writer"]
  ]
}
```

### CUDA Pipeline (GPU-accelerated)

```json
{
  "modules": {
    "reader": {
      "type": "FileReaderModule",
      "props": { "path": "./data/frame.jpg" }
    },
    "decoder": {
      "type": "JPEGDecoderNVJPEG",
      "props": {}
    },
    "blur": {
      "type": "GaussianBlurNPP",
      "props": { "kernelSize": 5 }
    },
    "encoder": {
      "type": "JPEGEncoderNVJPEG",
      "props": { "quality": 90 }
    },
    "writer": {
      "type": "FileWriterModule",
      "props": { "path": "./output_blurred.jpg" }
    }
  },
  "connections": [
    ["reader", "decoder"],
    ["decoder", "blur"],
    ["blur", "encoder"],
    ["encoder", "writer"]
  ]
}
```

## Validating Installation

Run the unit tests to verify your installation:

```bash
# Run all tests
./bin/aprapipesut

# Run specific test suite
./bin/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite
```

## Documentation

- [Pipeline Author Guide](https://github.com/Apra-Labs/ApraPipes/blob/main/docs/declarative-pipeline/PIPELINE_AUTHOR_GUIDE.md)
- [Developer Guide](https://github.com/Apra-Labs/ApraPipes/blob/main/docs/declarative-pipeline/DEVELOPER_GUIDE.md)
- [Node.js Examples](examples/node/README.md)

## Version

Check the `VERSION` file for the SDK version string.

Format: `{major}.{minor}.{patch}-g{commit-hash}` (e.g., `2.0.0-g6146afb`)

## License

See the main ApraPipes repository for license information.

## Support

- [GitHub Issues](https://github.com/Apra-Labs/ApraPipes/issues)
- [GitHub Discussions](https://github.com/Apra-Labs/ApraPipes/discussions)

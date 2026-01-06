# Pipeline Author Guide: Creating JSON Pipelines

> Complete guide for creating video processing pipelines using JSON configuration files.

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [JSON Pipeline Structure](#json-pipeline-structure)
4. [Module Configuration](#module-configuration)
5. [Connections](#connections)
6. [Using the CLI Tool](#using-the-cli-tool)
7. [Using the Schema Generator](#using-the-schema-generator)
8. [Using Node.js](#using-nodejs)
9. [Frame Type Compatibility](#frame-type-compatibility)
10. [Common Pipeline Patterns](#common-pipeline-patterns)
11. [Working with Properties](#working-with-properties)
12. [Troubleshooting](#troubleshooting)
13. [Example Pipelines](#example-pipelines)

---

## Overview

The declarative pipeline system allows you to define video processing pipelines using JSON configuration files instead of writing C++ code. This approach offers several benefits:

- **No compilation required** - Modify pipelines without rebuilding
- **Readable configuration** - JSON is human-friendly and widely supported
- **Validation before runtime** - Catch errors early
- **Documentation as code** - Pipeline files are self-documenting
- **LLM-friendly** - Easy for AI assistants to generate and modify

### What You Can Do

```json
{
  "pipeline": {
    "description": "Read a video, detect faces, and write results"
  },
  "modules": {
    "reader": {
      "type": "FileReaderModule",
      "props": {
        "strFullFileNameWithPattern": "./video.mp4"
      }
    },
    "decoder": {
      "type": "ImageDecoderCV"
    },
    "face_detect": {
      "type": "FaceDetectorXform"
    },
    "writer": {
      "type": "FileWriterModule",
      "props": {
        "strFullFileNameWithPattern": "./faces_output.raw"
      }
    }
  },
  "connections": [
    { "from": "reader", "to": "decoder" },
    { "from": "decoder", "to": "face_detect" },
    { "from": "face_detect", "to": "writer" }
  ]
}
```

---

## Getting Started

### Prerequisites

1. **Build ApraPipes** with declarative pipeline support
2. **CLI tool** - `aprapipes_cli` (built automatically)
3. **Schema generator** - `apra_schema_generator` (optional but recommended)

### Quick Start

1. **Create a simple pipeline file** (`my_pipeline.json`):

```json
{
  "pipeline": {
    "name": "my_first_pipeline"
  },
  "modules": {
    "generator": {
      "type": "TestSignalGenerator",
      "props": {
        "width": 640,
        "height": 480,
        "pattern": "CHECKERBOARD"
      }
    },
    "colorConvert": {
      "type": "ColorConversion",
      "props": {
        "conversionType": "YUV420PLANAR_TO_RGB"
      }
    },
    "encoder": {
      "type": "ImageEncoderCV"
    },
    "writer": {
      "type": "FileWriterModule",
      "props": {
        "strFullFileNameWithPattern": "./output/frame_????.jpg"
      }
    }
  },
  "connections": [
    { "from": "generator", "to": "colorConvert" },
    { "from": "colorConvert", "to": "encoder" },
    { "from": "encoder", "to": "writer" }
  ]
}
```

2. **Validate the pipeline**:

```bash
./aprapipes_cli validate my_pipeline.json
```

3. **Run the pipeline**:

```bash
./aprapipes_cli run my_pipeline.json
```

---

## JSON Pipeline Structure

A pipeline JSON file has three main sections:

### 1. Pipeline Metadata (Optional)

```json
{
  "pipeline": {
    "name": "face_detection_pipeline",
    "description": "Detects faces in video files"
  }
}
```

### 2. Module Definitions

```json
{
  "modules": {
    "reader": {
      "type": "FileReaderModule",
      "props": {
        "strFullFileNameWithPattern": "./video.mp4",
        "readLoop": false
      }
    },
    "decoder": {
      "type": "ImageDecoderCV"
    },
    "writer": {
      "type": "FileWriterModule",
      "props": {
        "strFullFileNameWithPattern": "./output.raw"
      }
    }
  }
}
```

### 3. Connections

```json
{
  "connections": [
    { "from": "reader", "to": "decoder" },
    { "from": "decoder", "to": "writer" }
  ]
}
```

---

## Module Configuration

### Module Definition

Every module needs at least a `type`:

```json
{
  "modules": {
    "my_instance_name": {
      "type": "ModuleTypeName"
    }
  }
}
```

- **Instance name** (`my_instance_name`): Your unique identifier for this module instance
- **Type** (`ModuleTypeName`): The registered module class name

### Properties

Properties are set in a `props` subsection:

```json
{
  "modules": {
    "reader": {
      "type": "FileReaderModule",
      "props": {
        "strFullFileNameWithPattern": "./data/video.mp4",
        "readLoop": true,
        "startIndex": 0
      }
    }
  }
}
```

### Property Types

| Type | JSON Syntax | Example |
|------|-------------|---------|
| String | `"value"` | `"path": "./video.mp4"` |
| Integer | `123` | `"width": 640` |
| Float | `1.5` | `"angle": 45.0` |
| Boolean | `true` or `false` | `"readLoop": true` |

### Finding Available Modules

Use the CLI to list all registered modules:

```bash
# List all modules
./aprapipes_cli list-modules

# Filter by category
./aprapipes_cli list-modules --category Source
./aprapipes_cli list-modules --category Transform
./aprapipes_cli list-modules --category Sink

# Filter by tag
./aprapipes_cli list-modules --tag opencv
./aprapipes_cli list-modules --tag video
```

### Getting Module Details

```bash
./aprapipes_cli describe FileReaderModule
```

Output shows:
- Category and description
- Input and output pins with frame types
- Available properties with types and defaults

---

## Connections

### Basic Connection

```json
{
  "connections": [
    { "from": "source_module", "to": "destination_module" }
  ]
}
```

### Connection with Pin Names

For modules with multiple inputs or outputs, specify pin names:

```json
{
  "connections": [
    { "from": "source_module.output_pin", "to": "destination_module.input_pin" }
  ]
}
```

### Fan-Out (One to Many)

```json
{
  "modules": {
    "source": { "type": "TestSignalGenerator" },
    "sink1": { "type": "StatSink" },
    "sink2": { "type": "StatSink" },
    "splitter": { "type": "Split" }
  },
  "connections": [
    { "from": "source", "to": "splitter" },
    { "from": "splitter.output_1", "to": "sink1" },
    { "from": "splitter.output_2", "to": "sink2" }
  ]
}
```

### Fan-In (Many to One)

```json
{
  "modules": {
    "source1": { "type": "TestSignalGenerator" },
    "source2": { "type": "TestSignalGenerator" },
    "merge": { "type": "Merge" },
    "sink": { "type": "StatSink" }
  },
  "connections": [
    { "from": "source1", "to": "merge.input_1" },
    { "from": "source2", "to": "merge.input_2" },
    { "from": "merge", "to": "sink" }
  ]
}
```

---

## Using the CLI Tool

### Commands

```bash
# Validate a pipeline (check for errors without running)
./aprapipes_cli validate pipeline.json

# Run a pipeline
./aprapipes_cli run pipeline.json

# List registered modules
./aprapipes_cli list-modules

# Get module details
./aprapipes_cli describe ModuleName

# JSON output for tooling
./aprapipes_cli list-modules --json
./aprapipes_cli describe ModuleName --json
```

### Runtime Property Overrides

Override properties without editing the JSON file:

```bash
./aprapipes_cli run pipeline.json --set reader.strFullFileNameWithPattern="./other_video.mp4"
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Validation error |
| 2 | Runtime error |
| 3 | Usage error |

---

## Using the Schema Generator

The schema generator exports module and frame type information for tooling and documentation.

### Generate All Schemas

```bash
./apra_schema_generator --all --output-dir ./schema
```

This creates:
- `modules.json` - All modules with properties
- `frame_types.json` - Frame type hierarchy
- `MODULES.md` - Markdown documentation
- `FRAME_TYPES.md` - Frame type documentation

### Generate Specific Outputs

```bash
# Just JSON
./apra_schema_generator --modules-json modules.json

# Just Markdown
./apra_schema_generator --modules-md MODULES.md
```

### Using Schema for Pipeline Authoring

The generated `modules.json` provides:
- All available module types
- Required and optional properties
- Property types, defaults, and constraints
- Input/output frame types

Example from `modules.json`:

```json
{
  "modules": {
    "FileReaderModule": {
      "category": "source",
      "description": "Reads frames from files matching a pattern",
      "inputs": [],
      "outputs": [
        {"name": "output", "frame_types": ["Frame"]}
      ],
      "properties": {
        "strFullFileNameWithPattern": {
          "type": "string",
          "required": true,
          "description": "File path pattern"
        },
        "readLoop": {
          "type": "bool",
          "default": "true",
          "description": "Loop back to start when reaching end"
        }
      }
    }
  }
}
```

### IDE Integration

Use the JSON schema with IDE plugins for:
- Autocomplete on module types
- Property validation
- Inline documentation

---

## Using Node.js

The declarative pipeline system also supports JavaScript/Node.js for building interactive applications.

### Loading the Addon

```javascript
const ap = require('./aprapipes.node');
```

### Creating and Running a Pipeline

```javascript
const ap = require('./aprapipes.node');

const pipeline = ap.createPipeline({
    modules: {
        source: {
            type: "TestSignalGenerator",
            props: { width: 640, height: 480, pattern: "GRID" }
        },
        colorConvert: {
            type: "ColorConversion",
            props: { conversionType: "YUV420PLANAR_TO_RGB" }
        },
        encoder: { type: "ImageEncoderCV" },
        writer: {
            type: "FileWriterModule",
            props: { strFullFileNameWithPattern: "./output/frame_????.jpg" }
        }
    },
    connections: [
        { from: "source", to: "colorConvert" },
        { from: "colorConvert", to: "encoder" },
        { from: "encoder", to: "writer" }
    ]
});

async function main() {
    // Initialize pipeline
    await pipeline.init();

    // Start processing
    pipeline.run();

    // Let it run for 3 seconds
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Stop and cleanup
    await pipeline.stop();
}

main();
```

### Dynamic Property Control

Node.js allows real-time property changes while the pipeline is running:

```javascript
const pipeline = ap.createPipeline({
    modules: {
        source: { type: "TestSignalGenerator", props: { width: 1920, height: 1080, pattern: "GRID" } },
        colorConvert: { type: "ColorConversion", props: { conversionType: "YUV420PLANAR_TO_RGB" } },
        ptz: { type: "VirtualPTZ" },
        encoder: { type: "ImageEncoderCV" },
        writer: { type: "FileWriterModule", props: { strFullFileNameWithPattern: "./output/ptz_????.jpg" } }
    },
    connections: [
        { from: "source", to: "colorConvert" },
        { from: "colorConvert", to: "ptz" },
        { from: "ptz", to: "encoder" },
        { from: "encoder", to: "writer" }
    ]
});

async function main() {
    await pipeline.init();
    pipeline.run();

    const ptz = pipeline.getModule('ptz');

    // Check if PTZ supports dynamic properties
    if (ptz.hasDynamicProperties()) {
        console.log('Available props:', ptz.getDynamicPropertyNames());

        // Zoom to 2x (50% of frame)
        ptz.setProperty('roiWidth', 0.5);
        ptz.setProperty('roiHeight', 0.5);
        ptz.setProperty('roiX', 0.25);
        ptz.setProperty('roiY', 0.25);

        await new Promise(r => setTimeout(r, 1000));

        // Pan to bottom-right
        ptz.setProperty('roiX', 0.5);
        ptz.setProperty('roiY', 0.5);

        await new Promise(r => setTimeout(r, 1000));
    }

    await pipeline.stop();
}

main();
```

### Event Handling

```javascript
pipeline
    .on('health', (event) => {
        console.log(`[Health] ${event.moduleId}: ${event.message}`);
    })
    .on('error', (event) => {
        console.error(`[Error] ${event.moduleId}: ${event.message}`);
    });
```

### When to Use Node.js vs CLI

| Use Case | Recommended |
|----------|-------------|
| Quick testing / one-off processing | CLI |
| Batch processing with same config | CLI |
| Interactive applications | Node.js |
| Real-time PTZ/parameter control | Node.js |
| Integration with web servers | Node.js |
| Scripted automation | Either |

For complete Node.js API documentation, see [Node.js API Reference](../node-api.md).

---

## Frame Type Compatibility

### Understanding Frame Types

Modules have typed input and output pins. The system validates that connected pins have compatible types.

**Frame Type Hierarchy:**

```
Frame (root - accepts anything)
├── RawImage (uncompressed pixels)
│   └── RawImagePlanar (YUV planar format)
├── EncodedImage (compressed image)
│   ├── H264Data (H.264 video frames)
│   ├── HEVCData (H.265 video frames)
│   └── BMPImage (BMP format)
├── Audio (audio samples)
├── Array (generic data)
└── AnalyticsFrame (detection results)
    ├── FaceDetectsInfo
    └── DefectsInfo
```

### Compatibility Rules

1. **Exact match**: Always works
2. **Subtype to parent**: Works (RawImagePlanar → RawImage)
3. **Different branches**: Fails (RawImage → Audio)

### Common Type Mismatches

**Problem**: Connecting a module that outputs `RawImagePlanar` to one expecting `RawImage`

**Solution**: Use ColorConversion module:

```json
{
  "modules": {
    "source": {
      "type": "TestSignalGenerator",
      "props": { "width": 640, "height": 480 }
    },
    "convert": {
      "type": "ColorConversion",
      "props": { "conversionType": "YUV420PLANAR_TO_RGB" }
    },
    "ptz": {
      "type": "VirtualPTZ"
    }
  },
  "connections": [
    { "from": "source", "to": "convert" },
    { "from": "convert", "to": "ptz" }
  ]
}
```

### Validation Suggestions

When validation detects a type mismatch, it suggests bridge modules:

```
[ERROR] E304 @ connections: Frame type mismatch: generator outputs 'RawImagePlanar',
        but ptz expects [RawImage]

  SUGGESTION: Insert ColorConversion module to convert RawImagePlanar → RawImage
```

---

## Common Pipeline Patterns

### Pattern 1: Simple Source to Sink

```json
{
  "modules": {
    "source": {
      "type": "TestSignalGenerator",
      "props": { "width": 640, "height": 480 }
    },
    "sink": {
      "type": "StatSink"
    }
  },
  "connections": [
    { "from": "source", "to": "sink" }
  ]
}
```

### Pattern 2: File Processing Chain

```json
{
  "modules": {
    "reader": {
      "type": "FileReaderModule",
      "props": {
        "strFullFileNameWithPattern": "./input.jpg",
        "readLoop": false,
        "outputFrameType": "EncodedImage"
      }
    },
    "decoder": {
      "type": "ImageDecoderCV"
    },
    "processor": {
      "type": "RotateCV",
      "props": { "angle": 90.0 }
    },
    "encoder": {
      "type": "ImageEncoderCV"
    },
    "writer": {
      "type": "FileWriterModule",
      "props": { "strFullFileNameWithPattern": "./output.jpg" }
    }
  },
  "connections": [
    { "from": "reader", "to": "decoder" },
    { "from": "decoder", "to": "processor" },
    { "from": "processor", "to": "encoder" },
    { "from": "encoder", "to": "writer" }
  ]
}
```

### Pattern 3: Video with Analytics

```json
{
  "modules": {
    "reader": {
      "type": "FileReaderModule",
      "props": {
        "strFullFileNameWithPattern": "./video.jpg",
        "outputFrameType": "EncodedImage"
      }
    },
    "decoder": {
      "type": "ImageDecoderCV"
    },
    "face_detect": {
      "type": "FaceDetectorXform"
    },
    "writer": {
      "type": "FileWriterModule",
      "props": { "strFullFileNameWithPattern": "./faces.raw" }
    }
  },
  "connections": [
    { "from": "reader", "to": "decoder" },
    { "from": "decoder", "to": "face_detect" },
    { "from": "face_detect", "to": "writer" }
  ]
}
```

### Pattern 4: Type Conversion Bridge

```json
{
  "pipeline": {
    "description": "When you need to connect modules with incompatible types"
  },
  "modules": {
    "source": {
      "type": "TestSignalGenerator",
      "props": { "width": 640, "height": 480 }
    },
    "convert": {
      "type": "ColorConversion",
      "props": { "conversionType": "YUV420PLANAR_TO_RGB" }
    },
    "ptz": {
      "type": "VirtualPTZ"
    },
    "sink": {
      "type": "StatSink"
    }
  },
  "connections": [
    { "from": "source", "to": "convert" },
    { "from": "convert", "to": "ptz" },
    { "from": "ptz", "to": "sink" }
  ]
}
```

### Pattern 5: Split and Merge

```json
{
  "modules": {
    "source": {
      "type": "TestSignalGenerator",
      "props": { "width": 640, "height": 480 }
    },
    "split": {
      "type": "Split"
    },
    "process1": {
      "type": "RotateCV",
      "props": { "angle": 45.0 }
    },
    "process2": {
      "type": "BrightnessContrastControl"
    },
    "sink1": {
      "type": "StatSink"
    },
    "sink2": {
      "type": "StatSink"
    }
  },
  "connections": [
    { "from": "source", "to": "split" },
    { "from": "split.output_1", "to": "process1" },
    { "from": "split.output_2", "to": "process2" },
    { "from": "process1", "to": "sink1" },
    { "from": "process2", "to": "sink2" }
  ]
}
```

---

## Working with Properties

### Required vs Optional Properties

**Required properties** must be set:
```json
{
  "props": {
    "strFullFileNameWithPattern": "./video.mp4"
  }
}
```

**Optional properties** have defaults:
```json
{
  "props": {
    "readLoop": true
  }
}
```

### Property Value Types

```json
{
  "props": {
    "path": "./data/video.mp4",
    "width": 640,
    "height": 480,
    "angle": 45.0,
    "scale": 1.5,
    "enabled": true,
    "loop": false
  }
}
```

### Enum Properties

Some properties accept only specific string values:

```json
{
  "props": {
    "conversionType": "YUV420PLANAR_TO_RGB"
  }
}
```

Use `./aprapipes_cli describe ModuleName` to see allowed values.

### Dynamic Properties

Some properties can be changed at runtime (marked as "dynamic" in module descriptions):

```json
{
  "props": {
    "roiX": 0.1,
    "roiY": 0.1,
    "roiWidth": 0.5,
    "roiHeight": 0.5
  }
}
```

---

## Troubleshooting

### "Module not found: XYZ"

**Cause**: Module type name is incorrect or module not registered.

**Solution**:
```bash
./aprapipes_cli list-modules | grep -i xyz
```

### "Frame type mismatch"

**Cause**: Output of one module doesn't match input of next module.

**Solution**:
1. Check the validation error for suggested bridge modules
2. Use ColorConversion for image type conversions
3. Use ImageDecoderCV for EncodedImage → RawImage

### "Unknown property: xyz"

**Cause**: Property name is misspelled or not supported.

**Solution**:
```bash
./aprapipes_cli describe ModuleName
```

### "Missing required property: xyz"

**Cause**: A required property was not set.

**Solution**: Add the property to your JSON file:
```json
{
  "props": {
    "strFullFileNameWithPattern": "./video.mp4"
  }
}
```

### "Cycle detected in pipeline"

**Cause**: Connections form a loop.

**Solution**: Check your connections for circular references.

### Pipeline runs but produces no output

**Causes**:
1. Source module can't find input files
2. readLoop is false and file was already processed
3. Sink module path is incorrect

**Solutions**:
1. Check file paths are correct and files exist
2. Set `"readLoop": true` for testing
3. Check output file permissions

---

## Example Pipelines

Example pipelines are in `docs/declarative-pipeline/examples/`:

| File | Description |
|------|-------------|
| `01_simple_source_sink.json` | Minimal pipeline (TestSignal → StatSink) |
| `02_three_module_chain.json` | FileReader → ImageDecoder → StatSink |
| `03_split_pipeline.json` | Fan-out with Split module |
| `04_ptz_with_conversion.json` | Type bridge example |
| `09_face_detection_demo.json` | Full face detection pipeline |

### Running Examples

```bash
# Validate an example
./aprapipes_cli validate docs/declarative-pipeline/examples/working/01_simple_source_sink.json

# Run an example
./aprapipes_cli run docs/declarative-pipeline/examples/working/01_simple_source_sink.json
```

---

## Quick Reference

### CLI Commands

```bash
./aprapipes_cli validate <file.json>        # Validate pipeline
./aprapipes_cli run <file.json>             # Run pipeline
./aprapipes_cli list-modules                # List all modules
./aprapipes_cli list-modules --category X   # Filter by category
./aprapipes_cli describe ModuleName         # Module details
```

### Categories

| Category | Description |
|----------|-------------|
| Source | Generate or read frames |
| Sink | Consume or write frames |
| Transform | Process/modify frames |
| Analytics | Detect/analyze content |
| Utility | Flow control, routing |

### Common Bridge Modules

| From Type | To Type | Bridge Module | Property |
|-----------|---------|---------------|----------|
| RawImagePlanar | RawImage | ColorConversion | YUV420PLANAR_TO_RGB |
| EncodedImage | RawImage | ImageDecoderCV | - |
| RawImage | EncodedImage | ImageEncoderCV | - |

---

## Next Steps

- Explore the [examples directory](./examples/)
- Run `./apra_schema_generator --all` for complete module documentation
- See [Developer Guide](./DEVELOPER_GUIDE.md) for creating custom modules

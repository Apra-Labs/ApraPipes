# Pipeline Author Guide: Creating TOML Pipelines

> Complete guide for creating video processing pipelines using TOML configuration files.

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [TOML Pipeline Structure](#toml-pipeline-structure)
4. [Module Configuration](#module-configuration)
5. [Connections](#connections)
6. [Using the CLI Tool](#using-the-cli-tool)
7. [Using the Schema Generator](#using-the-schema-generator)
8. [Frame Type Compatibility](#frame-type-compatibility)
9. [Common Pipeline Patterns](#common-pipeline-patterns)
10. [Working with Properties](#working-with-properties)
11. [Troubleshooting](#troubleshooting)
12. [Example Pipelines](#example-pipelines)

---

## Overview

The declarative pipeline system allows you to define video processing pipelines using TOML configuration files instead of writing C++ code. This approach offers several benefits:

- **No compilation required** - Modify pipelines without rebuilding
- **Readable configuration** - TOML is human-friendly
- **Validation before runtime** - Catch errors early
- **Documentation as code** - Pipeline files are self-documenting

### What You Can Do

```toml
# Read a video, detect faces, and write results
[modules.reader]
type = "FileReaderModule"
  [modules.reader.props]
  strFullFileNameWithPattern = "./video.mp4"

[modules.decoder]
type = "ImageDecoderCV"

[modules.face_detect]
type = "FaceDetectorXform"

[modules.writer]
type = "FileWriterModule"
  [modules.writer.props]
  strFullFileNameWithPattern = "./faces_output.raw"

[[connections]]
from = "reader"
to = "decoder"

[[connections]]
from = "decoder"
to = "face_detect"

[[connections]]
from = "face_detect"
to = "writer"
```

---

## Getting Started

### Prerequisites

1. **Build ApraPipes** with declarative pipeline support
2. **CLI tool** - `aprapipes_cli` (built automatically)
3. **Schema generator** - `apra_schema_generator` (optional but recommended)

### Quick Start

1. **Create a simple pipeline file** (`my_pipeline.toml`):

```toml
[pipeline]
name = "my_first_pipeline"

[modules.generator]
type = "TestSignalGenerator"
  [modules.generator.props]
  width = 640
  height = 480

[modules.stats]
type = "StatSink"

[[connections]]
from = "generator"
to = "stats"
```

2. **Validate the pipeline**:

```bash
./aprapipes_cli validate my_pipeline.toml
```

3. **Run the pipeline**:

```bash
./aprapipes_cli run my_pipeline.toml
```

---

## TOML Pipeline Structure

A pipeline TOML file has three main sections:

### 1. Pipeline Metadata (Optional)

```toml
[pipeline]
name = "face_detection_pipeline"
description = "Detects faces in video files"
```

### 2. Module Definitions

```toml
[modules.reader]
type = "FileReaderModule"
  [modules.reader.props]
  strFullFileNameWithPattern = "./video.mp4"
  readLoop = false

[modules.decoder]
type = "ImageDecoderCV"
# No props needed for this module

[modules.writer]
type = "FileWriterModule"
  [modules.writer.props]
  strFullFileNameWithPattern = "./output.raw"
```

### 3. Connections

```toml
[[connections]]
from = "reader"
to = "decoder"

[[connections]]
from = "decoder"
to = "writer"
```

---

## Module Configuration

### Module Definition

Every module needs at least a `type`:

```toml
[modules.my_instance_name]
type = "ModuleTypeName"
```

- **Instance name** (`my_instance_name`): Your unique identifier for this module instance
- **Type** (`ModuleTypeName`): The registered module class name

### Properties

Properties are set in a `props` subsection:

```toml
[modules.reader]
type = "FileReaderModule"
  [modules.reader.props]
  strFullFileNameWithPattern = "./data/video.mp4"
  readLoop = true
  startIndex = 0
```

### Property Types

| Type | TOML Syntax | Example |
|------|-------------|---------|
| String | `"value"` or `'value'` | `path = "./video.mp4"` |
| Integer | `123` | `width = 640` |
| Float | `1.5` | `angle = 45.0` |
| Boolean | `true` or `false` | `readLoop = true` |

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

```toml
[[connections]]
from = "source_module"
to = "destination_module"
```

### Connection with Pin Names

For modules with multiple inputs or outputs, specify pin names:

```toml
[[connections]]
from = "source_module.output_pin"
to = "destination_module.input_pin"
```

### Fan-Out (One to Many)

```toml
[modules.source]
type = "TestSignalGenerator"

[modules.sink1]
type = "StatSink"

[modules.sink2]
type = "StatSink"

[modules.splitter]
type = "Split"

[[connections]]
from = "source"
to = "splitter"

[[connections]]
from = "splitter.output_1"
to = "sink1"

[[connections]]
from = "splitter.output_2"
to = "sink2"
```

### Fan-In (Many to One)

```toml
[modules.merge]
type = "Merge"

[[connections]]
from = "source1"
to = "merge.input_1"

[[connections]]
from = "source2"
to = "merge.input_2"

[[connections]]
from = "merge"
to = "sink"
```

---

## Using the CLI Tool

### Commands

```bash
# Validate a pipeline (check for errors without running)
./aprapipes_cli validate pipeline.toml

# Run a pipeline
./aprapipes_cli run pipeline.toml

# List registered modules
./aprapipes_cli list-modules

# Get module details
./aprapipes_cli describe ModuleName

# JSON output for tooling
./aprapipes_cli list-modules --json
./aprapipes_cli describe ModuleName --json
```

### Runtime Property Overrides

Override properties without editing the TOML file:

```bash
./aprapipes_cli run pipeline.toml --set reader.strFullFileNameWithPattern="./other_video.mp4"
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

```toml
[modules.source]
type = "TestSignalGenerator"
  [modules.source.props]
  width = 640
  height = 480
# Outputs: RawImagePlanar

[modules.convert]
type = "ColorConversion"
  [modules.convert.props]
  conversionType = "YUV420PLANAR_TO_RGB"
# Converts: RawImagePlanar → RawImage

[modules.ptz]
type = "VirtualPTZ"
# Expects: RawImage

[[connections]]
from = "source"
to = "convert"

[[connections]]
from = "convert"
to = "ptz"
```

### Validation Suggestions

When validation detects a type mismatch, it suggests bridge modules:

```
[ERROR] E304 @ connections: Frame type mismatch: generator outputs 'RawImagePlanar',
        but ptz expects [RawImage]

  SUGGESTION: Insert ColorConversion module to convert RawImagePlanar → RawImage:

  [modules.convert_generator_to_ptz]
  type = "ColorConversion"
    [modules.convert_generator_to_ptz.props]
    conversionType = "YUV420PLANAR_TO_RGB"
```

---

## Common Pipeline Patterns

### Pattern 1: Simple Source to Sink

```toml
[modules.source]
type = "TestSignalGenerator"
  [modules.source.props]
  width = 640
  height = 480

[modules.sink]
type = "StatSink"

[[connections]]
from = "source"
to = "sink"
```

### Pattern 2: File Processing Chain

```toml
[modules.reader]
type = "FileReaderModule"
  [modules.reader.props]
  strFullFileNameWithPattern = "./input.jpg"
  readLoop = false
  outputFrameType = "EncodedImage"

[modules.decoder]
type = "ImageDecoderCV"

[modules.processor]
type = "RotateCV"
  [modules.processor.props]
  angle = 90.0

[modules.encoder]
type = "ImageEncoderCV"

[modules.writer]
type = "FileWriterModule"
  [modules.writer.props]
  strFullFileNameWithPattern = "./output.jpg"

[[connections]]
from = "reader"
to = "decoder"

[[connections]]
from = "decoder"
to = "processor"

[[connections]]
from = "processor"
to = "encoder"

[[connections]]
from = "encoder"
to = "writer"
```

### Pattern 3: Video with Analytics

```toml
[modules.reader]
type = "FileReaderModule"
  [modules.reader.props]
  strFullFileNameWithPattern = "./video.jpg"
  outputFrameType = "EncodedImage"

[modules.decoder]
type = "ImageDecoderCV"

[modules.face_detect]
type = "FaceDetectorXform"

[modules.writer]
type = "FileWriterModule"
  [modules.writer.props]
  strFullFileNameWithPattern = "./faces.raw"

[[connections]]
from = "reader"
to = "decoder"

[[connections]]
from = "decoder"
to = "face_detect"

[[connections]]
from = "face_detect"
to = "writer"
```

### Pattern 4: Type Conversion Bridge

```toml
# When you need to connect modules with incompatible types
[modules.source]
type = "TestSignalGenerator"
  [modules.source.props]
  width = 640
  height = 480

# TestSignalGenerator outputs RawImagePlanar
# VirtualPTZ expects RawImage
# Solution: ColorConversion bridge

[modules.convert]
type = "ColorConversion"
  [modules.convert.props]
  conversionType = "YUV420PLANAR_TO_RGB"

[modules.ptz]
type = "VirtualPTZ"

[modules.sink]
type = "StatSink"

[[connections]]
from = "source"
to = "convert"

[[connections]]
from = "convert"
to = "ptz"

[[connections]]
from = "ptz"
to = "sink"
```

### Pattern 5: Split and Merge

```toml
[modules.source]
type = "TestSignalGenerator"
  [modules.source.props]
  width = 640
  height = 480

[modules.split]
type = "Split"

[modules.process1]
type = "RotateCV"
  [modules.process1.props]
  angle = 45.0

[modules.process2]
type = "BrightnessContrastControl"

[modules.sink1]
type = "StatSink"

[modules.sink2]
type = "StatSink"

[[connections]]
from = "source"
to = "split"

[[connections]]
from = "split.output_1"
to = "process1"

[[connections]]
from = "split.output_2"
to = "process2"

[[connections]]
from = "process1"
to = "sink1"

[[connections]]
from = "process2"
to = "sink2"
```

---

## Working with Properties

### Required vs Optional Properties

**Required properties** must be set:
```toml
[modules.reader.props]
strFullFileNameWithPattern = "./video.mp4"  # Required - no default
```

**Optional properties** have defaults:
```toml
[modules.reader.props]
readLoop = true  # Optional - default is true
```

### Property Value Types

```toml
[modules.example.props]
# String (use quotes)
path = "./data/video.mp4"

# Integer (no quotes)
width = 640
height = 480

# Float (use decimal point)
angle = 45.0
scale = 1.5

# Boolean (true/false)
enabled = true
loop = false
```

### Enum Properties

Some properties accept only specific string values:

```toml
[modules.convert.props]
# Must be one of: RGB_TO_MONO, BGR_TO_RGB, YUV420PLANAR_TO_RGB, etc.
conversionType = "YUV420PLANAR_TO_RGB"
```

Use `./aprapipes_cli describe ModuleName` to see allowed values.

### Dynamic Properties

Some properties can be changed at runtime (marked as "dynamic" in module descriptions):

```toml
[modules.ptz.props]
roiX = 0.1
roiY = 0.1
roiWidth = 0.5
roiHeight = 0.5
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

**Solution**: Add the property to your TOML file:
```toml
[modules.reader.props]
strFullFileNameWithPattern = "./video.mp4"  # Add this
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
2. Set `readLoop = true` for testing
3. Check output file permissions

---

## Example Pipelines

Example pipelines are in `docs/declarative-pipeline/examples/`:

| File | Description |
|------|-------------|
| `01_simple_source_sink.toml` | Minimal pipeline (TestSignal → StatSink) |
| `02_three_module_chain.toml` | FileReader → ImageDecoder → StatSink |
| `03_split_pipeline.toml` | Fan-out with Split module |
| `04_ptz_with_conversion.toml` | Type bridge example |
| `09_face_detection_demo.toml` | Full face detection pipeline |

### Running Examples

```bash
# Validate an example
./aprapipes_cli validate docs/declarative-pipeline/examples/working/01_simple_source_sink.toml

# Run an example
./aprapipes_cli run docs/declarative-pipeline/examples/working/01_simple_source_sink.toml
```

---

## Quick Reference

### CLI Commands

```bash
./aprapipes_cli validate <file.toml>        # Validate pipeline
./aprapipes_cli run <file.toml>             # Run pipeline
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

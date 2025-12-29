# Declarative Pipeline Examples

This directory contains example TOML pipeline configurations demonstrating the declarative approach to building ApraPipes pipelines.

## Quick Start

Instead of writing C++ code like this:

```cpp
auto source = boost::shared_ptr<Module>(new FileReaderModule(readerProps));
auto decoder = boost::shared_ptr<Module>(new H264Decoder(decoderProps));
auto writer = boost::shared_ptr<Module>(new FileWriterModule(writerProps));

source->setNext(decoder);
decoder->setNext(writer);

PipeLine p("my_pipeline");
p.appendModule(source);
p.init();
p.run_all_threaded();
```

You can now write a TOML configuration:

```toml
[pipeline]
name = "my_pipeline"

[modules.source]
type = "FileReaderModule"
    [modules.source.props]
    strFullFileNameWithPattern = "/path/to/input.h264"

[modules.decoder]
type = "H264Decoder"

[modules.writer]
type = "FileWriterModule"
    [modules.writer.props]
    strFullFileNameWithPattern = "/path/to/output.raw"

[[connections]]
from = "source.output"
to = "decoder.input"

[[connections]]
from = "decoder.output"
to = "writer.input"
```

## Examples

| File | Description |
|------|-------------|
| `video_transcode.toml` | Basic 3-module pipeline: read -> decode -> write |
| `face_detection.toml` | Video decoding with face detection using DNN |
| `qr_code_reader.toml` | Scan video frames for QR codes and barcodes |
| `multi_output.toml` | Single source feeding multiple processing branches |

## TOML Structure

### Pipeline Settings

```toml
[pipeline]
name = "PipelineName"
version = "1.0"
description = "Human-readable description"

[pipeline.settings]
queue_size = 10          # Frame buffer size between modules
on_error = "restart_module"  # "stop_pipeline" | "skip_frame"
auto_start = false       # Start processing immediately after init
```

### Module Definitions

```toml
[modules.<instance_id>]
type = "ModuleClassName"
    [modules.<instance_id>.props]
    property1 = value1
    property2 = value2
```

### Connections

```toml
[[connections]]
from = "source_module.output_pin"
to = "dest_module.input_pin"
```

## Registered Modules

The following modules are currently registered with the declarative pipeline system:

| Module | Category | Description |
|--------|----------|-------------|
| `FileReaderModule` | Source | Reads frames from file sequences |
| `H264Decoder` | Transform | Decodes H.264 video to raw frames |
| `FaceDetectorXform` | Analytics | DNN-based face detection |
| `QRReader` | Analytics | QR code and barcode scanning |
| `FileWriterModule` | Sink | Writes frames to file sequences |

## Using the CLI Tool

```bash
# Parse and validate a pipeline
aprapipes_cli validate pipeline.toml

# Show pipeline information
aprapipes_cli info pipeline.toml

# Export module schema
aprapipes_cli schema --format json > modules.json
```

## Property Types

TOML supports the following property types:

```toml
[modules.example.props]
int_value = 42
float_value = 3.14
bool_value = true
string_value = "hello"
int_array = [1, 2, 3]
float_array = [1.0, 2.0, 3.0]
string_array = ["a", "b", "c"]
```

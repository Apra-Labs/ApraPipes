# Declarative Pipeline Construction

> Transform ApraPipes from imperative C++ construction to declarative configuration.

## Overview

This feature allows defining video processing pipelines in JSON instead of writing C++ code:

```json
{
  "pipeline": {
    "name": "face_detection",
    "description": "Face detection pipeline"
  },
  "modules": {
    "source": {
      "type": "FileReaderModule",
      "props": {
        "strFullFileNameWithPattern": "/video.mp4"
      }
    },
    "decoder": {
      "type": "H264Decoder"
    },
    "detector": {
      "type": "FaceDetectorXform",
      "props": {
        "confidenceThreshold": 0.8
      }
    }
  },
  "connections": [
    { "from": "source", "to": "decoder" },
    { "from": "decoder", "to": "detector" }
  ]
}
```

Then run:
```bash
aprapipes_cli run pipeline.json
```

## Documents

| Document | Description |
|----------|-------------|
| [PROGRESS.md](./PROGRESS.md) | Current progress and status tracker |
| [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) | Guide for module developers |
| [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md) | Guide for pipeline authors |
| [BUILD_OPTIONS.md](./BUILD_OPTIONS.md) | Build configuration options |
| [INTEGRATION_TESTS.md](./INTEGRATION_TESTS.md) | Test results and known issues |
| [CUDA_MEMTYPE_DESIGN.md](./CUDA_MEMTYPE_DESIGN.md) | Auto-bridging design document |

## Quick Links

- **GitHub Discussion:** [#471](https://github.com/Apra-Labs/ApraPipes/discussions/471)
- **Project Board:** [Declarative Pipeline Construction](https://github.com/orgs/Apra-Labs/projects)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                               │
├─────────────────────────────────────────────────────────────────────┤
│     JSON File     │    Node.js API    │    LLM Agent                │
└───────┬───────────┴────────┬──────────┴────────┬────────────────────┘
        │                    │                   │
        ▼                    ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND PARSERS                                  │
│                       JsonParser                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PIPELINE DESCRIPTION (IR)                            │
│  • ModuleInstance (id, type, properties)                             │
│  • Connection (from.pin → to.pin)                                    │
│  • PipelineSettings (name, queue_size, on_error)                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           │           ▼
┌───────────────────────┐       │       ┌───────────────────────┐
│  PIPELINE VALIDATOR   │       │       │   SCHEMA GENERATOR    │
│  (Optional, evolving) │       │       │   (Build-time)        │
└───────────────────────┘       │       └───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODULE FACTORY                                   │
│  • Query ModuleRegistry                                              │
│  • Instantiate modules                                               │
│  • Apply properties                                                  │
│  • Connect pins                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RUNNING PIPELINE                                  │
│                  (Existing ApraPipes)                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **C++ as Single Source of Truth** - Metadata lives in C++ headers, extracted at build time
2. **JSON-Only Format** - Simple, widely supported, LLM-friendly
3. **Validator is Non-Blocking** - Factory works without validation; rules added incrementally
4. **Tags for Multi-Dimensional Queries** - Modules and FrameTypes have tags for LLM/filtering
5. **Static Registration** - `REGISTER_MODULE` macro populates registry at program init

## Getting Started

### For Developers

1. Read the [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)
2. Study existing registrations in `base/src/declarative/ModuleRegistrations.cpp`
3. Register your module following the documented patterns

### For Pipeline Authors

1. Read the [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md)
2. Check the [examples](../../examples/) directory
3. Use `aprapipes_cli validate pipeline.json` to check your pipeline

### For Claude Code Agents

```bash
# Check current progress
cat docs/declarative-pipeline/PROGRESS.md

# Read the developer guide for registration patterns
cat docs/declarative-pipeline/DEVELOPER_GUIDE.md

# Verify changes with tests (Boost.Test)
./build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite
```

## Timeline

| Sprint | Description | Status |
|--------|-------------|--------|
| Sprint 1 | Foundations (Types, Registry, Parser) | ✅ Complete |
| Sprint 2 | MVP (Factory, CLI) | ✅ Complete |
| Sprint 3 | Polish (Validation, Docs) | ✅ Complete |
| Sprint 4 | JSON Migration + Node.js Addon | ✅ Complete |
| Sprint 5 | CUDA Module Registration | ✅ Complete |
| Sprint 6 | DRY Refactoring | ✅ Complete |
| Sprint 7 | Auto-Bridging (MemType, ImageType, PipelineAnalyzer) | ✅ Complete |

## Contributing

1. Check [PROGRESS.md](./PROGRESS.md) for current status and future work items
2. Read [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) for module registration patterns
3. Submit PR referencing the relevant GitHub issue

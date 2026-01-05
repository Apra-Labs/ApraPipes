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
| [RFC.md](./RFC.md) | Full RFC with design decisions |
| [PROJECT_PLAN.md](./PROJECT_PLAN.md) | Sprint plan and timeline |
| [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) | Guide for module developers |
| [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md) | Guide for pipeline authors |
| [tasks/](./tasks/) | Detailed task specifications |

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
2. Check [tasks/README.md](./tasks/README.md) for implementation details
3. Register your module in `ModuleRegistrations.cpp`

### For Pipeline Authors

1. Read the [PIPELINE_AUTHOR_GUIDE.md](./PIPELINE_AUTHOR_GUIDE.md)
2. Check the [examples](./examples/) directory
3. Use `aprapipes_cli validate pipeline.json` to check your pipeline

### For Claude Code Agents

```bash
# Read your assigned task
cat docs/declarative-pipeline/tasks/A1-core-metadata-types.md

# Implement in specified location
# Check "Files" section in the spec

# Verify
cd build && ctest -R <test_name>
```

## Timeline

| Sprint | Description | Status |
|--------|-------------|--------|
| Sprint 1 | Foundations (Types, Registry, Parser) | ✅ Complete |
| Sprint 2 | MVP (Factory, CLI) | ✅ Complete |
| Sprint 3 | Polish (Validation, Docs) | ✅ Complete |
| Sprint 4 | JSON Migration | ✅ Complete |

## Contributing

1. Pick a task from [tasks/](./tasks/)
2. Check dependencies are complete
3. Follow the spec's "Implementation Notes"
4. Ensure all acceptance criteria pass
5. Submit PR referencing the GitHub issue

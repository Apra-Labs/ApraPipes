# Declarative Pipeline Construction

> Transform ApraPipes from imperative C++ construction to declarative configuration.

## Overview

This feature allows defining video processing pipelines in TOML/YAML/JSON instead of writing C++ code:

```toml
[pipeline]
name = "face_detection"

[modules.source]
type = "FileReaderModule"
  [modules.source.props]
  path = "/video.mp4"

[modules.decoder]
type = "H264Decoder"

[modules.detector]
type = "FaceDetectorXform"
  [modules.detector.props]
  confidence_threshold = 0.8

[[connections]]
from = "source.output"
to = "decoder.input"

[[connections]]
from = "decoder.output"
to = "detector.input"
```

Then run:
```bash
aprapipes run pipeline.toml
```

## Documents

| Document | Description |
|----------|-------------|
| [RFC.md](./RFC.md) | Full RFC with design decisions |
| [PROJECT_PLAN.md](./PROJECT_PLAN.md) | Sprint plan and timeline |
| [tasks/](./tasks/) | Detailed task specifications |

## Quick Links

- **GitHub Discussion:** [#471](https://github.com/Apra-Labs/ApraPipes/discussions/471)
- **Project Board:** [Declarative Pipeline Construction](https://github.com/orgs/Apra-Labs/projects)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                               │
├─────────────────────────────────────────────────────────────────────┤
│  TOML File    │    YAML File    │    JSON File    │    LLM Agent    │
└───────┬───────┴────────┬────────┴────────┬────────┴────────┬────────┘
        │                │                 │                 │
        ▼                ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND PARSERS                                  │
│  TomlParser         YamlParser         JsonParser                    │
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
2. **Validator is Non-Blocking** - Factory works without validation; rules added incrementally
3. **Tags for Multi-Dimensional Queries** - Modules and FrameTypes have tags for LLM/filtering
4. **Static Registration** - `REGISTER_MODULE` macro populates registry at program init

## Getting Started

### For Developers

1. Read the [RFC](./RFC.md)
2. Check [tasks/README.md](./tasks/README.md) for implementation details
3. Start with A1 (Metadata.h) or B1 (PipelineDescription)

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

| Sprint | Weeks | Milestone | Status |
|--------|-------|-----------|--------|
| Sprint 1 | 1-2 | Foundations (Types, Registry, Parser) | 🔲 |
| Sprint 2 | 3-4 | MVP (Factory, CLI) | 🔲 |
| Sprint 3 | 5+ | Polish (Validation, Docs) | 🔲 |

## Contributing

1. Pick a task from [tasks/](./tasks/)
2. Check dependencies are complete
3. Follow the spec's "Implementation Notes"
4. Ensure all acceptance criteria pass
5. Submit PR referencing the GitHub issue

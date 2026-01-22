# ApraPipes Visual Editor — Design Questions

> **Date:** 2026-01-22
> **Branch:** `feat/visual-editor`
> **Status:** Awaiting answers to proceed with full specification

---

## Overview

This document captures design decisions needed before creating the full project specification for the ApraPipes Visual Editor.

**Vision:** A web-based visual editor that allows users to:
- Create pipelines by dragging modules onto a canvas
- Connect modules visually
- Configure properties via auto-generated forms
- Validate pipelines with real-time feedback
- Run pipelines and see live health metrics
- Use LLM assistance to generate/refine pipelines

**Existing Infrastructure:**
- `schema_generator` tool exports module metadata (properties, pins, types, constraints)
- `aprapipes.node` Node.js addon for pipeline execution
- JSON-based pipeline format with validation
- TypeScript type definitions

---

## Questions

### 1. Target Users & Experience Level

**Question:** Who is the primary audience?

| Option | Implications |
|--------|-------------|
| **A. Developers** | Code view toggle, terminal access, raw JSON editing acceptable |
| **B. Technical operators** | Wizards for common patterns, less raw JSON, more guardrails |
| **C. Non-technical users** | Heavy visual guidance, minimal configuration, templates-first |

This affects property editors (simple sliders vs. full JSON), error message verbosity, and how much we hide complexity.

**Your Answer:**
```
[x] A - Developers (JS developers who need image/video processing without deep domain knowledge)
```

---

### 2. Deployment Model

**Question:** How will users run this?

| Option | Architecture Impact |
|--------|---------------------|
| **A. Localhost only** (dev tool) | Single user, file access to local paths, no auth needed |
| **B. LAN/team server** | Multi-user considerations, shared pipeline storage, basic auth |
| **C. Cloud-hosted SaaS** | Full auth, remote storage, security isolation for pipeline execution |

**Your Answer:**
```
[x] A - Localhost only (designed for future cloud deployment behind Apache/nginx, but NO auth/security in this project)
```

---

### 3. Runtime Visualization Depth

The Node.js bindings currently expose:

| Metric | Available Now | Status |
|--------|---------------|--------|
| Pipeline status | `PL_RUNNING`, `PL_STOPPED`, etc. | ✅ Ready |
| FPS per module | `fps` in module props | ✅ Ready |
| Queue fill levels | `qlen`, `isInputQueFull()` | ✅ Ready |
| Health events | Via `on('health')` | ✅ Ready |
| Error events | Via `on('error')` | ✅ Ready |
| Frames processed count | Not exposed | ❌ Requires C++ work |
| Memory usage per module | Not exposed | ❌ Requires C++ work |
| Frame preview (thumbnails) | Not exposed | ❌ Significant C++ work |

**Question:** Should we scope Phase 1 to what's available now and add frame preview/memory later?

**Your Answer:**
```
[x] Yes - Use available metrics only for v1
    Future: WebRTC for video/audio rendering (mentioned in future plans only, no impact on v1)
```

---

### 4. LLM Integration Scope

**Question:** For the LLM assistant, what level of integration?

| Level | Description |
|-------|-------------|
| **A. Pipeline generator only** | User types "create face detection pipeline" → LLM returns JSON |
| **B. + Iterative refinement** | LLM can see validation errors and fix them automatically |
| **C. + Runtime debugging** | LLM sees health/error events and suggests fixes |
| **D. + Full conversation** | LLM can answer "why is my FPS low?" based on pipeline context |

**Your Answer:**
```
[x] C - + Runtime debugging
    Context: LLM receives system prompt + user conversation + schema-generator output + validator results + runtime log snippets
```

---

### 5. LLM Provider & API Keys

**Question:** How should LLM access be handled?

| Option | Pros | Cons |
|--------|------|------|
| **A. User provides API keys** | No cost to you, flexible | User friction, key management |
| **B. Hosted/proxied** | Seamless UX | Cost, infrastructure |
| **C. Both options** | Flexibility | More implementation work |

**Which providers to support initially?**

```
[ ] Anthropic (Claude)
[ ] OpenAI (GPT-4)
[ ] Google (Gemini)
[ ] Local/Ollama
[ ] All of the above
```

**Your Answer:**
```
API Keys: [x] A - User provides (follow ai-code-buddy pattern: https://github.com/Apra-Labs/ai-code-buddy)

Providers: Design should support easy addition of new LLM providers
```

---

### 6. Technology Stack

**Recommended stack:**

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | React + TypeScript | Large ecosystem, familiar to most |
| **Node Canvas** | React Flow | Purpose-built for node editors, MIT licensed |
| **State Management** | Zustand | Lightweight, works well with React Flow |
| **Styling** | Tailwind CSS | Fast development, utility-first |
| **UI Components** | shadcn/ui | Modern, accessible, composable |
| **Backend** | Express.js | Already have Node.js addon, simple |
| **Real-time** | WebSocket (ws) | For runtime metrics stream |
| **Build** | Vite | Fast, modern bundler |

**Question:** Any strong preferences or alternatives?

**Your Answer:**
```
[x] Approved as-is (React + TypeScript + React Flow + Zustand + Tailwind + shadcn/ui + Express.js + WebSocket + Vite)
```

---

### 7. JSON/Code View

**Question:** Should we include a split view showing JSON alongside the visual editor?

```
┌─────────────────────────────────────────────────────────────┐
│  [Visual]  [JSON]  [Split]                                  │
├─────────────────────────┬───────────────────────────────────┤
│                         │  {                                │
│    ┌─────┐   ┌─────┐    │    "modules": {                   │
│    │src  │───│sink │    │      "source": {                  │
│    └─────┘   └─────┘    │        "type": "TestSignal...",   │
│                         │      }                            │
│                         │    }                              │
│                         │  }                                │
└─────────────────────────┴───────────────────────────────────┘
```

**Your Answer:**
```
[x] Yes - Split view like VS Code markdown (text/preview)
    v1: Read-only JSON view
    Future: Editable JSON (design should accommodate this easily)
```

---

### 8. Validation Timing

The existing validator provides error codes, locations, and suggestions.

**Question:** How aggressive should validation be?

| Option | Behavior |
|--------|----------|
| **A. On-save only** | Validate when user clicks Save |
| **B. On-blur** | Validate each field when focus leaves |
| **C. On-change (debounced)** | Validate as user types, with 500ms delay |
| **D. Continuous** | Always show validation state, update on any change |

**Your Answer:**
```
[x] A - On-save only
    Visual feedback: Module/pin glyphs should clearly show errors/warnings so users know where to focus
```

---

### 9. Connection Handling

The pipeline schema supports:
- Named pins: `{ "from": "decoder.output", "to": "encoder.input" }`
- Default pins: `{ "from": "decoder", "to": "encoder" }`
- Sieve mode: `{ "from": "a", "to": "b", "sieve": true }`

**Question:** How sophisticated should connection visualization be?

| Feature | Include in v1? |
|---------|----------------|
| Simple lines between modules | Required |
| Colored by frame type (e.g., RGB=green, H264=blue) | [x] Yes |
| Show frame type label on hover | [x] Yes |
| Prevent incompatible connections (type mismatch) | [x] Yes |
| Multiple output pins visualized separately | [x] Yes |
| Sieve mode visual indicator | [x] Yes |

**Your Answer:**
```
All connection features included in v1 except "Colored by frame type" (No)
```

---

### 10. Persistence & Workspace

**Question:** How should pipelines be stored?

| Option | Pros | Cons |
|--------|------|------|
| **A. Open/Save file dialogs** | Familiar, flexible | User manages paths |
| **B. Workspace folder** | Organized, like VS Code | Slightly more setup |
| **C. Browser localStorage** | Zero config | Lost if cache cleared |
| **D. Project-based** | Full project with assets | More complex |

**Your Answer:**
```
[x] B - Workspace folder (each project can have its own files, organized structure)
```

---

### 11. Additional Features (Priority Check)

**Question:** Rank these features by priority (1=must have, 2=nice to have, 3=later):

| Feature | Priority (1/2/3) |
|---------|------------------|
| Undo/Redo | 1 (using localStorage, memento pattern) |
| Keyboard shortcuts (delete, copy/paste nodes) | 1 |
| Pipeline templates ("Start from Face Detection") | 3 (but allow opening example JSON files; future: .js files) |
| Module search/filter in palette | 2 |
| Minimap for large pipelines | 3 |
| Dark mode | 3 |
| Export pipeline as image (PNG/SVG) | 3 |
| Import existing JSON files | 1 |
| Recent files list | 1 |

---

### 12. Naming & Branding

**Question:** What should we call this tool?

Suggestions:
- ApraPipes Studio
- ApraPipes Designer
- ApraPipes Visual Editor
- Pipeline Studio
- Other: _______________

**Your Answer:** ApraPipes Studio

---

### 13. Project Location

**Question:** Where should the visual editor code live?

| Option | Path | Notes |
|--------|------|-------|
| **A. In ApraPipes repo** | `tools/visual-editor/` | Monorepo, shared build |
| **B. In ApraPipes repo** | `studio/` | Top-level, prominent |
| **C. Separate repo** | `ApraPipes-Studio` | Independent releases |

**Your Answer:**
```
[x] A - tools/visual-editor/ (distributed with SDK artifacts)
```

---

## Proposed Phase Plan

Once you answer the above, I'll create a detailed spec. Here's the rough phasing:

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| **Phase 1** | Core Editor | Module palette from schema, canvas with React Flow, property panel, save/load JSON |
| **Phase 2** | Validation | Inline errors, issue panel, connection type checking |
| **Phase 3** | Runtime | Start/stop controls, health overlay on nodes, FPS/queue display |
| **Phase 4** | LLM Basic | Chat panel, pipeline generation from natural language |
| **Phase 5** | LLM Advanced | Error remediation, context-aware suggestions |
| **Phase 6** | Polish | Templates, keyboard shortcuts, undo/redo, dark mode |

**Question:** Does this phasing make sense? Any phases to reorder or combine?

**Your Answer:**
```
Phase 6 (Polish) is MORE important than Phases 4-5 (LLM features).
Revised order: Phase 1 → 2 → 3 → 6 → 4 → 5

Additional requirement: Bottom pane for validation errors (like VS Code Problems panel)
```

---

## Next Steps

1. Fill in your answers above
2. Save this file
3. I'll create the full specification based on your answers:
   - `docs/visual-editor/SPECIFICATION.md` — Complete technical spec
   - `docs/visual-editor/ARCHITECTURE.md` — System architecture
   - `docs/visual-editor/PROJECT_PLAN.md` — Sprint breakdown with acceptance criteria

---

## Quick Reference: Existing APIs

### Schema Generator Output (modules.json)
```json
{
  "modules": {
    "TestSignalGenerator": {
      "category": "source",
      "description": "Generates test video signals",
      "inputs": [],
      "outputs": [{ "name": "output", "frame_types": ["RAW_IMAGE"] }],
      "properties": {
        "width": { "type": "int", "default": "640", "min": "1", "max": "4096" },
        "height": { "type": "int", "default": "480" },
        "pattern": { "type": "enum", "enum_values": ["GRADIENT", "CHECKERBOARD", "GRID"] }
      }
    }
  }
}
```

### Node.js API (aprapipes.node)
```javascript
// Create pipeline
const pipeline = ap.createPipeline(config);

// Lifecycle
await pipeline.init();
pipeline.run();
await pipeline.stop();

// Events
pipeline.on('health', (e) => { /* fps, queue info */ });
pipeline.on('error', (e) => { /* error details */ });

// Module access
const module = pipeline.getModule('source');
module.getProperty('width');
module.setProperty('roiX', 0.5);
```

### Validation API
```javascript
const result = ap.validatePipeline(config);
// { valid: false, issues: [{ level: 'error', code: 'E101', message: '...', location: 'modules.source.props.width' }] }
```

# ApraPipes Studio — Visual Pipeline Editor

> **Status:** Planning Complete → Ready for Implementation
> **Branch:** `feat/visual-editor`
> **Location:** `tools/visual-editor/`

---

## Overview

ApraPipes Studio is a web-based visual editor for creating, validating, and running ApraPipes video processing pipelines. It provides a drag-and-drop interface for JavaScript developers who need to build video pipelines without deep image processing knowledge.

**Key Features:**
- 🎨 Visual drag-and-drop pipeline editor (React Flow)
- 🔧 Auto-generated property editors from module schema
- ✅ Real-time validation with clear error reporting
- 📊 Live runtime monitoring (FPS, queue status, health events)
- 🤖 LLM-assisted pipeline generation and debugging
- ⚡ Undo/redo, keyboard shortcuts, workspace management

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[DESIGN_QUESTIONS.md](./DESIGN_QUESTIONS.md)** | Design decisions with user answers |
| **[SPECIFICATION.md](./SPECIFICATION.md)** | Complete technical specification |
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | System architecture for developers |
| **[PROJECT_PLAN.md](./PROJECT_PLAN.md)** | Sprint breakdown with acceptance criteria |
| **[PROGRESS.md](./PROGRESS.md)** | Current implementation status |

---

## Quick Start (After Implementation)

```bash
# Navigate to visual editor
cd tools/visual-editor

# Install dependencies
npm install

# Start development servers
npm run dev

# Open browser to http://localhost:5173
```

---

## Architecture Summary

```
┌─────────────────────────────────────────────────┐
│         Browser (React + TypeScript)            │
│  Canvas | Property Panel | JSON View | Problems │
├─────────────────────────────────────────────────┤
│         Express.js Backend (Node.js)            │
│  REST API | WebSocket | Pipeline Manager        │
├─────────────────────────────────────────────────┤
│         aprapipes.node (C++ Addon)              │
│  Pipeline Execution | Validation                │
└─────────────────────────────────────────────────┘
```

**Tech Stack:**
- Frontend: React 18, TypeScript, React Flow, Zustand, Tailwind CSS, shadcn/ui
- Backend: Express.js, WebSocket (ws)
- Build: Vite
- Native: aprapipes.node

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | 📋 Planned | Core editor (canvas, palette, properties, JSON view) |
| **Phase 2** | 📋 Planned | Validation (errors, problems panel, visual feedback) |
| **Phase 3** | 📋 Planned | Runtime (start/stop, live metrics, health monitoring) |
| **Phase 6** | 📋 Planned | Polish (undo/redo, shortcuts, recent files, import) |
| **Phase 4** | 📋 Planned | LLM Basic (pipeline generation) |
| **Phase 5** | 📋 Planned | LLM Advanced (error remediation, debugging) |

**Estimated Timeline:** 12-16 weeks

---

## Key Design Decisions

### Target Users
JavaScript developers building video pipelines without deep image processing knowledge.

### Deployment Model
Localhost-first (no auth), designed for future cloud deployment behind Apache/nginx.

### Runtime Visualization
v1 uses available metrics (FPS, queue, health). Future: WebRTC for video preview.

### LLM Integration
- User provides API keys (follows [ai-code-buddy](https://github.com/Apra-Labs/ai-code-buddy) pattern)
- Extensible provider system (Anthropic, OpenAI, Ollama)
- Context: schema + pipeline + validation + runtime logs

### JSON View
Read-only in v1 (VS Code-style split view). Future: editable with two-way sync.

### Validation
On-save only. Visual glyphs on modules/pins show errors/warnings.

### Connections
- Show frame type labels on hover
- Prevent incompatible type connections
- Multiple output pins visualized separately
- Sieve mode indicator

### Workspace
Project-based folders (like VS Code). Each project has `pipeline.json`, `data/`, `.studio/`.

### Priority 1 Features
- Undo/redo (localStorage, memento pattern)
- Keyboard shortcuts (Ctrl+S, Ctrl+Z, Delete, Ctrl+C/V, F5)
- Import existing JSON files
- Recent files list
- Module search/filter (priority 2)

### Phase Priority
Phase 6 (Polish) is MORE important than Phases 4-5 (LLM features).
Order: 1 → 2 → 3 → 6 → 4 → 5

---

## Project Structure (Post-Implementation)

```
tools/visual-editor/
├── client/                      # Frontend (Vite + React)
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── services/            # API clients
│   │   ├── store/               # Zustand state
│   │   ├── types/               # TypeScript types
│   │   └── utils/               # Helpers
│   ├── package.json
│   └── vite.config.ts
├── server/                      # Backend (Express.js)
│   ├── src/
│   │   ├── api/                 # REST routes
│   │   ├── services/            # Business logic
│   │   ├── websocket/           # WebSocket server
│   │   └── index.ts
│   ├── package.json
│   └── tsconfig.json
├── shared/                      # Shared types
├── package.json                 # Root workspace
├── start.sh                     # Launch script
└── README.md
```

---

## Integration with ApraPipes SDK

ApraPipes Studio will be distributed with SDK artifacts:

```
aprapipes-sdk-{platform}/
├── bin/
├── lib/
├── include/
├── examples/
├── data/
└── studio/                      # ← ApraPipes Studio
    ├── client/
    ├── server/
    ├── start.sh
    └── README.md
```

Users can launch Studio from the SDK:
```bash
cd aprapipes-sdk-linux-x64/studio
./start.sh
```

---

## Development Workflow

### Prerequisites
- Node.js 18+
- aprapipes.node installed
- `schema_generator` has run (modules.json available)

### Dev Setup
```bash
# Terminal 1: Backend
cd server
npm install
npm run dev  # nodemon

# Terminal 2: Frontend
cd client
npm install
npm run dev  # vite
```

### Testing
```bash
# Unit tests
npm run test

# Component tests
npm run test:components

# E2E tests
npm run test:e2e

# All tests
npm run test:all
```

### Build
```bash
# Production build
npm run build

# Output:
# - client/dist/  (frontend static files)
# - server/dist/  (backend JS)
```

---

## API Overview

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/schema` | GET | Get module schema (from schema_generator) |
| `/api/validate` | POST | Validate pipeline config |
| `/api/pipeline/create` | POST | Create pipeline instance |
| `/api/pipeline/:id/start` | POST | Start pipeline |
| `/api/pipeline/:id/stop` | POST | Stop pipeline |
| `/api/workspace/save` | POST | Save pipeline + layout |
| `/api/workspace/load` | POST | Load pipeline |

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `subscribe` | Client → Server | Subscribe to pipeline events |
| `health` | Server → Client | Module health metrics (FPS, queue) |
| `error` | Server → Client | Runtime errors |
| `status` | Server → Client | Pipeline status changes |

---

## Success Criteria

**Phase 1 Complete:**
- User can drag modules onto canvas
- User can connect modules via pins
- Property panel edits module properties
- JSON view displays pipeline
- Save/load workspace projects

**Phase 2 Complete:**
- Validation runs on save
- Errors in Problems panel
- Inline errors on canvas
- Type mismatch prevention

**Phase 3 Complete:**
- User can start/stop pipelines
- Live metrics stream via WebSocket
- Canvas shows FPS, queue, health
- Errors logged to Problems panel

**Phase 6 Complete:**
- Undo/redo with localStorage
- All keyboard shortcuts work
- Recent files list
- Import existing JSON

**Phases 4-5 Complete:**
- Generate pipelines via LLM
- LLM sees validation + runtime context
- Chat panel with history

---

## Contributing

### Code Style
- TypeScript strict mode
- ESLint + Prettier
- React functional components + hooks
- Zustand for state management

### Git Workflow
- Branch: `feat/visual-editor`
- Commit per sprint deliverable
- PR per phase completion

### Testing Requirements
- 80%+ unit test coverage
- Component tests for key UI
- Integration tests for workflows
- E2E tests for critical paths

---

## Future Enhancements (Post-v1)

| Feature | Phase | Priority |
|---------|-------|----------|
| WebRTC video preview | Phase 7 | High |
| Editable JSON (two-way sync) | Phase 8 | High |
| Templates & examples | Phase 9 | Medium |
| Dark mode | Phase 10 | Medium |
| Export as PNG/SVG | Phase 10 | Low |
| Minimap | Phase 10 | Low |
| Multi-user collaboration | Phase 11 | Future |

---

## Support

**Issues:** Report bugs in [ApraPipes GitHub Issues](https://github.com/Apra-Labs/ApraPipes/issues)

**Documentation:** See [SPECIFICATION.md](./SPECIFICATION.md) for full technical details

**Questions:** Ask in ApraPipes community channels

---

**License:** Same as ApraPipes project

**Maintainer:** ApraPipes Team

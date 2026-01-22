# ApraPipes Studio — Technical Specification

> **Version:** 1.0
> **Date:** 2026-01-22
> **Status:** Draft
> **Branch:** `feat/visual-editor`

---

## 1. Executive Summary

**ApraPipes Studio** is a web-based visual editor for creating, configuring, validating, and running ApraPipes video processing pipelines. The tool targets JavaScript developers who need to build video pipelines without deep image processing expertise.

**Key Goals:**
- Visual drag-and-drop pipeline construction
- Real-time validation with clear error reporting
- Live runtime monitoring (FPS, queue status, health events)
- LLM-assisted pipeline generation and debugging
- Export/import workspace-based projects

**Non-Goals (v1):**
- Video/audio/image rendering (future: WebRTC)
- Authentication or multi-user support
- Cloud deployment infrastructure

---

## 2. User Persona

**Primary User:** JavaScript Developer (Video Pipeline Builder)

| Attribute | Details |
|-----------|---------|
| **Experience** | Proficient in JavaScript/Node.js, minimal image processing knowledge |
| **Goals** | Build video processing pipelines quickly without learning C++ APIs |
| **Pain Points** | Manual JSON editing is error-prone; hard to visualize data flow; unclear error messages |
| **Expectations** | Visual tools similar to VS Code, Figma, or Node-RED |

---

## 3. System Architecture Overview

### 3.1 Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | React 18 + TypeScript | Component-based UI, strong typing, large ecosystem |
| **Canvas** | React Flow | Purpose-built for node editors, handles layout/zoom/pan |
| **State** | Zustand | Lightweight, works seamlessly with React Flow |
| **Styling** | Tailwind CSS | Utility-first, rapid prototyping |
| **UI Components** | shadcn/ui | Modern, accessible, composable components |
| **Backend** | Express.js | Simple REST API, integrates with aprapipes.node |
| **Real-time** | WebSocket (ws) | Streaming runtime metrics from pipelines |
| **Build** | Vite | Fast HMR, modern ESM bundler |
| **LLM Config** | Based on [ai-code-buddy](https://github.com/Apra-Labs/ai-code-buddy) | User-provided API keys, extensible provider system |

### 3.2 Deployment Model

- **Target:** Localhost only (port 3000 by default)
- **Future:** Cloud deployment behind Apache/nginx (NO auth/security in this project)
- **Distribution:** Bundled with SDK artifacts in `tools/visual-editor/`

### 3.3 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (React SPA)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Canvas (React Flow)  │  Property Panel  │  JSON View    │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Module Palette       │  Problems Panel                  │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Toolbar (Run/Stop/Validate/LLM)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ↕ HTTP/REST + WebSocket                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Express.js Backend                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  REST API          │  WebSocket Server                   │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Pipeline Manager  │  Validator  │  Schema Loader       │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  LLM Service (pluggable providers)                      │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Workspace Manager (file I/O)                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ↕ Node.js C++ Addon                                     │
├─────────────────────────────────────────────────────────────────┤
│                    aprapipes.node                               │
│  - createPipeline()   - validatePipeline()                     │
│  - getModule()        - on('health')  - on('error')             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Features

### 4.1 Visual Canvas (React Flow)

**Nodes:**
- Each node represents a module (source, transform, sink, etc.)
- Visual indicators: category color-coding, status badges (running/error/idle)
- Compact mode: show only module name/type
- Expanded mode: show key properties inline

**Connections:**
- Bezier curves between output → input pins
- Show frame type label on hover (e.g., "RAW_IMAGE", "H264Frame")
- Prevent incompatible connections (type mismatch → visual rejection)
- Sieve mode indicator (dashed line or icon)
- Multiple output pins visualized separately

**Interactions:**
- Drag modules from palette onto canvas
- Drag connections from output pins to input pins
- Select nodes → property panel updates
- Delete nodes (keyboard shortcut: Delete/Backspace)
- Copy/paste nodes (Ctrl+C / Ctrl+V)
- Undo/Redo (Ctrl+Z / Ctrl+Shift+Z)

### 4.2 Module Palette

**Source:** `schema_generator` output (modules.json)

**Display:**
- Grouped by category: source, transform, sink, etc.
- Search/filter (priority 2): type to find modules
- Drag-and-drop to canvas
- Show module description on hover

**Example Categories:**
```
Sources
  - TestSignalGenerator
  - FileReaderModule
  - RTSPClientSrc

Transforms
  - CUDAResizeModule
  - FaceDetectorXForm

Sinks
  - FileWriterModule
  - RTSPSink
```

### 4.3 Property Panel

**Trigger:** Select a node on canvas

**Display:**
- Module name (editable → unique ID requirement)
- Module type (read-only)
- Properties grouped by type:
  - **Primitives** (int, float, bool, string): text input, slider, checkbox
  - **Enums**: dropdown
  - **JSON objects**: collapsible JSON editor (future: schema-driven forms)
- Show validation errors inline (red border + tooltip)
- Reset to defaults button

**Example:**
```
Module: source
Type: TestSignalGenerator
─────────────────────────
Width:       [640      ] (int, min: 1, max: 4096)
Height:      [480      ]
Pattern:     [GRADIENT ▾] (enum)
```

### 4.4 JSON View (Read-Only in v1)

**Layout:** Split view toggle (Visual | JSON | Split)

**Behavior:**
- **Visual mode:** Canvas + property panel (full width)
- **JSON mode:** Syntax-highlighted JSON (full width)
- **Split mode:** Canvas on left, JSON on right (50/50 or adjustable)

**v1:** JSON is read-only, auto-synced from canvas state
**Future:** Two-way sync (edit JSON → update canvas)

**Implementation:** Monaco Editor or CodeMirror with JSON syntax highlighting

### 4.5 Problems Panel (Bottom Pane)

**Purpose:** Display validation errors/warnings (like VS Code Problems panel)

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  [Canvas Area]                                              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  PROBLEMS (3 errors, 1 warning)                     [Clear] │
├─────────────────────────────────────────────────────────────┤
│  ⚠ modules.source.props.width: Value 10000 exceeds max 4096│
│  ✖ modules.decoder: Missing required property 'codecType'  │
│  ✖ connections[0]: Type mismatch (H264Frame → RAW_IMAGE)   │
│  ⓘ modules.encoder.props.bitrate: Using default value 2000 │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- Click error → jump to node/connection on canvas
- Filter by severity (errors, warnings, info)
- Clear all button
- Collapsible/resizable

### 4.6 Validation

**Trigger:** On-save (manual validation via Save button or Ctrl+S)

**Backend:** Call `aprapipes.validatePipeline(config)`

**Error Display:**
1. **Canvas:** Red border on problematic nodes/connections
2. **Property Panel:** Inline error messages below invalid fields
3. **Problems Panel:** Aggregated list with clickable links
4. **Status Bar:** Summary (e.g., "3 errors, 1 warning")

**Error Codes:** Use existing validator error codes (E101, W201, etc.)

### 4.7 Runtime Monitoring

**Lifecycle:**
```
User clicks "Run" → Backend calls pipeline.init() → pipeline.run()
                 → WebSocket stream starts
                 → Canvas nodes show live status
User clicks "Stop" → Backend calls pipeline.stop()
                  → WebSocket stream ends
```

**Metrics (from aprapipes.node):**
- **Pipeline status:** PL_RUNNING, PL_STOPPED, etc. (status bar)
- **Per-module FPS:** Display on node (e.g., "24 fps")
- **Queue fill:** Progress bar or percentage (e.g., "Queue: 80%")
- **Health events:** `on('health')` → highlight healthy nodes (green pulse)
- **Error events:** `on('error')` → red badge on node, log to Problems panel

**Visual Indicators:**
- **Idle:** Gray border
- **Running:** Green pulsing border
- **Error:** Red border + error icon
- **Healthy:** Green checkmark badge

**No video/audio rendering in v1.** Future: WebRTC for frame preview.

### 4.8 Workspace Management

**Workspace Structure:**
```
~/aprapipes-studio/
├── project-1/
│   ├── pipeline.json
│   ├── data/
│   │   └── input.mp4
│   └── .studio/
│       ├── layout.json       (canvas positions, zoom level)
│       └── recent.json       (recent files list)
├── project-2/
│   └── pipeline.json
```

**Features:**
- **Open Workspace:** File picker to select workspace folder
- **New Project:** Create folder + empty pipeline.json
- **Save:** Write pipeline.json + layout.json
- **Recent Files:** List last 10 opened projects (priority 1)
- **Import JSON:** Open existing JSON files (priority 1)

### 4.9 Undo/Redo (Priority 1)

**Pattern:** Memento (state snapshots)

**Storage:** localStorage for persistence across sessions

**Implementation:**
- Store canvas state + pipeline config on each mutation
- Limit: 50 undo steps (configurable)
- Shortcuts: Ctrl+Z (undo), Ctrl+Shift+Z (redo)

**State to Track:**
- Add/delete/move nodes
- Add/delete connections
- Property edits
- Module renames

### 4.10 Keyboard Shortcuts (Priority 1)

| Shortcut | Action |
|----------|--------|
| **Ctrl+S** | Save pipeline |
| **Ctrl+Z** | Undo |
| **Ctrl+Shift+Z** | Redo |
| **Ctrl+C** | Copy selected nodes |
| **Ctrl+V** | Paste nodes |
| **Delete / Backspace** | Delete selected nodes |
| **Ctrl+F** | Focus module search |
| **Ctrl+Shift+P** | Open command palette (future) |
| **F5** | Run pipeline |
| **Shift+F5** | Stop pipeline |

---

## 5. LLM Integration

### 5.1 API Key Management

**Pattern:** Follow [ai-code-buddy](https://github.com/Apra-Labs/ai-code-buddy)

**UI:**
- Settings panel (gear icon in toolbar)
- User enters API keys for supported providers
- Keys stored in localStorage (NOT sent to backend unless needed)

**Supported Providers (Extensible):**
- Anthropic (Claude)
- OpenAI (GPT-4)
- Google (Gemini)
- Ollama (local)

**Provider Interface:**
```typescript
interface LLMProvider {
  name: string;
  validateApiKey(key: string): Promise<boolean>;
  generatePipeline(prompt: string, context: LLMContext): Promise<string>;
}
```

### 5.2 LLM Scope (Phase 4-5)

**Phase 4: Basic Generation**
- User types: "Create a face detection pipeline"
- LLM receives:
  - System prompt (ApraPipes module reference)
  - User message
  - Schema generator output (modules.json)
- Returns: pipeline JSON
- Frontend: Parse and load onto canvas

**Phase 5: Iterative Refinement + Debugging**
- LLM receives additional context:
  - Validator output (errors/warnings)
  - Runtime logs (health, error events)
  - Current pipeline state
- User asks: "Why is my FPS low?"
- LLM suggests: "Increase CUDA batch size in encoder"

**System Prompt Example:**
```
You are an expert in ApraPipes video processing pipelines.
Available modules:
- TestSignalGenerator (source): Generates test video signals
- FaceDetectorXForm (transform): Detects faces in frames
- FileWriterModule (sink): Writes frames to disk

User request: {user_message}
Current pipeline: {pipeline_json}
Validation errors: {validator_output}
Runtime logs: {last_50_log_lines}

Generate a valid pipeline JSON that addresses the user's request.
```

### 5.3 LLM UI

**Chat Panel:**
- Slide-out from right (similar to GitHub Copilot)
- Conversation history
- Code blocks with "Apply" button → update canvas
- Clear conversation button

---

## 6. Non-Functional Requirements

### 6.1 Performance

- Canvas with 100 nodes: smooth 60 FPS interaction
- Validation: < 500ms for typical pipelines
- WebSocket latency: < 100ms for runtime metrics

### 6.2 Usability

- First-time user can create a simple pipeline in < 5 minutes
- Error messages include actionable suggestions
- All actions reversible (undo/redo)

### 6.3 Compatibility

- Browsers: Chrome 100+, Firefox 100+, Safari 15+, Edge 100+
- Node.js: 18+ (required for aprapipes.node)
- OS: Windows, Linux, macOS (wherever aprapipes.node runs)

### 6.4 Security

- No authentication in v1 (localhost only)
- Sanitize file paths (prevent directory traversal)
- Validate JSON before passing to aprapipes.node

---

## 7. API Specification

### 7.1 REST Endpoints

#### `GET /api/schema`
Returns module schema from schema_generator.

**Response:**
```json
{
  "modules": {
    "TestSignalGenerator": { ... }
  }
}
```

#### `POST /api/validate`
Validates pipeline configuration.

**Request:**
```json
{
  "config": { "modules": { ... }, "connections": [ ... ] }
}
```

**Response:**
```json
{
  "valid": true,
  "issues": []
}
```

#### `POST /api/pipeline/create`
Creates a new pipeline instance.

**Request:**
```json
{
  "config": { ... }
}
```

**Response:**
```json
{
  "pipelineId": "abc123"
}
```

#### `POST /api/pipeline/:id/start`
Starts pipeline execution.

**Response:**
```json
{
  "status": "running"
}
```

#### `POST /api/pipeline/:id/stop`
Stops pipeline execution.

#### `GET /api/workspace/:path`
Lists files in workspace folder.

#### `POST /api/workspace/save`
Saves pipeline.json and layout.json.

**Request:**
```json
{
  "path": "/path/to/project",
  "pipeline": { ... },
  "layout": { ... }
}
```

### 7.2 WebSocket Events

#### Client → Server

**`subscribe`**
```json
{
  "event": "subscribe",
  "pipelineId": "abc123"
}
```

#### Server → Client

**`health`**
```json
{
  "event": "health",
  "moduleId": "source",
  "fps": 30,
  "qlen": 5,
  "isQueueFull": false
}
```

**`error`**
```json
{
  "event": "error",
  "moduleId": "decoder",
  "message": "Failed to decode frame",
  "timestamp": 1706000000
}
```

**`status`**
```json
{
  "event": "status",
  "pipelineStatus": "PL_RUNNING"
}
```

---

## 8. Data Models

### 8.1 Pipeline Config (JSON)

```typescript
interface PipelineConfig {
  modules: Record<string, ModuleConfig>;
  connections: Connection[];
}

interface ModuleConfig {
  type: string;
  props?: Record<string, any>;
}

interface Connection {
  from: string;  // "moduleId" or "moduleId.pinName"
  to: string;
  sieve?: boolean;
}
```

### 8.2 Canvas Layout (JSON)

```typescript
interface CanvasLayout {
  nodes: Record<string, NodePosition>;
  zoom: number;
  pan: { x: number; y: number };
}

interface NodePosition {
  x: number;
  y: number;
}
```

### 8.3 Module Schema (from schema_generator)

```typescript
interface ModuleSchema {
  category: string;
  description: string;
  inputs: Pin[];
  outputs: Pin[];
  properties: Record<string, PropertySchema>;
}

interface Pin {
  name: string;
  frame_types: string[];
}

interface PropertySchema {
  type: "int" | "float" | "bool" | "string" | "enum" | "json";
  default?: string;
  min?: string;
  max?: string;
  enum_values?: string[];
}
```

---

## 9. Error Handling

### 9.1 Error Categories

| Code | Category | Example |
|------|----------|---------|
| **E1xx** | Config errors | Missing required property |
| **E2xx** | Connection errors | Type mismatch |
| **E3xx** | Runtime errors | Module init failure |
| **W1xx** | Warnings | Using default value |
| **I1xx** | Info | Successful operation |

### 9.2 Error Display Strategy

1. **Critical errors** (E3xx): Modal dialog, block execution
2. **Config errors** (E1xx, E2xx): Problems panel + inline
3. **Warnings** (W1xx): Problems panel only
4. **Info** (I1xx): Status bar or console

---

## 10. Testing Strategy

### 10.1 Unit Tests

- **Frontend:** React Testing Library for components
- **Backend:** Jest for API endpoints, pipeline manager

### 10.2 Integration Tests

- Load example JSON → verify canvas renders correctly
- Validate invalid pipeline → check error display
- Run simple pipeline → verify runtime metrics stream

### 10.3 E2E Tests

- Playwright for full user workflows:
  - Create pipeline from scratch
  - Add modules, connect, configure
  - Validate, fix errors, run
  - Stop, save, reopen

---

## 11. Deployment & Distribution

### 11.1 Build Artifacts

```
tools/visual-editor/
├── client/          (Vite build output)
│   ├── index.html
│   ├── assets/
├── server/          (Express.js)
│   ├── index.js
│   ├── api/
│   ├── services/
├── package.json
├── README.md
└── start.sh         (Launch script)
```

### 11.2 Startup Script

**`start.sh`:**
```bash
#!/bin/bash
cd "$(dirname "$0")"
npm install --production
node server/index.js
```

Opens browser to `http://localhost:3000`

### 11.3 SDK Integration

Add to SDK packaging scripts (`.github/workflows/build-test*.yml`):
```yaml
- name: Package Studio
  run: |
    cp -r tools/visual-editor $SDK_DIR/studio
    cd $SDK_DIR/studio
    npm install --production
```

---

## 12. Future Enhancements (Out of Scope for v1)

| Feature | Description | Phase |
|---------|-------------|-------|
| **Video Preview** | WebRTC-based frame streaming | Phase 7 |
| **Editable JSON** | Two-way sync between JSON and canvas | Phase 8 |
| **Templates** | Pre-built pipelines (face detection, streaming, etc.) | Phase 6 (priority 3) |
| **Dark Mode** | Theme switcher | Phase 6 (priority 3) |
| **Export as Image** | PNG/SVG export of canvas | Phase 6 (priority 3) |
| **Minimap** | Overview for large pipelines | Phase 6 (priority 3) |
| **Multi-user** | Collaborative editing | Phase 9+ |
| **Cloud Deployment** | Auth, remote storage | Phase 9+ |

---

## 13. Open Questions

1. **Module Icons:** Should each module have a custom icon, or use category colors only?
2. **Property Validation:** Real-time (on-change) or batch (on-save)?
   - **Decision:** On-save (per requirements)
3. **LLM Rate Limits:** How to handle API rate limits gracefully?
   - **Suggestion:** Show retry countdown, allow manual retry

---

## 14. Success Criteria

**Phase 1 Complete When:**
- [x] User can drag modules onto canvas
- [x] User can connect modules via pins
- [x] Property panel shows/edits module properties
- [x] JSON view displays read-only pipeline config
- [x] Save/load workspace projects

**Phase 2 Complete When:**
- [x] Validation runs on save
- [x] Errors displayed in Problems panel
- [x] Inline errors shown on canvas nodes
- [x] Type mismatch prevents connections

**Phase 3 Complete When:**
- [x] User can start/stop pipeline
- [x] Runtime metrics stream via WebSocket
- [x] Canvas shows FPS, queue status, health
- [x] Errors logged to Problems panel

**Phase 6 Complete When:**
- [x] Undo/redo works with localStorage
- [x] Keyboard shortcuts functional
- [x] Recent files list
- [x] Import existing JSON

**Phase 4-5 Complete When:**
- [x] User can generate pipelines via LLM
- [x] LLM sees validation errors and runtime logs
- [x] Chat panel with conversation history

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **Module** | A processing unit (source, transform, sink) |
| **Pin** | Input or output connection point on a module |
| **Connection** | Data flow link between two modules |
| **Sieve** | Pass-through mode for connections |
| **Frame Type** | Data format (RAW_IMAGE, H264Frame, etc.) |
| **Schema Generator** | Tool that exports module metadata |
| **Workspace** | Folder containing project files |

---

**End of Specification**

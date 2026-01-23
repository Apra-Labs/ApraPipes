# ApraPipes Studio

A visual pipeline editor for building ApraPipes video processing pipelines.

## Overview

ApraPipes Studio provides a web-based graphical interface for creating, editing, and managing ApraPipes pipelines. Users can drag and drop modules onto a canvas, connect them together, configure properties, and export the configuration as JSON.

## Features

### Phase 1 (Complete)

- **Module Palette**: Browse available modules grouped by category (Source, Transform, Sink, Filter, Analytics)
- **Canvas Editor**: Drag-drop interface using React Flow
- **Custom Nodes**: Visual representation of modules with input/output pins
- **Connection Management**: Create connections by dragging between pins
- **Property Panel**: Configure module properties with type-specific editors
- **JSON View**: Monaco Editor integration for viewing/editing raw JSON
- **View Modes**: Visual, JSON, or Split view options
- **Workspace Management**: New, Open, Save workspace operations

### Phase 2 (In Progress)

- **Pipeline Validation**: Validate pipeline configuration against schema
- **Problems Panel**: View validation issues (errors, warnings, info)
- **Visual Error Feedback**: Nodes show error/warning badges
- **Click to Navigate**: Click an issue to jump to the affected node
- **Validation API**: `POST /api/validate` endpoint for backend validation

## Getting Started

### Prerequisites

- Node.js 18+
- npm 9+

### Installation

```bash
# Navigate to the visual editor directory
cd tools/visual-editor

# Install dependencies for both client and server
cd client && npm install && cd ..
cd server && npm install && cd ..
```

### Development

Run the server and client in separate terminals:

```bash
# Terminal 1: Start the server (port 3000)
cd tools/visual-editor/server
npm run dev

# Terminal 2: Start the client (port 5173)
cd tools/visual-editor/client
npm run dev
```

Then open http://localhost:5173 in your browser.

### Building for Production

```bash
# Build server
cd tools/visual-editor/server
npm run build

# Build client
cd tools/visual-editor/client
npm run build
```

## Usage

### Creating a Pipeline

1. **Add Modules**: Drag modules from the left palette onto the canvas
2. **Connect Modules**: Drag from output pins (green, right side) to input pins (blue, left side)
3. **Configure Properties**: Select a node and use the property panel on the right
4. **Rename Modules**: Edit the module name in the property panel header

### View Modes

Toggle between views using the toolbar buttons:

- **Visual**: Canvas-only view for editing
- **JSON**: Monaco editor showing the pipeline JSON configuration
- **Split**: Side-by-side visual and JSON views

### File Operations

- **New**: Create a new empty workspace (Ctrl+N)
- **Open**: Open an existing workspace file
- **Save**: Save the current workspace (Ctrl+S)

### Module Categories

| Category | Color | Description |
|----------|-------|-------------|
| Source | Blue | Input modules (file, camera, network) |
| Transform | Green | Processing modules (resize, convert, encode) |
| Sink | Red | Output modules (file writer, display) |
| Filter | Yellow | Filtering modules (crop, overlay) |
| Analytics | Purple | Analysis modules (detection, classification) |

### Keyboard Shortcuts (Coming Soon)

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New workspace |
| Ctrl+O | Open workspace |
| Ctrl+S | Save workspace |
| Delete | Delete selected node |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |

## Architecture

```
tools/visual-editor/
├── client/                 # React frontend (Vite + TypeScript)
│   ├── src/
│   │   ├── components/     # UI components
│   │   │   ├── Canvas/     # React Flow canvas, ModuleNode
│   │   │   ├── Panels/     # ModulePalette, PropertyPanel, JsonView, ProblemsPanel
│   │   │   └── Toolbar/    # Toolbar, StatusBar
│   │   ├── store/          # Zustand state management
│   │   │   ├── canvasStore.ts      # Canvas nodes/edges
│   │   │   ├── pipelineStore.ts    # Pipeline config
│   │   │   ├── workspaceStore.ts   # File operations
│   │   │   └── uiStore.ts          # UI state
│   │   ├── services/       # API client
│   │   └── types/          # TypeScript types
│   └── package.json
├── server/                 # Express backend
│   ├── src/
│   │   ├── api/            # REST API routes
│   │   ├── services/       # Business logic
│   │   │   ├── SchemaLoader.ts     # Module schema loading
│   │   │   ├── WorkspaceManager.ts # File I/O
│   │   │   └── Validator.ts        # Pipeline validation
│   │   └── utils/          # Logging, helpers
│   └── package.json
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/schema | Get module schema |
| POST | /api/workspace/load | Load workspace file |
| POST | /api/workspace/save | Save workspace file |
| GET | /api/workspace/list | List workspace files |
| POST | /api/validate | Validate pipeline configuration |

## Development

### Running Tests

```bash
# Run client tests
cd tools/visual-editor/client
npm test

# Run server tests
cd tools/visual-editor/server
npm test

# Run with coverage
npm run test:coverage
```

### Linting

```bash
# Client
cd tools/visual-editor/client
npm run lint

# Server
cd tools/visual-editor/server
npm run lint
```

### Type Checking

```bash
# Client
cd tools/visual-editor/client
npm run type-check

# Server
cd tools/visual-editor/server
npm run type-check
```

## Roadmap

- **Phase 2**: Validation - Pipeline validation with visual error feedback
- **Phase 3**: Runtime - Pipeline execution and live metrics monitoring
- **Phase 4**: LLM Basic - AI-assisted pipeline generation
- **Phase 5**: LLM Advanced - Iterative refinement and debugging
- **Phase 6**: Polish - Undo/redo, keyboard shortcuts, search

## Contributing

See the main ApraPipes CONTRIBUTING.md for guidelines.

## License

See the main ApraPipes LICENSE file.

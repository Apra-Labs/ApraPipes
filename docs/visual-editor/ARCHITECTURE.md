# ApraPipes Studio — System Architecture

> **Version:** 1.0
> **Date:** 2026-01-22
> **Purpose:** Detailed technical architecture for developers

---

## 1. Overview

ApraPipes Studio is a full-stack web application with three main layers:

1. **Frontend (React SPA):** Visual editor UI, canvas, property panels
2. **Backend (Express.js):** REST API, WebSocket server, pipeline lifecycle management
3. **Native Layer (aprapipes.node):** C++ addon providing pipeline execution, validation

---

## 2. Directory Structure

```
tools/visual-editor/
├── client/                      # Frontend (Vite + React)
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── Canvas/
│   │   │   │   ├── Canvas.tsx
│   │   │   │   ├── ModuleNode.tsx
│   │   │   │   ├── ConnectionEdge.tsx
│   │   │   │   └── NodeToolbar.tsx
│   │   │   ├── Panels/
│   │   │   │   ├── ModulePalette.tsx
│   │   │   │   ├── PropertyPanel.tsx
│   │   │   │   ├── ProblemsPanel.tsx
│   │   │   │   ├── JsonView.tsx
│   │   │   │   └── LLMChat.tsx
│   │   │   ├── Toolbar/
│   │   │   │   ├── Toolbar.tsx
│   │   │   │   └── StatusBar.tsx
│   │   │   └── Settings/
│   │   │       └── LLMSettings.tsx
│   │   ├── services/            # API clients
│   │   │   ├── api.ts           # REST API wrapper
│   │   │   ├── websocket.ts     # WebSocket client
│   │   │   └── llm.ts           # LLM provider interface
│   │   ├── store/               # Zustand state management
│   │   │   ├── canvasStore.ts   # Canvas state (nodes, edges, selection)
│   │   │   ├── pipelineStore.ts # Pipeline config, validation
│   │   │   ├── runtimeStore.ts  # Runtime metrics, status
│   │   │   └── workspaceStore.ts# Workspace, recent files
│   │   ├── types/               # TypeScript definitions
│   │   │   ├── pipeline.ts
│   │   │   ├── schema.ts
│   │   │   └── runtime.ts
│   │   ├── utils/               # Helpers
│   │   │   ├── validator.ts     # Frontend validation helpers
│   │   │   ├── history.ts       # Undo/redo memento
│   │   │   └── storage.ts       # localStorage wrapper
│   │   ├── App.tsx              # Root component
│   │   ├── main.tsx             # Entry point
│   │   └── index.css            # Global styles (Tailwind)
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── server/                      # Backend (Express.js)
│   ├── src/
│   │   ├── api/                 # REST routes
│   │   │   ├── schema.ts        # GET /api/schema
│   │   │   ├── validate.ts      # POST /api/validate
│   │   │   ├── pipeline.ts      # Pipeline lifecycle endpoints
│   │   │   └── workspace.ts     # Workspace file operations
│   │   ├── services/            # Business logic
│   │   │   ├── PipelineManager.ts # Manages pipeline instances
│   │   │   ├── SchemaLoader.ts   # Loads modules.json from schema_generator
│   │   │   ├── Validator.ts      # Wraps aprapipes.validatePipeline
│   │   │   ├── WorkspaceManager.ts # File I/O for projects
│   │   │   └── LLMService.ts     # LLM provider abstraction
│   │   ├── websocket/           # WebSocket server
│   │   │   └── MetricsStream.ts # Streams runtime metrics
│   │   ├── types/               # Shared types
│   │   │   └── index.ts
│   │   ├── index.ts             # Express app entry
│   │   └── config.ts            # Configuration (ports, paths)
│   ├── package.json
│   └── tsconfig.json
├── shared/                      # Shared types/utils (client + server)
│   └── types.ts
├── package.json                 # Root package (workspace setup)
├── README.md
├── start.sh                     # Launch script
└── .env.example                 # Example config
```

---

## 3. Frontend Architecture

### 3.1 Component Hierarchy

```
App
├── Toolbar
│   ├── RunButton
│   ├── StopButton
│   ├── ValidateButton
│   ├── LLMButton
│   └── SettingsButton
├── StatusBar
│   └── PipelineStatus
├── MainLayout (split view)
│   ├── LeftPane
│   │   └── ModulePalette
│   │       ├── CategoryGroup (collapsible)
│   │       └── ModuleItem (draggable)
│   ├── CenterPane
│   │   ├── ViewModeToggle (Visual | JSON | Split)
│   │   ├── Canvas (React Flow)
│   │   │   ├── ModuleNode (custom node)
│   │   │   ├── ConnectionEdge (custom edge)
│   │   │   └── MiniMap (optional)
│   │   └── JsonView (Monaco Editor)
│   ├── RightPane
│   │   └── PropertyPanel
│   │       ├── ModuleInfo
│   │       └── PropertyEditor
│   │           ├── IntInput
│   │           ├── FloatInput
│   │           ├── BoolCheckbox
│   │           ├── StringInput
│   │           ├── EnumDropdown
│   │           └── JsonEditor
│   └── BottomPane
│       └── ProblemsPanel
│           └── ProblemItem (clickable)
└── LLMChat (slide-out)
    ├── ConversationHistory
    ├── ChatInput
    └── ApplyButton
```

### 3.2 State Management (Zustand)

#### Canvas Store (`canvasStore.ts`)

```typescript
interface CanvasStore {
  nodes: Node[];               // React Flow nodes
  edges: Edge[];               // React Flow edges
  selectedNodeId: string | null;

  addNode: (node: Node) => void;
  removeNode: (id: string) => void;
  updateNodePosition: (id: string, position: XYPosition) => void;

  addEdge: (edge: Edge) => void;
  removeEdge: (id: string) => void;

  selectNode: (id: string) => void;
  clearSelection: () => void;

  // Undo/redo
  history: StateSnapshot[];
  historyIndex: number;
  undo: () => void;
  redo: () => void;
  snapshot: () => void;
}
```

#### Pipeline Store (`pipelineStore.ts`)

```typescript
interface PipelineStore {
  config: PipelineConfig;      // Current pipeline JSON
  schema: ModuleSchema[];      // From schema_generator

  updateModuleProperty: (moduleId: string, key: string, value: any) => void;
  renameModule: (oldId: string, newId: string) => void;

  validationResult: ValidationResult | null;
  validate: () => Promise<void>;

  toJSON: () => PipelineConfig;
  fromJSON: (config: PipelineConfig) => void;
}
```

#### Runtime Store (`runtimeStore.ts`)

```typescript
interface RuntimeStore {
  pipelineId: string | null;
  status: PipelineStatus;      // IDLE, RUNNING, STOPPED, ERROR

  moduleMetrics: Record<string, ModuleMetrics>; // { fps, qlen, isQueueFull }
  errors: RuntimeError[];

  connect: (pipelineId: string) => void;
  disconnect: () => void;

  onHealthEvent: (event: HealthEvent) => void;
  onErrorEvent: (event: ErrorEvent) => void;
  onStatusEvent: (event: StatusEvent) => void;
}

interface ModuleMetrics {
  fps: number;
  qlen: number;
  isQueueFull: boolean;
}
```

#### Workspace Store (`workspaceStore.ts`)

```typescript
interface WorkspaceStore {
  currentPath: string | null;
  recentFiles: string[];

  openWorkspace: (path: string) => Promise<void>;
  saveWorkspace: () => Promise<void>;
  addToRecent: (path: string) => void;

  loadPipeline: (path: string) => Promise<PipelineConfig>;
  savePipeline: (path: string, config: PipelineConfig) => Promise<void>;
}
```

### 3.3 Custom React Flow Nodes

#### Module Node (`ModuleNode.tsx`)

```typescript
interface ModuleNodeProps {
  id: string;
  data: {
    type: string;           // Module type (e.g., "TestSignalGenerator")
    label: string;          // Display name
    category: string;       // "source" | "transform" | "sink"
    inputs: Pin[];
    outputs: Pin[];
    status: NodeStatus;     // "idle" | "running" | "error"
    metrics?: ModuleMetrics;
    errors?: string[];
  };
}

const ModuleNode: React.FC<ModuleNodeProps> = ({ id, data }) => {
  return (
    <div className={`module-node ${data.status}`}>
      <div className="module-header">
        <span className="category-badge">{data.category}</span>
        <h3>{data.label}</h3>
        {data.status === 'error' && <ErrorIcon />}
      </div>

      <div className="pins">
        {data.inputs.map(pin => (
          <Handle type="target" position={Position.Left} id={pin.name} />
        ))}
        {data.outputs.map(pin => (
          <Handle type="source" position={Position.Right} id={pin.name} />
        ))}
      </div>

      {data.metrics && (
        <div className="metrics">
          <span>FPS: {data.metrics.fps}</span>
          <span>Queue: {data.metrics.qlen}</span>
        </div>
      )}
    </div>
  );
};
```

**Styling Strategy:**
- Use Tailwind classes for base styles
- Category-specific colors via CSS classes (`.source`, `.transform`, `.sink`)
- Status indicators via border colors (green=running, red=error, gray=idle)

### 3.4 Connection Validation

**Frontend Logic (`canvasStore.ts`):**

```typescript
const addEdge = (edge: Edge) => {
  const sourceNode = nodes.find(n => n.id === edge.source);
  const targetNode = nodes.find(n => n.id === edge.target);

  const sourcePin = sourceNode.data.outputs.find(p => p.name === edge.sourceHandle);
  const targetPin = targetNode.data.inputs.find(p => p.name === edge.targetHandle);

  // Check frame type compatibility
  const compatible = sourcePin.frame_types.some(ft =>
    targetPin.frame_types.includes(ft)
  );

  if (!compatible) {
    showError(`Type mismatch: ${sourcePin.frame_types} → ${targetPin.frame_types}`);
    return;
  }

  set(state => ({ edges: [...state.edges, edge] }));
};
```

### 3.5 Undo/Redo Implementation

**Memento Pattern (`utils/history.ts`):**

```typescript
interface StateSnapshot {
  timestamp: number;
  nodes: Node[];
  edges: Edge[];
  config: PipelineConfig;
}

class History {
  private snapshots: StateSnapshot[] = [];
  private currentIndex = -1;
  private maxSize = 50;

  snapshot(state: CanvasStore & PipelineStore): void {
    // Remove future snapshots if in middle of history
    this.snapshots = this.snapshots.slice(0, this.currentIndex + 1);

    const snapshot: StateSnapshot = {
      timestamp: Date.now(),
      nodes: cloneDeep(state.nodes),
      edges: cloneDeep(state.edges),
      config: cloneDeep(state.config),
    };

    this.snapshots.push(snapshot);
    this.currentIndex++;

    // Limit size
    if (this.snapshots.length > this.maxSize) {
      this.snapshots.shift();
      this.currentIndex--;
    }

    // Persist to localStorage
    localStorage.setItem('history', JSON.stringify(this.snapshots));
  }

  undo(): StateSnapshot | null {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      return this.snapshots[this.currentIndex];
    }
    return null;
  }

  redo(): StateSnapshot | null {
    if (this.currentIndex < this.snapshots.length - 1) {
      this.currentIndex++;
      return this.snapshots[this.currentIndex];
    }
    return null;
  }
}
```

---

## 4. Backend Architecture

### 4.1 Express.js App Structure

**`server/src/index.ts`:**

```typescript
import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import schemaRouter from './api/schema';
import validateRouter from './api/validate';
import pipelineRouter from './api/pipeline';
import workspaceRouter from './api/workspace';
import { MetricsStream } from './websocket/MetricsStream';

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

app.use(cors());
app.use(express.json());

// Serve frontend static files
app.use(express.static('../client/dist'));

// API routes
app.use('/api/schema', schemaRouter);
app.use('/api/validate', validateRouter);
app.use('/api/pipeline', pipelineRouter);
app.use('/api/workspace', workspaceRouter);

// WebSocket for runtime metrics
const metricsStream = new MetricsStream(wss);

server.listen(3000, () => {
  console.log('ApraPipes Studio running at http://localhost:3000');
});
```

### 4.2 Pipeline Manager

**`server/src/services/PipelineManager.ts`:**

```typescript
import aprapipes from 'aprapipes';

interface PipelineInstance {
  id: string;
  pipeline: any;  // aprapipes.Pipeline
  config: PipelineConfig;
  status: PipelineStatus;
}

class PipelineManager {
  private instances = new Map<string, PipelineInstance>();

  async create(config: PipelineConfig): Promise<string> {
    const id = generateId();
    const pipeline = aprapipes.createPipeline(config);

    // Set up event listeners
    pipeline.on('health', (event) => {
      this.onHealthEvent(id, event);
    });

    pipeline.on('error', (event) => {
      this.onErrorEvent(id, event);
    });

    await pipeline.init();

    this.instances.set(id, {
      id,
      pipeline,
      config,
      status: 'INITIALIZED',
    });

    return id;
  }

  async start(id: string): Promise<void> {
    const instance = this.instances.get(id);
    if (!instance) throw new Error('Pipeline not found');

    instance.pipeline.run();
    instance.status = 'RUNNING';
  }

  async stop(id: string): Promise<void> {
    const instance = this.instances.get(id);
    if (!instance) throw new Error('Pipeline not found');

    await instance.pipeline.stop();
    instance.status = 'STOPPED';
  }

  get(id: string): PipelineInstance | undefined {
    return this.instances.get(id);
  }

  private onHealthEvent(pipelineId: string, event: any): void {
    // Broadcast to WebSocket clients
    MetricsStream.broadcast({
      pipelineId,
      event: 'health',
      data: event,
    });
  }

  private onErrorEvent(pipelineId: string, event: any): void {
    MetricsStream.broadcast({
      pipelineId,
      event: 'error',
      data: event,
    });
  }
}

export default new PipelineManager();
```

### 4.3 Validator Service

**`server/src/services/Validator.ts`:**

```typescript
import aprapipes from 'aprapipes';

interface ValidationResult {
  valid: boolean;
  issues: ValidationIssue[];
}

interface ValidationIssue {
  level: 'error' | 'warning' | 'info';
  code: string;
  message: string;
  location: string;  // JSONPath (e.g., "modules.source.props.width")
  suggestion?: string;
}

class Validator {
  validate(config: PipelineConfig): ValidationResult {
    try {
      const result = aprapipes.validatePipeline(config);
      return result;
    } catch (error) {
      return {
        valid: false,
        issues: [{
          level: 'error',
          code: 'E000',
          message: error.message,
          location: 'root',
        }],
      };
    }
  }
}

export default new Validator();
```

### 4.4 WebSocket Metrics Stream

**`server/src/websocket/MetricsStream.ts`:**

```typescript
import { WebSocketServer, WebSocket } from 'ws';

interface Client {
  ws: WebSocket;
  subscribedPipelines: Set<string>;
}

class MetricsStream {
  private clients = new Map<WebSocket, Client>();

  constructor(private wss: WebSocketServer) {
    this.wss.on('connection', (ws) => {
      this.clients.set(ws, { ws, subscribedPipelines: new Set() });

      ws.on('message', (data) => {
        const msg = JSON.parse(data.toString());
        this.handleMessage(ws, msg);
      });

      ws.on('close', () => {
        this.clients.delete(ws);
      });
    });
  }

  private handleMessage(ws: WebSocket, msg: any): void {
    if (msg.event === 'subscribe') {
      const client = this.clients.get(ws);
      client?.subscribedPipelines.add(msg.pipelineId);
    } else if (msg.event === 'unsubscribe') {
      const client = this.clients.get(ws);
      client?.subscribedPipelines.delete(msg.pipelineId);
    }
  }

  broadcast(message: any): void {
    const payload = JSON.stringify(message);

    this.clients.forEach((client) => {
      if (client.subscribedPipelines.has(message.pipelineId)) {
        client.ws.send(payload);
      }
    });
  }
}

export { MetricsStream };
```

### 4.5 Schema Loader

**`server/src/services/SchemaLoader.ts`:**

```typescript
import fs from 'fs/promises';
import path from 'path';

class SchemaLoader {
  private schema: any = null;
  private schemaPath: string;

  constructor() {
    // Look for modules.json in expected locations
    this.schemaPath = this.findSchemaPath();
  }

  private findSchemaPath(): string {
    const candidates = [
      './modules.json',                     // Current dir
      '../data/modules.json',               // Relative to server
      '../../data/modules.json',            // SDK structure
      process.env.APRAPIPES_SCHEMA_PATH,    // Env override
    ];

    for (const candidate of candidates) {
      if (candidate && fs.existsSync(candidate)) {
        return candidate;
      }
    }

    throw new Error('modules.json not found. Run schema_generator first.');
  }

  async load(): Promise<any> {
    if (!this.schema) {
      const content = await fs.readFile(this.schemaPath, 'utf-8');
      this.schema = JSON.parse(content);
    }
    return this.schema;
  }

  async reload(): Promise<void> {
    this.schema = null;
    await this.load();
  }
}

export default new SchemaLoader();
```

---

## 5. LLM Integration Architecture

### 5.1 Provider Abstraction

**`client/src/services/llm.ts`:**

```typescript
interface LLMProvider {
  name: string;
  configFields: ConfigField[];  // For settings UI
  generatePipeline(prompt: string, context: LLMContext): Promise<string>;
}

interface LLMContext {
  schema: ModuleSchema[];
  currentPipeline?: PipelineConfig;
  validationErrors?: ValidationIssue[];
  runtimeLogs?: string[];
}

class AnthropicProvider implements LLMProvider {
  name = 'Anthropic (Claude)';
  configFields = [
    { name: 'apiKey', type: 'password', label: 'API Key' },
    { name: 'model', type: 'select', options: ['claude-3-opus', 'claude-3-sonnet'] },
  ];

  async generatePipeline(prompt: string, context: LLMContext): Promise<string> {
    const systemPrompt = this.buildSystemPrompt(context);

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.getApiKey(),
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: this.getModel(),
        max_tokens: 4096,
        system: systemPrompt,
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    const data = await response.json();
    return this.extractJSON(data.content[0].text);
  }

  private buildSystemPrompt(context: LLMContext): string {
    let prompt = 'You are an expert in ApraPipes video processing pipelines.\n\n';

    // Add schema info
    prompt += 'Available modules:\n';
    for (const [name, schema] of Object.entries(context.schema.modules)) {
      prompt += `- ${name} (${schema.category}): ${schema.description}\n`;
    }

    // Add current pipeline if exists
    if (context.currentPipeline) {
      prompt += `\nCurrent pipeline:\n${JSON.stringify(context.currentPipeline, null, 2)}\n`;
    }

    // Add validation errors if exist
    if (context.validationErrors?.length) {
      prompt += `\nValidation errors:\n`;
      context.validationErrors.forEach(err => {
        prompt += `- ${err.code}: ${err.message} (${err.location})\n`;
      });
    }

    // Add runtime logs if exist
    if (context.runtimeLogs?.length) {
      prompt += `\nRecent logs:\n${context.runtimeLogs.join('\n')}\n`;
    }

    prompt += '\nGenerate a valid ApraPipes pipeline JSON that addresses the user\'s request.';
    return prompt;
  }

  private extractJSON(text: string): string {
    // Extract JSON from markdown code blocks
    const match = text.match(/```json\n([\s\S]*?)\n```/);
    return match ? match[1] : text;
  }

  private getApiKey(): string {
    return localStorage.getItem('llm.anthropic.apiKey') || '';
  }

  private getModel(): string {
    return localStorage.getItem('llm.anthropic.model') || 'claude-3-sonnet';
  }
}

// Registry pattern for easy extension
class LLMRegistry {
  private providers = new Map<string, LLMProvider>();

  register(provider: LLMProvider): void {
    this.providers.set(provider.name, provider);
  }

  get(name: string): LLMProvider | undefined {
    return this.providers.get(name);
  }

  list(): LLMProvider[] {
    return Array.from(this.providers.values());
  }
}

const registry = new LLMRegistry();
registry.register(new AnthropicProvider());
registry.register(new OpenAIProvider());
registry.register(new OllamaProvider());

export { registry as LLMRegistry };
```

### 5.2 Settings UI

**`client/src/components/Settings/LLMSettings.tsx`:**

```typescript
const LLMSettings: React.FC = () => {
  const [selectedProvider, setSelectedProvider] = useState('Anthropic (Claude)');
  const provider = LLMRegistry.get(selectedProvider);

  return (
    <div className="llm-settings">
      <h2>LLM Configuration</h2>

      <label>Provider</label>
      <select value={selectedProvider} onChange={(e) => setSelectedProvider(e.target.value)}>
        {LLMRegistry.list().map(p => (
          <option key={p.name} value={p.name}>{p.name}</option>
        ))}
      </select>

      {provider?.configFields.map(field => (
        <div key={field.name}>
          <label>{field.label}</label>
          {field.type === 'password' && (
            <input
              type="password"
              value={localStorage.getItem(`llm.${provider.name}.${field.name}`) || ''}
              onChange={(e) => localStorage.setItem(`llm.${provider.name}.${field.name}`, e.target.value)}
            />
          )}
          {field.type === 'select' && (
            <select
              value={localStorage.getItem(`llm.${provider.name}.${field.name}`) || field.options[0]}
              onChange={(e) => localStorage.setItem(`llm.${provider.name}.${field.name}`, e.target.value)}
            >
              {field.options.map(opt => <option key={opt}>{opt}</option>)}
            </select>
          )}
        </div>
      ))}
    </div>
  );
};
```

---

## 6. Data Flow Examples

### 6.1 Creating a Pipeline

```
1. User drags "TestSignalGenerator" from palette onto canvas
   → canvasStore.addNode(node)
   → pipelineStore.config.modules["source"] = { type: "TestSignalGenerator" }

2. User drags "FileWriterModule" onto canvas
   → canvasStore.addNode(node)
   → pipelineStore.config.modules["writer"] = { type: "FileWriterModule" }

3. User drags connection from source.output → writer.input
   → canvasStore.addEdge(edge)
   → Frontend checks type compatibility
   → pipelineStore.config.connections.push({ from: "source", to: "writer" })

4. User clicks "Save"
   → History.snapshot(canvasStore, pipelineStore)
   → POST /api/workspace/save { pipeline: config, layout: canvasPositions }
   → Backend writes to workspace/pipeline.json
```

### 6.2 Running a Pipeline

```
1. User clicks "Run"
   → Frontend: POST /api/pipeline/create { config }
   → Backend: PipelineManager.create(config)
   → Returns pipelineId

2. Frontend: POST /api/pipeline/:id/start
   → Backend: PipelineManager.start(id)
   → Pipeline.run() starts execution

3. Frontend: WebSocket connect
   → Send: { event: "subscribe", pipelineId }
   → Backend: MetricsStream adds client to subscribers

4. Pipeline emits health event
   → pipeline.on('health') fires
   → PipelineManager.onHealthEvent()
   → MetricsStream.broadcast({ event: 'health', data })
   → Frontend: runtimeStore.onHealthEvent()
   → canvasStore updates node metrics
   → Canvas re-renders node with FPS badge

5. User clicks "Stop"
   → POST /api/pipeline/:id/stop
   → Backend: PipelineManager.stop(id)
   → Pipeline.stop()
   → Status → STOPPED
```

### 6.3 LLM-Assisted Pipeline Generation

```
1. User clicks LLM button, types: "Create a face detection pipeline"
   → LLMChat component opens

2. User clicks "Generate"
   → Frontend: LLMService.generatePipeline(prompt, context)
   → Context includes:
      - schema (modules.json)
      - currentPipeline (if editing)
      - validationErrors (if any)

3. LLM returns JSON:
   {
     "modules": {
       "source": { "type": "FileReaderModule", "props": { ... } },
       "detector": { "type": "FaceDetectorXForm" },
       "sink": { "type": "FileWriterModule" }
     },
     "connections": [ ... ]
   }

4. User clicks "Apply"
   → pipelineStore.fromJSON(llmResponse)
   → canvasStore reconstructs nodes/edges
   → Canvas updates with new pipeline

5. User clicks "Validate"
   → POST /api/validate { config }
   → Backend returns errors (if any)
   → ProblemsPanel displays issues

6. If errors exist, user clicks "Fix with LLM"
   → LLM receives original prompt + validation errors
   → Returns corrected JSON
   → Repeat from step 4
```

---

## 7. Security Considerations

### 7.1 File Path Validation

**`server/src/services/WorkspaceManager.ts`:**

```typescript
import path from 'path';

class WorkspaceManager {
  private baseDir = process.env.WORKSPACE_DIR || '~/aprapipes-studio';

  private sanitizePath(userPath: string): string {
    const resolved = path.resolve(this.baseDir, userPath);

    // Prevent directory traversal
    if (!resolved.startsWith(this.baseDir)) {
      throw new Error('Invalid path: outside workspace');
    }

    return resolved;
  }

  async readFile(relativePath: string): Promise<string> {
    const safePath = this.sanitizePath(relativePath);
    return fs.readFile(safePath, 'utf-8');
  }
}
```

### 7.2 JSON Validation

Always validate user-provided JSON before passing to `aprapipes.node`:

```typescript
import Ajv from 'ajv';

const ajv = new Ajv();
const pipelineSchema = { /* JSON Schema for PipelineConfig */ };
const validate = ajv.compile(pipelineSchema);

app.post('/api/pipeline/create', (req, res) => {
  const { config } = req.body;

  if (!validate(config)) {
    return res.status(400).json({ error: validate.errors });
  }

  // Safe to pass to aprapipes
  const id = PipelineManager.create(config);
  res.json({ pipelineId: id });
});
```

---

## 8. Performance Optimizations

### 8.1 Frontend

1. **React Flow optimization:**
   - Use `nodesDraggable={false}` for read-only mode
   - `nodesFocusable={false}` when not needed
   - Virtual scrolling for large module palettes

2. **Memoization:**
   ```typescript
   const ModuleNode = React.memo(({ data }) => {
     // Only re-render if data changes
   });
   ```

3. **Debounced validation:**
   - Even though validation is on-save, debounce any client-side checks

4. **WebSocket throttling:**
   - Limit health event updates to 10 Hz (100ms interval)
   ```typescript
   let lastUpdate = 0;
   ws.on('message', (msg) => {
     const now = Date.now();
     if (now - lastUpdate < 100) return;
     lastUpdate = now;
     // Process message
   });
   ```

### 8.2 Backend

1. **Pipeline instance pooling:**
   - Reuse stopped pipelines instead of destroying immediately
   - Max 10 concurrent running pipelines (configurable)

2. **Schema caching:**
   - Load `modules.json` once at startup
   - Reload only on manual trigger (dev mode)

3. **WebSocket message batching:**
   - Collect health events for 100ms, send batch
   ```typescript
   private batch: any[] = [];
   private batchTimer: NodeJS.Timeout | null = null;

   broadcast(message: any): void {
     this.batch.push(message);

     if (!this.batchTimer) {
       this.batchTimer = setTimeout(() => {
         this.flushBatch();
       }, 100);
     }
   }
   ```

---

## 9. Testing Architecture

### 9.1 Frontend Tests

**Component Tests (React Testing Library):**

```typescript
// ModuleNode.test.tsx
describe('ModuleNode', () => {
  it('displays module name and category', () => {
    const data = {
      type: 'TestSignalGenerator',
      label: 'source',
      category: 'source',
      inputs: [],
      outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
      status: 'idle',
    };

    render(<ModuleNode id="test" data={data} />);

    expect(screen.getByText('source')).toBeInTheDocument();
    expect(screen.getByText('source')).toHaveClass('category-badge');
  });

  it('shows error icon when status is error', () => {
    const data = { ...baseData, status: 'error' };
    render(<ModuleNode id="test" data={data} />);

    expect(screen.getByTestId('error-icon')).toBeInTheDocument();
  });
});
```

**Store Tests (Zustand):**

```typescript
// canvasStore.test.ts
describe('canvasStore', () => {
  beforeEach(() => {
    useCanvasStore.getState().reset();
  });

  it('adds node to canvas', () => {
    const node = { id: 'test', type: 'module', data: {} };
    useCanvasStore.getState().addNode(node);

    expect(useCanvasStore.getState().nodes).toHaveLength(1);
    expect(useCanvasStore.getState().nodes[0].id).toBe('test');
  });

  it('prevents incompatible connections', () => {
    // ... test edge validation
  });
});
```

### 9.2 Backend Tests

**API Tests (Jest + Supertest):**

```typescript
// pipeline.test.ts
describe('POST /api/pipeline/create', () => {
  it('creates pipeline and returns ID', async () => {
    const config = {
      modules: {
        source: { type: 'TestSignalGenerator' },
      },
      connections: [],
    };

    const res = await request(app)
      .post('/api/pipeline/create')
      .send({ config })
      .expect(200);

    expect(res.body.pipelineId).toBeDefined();
  });

  it('rejects invalid config', async () => {
    const config = { invalid: true };

    await request(app)
      .post('/api/pipeline/create')
      .send({ config })
      .expect(400);
  });
});
```

### 9.3 E2E Tests (Playwright)

```typescript
// e2e/create-pipeline.spec.ts
test('create simple pipeline from scratch', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Drag module from palette
  await page.dragAndDrop('[data-module="TestSignalGenerator"]', '[data-canvas]');

  // Verify node appears
  await expect(page.locator('.module-node')).toBeVisible();

  // Add another module
  await page.dragAndDrop('[data-module="FileWriterModule"]', '[data-canvas]');

  // Connect modules
  await page.hover('[data-handle="source.output"]');
  await page.mouse.down();
  await page.hover('[data-handle="writer.input"]');
  await page.mouse.up();

  // Save
  await page.click('[data-action="save"]');

  // Verify saved
  await expect(page.locator('[data-status="saved"]')).toBeVisible();
});
```

---

## 10. Deployment Architecture

### 10.1 Development Mode

```bash
# Terminal 1: Backend
cd server
npm run dev  # nodemon index.ts

# Terminal 2: Frontend
cd client
npm run dev  # vite

# Opens http://localhost:5173 (Vite dev server)
# Proxies API calls to http://localhost:3000 (Express)
```

**`client/vite.config.ts`:**
```typescript
export default {
  server: {
    proxy: {
      '/api': 'http://localhost:3000',
      '/ws': {
        target: 'ws://localhost:3000',
        ws: true,
      },
    },
  },
};
```

### 10.2 Production Build

```bash
# Build frontend
cd client
npm run build  # → dist/

# Build backend (TypeScript → JavaScript)
cd server
npm run build  # → dist/

# Package
mkdir -p tools/visual-editor
cp -r client/dist tools/visual-editor/client
cp -r server/dist tools/visual-editor/server
cp -r server/node_modules tools/visual-editor/server/
cp start.sh tools/visual-editor/
```

**`start.sh`:**
```bash
#!/bin/bash
cd "$(dirname "$0")/server"
node index.js
```

### 10.3 SDK Integration

Add to `.github/workflows/build-test.yml`:

```yaml
- name: Build Visual Editor
  run: |
    cd tools/visual-editor/client
    npm ci
    npm run build

    cd ../server
    npm ci
    npm run build

- name: Package SDK with Studio
  run: |
    cp -r tools/visual-editor $SDK_DIR/studio
    echo "ApraPipes Studio included. Run: cd studio && ./start.sh" > $SDK_DIR/studio/README.txt
```

---

## 11. Future Architecture Considerations

### 11.1 Video Preview (WebRTC)

**Architecture:**
```
Pipeline → FrameExporter module → WebRTC sender
                                  ↓
                                  WebSocket signaling
                                  ↓
                        Frontend ← WebRTC receiver → <video> element
```

**Implementation sketch:**
```typescript
// Backend: Add WebRTC sender
class FrameStreamer {
  private peer: RTCPeerConnection;

  async streamFromModule(pipelineId: string, moduleId: string): Promise<void> {
    const module = pipeline.getModule(moduleId);

    // Tap into module's frame output
    module.on('frame', (frameData) => {
      // Send via data channel or encode as video track
      this.peer.send(frameData);
    });
  }
}

// Frontend: Video component
const VideoPreview: React.FC<{ moduleId: string }> = ({ moduleId }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const peer = new RTCPeerConnection();
    peer.ontrack = (event) => {
      videoRef.current.srcObject = event.streams[0];
    };

    // WebSocket signaling to negotiate connection
    ws.send({ event: 'start-stream', moduleId });
  }, [moduleId]);

  return <video ref={videoRef} autoPlay />;
};
```

### 11.2 Collaborative Editing (Multi-user)

**Architecture:**
- Use Yjs or Automerge for CRDT-based state sync
- WebSocket broadcast canvas operations
- Presence indicators (colored cursors)

**Conflict resolution:**
- Last-write-wins for properties
- Operational transforms for canvas positions

---

**End of Architecture Document**

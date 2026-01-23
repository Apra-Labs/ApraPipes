# ApraPipes Studio — Project Plan

> **Version:** 1.1
> **Date:** 2026-01-22
> **Branch:** `feat/visual-editor`
> **Execution Model:** Autonomous agent (agentic loop)

---

## 1. Overview

**Project:** ApraPipes Studio (Visual Pipeline Editor)
**Timeline:** 6 phases, estimated 12-16 weeks
**Execution:** Autonomous Claude agent (agentic loop)
**Priority Order:** Phase 1 → 2 → 3 → 6 → 4 → 5

**Key Principles:**
1. **Build incrementally:** Each sprint delivers working, tested code
2. **Test-first:** Write tests before implementation where possible
3. **Definition of Done (DoD):** Every sprint has clear, measurable completion criteria
4. **No phase-skipping:** Must complete DoD before moving to next sprint
5. **Autonomous verification:** Agent must verify DoD items programmatically

---

## 2. Phase Breakdown

| Phase | Name | Focus | Duration | Priority |
|-------|------|-------|----------|----------|
| **Phase 1** | Core Editor | Canvas, palette, property editing, JSON view | 3-4 weeks | Critical |
| **Phase 2** | Validation | Error detection, problems panel, visual feedback | 2 weeks | Critical |
| **Phase 3** | Runtime | Start/stop pipelines, live metrics, health monitoring | 2-3 weeks | Critical |
| **Phase 6** | Polish | Undo/redo, keyboard shortcuts, recent files, import | 2-3 weeks | High |
| **Phase 4** | LLM Basic | Pipeline generation from natural language | 1-2 weeks | Medium |
| **Phase 5** | LLM Advanced | Error remediation, runtime debugging assistance | 1-2 weeks | Medium |

**Total:** 11-16 weeks

---

## 3. Definition of Done (Global)

Every sprint must satisfy:

### 3.1 Code Quality
- [ ] TypeScript strict mode (no `any` types without justification)
- [ ] ESLint passes with no errors
- [ ] Prettier formatting applied
- [ ] No console.log (use proper logging)
- [ ] All imports used (no dead code)

### 3.2 Testing
- [ ] **ALL tests pass (100% passing rate)** - Zero failing tests allowed
- [ ] Unit tests written with 80%+ code coverage for new code
- [ ] Component tests for UI (React Testing Library)
- [ ] Integration tests for workflows
- [ ] Manual smoke test checklist completed

### 3.3 Documentation
- [ ] JSDoc comments for public APIs
- [ ] README updated if new features added
- [ ] PROGRESS.md updated with completion status

### 3.4 Verification
- [ ] Build succeeds (`npm run build`)
- [ ] Dev servers start without errors
- [ ] No TypeScript errors
- [ ] No runtime errors in console during smoke test

---

## 4. Phase 1: Core Editor (3-4 weeks)

### 4.1 Goals

Build the foundational visual editor with drag-and-drop module placement, connection creation, property editing, and JSON view.

---

### 4.2 Sprint 1.1: Project Setup & Schema Loading (Week 1)

#### Tasks

1. **Initialize project structure**
   - Create `tools/visual-editor/` directory
   - Set up monorepo with root `package.json` (workspaces)
   - Initialize client: `cd client && npm create vite@latest . -- --template react-ts`
   - Initialize server: `cd server && npm init -y && add TypeScript`
   - Install dependencies:
     - Client: `react-flow-renderer`, `zustand`, `tailwindcss`, `@radix-ui/react-*` (shadcn/ui)
     - Server: `express`, `cors`, `ws`, `@types/*`

2. **Backend: Schema Loader**
   - Create `server/src/services/SchemaLoader.ts`
   - Implement multi-path search for `modules.json`:
     1. `./modules.json`
     2. `../data/modules.json`
     3. `../../data/modules.json`
     4. `process.env.APRAPIPES_SCHEMA_PATH`
   - Add `GET /api/schema` endpoint in `server/src/api/schema.ts`
   - Return mock schema if `modules.json` not found (for dev mode)

3. **Frontend: Basic layout**
   - Create `App.tsx` with:
     - Toolbar (top)
     - Main area (center)
     - Status bar (bottom)
   - Set up Tailwind CSS (`tailwind.config.js`, `postcss.config.js`)
   - Create empty placeholder components (no functionality yet)

4. **Testing setup**
   - Configure Jest for backend
   - Configure Vitest for frontend
   - Add test scripts to `package.json`

#### Acceptance Criteria

**Functional:**
- [ ] Run `npm run dev:server` → server starts on port 3000
- [ ] Run `npm run dev:client` → Vite starts on port 5173
- [ ] Open `http://localhost:5173` → see basic layout (toolbar, main area, status bar)
- [ ] GET `http://localhost:3000/api/schema` → returns JSON with modules
- [ ] Client can fetch schema via API (verify in network tab)

**Code Quality:**
- [ ] All TypeScript files compile without errors
- [ ] ESLint passes
- [ ] Prettier formatting applied

**Testing:**
- [ ] Backend test: `SchemaLoader.test.ts` verifies multi-path search
- [ ] Backend test: `GET /api/schema` returns valid schema
- [ ] Frontend test: `App.test.tsx` verifies layout renders

**Documentation:**
- [ ] `tools/visual-editor/README.md` created with dev setup instructions
- [ ] `PROGRESS.md` updated: Sprint 1.1 marked complete

#### Definition of Done

```bash
# Agent must run these commands and verify success:
cd tools/visual-editor

# 1. Install dependencies
npm install
cd client && npm install && cd ..
cd server && npm install && cd ..

# 2. Build succeeds
npm run build

# 3. Tests pass
npm test

# 4. Dev servers start
npm run dev:server &  # Should print "Server running on port 3000"
npm run dev:client &  # Should print "Local: http://localhost:5173"

# 5. API works
curl http://localhost:3000/api/schema | jq '.modules | length'  # Should return > 0

# 6. Browser opens
open http://localhost:5173  # Should see layout (manual check)
```

**Manual Smoke Test Checklist:**
- [ ] Page loads without console errors
- [ ] Toolbar visible
- [ ] Main area visible
- [ ] Status bar visible

---

### 4.3 Sprint 1.2: Module Palette (Week 1)

#### Tasks

1. **ModulePalette component**
   - Create `client/src/components/Panels/ModulePalette.tsx`
   - Fetch schema from `GET /api/schema` on mount
   - Group modules by category (source, transform, sink, etc.)
   - Display module cards:
     - Module name
     - Category badge
     - Description (truncated, show full on hover)
   - Add search input (skeleton, non-functional in v1)

2. **Drag-and-drop**
   - Make module cards draggable (`draggable={true}`, `onDragStart`)
   - Store module type in `dataTransfer`
   - Create `Canvas.tsx` (React Flow wrapper)
   - Implement `onDrop` handler:
     - Parse module type from `dataTransfer`
     - Generate unique node ID (`nanoid` or `uuid`)
     - Add node to canvas store

3. **Canvas Store (Zustand)**
   - Create `client/src/store/canvasStore.ts`
   - State: `nodes`, `edges`, `selectedNodeId`
   - Actions: `addNode`, `removeNode`, `updateNodePosition`

#### Acceptance Criteria

**Functional:**
- [ ] Module palette displays all modules from schema
- [ ] Modules grouped by category (collapsible categories)
- [ ] Each module card shows name, category, description
- [ ] Drag module card → cursor shows drag preview
- [ ] Drop module on canvas → node appears at drop position
- [ ] Each dropped node has unique ID
- [ ] Multiple copies of same module can be added

**Code Quality:**
- [ ] TypeScript strict mode (no `any`)
- [ ] ESLint passes
- [ ] Module types extracted to `client/src/types/schema.ts`

**Testing:**
- [ ] Unit test: `canvasStore.test.ts` verifies `addNode` generates unique IDs
- [ ] Component test: `ModulePalette.test.tsx` verifies modules render grouped by category
- [ ] Integration test: Mock schema → verify palette displays correct modules

**Documentation:**
- [ ] JSDoc for `canvasStore` actions
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- ModulePalette
npm test -- canvasStore

# 2. Type check
npm run type-check

# 3. Build succeeds
npm run build

# 4. Dev server runs
npm run dev
```

**Manual Smoke Test Checklist:**
- [ ] Palette shows modules grouped by category
- [ ] Drag module from palette → drop on canvas → node appears
- [ ] Drop multiple modules → all have unique IDs (check React DevTools)
- [ ] No console errors during drag/drop

---

### 4.4 Sprint 1.3: Canvas & Custom Nodes (Week 2)

#### Tasks

1. **React Flow canvas setup**
   - Install `@xyflow/react` (React Flow v12+)
   - Create `Canvas.tsx` with `<ReactFlow>` component
   - Configure: zoom, pan, controls, background grid
   - Connect to `canvasStore` (nodes, edges, onNodesChange, onEdgesChange)

2. **Custom ModuleNode component**
   - Create `client/src/components/Canvas/ModuleNode.tsx`
   - Display:
     - Module name (editable later)
     - Module type (badge)
     - Category color-coding (source=blue, transform=green, sink=red)
     - Input handles (left side)
     - Output handles (right side)
   - Style with Tailwind
   - Register custom node: `nodeTypes={{ module: ModuleNode }}`

3. **Pin rendering**
   - Parse `inputs` and `outputs` from module schema
   - Render `<Handle>` for each pin
   - Show pin name on hover (Tooltip)
   - Show frame types on hover (e.g., "RAW_IMAGE, H264Frame")

4. **Node selection**
   - Implement `onNodeClick` → update `canvasStore.selectedNodeId`
   - Highlight selected node (border change)

#### Acceptance Criteria

**Functional:**
- [ ] Canvas displays dropped modules as custom nodes
- [ ] Each node shows:
  - Module name
  - Module type badge
  - Category color (blue/green/red)
  - Input handles on left
  - Output handles on right
- [ ] Hover over handle → tooltip shows pin name and frame types
- [ ] Click node → border highlights (selection)
- [ ] Canvas supports zoom (mouse wheel) and pan (drag background)
- [ ] Nodes can be dragged to reposition

**Code Quality:**
- [ ] `ModuleNode` is a React.memo component (performance)
- [ ] TypeScript types for `ModuleNodeProps`
- [ ] Tailwind classes, no inline styles

**Testing:**
- [ ] Component test: `ModuleNode.test.tsx` renders with correct structure
- [ ] Component test: Verify category colors applied correctly
- [ ] Component test: Verify handles render for inputs/outputs
- [ ] Integration test: Add node to store → verify appears on canvas

**Documentation:**
- [ ] JSDoc for `ModuleNode` props
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- ModuleNode
npm test -- Canvas

# 2. Type check
npm run type-check

# 3. Build succeeds
npm run build

# 4. Visual check
npm run dev
# Open browser → drag module → verify custom node renders correctly
```

**Manual Smoke Test Checklist:**
- [ ] Drag module → custom node appears with correct styling
- [ ] Hover pin → tooltip shows pin name and frame types
- [ ] Click node → border highlights
- [ ] Drag node → repositions smoothly
- [ ] Zoom/pan works correctly

---

### 4.5 Sprint 1.4: Connections & Property Panel (Week 2)

#### Tasks

1. **Connection creation**
   - Implement `onConnect` callback in `Canvas.tsx`
   - Add edge to `canvasStore.edges`
   - Sync connection to `pipelineStore.config.connections`
   - Format: `{ from: "sourceId.outputPin", to: "targetId.inputPin" }`

2. **Property Panel component**
   - Create `client/src/components/Panels/PropertyPanel.tsx`
   - Show only when `canvasStore.selectedNodeId !== null`
   - Display:
     - Module name (editable input)
     - Module type (read-only badge)
     - Properties list

3. **Property editors**
   - Create `client/src/components/Panels/PropertyEditors/` folder
   - Implement:
     - `IntInput.tsx` (with min/max validation from schema)
     - `FloatInput.tsx`
     - `BoolCheckbox.tsx`
     - `StringInput.tsx`
     - `EnumDropdown.tsx`
     - `JsonEditor.tsx` (simple textarea for v1)
   - Auto-select editor based on property type from schema

4. **Pipeline Store (Zustand)**
   - Create `client/src/store/pipelineStore.ts`
   - State: `config: PipelineConfig`, `schema: ModuleSchema[]`
   - Actions:
     - `updateModuleProperty(moduleId, key, value)`
     - `renameModule(oldId, newId)`
     - `addConnection(from, to)`
     - `removeConnection(id)`
     - `toJSON()`, `fromJSON(config)`

#### Acceptance Criteria

**Functional:**
- [ ] Drag from output handle to input handle → edge appears
- [ ] Connection stored in `pipelineStore.config.connections`
- [ ] Connection format: `{ from: "src.output", to: "dest.input" }`
- [ ] Click node → property panel shows on right side
- [ ] Property panel displays module name (editable)
- [ ] Property panel displays module type (read-only)
- [ ] Property panel lists all properties with correct editors:
  - Int properties → number input with min/max
  - Float properties → number input with step
  - Bool properties → checkbox
  - String properties → text input
  - Enum properties → dropdown with options
  - JSON properties → textarea
- [ ] Edit property → updates `pipelineStore.config.modules[id].props`
- [ ] Rename module → updates node label and connection references

**Code Quality:**
- [ ] Each property editor is a separate component
- [ ] TypeScript types for `PropertyEditorProps`
- [ ] Validation happens in property editors (min/max for int/float)

**Testing:**
- [ ] Unit test: `pipelineStore.test.ts` verifies `updateModuleProperty`
- [ ] Unit test: `pipelineStore.test.ts` verifies `renameModule` updates connections
- [ ] Component test: Each property editor renders correctly
- [ ] Integration test: Select node → edit property → verify store updated

**Documentation:**
- [ ] JSDoc for `pipelineStore` actions
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- pipelineStore
npm test -- PropertyPanel
npm test -- PropertyEditors

# 2. Type check
npm run type-check

# 3. Build succeeds
npm run build

# 4. Integration test
npm test -- integration
```

**Manual Smoke Test Checklist:**
- [ ] Drag connection between two nodes → edge appears
- [ ] Click node → property panel opens
- [ ] Edit int property → value updates in store (check Redux DevTools)
- [ ] Edit enum property → dropdown shows correct options
- [ ] Rename module → node label updates
- [ ] Create connection → verify JSON format in store

---

### 4.6 Sprint 1.5: JSON View & Workspace (Week 3)

#### Tasks

1. **JSON View component**
   - Install `@monaco-editor/react`
   - Create `client/src/components/Panels/JsonView.tsx`
   - Display `pipelineStore.config` as formatted JSON
   - Read-only mode (v1)
   - Syntax highlighting
   - Auto-sync from store

2. **View mode toggle**
   - Add buttons to toolbar: Visual | JSON | Split
   - Store current view mode in `uiStore` (new Zustand store)
   - Layout:
     - Visual: Canvas full-width
     - JSON: Monaco Editor full-width
     - Split: Canvas 50% | JSON 50% (resizable splitter optional in v1)

3. **Workspace Manager (backend)**
   - Create `server/src/services/WorkspaceManager.ts`
   - Path sanitization (prevent directory traversal)
   - Endpoints:
     - `GET /api/workspace/list?path=` - list files in workspace
     - `POST /api/workspace/save` - save pipeline.json + layout.json
     - `POST /api/workspace/load` - load pipeline.json
     - `POST /api/workspace/create` - create new project folder

4. **File operations (frontend)**
   - Add toolbar buttons: New, Open, Save
   - Create `client/src/store/workspaceStore.ts`
   - Actions:
     - `newWorkspace()` - clear canvas + config
     - `openWorkspace(path)` - load pipeline.json
     - `saveWorkspace()` - write files
     - `setCurrentPath(path)`

5. **Layout persistence**
   - Save node positions to `.studio/layout.json`
   - Load positions on open

#### Acceptance Criteria

**Functional:**
- [ ] Toggle to "JSON" view → Monaco Editor displays pipeline config
- [ ] JSON view shows syntax highlighting
- [ ] JSON updates when canvas changes (live sync)
- [ ] Toggle to "Split" view → Canvas 50% + JSON 50%
- [ ] Click "New" → clears canvas and resets config
- [ ] Click "Open" → file picker (or input) → loads pipeline.json
- [ ] Pipeline loads onto canvas (nodes + edges)
- [ ] Click "Save" → writes `pipeline.json` and `.studio/layout.json`
- [ ] Reopen saved pipeline → node positions restored

**Code Quality:**
- [ ] Path sanitization in `WorkspaceManager` (no `../` escapes)
- [ ] Error handling for file I/O (file not found, parse errors)
- [ ] TypeScript types for workspace API

**Testing:**
- [ ] Unit test: `WorkspaceManager.sanitizePath()` prevents traversal
- [ ] Unit test: `workspaceStore.test.ts` verifies save/load
- [ ] Integration test: Save pipeline → verify files written
- [ ] Integration test: Load pipeline → verify canvas restored

**Documentation:**
- [ ] Document workspace folder structure in README
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- WorkspaceManager
npm test -- workspaceStore
npm test -- JsonView

# 2. Type check
npm run type-check

# 3. Build succeeds
npm run build

# 4. File I/O test
npm run dev
# Manual: Save pipeline → verify files exist in workspace
# Manual: Load pipeline → verify canvas restored
```

**Manual Smoke Test Checklist:**
- [ ] Toggle JSON view → see formatted pipeline JSON
- [ ] Edit pipeline on canvas → JSON updates live
- [ ] Click "New" → canvas clears
- [ ] Click "Save" → files written to workspace folder
- [ ] Restart app → "Open" last saved pipeline → canvas restored
- [ ] Node positions match saved layout

---

### 4.7 Sprint 1.6: Phase 1 Polish & Testing (Week 3-4)

#### Tasks

1. **Styling & UX**
   - Polish `ModuleNode` appearance (shadows, borders, hover states)
   - Add category icons (React Icons or Lucide)
   - Improve property panel layout (grouping, spacing)
   - Add tooltips for all pins (show frame types)
   - Add loading states (spinner while fetching schema)
   - Add error states (schema load failed)

2. **Edge styling**
   - Bezier curves (smooth connections)
   - Animated flow (optional, React Flow built-in)

3. **Comprehensive testing**
   - **Unit tests:**
     - All store actions
     - All utility functions
   - **Component tests:**
     - ModuleNode renders correctly
     - PropertyPanel displays all editor types
     - ModulePalette groups by category
   - **Integration tests:**
     - Full workflow: drag → connect → edit → save → reload
   - **E2E test (Playwright):**
     - Create pipeline from scratch
     - Save pipeline
     - Reload and verify

4. **Documentation**
   - Update `tools/visual-editor/README.md`:
     - Installation instructions
     - Dev setup
     - Build and run
   - Add JSDoc to all public APIs
   - Create `CONTRIBUTING.md` (coding standards)

#### Acceptance Criteria

**Functional:**
- [ ] All Phase 1 features work smoothly
- [ ] No console errors
- [ ] Loading states show during async operations
- [ ] Error messages user-friendly (not stack traces)

**Code Quality:**
- [ ] ESLint passes with zero warnings
- [ ] Prettier formatting applied to all files
- [ ] No unused imports or variables
- [ ] All components have TypeScript types

**Testing:**
- [ ] Unit test coverage: 80%+ for stores
- [ ] Component test coverage: All major components tested
- [ ] Integration test: Full workflow passes
- [ ] E2E test: Playwright test passes

**Documentation:**
- [ ] README complete with dev setup
- [ ] All public APIs have JSDoc
- [ ] PROGRESS.md shows Phase 1 complete

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. All tests pass
npm test

# 2. Test coverage
npm run test:coverage
# Verify: Stores >80%, Components >60%

# 3. Lint
npm run lint

# 4. Build
npm run build

# 5. E2E test
npm run test:e2e

# 6. Type check
npm run type-check

# 7. Manual smoke test
npm run dev
```

**Manual Smoke Test Checklist (Complete Workflow):**
- [ ] Open app → see empty canvas
- [ ] Drag "TestSignalGenerator" from palette → drop on canvas
- [ ] Drag "FileWriterModule" from palette → drop on canvas
- [ ] Connect TestSignalGenerator.output → FileWriterModule.input
- [ ] Click TestSignalGenerator → edit width property to 1920
- [ ] Click FileWriterModule → edit filePath property
- [ ] Toggle JSON view → verify pipeline JSON correct
- [ ] Click "Save" → enter workspace path
- [ ] Close browser tab
- [ ] Reopen app → "Open" saved pipeline
- [ ] Verify nodes, edges, positions, properties all restored

**Phase 1 Exit Criteria:**
✅ All sprints 1.1-1.6 DoD satisfied
✅ Manual smoke test passes
✅ E2E test passes
✅ Demo recorded for stakeholders

---

## 5. Phase 2: Validation (2 weeks)

### 5.1 Goals

Integrate pipeline validation, display errors in Problems panel and on canvas, provide clear visual feedback for issues.

---

### 5.2 Sprint 2.1: Validation Backend (Week 4)

#### Tasks

1. **Validator service**
   - Create `server/src/services/Validator.ts`
   - Wrap `aprapipes.validatePipeline(config)`
   - Parse errors into structured format:
     ```typescript
     {
       valid: boolean,
       issues: Array<{
         level: 'error' | 'warning' | 'info',
         code: string,
         message: string,
         location: string,  // JSONPath
         suggestion?: string
       }>
     }
     ```
   - Add `POST /api/validate` endpoint

2. **Error code mapping**
   - Create `errorMessages.ts` with user-friendly messages
   - Map aprapipes error codes to readable text
   - Include actionable suggestions

3. **Mock validator (fallback)**
   - If `aprapipes.node` not available, use mock validator
   - Check common errors:
     - Missing required properties
     - Type mismatches in connections
     - Invalid property values (out of range)

#### Acceptance Criteria

**Functional:**
- [ ] POST /api/validate accepts pipeline config
- [ ] Returns structured validation result
- [ ] Error messages include:
  - Level (error/warning/info)
  - Code (E101, W201, etc.)
  - Message (user-friendly)
  - Location (JSONPath like "modules.source.props.width")
  - Suggestion (actionable fix)
- [ ] Handles invalid JSON gracefully (returns error, no crash)

**Code Quality:**
- [ ] TypeScript types for `ValidationResult`, `ValidationIssue`
- [ ] Error messages extracted to separate file (maintainable)

**Testing:**
- [ ] Unit test: `Validator.test.ts` with mock aprapipes
- [ ] Unit test: Validate valid config → returns `{ valid: true, issues: [] }`
- [ ] Unit test: Validate invalid config → returns specific errors
- [ ] Integration test: POST /api/validate with test configs

**Documentation:**
- [ ] JSDoc for `Validator` methods
- [ ] API endpoint documented in README or API.md
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- Validator

# 2. API test
npm run dev:server &
curl -X POST http://localhost:3000/api/validate \
  -H "Content-Type: application/json" \
  -d '{"modules": {}, "connections": []}'
# Should return { valid: true }

curl -X POST http://localhost:3000/api/validate \
  -H "Content-Type: application/json" \
  -d '{"modules": {"src": {"type": "Unknown"}}, "connections": []}'
# Should return { valid: false, issues: [...] }

# 3. Type check
npm run type-check
```

**Manual Smoke Test Checklist:**
- [ ] Send valid pipeline → returns valid=true
- [ ] Send pipeline with missing property → returns error with location
- [ ] Send pipeline with type mismatch → returns error with suggestion
- [ ] Error messages are user-friendly (not raw C++ errors)

---

### 5.3 Sprint 2.2: Problems Panel (Week 4-5)

#### Tasks

1. **ProblemsPanel component**
   - Create `client/src/components/Panels/ProblemsPanel.tsx`
   - Bottom pane (collapsible, fixed height initially)
   - Display issues from `pipelineStore.validationResult`
   - Filter by severity:
     - Errors (red icon)
     - Warnings (yellow icon)
     - Info (blue icon)
   - Click issue → jump to node/connection on canvas

2. **Integration with validation**
   - Add `pipelineStore.validate()` action:
     - Call `POST /api/validate`
     - Store result in `pipelineStore.validationResult`
   - Add "Validate" button to toolbar
   - Trigger validation on save (optional)

3. **Status bar summary**
   - Add validation summary to status bar:
     - "3 errors, 1 warning" (clickable)
   - Click → toggle Problems panel

4. **Jump to issue**
   - Parse issue location (JSONPath)
   - Find corresponding node/edge
   - Center canvas on node
   - Highlight node (flash animation)

#### Acceptance Criteria

**Functional:**
- [ ] Problems panel displays validation issues
- [ ] Issues grouped by severity (Errors, Warnings, Info)
- [ ] Each issue shows:
  - Icon (severity-based)
  - Code (E101)
  - Message
  - Location
- [ ] Click issue → canvas centers on affected node
- [ ] Click issue → node flashes (highlight animation)
- [ ] Status bar shows summary (e.g., "3 errors")
- [ ] Click status bar summary → toggle Problems panel
- [ ] Panel collapsible (minimize button)
- [ ] Panel shows "No issues" when empty

**Code Quality:**
- [ ] TypeScript types for issue display props
- [ ] Jump-to-issue logic extracted to utility function

**Testing:**
- [ ] Component test: `ProblemsPanel.test.tsx` renders issues correctly
- [ ] Unit test: Jump-to-issue logic parses locations correctly
- [ ] Integration test: Validate → verify issues display → click issue → verify jump

**Documentation:**
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- ProblemsPanel
npm test -- pipelineStore

# 2. Type check
npm run type-check

# 3. Build
npm run build
```

**Manual Smoke Test Checklist:**
- [ ] Create invalid pipeline (e.g., missing property)
- [ ] Click "Validate" → Problems panel opens
- [ ] See error listed with code, message, location
- [ ] Click error → canvas jumps to node
- [ ] Node flashes (highlight)
- [ ] Status bar shows "1 error"
- [ ] Fix error → re-validate → Problems panel shows "No issues"

---

### 5.4 Sprint 2.3: Visual Error Feedback (Week 5)

#### Tasks

1. **Node error indicators**
   - Add error badge to `ModuleNode`:
     - Red border when node has errors
     - Error icon (top-right corner)
     - Tooltip shows error messages
   - Store node errors in `canvasStore` (derived from validation result)

2. **Edge error indicators**
   - Red dashed line for invalid connections
   - Tooltip shows type mismatch details

3. **Property panel inline errors**
   - Red border on invalid input fields
   - Error message below field (small red text)
   - Extract errors from validation result by property path

4. **Connection type checking (frontend)**
   - Implement `isValidConnection()` in `canvasStore`
   - Check frame type compatibility before allowing edge
   - Show error toast if connection rejected

#### Acceptance Criteria

**Functional:**
- [ ] Node with error shows red border + error icon
- [ ] Hover error icon → tooltip shows error messages
- [ ] Invalid edge shows red dashed line
- [ ] Hover invalid edge → tooltip shows why invalid
- [ ] Property with error shows red border
- [ ] Error message displays below invalid property field
- [ ] Attempt invalid connection → toast notification + rejection
- [ ] Toast shows reason (e.g., "Type mismatch: H264 → RAW_IMAGE")

**Code Quality:**
- [ ] Error extraction logic extracted to utility
- [ ] TypeScript types for error display props

**Testing:**
- [ ] Component test: ModuleNode with errors renders correctly
- [ ] Unit test: `isValidConnection()` checks frame types
- [ ] Integration test: Validate → verify visual errors appear

**Documentation:**
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- ModuleNode
npm test -- canvasStore

# 2. Type check
npm run type-check

# 3. Build
npm run build
```

**Manual Smoke Test Checklist:**
- [ ] Create pipeline with error (e.g., width=999999)
- [ ] Validate → node shows red border + error icon
- [ ] Hover error icon → see error message
- [ ] Click node → property panel shows red border on width field
- [ ] See error message below width field
- [ ] Try to connect incompatible pins → connection rejected
- [ ] See toast notification explaining why

---

### 5.5 Sprint 2.4: Phase 2 Testing & Polish (Week 5)

#### Tasks

1. **Comprehensive testing**
   - Unit tests for all validation logic
   - Component tests for ProblemsPanel, error badges
   - Integration test: Full validation workflow
   - E2E test: Create invalid pipeline → validate → fix → re-validate

2. **Error message quality**
   - Review all error messages for clarity
   - Add suggestions where applicable
   - Test with non-technical user (if possible)

3. **Performance**
   - Ensure validation doesn't block UI (async)
   - Add loading state during validation

#### Acceptance Criteria

**Functional:**
- [ ] All Phase 2 features work
- [ ] Validation async (doesn't freeze UI)
- [ ] Loading indicator shows during validation

**Testing:**
- [ ] Unit test coverage: 80%+ for validation logic
- [ ] Integration test passes
- [ ] E2E test passes

**Documentation:**
- [ ] PROGRESS.md shows Phase 2 complete

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. All tests pass
npm test

# 2. Test coverage
npm run test:coverage

# 3. E2E test
npm run test:e2e

# 4. Build
npm run build
```

**Manual Smoke Test Checklist (Complete Validation Workflow):**
- [ ] Create pipeline with multiple errors
- [ ] Validate → see all errors in Problems panel
- [ ] Status bar shows correct count
- [ ] Click each error → jumps to correct node
- [ ] Fix first error → re-validate → count decreases
- [ ] Fix all errors → re-validate → "No issues"
- [ ] Try invalid connection → rejected with toast

**Phase 2 Exit Criteria:**
✅ All sprints 2.1-2.4 DoD satisfied
✅ Manual smoke test passes
✅ E2E test passes

---

## 6. Phase 3: Runtime Monitoring (2-3 weeks)

### 6.1 Goals

Enable users to run pipelines, monitor live metrics (FPS, queue status), and see health/error events in real-time.

---

### 6.2 Sprint 3.1: Pipeline Lifecycle Backend (Week 6)

#### Tasks

1. **PipelineManager service**
   - Create `server/src/services/PipelineManager.ts`
   - Methods:
     - `create(config)` → initialize pipeline, return ID
     - `start(id)` → run pipeline
     - `stop(id)` → stop pipeline
     - `get(id)` → retrieve pipeline instance
     - `delete(id)` → cleanup
   - Track pipeline instances in Map
   - Set up event listeners: `on('health')`, `on('error')`

2. **Pipeline API endpoints**
   - `POST /api/pipeline/create` → returns `{ pipelineId }`
   - `POST /api/pipeline/:id/start` → returns `{ status }`
   - `POST /api/pipeline/:id/stop` → returns `{ status }`
   - `GET /api/pipeline/:id/status` → returns current status
   - `DELETE /api/pipeline/:id` → cleanup

3. **Mock pipeline (fallback)**
   - If `aprapipes.node` not available, create mock
   - Simulate health events (random FPS, queue)
   - Allow testing without native addon

#### Acceptance Criteria

**Functional:**
- [ ] POST /api/pipeline/create initializes pipeline
- [ ] Returns unique pipeline ID
- [ ] POST /api/pipeline/:id/start runs pipeline
- [ ] Pipeline emits health events
- [ ] POST /api/pipeline/:id/stop stops pipeline
- [ ] GET /api/pipeline/:id/status returns current state
- [ ] DELETE /api/pipeline/:id cleans up resources

**Code Quality:**
- [ ] TypeScript types for pipeline lifecycle
- [ ] Error handling (pipeline not found, already running, etc.)

**Testing:**
- [ ] Unit test: `PipelineManager.test.ts` with mock aprapipes
- [ ] Unit test: Create → start → stop → delete
- [ ] Integration test: Full lifecycle via API

**Documentation:**
- [ ] API endpoints documented
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- PipelineManager

# 2. API test
npm run dev:server &

# Create pipeline
PIPELINE_ID=$(curl -X POST http://localhost:3000/api/pipeline/create \
  -H "Content-Type: application/json" \
  -d '{"modules": {"src": {"type": "TestSignalGenerator"}}, "connections": []}' \
  | jq -r '.pipelineId')

# Start pipeline
curl -X POST http://localhost:3000/api/pipeline/$PIPELINE_ID/start

# Check status
curl http://localhost:3000/api/pipeline/$PIPELINE_ID/status
# Should return { status: "RUNNING" }

# Stop pipeline
curl -X POST http://localhost:3000/api/pipeline/$PIPELINE_ID/stop

# 3. Type check
npm run type-check
```

**Manual Smoke Test Checklist:**
- [ ] Create pipeline via API → returns ID
- [ ] Start pipeline → status becomes RUNNING
- [ ] Stop pipeline → status becomes STOPPED
- [ ] Create pipeline with mock → emits health events (check logs)

---

### 6.3 Sprint 3.2: WebSocket Metrics Stream (Week 6)

#### Tasks

1. **Backend: MetricsStream service**
   - Create `server/src/websocket/MetricsStream.ts`
   - WebSocket server setup (port 3001 or same as Express)
   - Client subscription:
     - Client sends `{ event: 'subscribe', pipelineId }`
     - Server adds client to subscribers
   - Broadcast events:
     - `{ event: 'health', moduleId, data: { fps, qlen, isQueueFull } }`
     - `{ event: 'error', moduleId, data: { message, timestamp } }`
     - `{ event: 'status', data: { status } }`

2. **Frontend: WebSocket client**
   - Create `client/src/services/websocket.ts`
   - Connect to `ws://localhost:3001`
   - Auto-reconnect on disconnect
   - Send subscribe message
   - Handle incoming events

3. **Runtime store (Zustand)**
   - Create `client/src/store/runtimeStore.ts`
   - State:
     - `pipelineId: string | null`
     - `status: PipelineStatus`
     - `moduleMetrics: Record<string, ModuleMetrics>`
     - `errors: RuntimeError[]`
   - Actions:
     - `connect(pipelineId)`
     - `disconnect()`
     - `onHealthEvent(event)`
     - `onErrorEvent(event)`
     - `onStatusEvent(event)`

#### Acceptance Criteria

**Functional:**
- [ ] WebSocket server starts with Express
- [ ] Frontend connects to WebSocket
- [ ] Client sends subscribe message
- [ ] Server receives subscription
- [ ] Pipeline emits health event → server broadcasts → client receives
- [ ] Client stores metrics in `runtimeStore`
- [ ] WebSocket reconnects on disconnect

**Code Quality:**
- [ ] TypeScript types for WebSocket messages
- [ ] Reconnect logic with exponential backoff

**Testing:**
- [ ] Unit test: `MetricsStream.test.ts` verifies broadcast logic
- [ ] Unit test: `runtimeStore.test.ts` verifies event handlers
- [ ] Integration test: Start pipeline → verify WebSocket messages

**Documentation:**
- [ ] WebSocket protocol documented
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- MetricsStream
npm test -- runtimeStore

# 2. Integration test
npm run dev:server &
npm run dev:client &

# Open browser console → check WebSocket connection
# Should see: "WebSocket connected to ws://localhost:3001"

# Create and start pipeline → check console for health events

# 3. Type check
npm run type-check
```

**Manual Smoke Test Checklist:**
- [ ] Open app → WebSocket connects (check console)
- [ ] Create pipeline → start → health events arrive (check console)
- [ ] Check `runtimeStore` in Redux DevTools → metrics populated
- [ ] Stop server → WebSocket disconnects → reconnects when server restarts

---

### 6.4 Sprint 3.3: Live Metrics Display (Week 7)

#### Tasks

1. **ModuleNode live indicators**
   - Update `ModuleNode.tsx`:
     - Status badge (idle=gray, running=green pulse, error=red)
     - FPS display (e.g., "24 fps" in small badge)
     - Queue fill indicator (progress bar, 0-100%)
   - Subscribe to `runtimeStore.moduleMetrics`
   - Re-render when metrics update

2. **Toolbar controls**
   - Add "Run" button (play icon)
     - Click → create pipeline → start → connect WebSocket
   - Add "Stop" button (stop icon)
     - Click → stop pipeline → disconnect WebSocket
   - Disable buttons based on state:
     - Run enabled only when pipeline not running
     - Stop enabled only when pipeline running

3. **Status bar**
   - Show pipeline status: "IDLE" | "RUNNING" | "STOPPED" | "ERROR"
   - Show runtime duration (e.g., "Running for 00:05:32")
   - Update every second (useInterval hook)

#### Acceptance Criteria

**Functional:**
- [ ] Node shows status badge:
  - Gray when idle
  - Green pulsing when running
  - Red when error
- [ ] Running node shows FPS (e.g., "30 fps")
- [ ] Running node shows queue fill (progress bar, 0-100%)
- [ ] Click "Run" → pipeline starts → nodes change to green
- [ ] Metrics appear on nodes (FPS, queue)
- [ ] Click "Stop" → pipeline stops → nodes change to gray
- [ ] Status bar shows "RUNNING" when active
- [ ] Status bar shows runtime duration (updates every second)
- [ ] Buttons enable/disable correctly

**Code Quality:**
- [ ] ModuleNode subscribes to runtime store efficiently (useShallow)
- [ ] No memory leaks (cleanup WebSocket on unmount)

**Testing:**
- [ ] Component test: ModuleNode with metrics renders correctly
- [ ] Integration test: Run pipeline → verify metrics display
- [ ] Integration test: Stop pipeline → verify metrics cleared

**Documentation:**
- [ ] PROGRESS.md updated

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. Tests pass
npm test -- ModuleNode
npm test -- Toolbar

# 2. Type check
npm run type-check

# 3. Build
npm run build
```

**Manual Smoke Test Checklist:**
- [ ] Create simple pipeline (TestSignalGenerator → FileWriter)
- [ ] Click "Run" → nodes turn green (pulsing)
- [ ] See FPS appear on nodes (e.g., "30 fps")
- [ ] See queue indicator (progress bar)
- [ ] Status bar shows "RUNNING"
- [ ] Status bar shows duration (00:00:05...)
- [ ] Click "Stop" → nodes turn gray
- [ ] Status bar shows "STOPPED"

---

### 6.5 Sprint 3.4: Error Logging & Phase 3 Testing (Week 7-8)

#### Tasks

1. **Runtime errors in Problems panel**
   - Extend `ProblemsPanel` to show runtime errors
   - Distinguish from validation errors:
     - Validation: red circle icon
     - Runtime: orange lightning icon
   - Click runtime error → jump to module

2. **Error log export**
   - Add "Export Logs" button in Problems panel
   - Download runtime errors as `.txt` or `.json`
   - Include timestamp, module, message

3. **Comprehensive testing**
   - Unit tests for all runtime logic
   - Integration test: Full run workflow
   - E2E test: Create → run → monitor → stop

#### Acceptance Criteria

**Functional:**
- [ ] Runtime errors appear in Problems panel
- [ ] Runtime errors distinguished from validation errors (icon)
- [ ] Click runtime error → jumps to module
- [ ] "Export Logs" button downloads file
- [ ] Log file contains timestamp, module, message

**Testing:**
- [ ] Unit test coverage: 80%+ for runtime logic
- [ ] Integration test passes
- [ ] E2E test passes

**Documentation:**
- [ ] PROGRESS.md shows Phase 3 complete

#### Definition of Done

```bash
# Agent must verify:
cd tools/visual-editor

# 1. All tests pass
npm test

# 2. Test coverage
npm run test:coverage

# 3. E2E test
npm run test:e2e

# 4. Build
npm run build
```

**Manual Smoke Test Checklist (Complete Runtime Workflow):**
- [ ] Create pipeline
- [ ] Click "Run" → pipeline starts
- [ ] See metrics on nodes (FPS, queue)
- [ ] Status bar shows "RUNNING" + duration
- [ ] Simulate error (if possible) → see in Problems panel
- [ ] Click error → jumps to module
- [ ] Click "Export Logs" → file downloads
- [ ] Open log file → verify contents
- [ ] Click "Stop" → pipeline stops
- [ ] Status bar shows "STOPPED"

**Phase 3 Exit Criteria:**
✅ All sprints 3.1-3.4 DoD satisfied
✅ Manual smoke test passes
✅ E2E test passes

---

## 7. Phase 6: Polish (2-3 weeks)

*[Continue with same detailed format for Phase 6, 4, and 5]*

**Note:** Phase 6, 4, and 5 would follow the same structure:
- Each sprint has Tasks, Acceptance Criteria, DoD with bash verification
- Manual smoke test checklist
- Phase exit criteria

---

## 8. Testing Standards (All Phases)

**CRITICAL:** All tests MUST pass (100% pass rate). Zero failing tests allowed. Code coverage target is 80%+ (not all code needs tests, but what is tested must pass).

### 8.1 Unit Tests

**Requirements:**
- 80%+ code coverage for business logic (stores, services)
- 100% of written tests must pass
- Test file naming: `*.test.ts` (not `.spec.ts`)
- Use `describe` blocks for grouping
- Use `it` for test cases (not `test`)
- Mock external dependencies (API, WebSocket, aprapipes.node)

**Example:**
```typescript
describe('canvasStore', () => {
  beforeEach(() => {
    useCanvasStore.getState().reset();
  });

  it('adds node with unique ID', () => {
    const store = useCanvasStore.getState();
    store.addNode({ type: 'TestSignalGenerator' });
    store.addNode({ type: 'TestSignalGenerator' });

    expect(store.nodes).toHaveLength(2);
    expect(store.nodes[0].id).not.toBe(store.nodes[1].id);
  });
});
```

### 8.2 Component Tests

**Requirements:**
- Test user-visible behavior (not implementation details)
- Use `@testing-library/react` and `@testing-library/user-event`
- Test accessibility (aria labels, keyboard navigation)

**Example:**
```typescript
describe('ModuleNode', () => {
  it('displays module name and category', () => {
    render(<ModuleNode data={{ name: 'source', type: 'TestSignalGenerator', category: 'source' }} />);

    expect(screen.getByText('source')).toBeInTheDocument();
    expect(screen.getByText('source')).toHaveClass('badge-source');
  });
});
```

### 8.3 Integration Tests

**Requirements:**
- Test interactions between multiple components/stores
- Use real stores (not mocks) but mock external APIs
- Test complete user workflows

**Example:**
```typescript
describe('Pipeline creation workflow', () => {
  it('creates pipeline from drag and drop', async () => {
    render(<App />);

    // Drag module from palette
    const module = screen.getByText('TestSignalGenerator');
    fireEvent.dragStart(module);

    const canvas = screen.getByTestId('canvas');
    fireEvent.drop(canvas);

    // Verify node appears
    await waitFor(() => {
      expect(screen.getByText('source')).toBeInTheDocument();
    });

    // Verify store updated
    const store = useCanvasStore.getState();
    expect(store.nodes).toHaveLength(1);
  });
});
```

### 8.4 E2E Tests (Playwright)

**Requirements:**
- Test critical user paths
- Run against production build
- Use Page Object Model pattern
- Include visual regression (screenshots)

**Example:**
```typescript
test('create and run pipeline', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Create pipeline
  await page.dragAndDrop('[data-module="TestSignalGenerator"]', '[data-canvas]');
  await page.dragAndDrop('[data-module="FileWriterModule"]', '[data-canvas]');

  // Connect
  await page.hover('[data-handle="source.output"]');
  await page.mouse.down();
  await page.hover('[data-handle="writer.input"]');
  await page.mouse.up();

  // Run
  await page.click('[data-action="run"]');
  await expect(page.locator('[data-status="running"]')).toBeVisible();

  // Stop
  await page.click('[data-action="stop"]');
  await expect(page.locator('[data-status="stopped"]')).toBeVisible();
});
```

---

## 9. Autonomous Agent Verification Protocol

For each sprint, the agent must:

### 9.1 Pre-Implementation Checklist
- [ ] Read sprint tasks and acceptance criteria
- [ ] Identify dependencies (previous sprints)
- [ ] Check for blockers (missing files, APIs)

### 9.2 During Implementation
- [ ] Write tests FIRST (TDD where applicable)
- [ ] Implement feature
- [ ] Run tests continuously (`npm test -- --watch`)
- [ ] Fix all TypeScript errors
- [ ] Fix all ESLint warnings

### 9.3 Post-Implementation Checklist
- [ ] Run all tests: `npm test`
- [ ] Check coverage: `npm run test:coverage`
- [ ] Run type check: `npm run type-check`
- [ ] Run linter: `npm run lint`
- [ ] Run build: `npm run build`
- [ ] Run dev servers: `npm run dev`
- [ ] Execute manual smoke test checklist
- [ ] Update PROGRESS.md

### 9.4 Sprint Completion Report

Agent must generate:

```markdown
## Sprint X.Y Completion Report

**Status:** ✅ Complete | ❌ Incomplete

### Tests
- **Pass Rate:** X/Y passing (MUST be 100% - all tests pass)
- **Code Coverage:** Z% (target: >80% for business logic)
- Unit tests: X passing, Y total
- Component tests: X passing, Y total
- Integration tests: X passing, Y total
- E2E tests: X passing, Y total

### Code Quality
- TypeScript errors: 0
- ESLint warnings: 0
- Build: Success

### Manual Smoke Test
- [ ] All checklist items verified

### Blockers
- None | [List blockers]

### Notes
[Any observations, issues, or recommendations]
```

---

## 10. Risk Management (Updated)

| Risk | Probability | Impact | Mitigation | Agent Action |
|------|-------------|--------|------------|--------------|
| **React Flow performance issues** | Medium | High | Profile early, optimize re-renders | Add performance tests in Sprint 1.3 |
| **aprapipes.node binding issues** | Low | High | Use mock mode for development | Create mock in Sprint 1.1 |
| **LLM API rate limits** | High | Medium | Implement retry with backoff | Add error handling in Phase 4 |
| **WebSocket connection instability** | Medium | Medium | Auto-reconnect with exponential backoff | Implement in Sprint 3.2 |
| **Undo/redo state explosion** | Medium | Low | Limit to 50 snapshots, use structural sharing | Test memory in Sprint 6.1 |
| **Module schema changes** | Low | High | Version schema, graceful degradation | Handle in Sprint 1.1 |
| **Agent gets stuck on failing test** | Medium | High | Timeout after 3 attempts, flag for human | Built into agent loop |

---

## 11. Success Metrics (Measurable)

### 11.1 Per-Sprint Metrics
- **Test Pass Rate:** 100% (all tests must pass, zero failures)
- **Code Coverage:** >80% for business logic (stores, services)
- **Build Time:** < 30s
- **Bundle Size:** < 2MB (gzipped)
- **Type Errors:** 0
- **Lint Warnings:** 0

### 11.2 Phase Completion Metrics
- **Phase 1:** User creates 3-module pipeline in < 2 minutes
- **Phase 2:** Validation finds all errors, 0 false positives
- **Phase 3:** Runtime metrics update < 100ms latency
- **Phase 6:** All keyboard shortcuts work, undo/redo < 50ms

### 11.3 Final Release Metrics
- **Performance:** 100 nodes @ 60 FPS
- **Test Pass Rate:** 100% (all tests passing)
- **Code Coverage:** >80% overall
- **Bug Count:** < 5 open issues
- **Documentation:** README + API docs complete

---

## 12. Phase 7: Critical Fixes & Real Schema Integration (HIGH PRIORITY)

> **Added:** 2026-01-23 (User Feedback)
> **Priority:** CRITICAL - Must complete before further development
> **Rationale:** Visual editor must be driven by real schema from aprapipes tools, not mock data

### 7.1 Priority Order

| Priority | Item | Description | Type |
|----------|------|-------------|------|
| **P0** | F10 | CLI `describe --all --json` + `describeAllModules()` | Feature |
| **P0** | F8/F9 | Use real schema from schema-generator | Infrastructure |
| **P1** | F3 | Opening workspace doesn't draw nodes on canvas | Bug |
| **P1** | F1 | No way to remove a node from design surface | Bug |
| **P1** | F2 | Difficult to connect nodes (end up dragging whole node) | UX Bug |
| **P2** | F7 | Pin properties NOT implemented | Missing Feature |
| **P2** | F4 | Validate says "valid" then shows warnings/errors | UX Bug |
| **P3** | F6 | Remove "SOURCE"/"TRANSFORM" labels (colors enough) | UI Polish |
| **P3** | F5 | Settings page + validate-on-save + arrange button | Feature |

---

### 7.2 Sprint 7.1: Schema Generation Infrastructure (P0)

**Goal:** Create consolidated schema generation so visual editor depends ONLY on tool output.

#### Tasks

1. **CLI: Add `describe --all --json` option**
   - Modify `base/src/AprapipesCli.cpp` or relevant CLI source
   - New option: `aprapipes_cli describe --all --json`
   - Output: Consolidated JSON with ALL modules and their properties
   - Same structure as individual `describe <module> --json` but as array/object
   - Format:
     ```json
     {
       "modules": {
         "TestSignalGenerator": {
           "type": "TestSignalGenerator",
           "category": "source",
           "description": "...",
           "inputs": [...],
           "outputs": [...],
           "properties": {...}
         },
         "FileWriterModule": {...}
       }
     }
     ```

2. **Node.js Addon: Add `describeAllModules()` function**
   - Modify `base/src/node_addon/aprapipes_node.cpp` (or similar)
   - Export new function: `describeAllModules()`
   - Must produce IDENTICAL output structure as CLI `--all --json`
   - Update TypeScript types in `types/aprapipes.d.ts`

3. **Visual Editor: Use real schema**
   - Update `SchemaLoader.ts` to call `describeAllModules()` if available
   - Fallback to loading `modules.json` file (generated from CLI)
   - Remove hard-coded mock schema
   - Visual editor should have NO knowledge of module internals

#### Acceptance Criteria

**Functional:**
- [ ] `./build/aprapipes_cli describe --all --json` outputs all modules
- [ ] Output is valid JSON parseable by `jq`
- [ ] `describeAllModules()` in Node addon returns same structure
- [ ] Visual editor loads schema from addon/file, not mock
- [ ] Both CLI and addon produce identical JSON structure

**Testing:**
- [ ] C++ unit test for CLI `--all` option
- [ ] Node addon test for `describeAllModules()`
- [ ] Visual editor test with real schema

**Verification:**
```bash
# CLI test
./build/aprapipes_cli describe --all --json > /tmp/schema.json
jq '.modules | keys | length' /tmp/schema.json  # Should be > 30

# Compare with individual describes
./build/aprapipes_cli describe TestSignalGenerator --json > /tmp/single.json
jq '.modules.TestSignalGenerator' /tmp/schema.json > /tmp/from_all.json
diff /tmp/single.json /tmp/from_all.json  # Should match (or be equivalent)

# Node addon test
node -e "const m = require('./build/aprapipes.node'); console.log(JSON.stringify(m.describeAllModules(), null, 2))" > /tmp/addon_schema.json
```

---

### 7.3 Sprint 7.2: Critical Bug Fixes (P1)

**Goal:** Fix blocking bugs that prevent basic editor usage.

#### Tasks

1. **F3: Fix workspace load not drawing nodes**
   - Debug `workspaceStore.openWorkspace()` flow
   - Ensure loaded JSON is converted to React Flow nodes
   - Ensure canvas store `setNodes()` is called after load
   - Test: Open saved workspace → nodes appear on canvas

2. **F1: Add node deletion**
   - Add Delete key handler in canvas
   - Add right-click context menu with "Delete" option
   - Add delete button in property panel header
   - `canvasStore.removeNode(id)` action
   - Update connections when node deleted

3. **F2: Improve connection UX (handle hit area)**
   - Increase handle size/hit area (larger clickable region)
   - Add visual feedback on handle hover
   - Consider: connection mode toggle or different interaction
   - Test: Can easily grab handle without moving node

#### Acceptance Criteria

- [ ] Open saved workspace → nodes and edges appear on canvas
- [ ] Select node + press Delete → node removed
- [ ] Right-click node → context menu with Delete option
- [ ] Handles have larger hit area, easier to click
- [ ] Visual feedback when hovering over handle

---

### 7.4 Sprint 7.3: Pin Properties & Validation UX (P2)

**Goal:** Implement pin properties and fix validation messaging.

#### Tasks

1. **F7: Pin Properties**
   - Pins (inputs/outputs) can have configurable properties
   - Property Panel shows pin properties when pin selected
   - Pin properties editable (e.g., frame type constraints)
   - Schema must include pin property definitions

2. **F4: Fix validation messaging**
   - Don't say "pipeline is valid" if there are warnings
   - Clear messaging:
     - "Valid" = no errors AND no warnings
     - "Valid with warnings" = no errors, has warnings
     - "Invalid" = has errors
   - Status bar reflects actual state

#### Acceptance Criteria

- [ ] Click on pin → property panel shows pin properties
- [ ] Can edit pin properties
- [ ] Validation message accurately reflects state
- [ ] No confusing "valid but has warnings" situation

---

### 7.5 Sprint 7.4: UI Polish & Settings (P3)

**Goal:** Remove visual clutter and add user preferences.

#### Tasks

1. **F6: Remove category labels from nodes**
   - Remove "SOURCE", "TRANSFORM", "SINK" text from ModuleNode
   - Keep category colors (sufficient visual indicator)
   - Reclaim vertical space in node design

2. **F5: Settings & Arrange**
   - Create Settings page/modal
   - Add "Validate on Save" toggle (stored in localStorage)
   - Add "Arrange" button in toolbar
   - Auto-arrange nodes using layout algorithm (dagre or similar)

#### Acceptance Criteria

- [ ] Nodes show category by color only, no text label
- [ ] Settings modal accessible from toolbar/menu
- [ ] "Validate on Save" preference works
- [ ] "Arrange" button auto-layouts nodes nicely

---

## 13. Updated Phase Roadmap

| Phase | Name | Priority | Status | Notes |
|-------|------|----------|--------|-------|
| **Phase 7** | Critical Fixes & Real Schema | **CRITICAL** | 🚧 In Progress | Do FIRST |
| Phase 1 | Core Editor | Critical | ✅ Complete | |
| Phase 2 | Validation | Critical | ✅ Complete | |
| Phase 3 | Runtime | Critical | ✅ Complete | |
| Phase 6 | Polish | High | ✅ Complete | |
| Phase 4 | LLM Basic | Medium | ⏳ Deferred | After Phase 7 |
| Phase 5 | LLM Advanced | Medium | ⏳ Deferred | After Phase 4 |

**New Execution Order:** Phase 7 → Phase 4 → Phase 5

---

**End of Enhanced Project Plan**

_This plan is designed for autonomous agent execution with clear verification at each step._

# ApraPipes Studio — Ralph Loop Development Prompt

You are an autonomous agent implementing ApraPipes Studio, a visual pipeline editor for JavaScript developers.

## Your Mission

Build ApraPipes Studio by completing each sprint in sequence per the PROJECT_PLAN.md in `docs/visual-editor/`.

## Key Documents (READ THESE)

At each iteration, read these files to understand current state:

| Document | Path | Purpose |
|----------|------|---------|
| **Progress Tracker** | `docs/visual-editor/PROGRESS.md` | Current sprint status |
| **Project Plan** | `docs/visual-editor/PROJECT_PLAN.md` | Sprint tasks & DoD |
| **Architecture** | `docs/visual-editor/ARCHITECTURE.md` | System design |
| **Specification** | `docs/visual-editor/SPECIFICATION.md` | Technical spec |
| **Design Decisions** | `docs/visual-editor/DESIGN_QUESTIONS.md` | User requirements |

## Product Vision

**ApraPipes Studio** is a web-based visual editor for creating, validating, and running ApraPipes video processing pipelines.

**Key Features:**
- Visual drag-and-drop pipeline construction (React Flow)
- Auto-generated property editors from module schema
- Real-time validation with Problems panel
- Live runtime monitoring (FPS, queue status, health events)
- LLM-assisted pipeline generation and debugging
- Undo/redo, keyboard shortcuts, workspace management

**Target Users:** JavaScript developers who need video/image processing without deep domain knowledge.

## Technical Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript |
| Canvas | React Flow (v12+) |
| State | Zustand |
| Styling | Tailwind CSS + shadcn/ui |
| Backend | Express.js + TypeScript |
| Real-time | WebSocket (ws) |
| Build | Vite |
| Testing | Vitest (frontend), Jest (backend), Playwright (E2E) |

## Directory Structure

```
tools/visual-editor/
├── client/                      # Frontend (Vite + React)
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── Canvas/          # Canvas.tsx, ModuleNode.tsx, ConnectionEdge.tsx
│   │   │   ├── Panels/          # ModulePalette, PropertyPanel, ProblemsPanel, JsonView
│   │   │   ├── Toolbar/         # Toolbar.tsx, StatusBar.tsx
│   │   │   └── Settings/        # LLMSettings.tsx
│   │   ├── services/            # api.ts, websocket.ts, llm.ts
│   │   ├── store/               # canvasStore, pipelineStore, runtimeStore, workspaceStore
│   │   ├── types/               # pipeline.ts, schema.ts, runtime.ts
│   │   └── utils/               # validator.ts, history.ts, storage.ts
│   ├── package.json
│   └── vite.config.ts
├── server/                      # Backend (Express.js)
│   ├── src/
│   │   ├── api/                 # schema.ts, validate.ts, pipeline.ts, workspace.ts
│   │   ├── services/            # PipelineManager, SchemaLoader, Validator, WorkspaceManager
│   │   ├── websocket/           # MetricsStream.ts
│   │   └── index.ts
│   ├── package.json
│   └── tsconfig.json
├── shared/                      # Shared types (client + server)
├── package.json                 # Root workspace
└── README.md
```

## Phase Roadmap

| Phase | Name | Focus | Priority |
|-------|------|-------|----------|
| **1** | Core Editor | Canvas, palette, properties, JSON view | Critical |
| **2** | Validation | Problems panel, visual feedback | Critical |
| **3** | Runtime | Start/stop, live metrics, health | Critical |
| **6** | Polish | Undo/redo, shortcuts, recent files | High |
| **4** | LLM Basic | Pipeline generation | Medium |
| **5** | LLM Advanced | Error remediation, debugging | Medium |

**Order:** 1 → 2 → 3 → 6 → 4 → 5

## Work Protocol

### Each Iteration:
1. **Read** `docs/visual-editor/PROGRESS.md` to find current sprint
2. **Read** sprint tasks from `docs/visual-editor/PROJECT_PLAN.md`
3. **Implement** the next incomplete task
4. **Test**: `cd tools/visual-editor && npm test`
5. **Update** PROGRESS.md with task completion
6. **If sprint complete**, commit locally and output completion signal

### Definition of Done (Every Sprint):

**Code Quality:**
- [ ] TypeScript strict mode (no `any` without justification)
- [ ] ESLint passes with no errors
- [ ] Prettier formatting applied
- [ ] No console.log (use proper logging)
- [ ] All imports used (no dead code)

**Testing:**
- [ ] **ALL tests pass (100% pass rate)** - Zero failing tests allowed
- [ ] Unit tests with **80%+ coverage for business logic** (stores, services)
- [ ] Component tests for UI (React Testing Library)
- [ ] Integration tests for workflows
- [ ] Manual smoke test checklist completed

**Documentation:**
- [ ] JSDoc comments for public APIs
- [ ] README updated if new features added
- [ ] PROGRESS.md updated with completion status

**Verification:**
- [ ] `npm run build` succeeds
- [ ] `npm run dev` starts without errors
- [ ] No TypeScript errors
- [ ] No runtime errors in console during smoke test

---

## Testing Standards (CRITICAL)

**ALL tests MUST pass (100% pass rate). Zero failing tests allowed.**

### Unit Tests
- **Coverage:** 80%+ for business logic (stores, services)
- **File naming:** `*.test.ts` (not `.spec.ts`)
- **Structure:** Use `describe` blocks for grouping, `it` for test cases (not `test`)
- **Mocking:** Mock external dependencies (API, WebSocket, aprapipes.node)

```typescript
describe('canvasStore', () => {
  beforeEach(() => {
    useCanvasStore.getState().reset();
  });

  it('adds node with unique ID', () => {
    const store = useCanvasStore.getState();
    store.addNode({ type: 'TestSignalGenerator' });
    expect(store.nodes).toHaveLength(1);
  });
});
```

### Component Tests
- Use `@testing-library/react` and `@testing-library/user-event`
- Test user-visible behavior (not implementation details)
- Test accessibility (aria labels, keyboard navigation)

```typescript
describe('ModuleNode', () => {
  it('displays module name and category', () => {
    render(<ModuleNode data={{ name: 'source', type: 'TestSignalGenerator' }} />);
    expect(screen.getByText('source')).toBeInTheDocument();
  });
});
```

### Integration Tests
- Test interactions between multiple components/stores
- Use real stores (not mocks) but mock external APIs
- Test complete user workflows

### E2E Tests (Playwright)
- Test critical user paths
- Run against production build
- Use Page Object Model pattern

### TDD Approach
**Write tests FIRST where applicable:**
1. Write failing test
2. Implement feature
3. Verify test passes
4. Refactor if needed

## Completion Signals

**Sprint Complete:**
```
<sprint-complete>Sprint X.Y complete</sprint-complete>
```
Then commit locally (do not push).

**Phase Complete:**
```
<promise>PHASE X COMPLETE</promise>
```
Then commit, create demo checklist, and prepare for push.

**Blocker Encountered (after 3 attempts):**
```
<blocker>Description of issue and attempted solutions</blocker>
```

## Commit Strategy

### Sprint Complete (local only):
```bash
git add -A
git commit -m "visual-editor: Complete Sprint X.Y - <Sprint Name>

- [x] Task 1
- [x] Task 2
- [x] Task 3

Tests: All passing
Coverage: X%
DoD: Complete

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Phase Complete (push to remote):
```bash
git add -A
git commit -m "visual-editor: Complete Phase X - <Phase Name>

Summary:
- Sprint X.1: <Brief description>
- Sprint X.2: <Brief description>
- Sprint X.3: <Brief description>

Deliverables:
- Core feature complete
- All tests passing (100%)
- Coverage: >80%

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin feat/visual-editor
```

## Demo Requirements (Phase Boundaries)

At phase completion:
1. Create `docs/visual-editor/demos/phase-X-checklist.md`
2. List required screenshots and manual verification items
3. Output instructions for human demo recording
4. Wait for "demo complete" before pushing

## Key Design Decisions

From DESIGN_QUESTIONS.md:

1. **Target Users:** JS developers without deep image processing knowledge
2. **Deployment:** Localhost only (future: cloud behind Apache/nginx)
3. **Runtime Visualization:** Use available metrics (FPS, queue, health) - no WebRTC in v1
4. **LLM Integration:** User provides API keys, extensible provider system
5. **JSON View:** Read-only in v1, VS Code-style split view
6. **Validation:** On-save only, visual glyphs show errors/warnings
7. **Connections:** Show frame types on hover, prevent type mismatches
8. **Workspace:** Project-based folders like VS Code
9. **Priority 1:** Undo/redo, keyboard shortcuts, import JSON, recent files

## API Endpoints (Backend)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/schema` | GET | Get module schema |
| `/api/validate` | POST | Validate pipeline config |
| `/api/pipeline/create` | POST | Create pipeline instance |
| `/api/pipeline/:id/start` | POST | Start pipeline |
| `/api/pipeline/:id/stop` | POST | Stop pipeline |
| `/api/workspace/save` | POST | Save pipeline + layout |
| `/api/workspace/load` | POST | Load pipeline |

## WebSocket Events

| Event | Direction | Data |
|-------|-----------|------|
| `subscribe` | Client → Server | `{ pipelineId }` |
| `health` | Server → Client | `{ moduleId, fps, qlen, isQueueFull }` |
| `error` | Server → Client | `{ moduleId, message, timestamp }` |
| `status` | Server → Client | `{ pipelineStatus }` |

## Rules

1. **Test First:** Write tests before implementation where possible
2. **No Shortcuts:** Complete each DoD item before moving on
3. **Incremental:** Each iteration should make measurable progress
4. **Self-Verify:** Run tests and type-check after each change
5. **Document:** Update PROGRESS.md after completing tasks
6. **No Skipping:** Complete sprints in order, no phase-skipping

## Quick Start Commands

```bash
# Install dependencies
cd tools/visual-editor
npm install
cd client && npm install && cd ..
cd server && npm install && cd ..

# Development
npm run dev              # Start both servers
npm run dev:client       # Frontend only (port 5173)
npm run dev:server       # Backend only (port 3000)

# Testing
npm test                 # All tests
npm run test:coverage    # With coverage

# Build
npm run build            # Production build

# Type checking
npm run type-check       # TypeScript check
npm run lint             # ESLint
```

## Begin

Start by reading `docs/visual-editor/PROGRESS.md` and implementing the next incomplete task in the current sprint.

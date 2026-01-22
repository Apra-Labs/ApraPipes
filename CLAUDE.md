# CLAUDE.md - ApraPipes Project Instructions

> Instructions for Claude Code agents working on the ApraPipes project.

---

## Active Projects

### 1. Visual Editor (ApraPipes Studio) - PRIMARY FOCUS

**Branch:** `feat/visual-editor`
**Documentation:** `docs/visual-editor/`
**Status:** Planning Complete → Ready for Phase 1 Implementation

**Mission:** Build a web-based visual pipeline editor for JavaScript developers.

**Key Documents:**
- `docs/visual-editor/PROJECT_PLAN.md` - Sprint breakdown with DoD
- `docs/visual-editor/PROGRESS.md` - Current status tracker
- `docs/visual-editor/SPECIFICATION.md` - Technical spec
- `docs/visual-editor/ARCHITECTURE.md` - System architecture
- `docs/visual-editor/DESIGN_QUESTIONS.md` - Design decisions (reference)
- `docs/visual-editor/README.md` - Project overview

**Current Phase:** Phase 1 - Core Editor (Not Started)

---

## Ralph Loop Configuration (Visual Editor)

The Visual Editor uses Ralph Loop for 24x7 autonomous development.

### Starting the Loop

```bash
/ralph-loop "Implement ApraPipes Studio per docs/visual-editor/PROJECT_PLAN.md. Read PROGRESS.md to find current sprint, implement next incomplete task, run tests, update PROGRESS.md. Output <sprint-complete>Sprint X.Y</sprint-complete> when sprint done. Output <promise>PHASE COMPLETE</promise> when phase done." --max-iterations 100 --completion-promise "PHASE COMPLETE"
```

### Loop Behavior

**Each Iteration:**
1. Read `docs/visual-editor/PROGRESS.md` → find current sprint
2. Read `docs/visual-editor/PROJECT_PLAN.md` → find sprint tasks
3. Implement next incomplete task
4. Run tests: `cd tools/visual-editor && npm test`
5. Update PROGRESS.md with completion status

**Sprint Complete:** Output `<sprint-complete>Sprint X.Y</sprint-complete>`, commit locally
**Phase Complete:** Output `<promise>PHASE COMPLETE</promise>`, commit + push

### Commit Strategy

**Sprint Complete (local only):**
```bash
git add -A
git commit -m "visual-editor: Complete Sprint X.Y - <Sprint Name>

- [x] Task 1
- [x] Task 2

Tests: All passing
Coverage: X%

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Phase Complete (push to remote):**
```bash
git add -A
git commit -m "visual-editor: Complete Phase X - <Phase Name>

Summary:
- Sprint X.1: <Brief>
- Sprint X.2: <Brief>

Tests: All passing
Coverage: X%

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin feat/visual-editor
```

### Demo Requirements

At phase completion, pause for human demo recording:
1. Create `docs/visual-editor/demos/phase-X-checklist.md` with screenshot requirements
2. Output instructions for human to record demo
3. Wait for "demo complete" confirmation before pushing

### Canceling the Loop

```bash
/cancel-ralph
```

---

## Visual Editor Technical Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript |
| Canvas | React Flow |
| State | Zustand |
| Styling | Tailwind CSS + shadcn/ui |
| Backend | Express.js |
| Real-time | WebSocket (ws) |
| Build | Vite |
| Testing | Vitest (frontend), Jest (backend), Playwright (E2E) |

---

## Visual Editor Phase Roadmap

| Phase | Name | Priority | Status |
|-------|------|----------|--------|
| 1 | Core Editor | Critical | Not Started |
| 2 | Validation | Critical | Not Started |
| 3 | Runtime | Critical | Not Started |
| 6 | Polish | High | Not Started |
| 4 | LLM Basic | Medium | Not Started |
| 5 | LLM Advanced | Medium | Not Started |

**Order:** 1 → 2 → 3 → 6 → 4 → 5

---

## Visual Editor Definition of Done (All Sprints)

### Code Quality
- [ ] TypeScript strict mode (no `any` without justification)
- [ ] ESLint passes with no errors
- [ ] Prettier formatting applied
- [ ] No console.log (use proper logging)
- [ ] All imports used (no dead code)

### Testing (CRITICAL)
**ALL tests MUST pass (100% pass rate). Zero failing tests allowed.**

- [ ] **100% pass rate** - All tests must pass
- [ ] **80%+ coverage** for business logic (stores, services)
- [ ] Component tests for UI (React Testing Library)
- [ ] Integration tests for workflows
- [ ] Manual smoke test checklist completed

**Testing Conventions:**
- File naming: `*.test.ts` (not `.spec.ts`)
- Use `describe` blocks for grouping, `it` for test cases
- Mock external dependencies (API, WebSocket, aprapipes.node)
- Write tests FIRST (TDD where applicable)

### Documentation
- [ ] JSDoc comments for public APIs
- [ ] README updated if new features added
- [ ] PROGRESS.md updated with completion status

### Verification
- [ ] Build succeeds (`npm run build`)
- [ ] Dev servers start without errors
- [ ] No TypeScript errors
- [ ] No runtime errors in console during smoke test

---

## Visual Editor Directory Structure

```
tools/visual-editor/
├── client/                      # Frontend (Vite + React)
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── Canvas/          # ModuleNode, ConnectionEdge
│   │   │   ├── Panels/          # ModulePalette, PropertyPanel, etc.
│   │   │   └── Toolbar/         # Toolbar, StatusBar
│   │   ├── services/            # API clients
│   │   ├── store/               # Zustand stores
│   │   ├── types/               # TypeScript types
│   │   └── utils/               # Helpers
│   ├── package.json
│   └── vite.config.ts
├── server/                      # Backend (Express.js)
│   ├── src/
│   │   ├── api/                 # REST routes
│   │   ├── services/            # Business logic
│   │   └── websocket/           # WebSocket server
│   ├── package.json
│   └── tsconfig.json
├── shared/                      # Shared types
├── package.json                 # Root workspace
├── PROMPT.md                    # Ralph loop prompt
└── README.md
```

---

## 2. SDK Packaging (Declarative Pipeline) - MAINTENANCE

**Branch:** `feat/sdk-packaging`
**Documentation:** `docs/declarative-pipeline/`
**Status:** Complete (maintenance only)

**Only work on this if explicitly requested.**

---

## SDK Structure

```
aprapipes-sdk-{platform}/
├── bin/
│   ├── aprapipes_cli              # CLI tool
│   ├── aprapipesut                # Unit tests
│   ├── aprapipes.node             # Node.js addon
│   └── *.so / *.dll / *.dylib     # Shared libraries
├── lib/
│   └── *.a / *.lib                # Static libraries
├── include/
│   └── *.h                        # Header files
├── examples/
│   ├── basic/                     # JSON pipeline examples
│   ├── cuda/                      # CUDA examples (if applicable)
│   ├── jetson/                    # Jetson examples (ARM64 only)
│   └── node/                      # Node.js examples
├── data/
│   ├── frame.jpg                  # Sample input files
│   └── faces.jpg                  # For examples to work out of box
├── README.md                      # SDK usage documentation
└── VERSION                        # Version info
```

---

## Critical Rules

### 1. Build and Test Before Commit (MANDATORY)

**NEVER commit code without verifying build and tests pass.**

```bash
# For Visual Editor
cd tools/visual-editor
npm install
npm run build
npm test

# For C++ code
cmake --build build -j$(nproc)
./build/aprapipesut --run_test="<RelevantSuite>/*" --log_level=test_suite
```

If build/tests fail: fix first, then commit. No exceptions.

### 2. Wait for CI Before Push

Before pushing to this branch, verify all current CI runs are complete:

```bash
gh run list --limit 10 --json status,name,conclusion,headBranch | jq -r '.[] | select(.status != "completed") | "\(.name) (\(.headBranch))"'
```

### 3. Platform Protection

**Keep all 4 CI workflows GREEN:**
- CI-Windows, CI-Linux, CI-Linux-ARM64, CI-MacOSX-NoCUDA

**Visual Editor Exception:** `tools/visual-editor/**` is excluded from CI until ready.

### 4. Code Review Before Commit

```bash
git diff --staged          # Review ALL changes
git diff --staged --stat   # Check which files changed
```

Check for: debug code, temporary hacks, commented-out code, unrelated changes.

---

## Jetson Development

### Device Rules

When working on Jetson (ssh akhil@192.168.1.18):
- **NEVER** modify `/data/action-runner/` (GitHub Actions)
- **NEVER** delete `/data/.cache/` (vcpkg cache shared with CI)
- **ALWAYS** work in `/data/ws/`

### Build Commands

```bash
ssh akhil@192.168.1.18
cd /data/ws/ApraPipes

# Configure
cmake -B _build -S base \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ARM64=ON \
  -DENABLE_CUDA=ON

# Build (use -j2 to avoid OOM)
TMPDIR=/data/.cache/tmp cmake --build _build -j2

# Test
./_build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite
```

---

## Quick Reference

```bash
# Visual Editor Development
cd tools/visual-editor
npm install
npm run dev          # Start dev servers
npm test             # Run tests
npm run build        # Production build

# Check progress
cat docs/visual-editor/PROGRESS.md

# Check CI status
gh run list --limit 8

# C++ Build
cmake --build build -j$(nproc)

# Test specific suite
./build/aprapipesut --run_test="ModuleRegistryTests/*" --log_level=test_suite

# Run CLI
./build/aprapipes_cli list-modules
./build/aprapipes_cli run examples/simple.json
```

---

## Key Documentation

### Visual Editor (ApraPipes Studio)
| Document | Purpose |
|----------|---------|
| `docs/visual-editor/PROJECT_PLAN.md` | **Sprint breakdown with acceptance criteria** |
| `docs/visual-editor/PROGRESS.md` | **Current sprint status and completion tracking** |
| `docs/visual-editor/SPECIFICATION.md` | Technical specification |
| `docs/visual-editor/ARCHITECTURE.md` | System architecture details |
| `docs/visual-editor/DESIGN_QUESTIONS.md` | Design decisions (reference) |
| `docs/visual-editor/README.md` | Project overview |

### Declarative Pipeline (SDK)
| Document | Purpose |
|----------|---------|
| `docs/declarative-pipeline/SDK_PACKAGING_PLAN.md` | SDK packaging plan |
| `docs/declarative-pipeline/PROGRESS.md` | Current status, sprint progress |
| `docs/declarative-pipeline/PROJECT_PLAN.md` | Sprint overview, objectives |
| `.github/workflows/build-test.yml` | Windows/Linux x64 workflow |
| `.github/workflows/build-test-macosx.yml` | macOS workflow |
| `.github/workflows/build-test-lin.yml` | ARM64 workflow |

# DevOps Operating Mode: Autonomous with Institutional Memory

## Operating Philosophy
- **Tesla FSD mode**: Keep moving autonomously, user observes and can intervene
- **Every run teaches**: Document learnings for future sessions
- **Survive /clear**: State persists in files, not just context

---

## Core Loop
```
while task not complete:
    1. Check LEARNINGS.md — don't repeat past mistakes
    2. Analyze current state
    3. Make changes (YAML, scripts, etc.)
    4. Update CURRENT_STATE.md with what you're about to do
    5. Commit and trigger build using BLOCKING command (see below)
    6. WAIT for build to complete — do not proceed
    7. Analyze results
    8. Update CURRENT_STATE.md with outcome
    9. If new lesson learned → append to LEARNINGS.md
    10. Continue to step 1 (pass → next task, fail → fix and retry)
```

---

## Build Commands (BLOCKING)

### Push and Watch (USE THIS)
```bash
git push origin HEAD && sleep 5 && \
  RUN_ID=$(gh run list -L1 --json databaseId -q '.[0].databaseId') && \
  echo ">>> Watching run $RUN_ID <<<" && \
  gh run watch $RUN_ID -i 120 --exit-status && \
  echo "✓ PASSED" || \
  (echo "✗ FAILED" && gh run view $RUN_ID --log-failed)
```

### Trigger Workflow and Watch
```bash
gh workflow run <workflow.yml> && sleep 5 && \
  RUN_ID=$(gh run list -L1 --json databaseId -q '.[0].databaseId') && \
  gh run watch $RUN_ID -i 120 --exit-status && \
  echo "✓ PASSED" || \
  (echo "✗ FAILED" && gh run view $RUN_ID --log-failed)
```

### Rules
- These commands BLOCK until complete — stay with them
- Do NOT background with `&`
- Do NOT proceed until exit code is known
- User can type during long waits — respond briefly, then continue waiting

---

## Persistent Memory System

### Files
| File | Purpose | When to Update |
|------|---------|----------------|
| `.claude/LEARNINGS.md` | Institutional knowledge (timeless) | Append after discovering new patterns |
| `.claude/CURRENT_STATE.md` | Current session state (ephemeral) | Overwrite after each significant action |

### On Session Start
1. **FIRST: Verify branch context matches**
   ```bash
   git branch --show-current
   ```
   - Compare actual branch with CURRENT_STATE.md branch
   - If mismatch: **DO NOT trust context from previous session**
   - Background processes, workflow IDs, and state may belong to a DIFFERENT agent on a DIFFERENT branch
   - When in doubt, ask user what task they want to work on

2. Read `.claude/LEARNINGS.md` — know what's been tried before
3. Read `.claude/CURRENT_STATE.md` — know where we left off (only if branch matches!)
4. Resume from documented state

### After Every Workflow Run
1. Update `CURRENT_STATE.md` with:
   - Current branch and goal
   - What just happened
   - Pass/fail status
   - Next steps

2. If something NEW was learned (failure mode, gotcha, fix), append to `LEARNINGS.md` as a **timeless principle**:
```markdown
### Category | Issue Title
**Symptom:** What you observed
**Root cause:** Why it happened
**Fix:** What resolved it
**Rule:** One-line principle for future reference
```

### Before Making Changes
- Check LEARNINGS.md for prior attempts at similar changes
- Don't repeat documented failures
- Reference past solutions

### On /clear or Context Reset
User will say "continue", "resume", or use `/project:resume`
→ Read CURRENT_STATE.md and LEARNINGS.md
→ Resume from documented state
→ Keep moving without waiting for approval

---

## Behavioral Rules

### Keep Moving
- Do NOT wait for user approval between steps
- Do NOT ask "should I proceed?" — just proceed
- User will interject if needed — assume green light otherwise

### User Interjection
- User may type questions/commands mid-operation
- Answer concisely, then resume the loop
- If user redirects, acknowledge and adapt
- Common queries during long builds:
  - "status" → summarize current state
  - "what happened?" → summarize last run
  - "hold" → pause and explain

### Long Builds (1-3 hrs)
- While blocked on `gh run watch`, output scrolls — this is expected
- User may check in during this time
- When build completes, summarize result and continue

### Never Do
- `git push &` or any backgrounding
- `gh workflow run X` without subsequent watch
- Move to next task before build result is known
- Repeat a mistake documented in LEARNINGS.md
- Stop without updating CURRENT_STATE.md
- **Trust background process IDs or workflow state from a resumed session without first verifying the current git branch matches CURRENT_STATE.md** — different agents may run on different branches simultaneously

---

## Quick Reference Commands

| Command | Purpose |
|---------|---------|
| `/project:status` | Quick state check during long builds |
| `/project:resume` | Pick up after /clear |
| `/project:checkpoint` | Force save current state |
| `/project:last-run` | Summarize most recent build |
| `/project:learned <note>` | Quick-add a learning |

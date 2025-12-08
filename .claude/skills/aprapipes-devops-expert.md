# ApraPipes DevOps Expert Agent

**Purpose**: Autonomous agent for monitoring and managing ApraPipes CI/CD workflows across all platforms (Windows, Linux x64/ARM64, Jetson, Docker, WSL).

## Core Responsibilities

1. **Continuous Workflow Monitoring**
   - Monitor all GitHub Actions workflows 24/7
   - Detect and cancel duplicate/stuck workflows
   - Alert on failures with actionable diagnostics
   - Track build times and resource usage

2. **Workflow Management**
   - Validate workflow YAML syntax before commits
   - Manage branch-specific workflow triggers
   - Coordinate multi-platform builds
   - Handle workflow dependencies

3. **Platform-Specific Expertise**
   - Jetson ARM64: curl HTTP/1.1 workarounds, gcc-11, vcpkg issues
   - Windows CUDA: Self-hosted runner management, cache strategies
   - Linux containers: Docker build optimization
   - WSL: Hybrid Windows/Linux build coordination

## Critical Rules Learned (Last 3 Days)

### GitHub Actions Syntax
```yaml
# ❌ WRONG - Cannot use both
pull_request:
  branches: [ main ]
  branches-ignore: [ fix-branch ]

# ✅ CORRECT - Use only one
pull_request:
  branches-ignore: [ fix-branch ]
```

**Rule**: GitHub Actions allows EITHER `branches` OR `branches-ignore`, NEVER both.

### Workflow Verification Protocol

**ALWAYS follow after ANY workflow change:**

```bash
# 1. Push changes
git push

# 2. Wait 5-10 seconds for GitHub to process
sleep 8

# 3. Immediately verify what's running
gh run list --limit 10 --branch <branch> --json databaseId,status,name,createdAt

# 4. Cancel unintended workflows IMMEDIATELY
# (duplicates, wrong platforms, old runs)
gh run cancel <ID>

# 5. Verify only intended workflows remain
gh run list --limit 5 --json name,status | jq '.[] | select(.status == "queued" or .status == "in_progress")'
```

**Never assume workflows are correct**. Always verify.

### Duplicate Workflow Problem

**Issue**: Every workflow file change triggers workflows using BOTH old and new definitions.

**Solution**:
1. Immediately cancel workflows started BEFORE the latest commit
2. Keep only workflows created AFTER the fix commit
3. Use timestamps to identify which to cancel

### Platform-Specific Issues

#### Jetson ARM64 (Self-Hosted)
```bash
# Issue: curl 7.58.0 has HTTP/2 framing bugs
# Solution: Force HTTP/1.1 globally
echo "http1.1" > ~/.curlrc

# Issue: vcpkg needs C++20
# Solution: Use gcc-11/g++-11
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11

# Issue: curl 8.11.0 upgrade breaks with symbol errors
# Solution: Keep system curl 7.58.0 + HTTP/1.1 workaround
```

#### Windows CUDA (Self-Hosted)
```yaml
# Issue: Prep phase needed for vcpkg cache
# Solution: Two-phase build
jobs:
  prep:
    is-prep-phase: true
  build:
    needs: prep
    is-prep-phase: false
```

#### All Platforms
- **Cache deletion**: Requires `actions: write` permission at workflow level
- **Force cache update**: Use `workflow_dispatch` input parameter
- **Self-hosted runners**: Set `is-selfhosted: true` to skip sudo commands

## Monitoring Strategy

### Continuous Monitor Script
```bash
#!/bin/bash
# Run indefinitely, adjust interval based on activity

while true; do
  # Get active workflows
  active=$(gh run list --limit 20 --json databaseId,status,name,createdAt,conclusion \
    | jq '[.[] | select(.status == "queued" or .status == "in_progress")]')

  # Detect duplicates (same name, different IDs)
  # Cancel older ones

  # Detect stuck workflows (running > 2 hours)
  # Alert and possibly cancel

  # Check for failures
  # Download logs and analyze

  # Adjust sleep based on activity
  if [ $(echo "$active" | jq 'length') -gt 0 ]; then
    sleep 60  # Check every minute when active
  else
    sleep 300  # Check every 5 minutes when idle
  fi
done
```

### Failure Analysis

When a workflow fails:
1. Download logs: `gh run view <ID> --log`
2. Search for known error patterns:
   - `HTTP/2 framing layer error` → curl issue
   - `undefined symbol` → library version mismatch
   - `vcpkg install failed` → network or dependencies
   - `No space left on device` → runner disk full
3. Apply known fixes automatically
4. Report to user if unknown failure

## Branch-Specific Strategies

### fix/ci-additional-workflows
- **Goal**: Fix ARM64 build only
- **Strategy**: Disable all non-ARM64 workflows temporarily
- **Method**: Add `branches-ignore: [fix/ci-additional-workflows]` to pull_request triggers
- **Problem**: This DOESN'T work for PR workflows (they still trigger)
- **Better Solution**: Temporarily rename/disable workflow files directly

### main branch
- **Goal**: Run all platforms
- **Strategy**: Full matrix builds
- **Coordination**: Self-hosted runners may be offline, queue gracefully

## Common Mistakes to Avoid

1. ❌ **Making workflow changes without immediate verification**
   - Always check `gh run list` within 10 seconds of pushing

2. ❌ **Leaving duplicate workflows running**
   - Wastes runner resources
   - Creates confusing build status

3. ❌ **Assuming branches-ignore works for pull_request**
   - It does NOT prevent PR-triggered workflows
   - Use different approach (rename files, use paths-ignore)

4. ❌ **Upgrading system packages without testing**
   - curl 8.11.0 upgrade broke with symbol errors
   - Always test in isolation first

5. ❌ **Not monitoring background processes**
   - Long builds can fail silently
   - Always have active monitoring

## Workflow File Inventory

```
.github/workflows/
├── CI-Linux-ARM64.yml        # Jetson self-hosted
├── CI-Linux-CUDA.yml          # Linux CUDA self-hosted
├── CI-Linux-CUDA-Docker.yml   # Containerized CUDA build
├── CI-Linux-CUDA-wsl.yml      # WSL on Windows CUDA runner
├── CI-Linux-NoCUDA.yml        # Ubuntu 22.04 GitHub-hosted
├── CI-Win-CUDA.yml            # Windows CUDA self-hosted
├── CI-Win-NoCUDA.yml          # Windows GitHub-hosted (2-phase)
├── build-test-lin.yml         # Reusable Linux build
├── build-test-lin-wsl.yml     # Reusable WSL build
├── build-test-win.yml         # Reusable Windows build
└── publish-test.yml           # Test result publishing
```

## Environment Variables

```bash
# Force vcpkg to use system binaries (ARM64)
export VCPKG_FORCE_SYSTEM_BINARIES=1

# Force curl HTTP/1.1 (alternative to .curlrc)
export CURL_HTTP_VERSION=HTTP/1.1

# Allow old Node.js on Jetson
export ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true
```

## Self-Hosted Runner Maintenance

### Jetson ARM64 (AGX)
```bash
# Location: /mnt/disks/actions-runner
# Access: SSH via openrport tunnel
# Port: 28032 (current), changes dynamically

# Verify runner health
cd /mnt/disks/actions-runner
./run.sh --check

# Check disk space (critical)
df -h /mnt/disks

# Clean old build artifacts
rm -rf /tmp/vcpkg-* /tmp/_temp
```

### Windows CUDA
```bash
# Self-hosted, runs as service
# Cache strategy: Prep phase builds vcpkg, main phase uses cache
```

## Future Improvements

1. **Automated Log Analysis**
   - Parse logs for error patterns
   - Suggest fixes automatically

2. **Predictive Monitoring**
   - Learn typical build times
   - Alert when builds exceed expected duration

3. **Resource Optimization**
   - Track cache hit rates
   - Optimize build parallelism (nProc settings)

4. **Cross-Platform Coordination**
   - Don't start expensive builds if likely to fail on other platforms
   - Fast-fail strategy

## Integration Points

### With Claude Code
```bash
# User asks: "How's the CI doing?"
# Agent checks workflows and reports status

# User asks: "Fix the ARM64 build"
# Agent knows to:
# 1. Check current failures
# 2. Apply known fixes
# 3. Monitor until success
```

### With GitHub CLI
```bash
# All operations via gh CLI for consistency
gh run list
gh run view <ID>
gh run cancel <ID>
gh run download <ID>
gh workflow list
```

### With MCP (Future)
- Could be an MCP server providing workflow monitoring tools
- Claude Code connects via MCP protocol
- Persistent state tracking

---

## Quick Reference Commands

```bash
# Check active workflows
gh run list --limit 10 --json name,status,conclusion,createdAt

# Cancel a workflow
gh run cancel <ID>

# View workflow logs
gh run view <ID> --log

# Download workflow logs
gh run download <ID>

# Watch workflow in real-time
watch -n 10 'gh run view <ID> --json status,conclusion | jq'

# List all workflows
gh workflow list

# Trigger manual workflow
gh workflow run <workflow.yml>

# Cancel ALL active workflows on a branch
gh run list --branch <branch> --json databaseId --limit 50 \
  | jq -r '.[] | .databaseId' | xargs -I {} gh run cancel {}
```

---

**Last Updated**: 2025-12-07 (Session covering ARM64 curl HTTP/1.1 fixes, workflow syntax errors, duplicate workflow management)

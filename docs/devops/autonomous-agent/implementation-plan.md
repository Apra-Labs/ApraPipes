# ApraPipes DevOps Expert Agent - Implementation Plan

## Goal
Create a persistent, autonomous agent that monitors and manages ApraPipes CI/CD workflows 24/7.

## Architecture Options

### Option 1: MCP Server (Recommended)
**Pros:**
- Native Claude Code integration
- Persistent state across sessions
- Can run 24/7 independently
- Provides tools to any Claude instance

**Cons:**
- Requires Node.js/TypeScript development
- Learning curve for MCP protocol

**Implementation:**
```typescript
// mcp-server/aprapipes-devops/src/index.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "aprapipes-devops",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});

// Tools:
// - monitor_workflows: Get current workflow status
// - analyze_failure: Download and analyze failure logs
// - cancel_duplicate: Cancel duplicate workflows
// - validate_yaml: Check workflow syntax before commit
```

### Option 2: Standalone Daemon + Skill
**Pros:**
- Simpler to implement (pure Bash/Python)
- Can start immediately
- No MCP complexity

**Cons:**
- Less integrated with Claude Code
- Requires manual startup
- State management more complex

**Implementation:**
```bash
#!/bin/bash
# /Users/akhil/bin/aprapipes-devops-daemon.sh

cd /Users/akhil/git/ApraPipes

while true; do
  # Monitor workflows
  python3 scripts/devops-monitor.py

  # Sleep based on activity
  sleep 60
done
```

### Option 3: Hybrid (Start Here)
**Phase 1**: Standalone Python script
**Phase 2**: Convert to MCP server when stable

## Phase 1: Python Monitoring Script

### 1. Create Monitoring Script
```python
# scripts/devops-monitor.py
import subprocess
import json
from datetime import datetime, timedelta

class WorkflowMonitor:
    def __init__(self, repo="Apra-Labs/ApraPipes"):
        self.repo = repo
        self.state_file = ".devops-state.json"

    def get_active_workflows(self):
        """Get all active workflows"""
        cmd = f"gh run list --repo {self.repo} --limit 50 --json databaseId,status,name,createdAt,event,conclusion"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def detect_duplicates(self, workflows):
        """Find duplicate workflows (same name, both active)"""
        active = [w for w in workflows if w['status'] in ['queued', 'in_progress']]
        by_name = {}
        for w in active:
            name = w['name']
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(w)

        duplicates = {name: wfs for name, wfs in by_name.items() if len(wfs) > 1}
        return duplicates

    def cancel_older_duplicates(self, duplicates):
        """Cancel older workflow when duplicates detected"""
        for name, workflows in duplicates.items():
            # Sort by creation time, keep newest
            sorted_wfs = sorted(workflows, key=lambda w: w['createdAt'], reverse=True)
            to_cancel = sorted_wfs[1:]  # All except newest

            for wf in to_cancel:
                print(f"Canceling duplicate {name} (ID: {wf['databaseId']})")
                subprocess.run(f"gh run cancel {wf['databaseId']}", shell=True)

    def analyze_failures(self, workflows):
        """Analyze recent failures and suggest fixes"""
        recent_failures = [
            w for w in workflows
            if w['conclusion'] == 'failure'
            and self._is_recent(w['createdAt'], hours=1)
        ]

        for wf in recent_failures:
            print(f"\nAnalyzing failure: {wf['name']} (ID: {wf['databaseId']})")
            # Download logs and analyze
            self._analyze_logs(wf['databaseId'])

    def _is_recent(self, timestamp, hours=1):
        """Check if timestamp is within last N hours"""
        created = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return datetime.now(created.tzinfo) - created < timedelta(hours=hours)

    def _analyze_logs(self, run_id):
        """Download and analyze logs for known error patterns"""
        # TODO: Implement log analysis
        pass

    def run_once(self):
        """Single monitoring cycle"""
        workflows = self.get_active_workflows()

        # Check for duplicates
        duplicates = self.detect_duplicates(workflows)
        if duplicates:
            print(f"Found {len(duplicates)} sets of duplicate workflows")
            self.cancel_older_duplicates(duplicates)

        # Analyze failures
        self.analyze_failures(workflows)

        # Report status
        active = [w for w in workflows if w['status'] in ['queued', 'in_progress']]
        print(f"\nActive workflows: {len(active)}")
        for w in active:
            print(f"  - {w['name']} [{w['status']}]")

if __name__ == "__main__":
    monitor = WorkflowMonitor()
    monitor.run_once()
```

### 2. Create Daemon Wrapper
```bash
#!/bin/bash
# bin/start-devops-monitor.sh

cd /Users/akhil/git/ApraPipes

LOG_FILE="logs/devops-monitor.log"
mkdir -p logs

echo "Starting ApraPipes DevOps Monitor at $(date)" | tee -a "$LOG_FILE"

while true; do
  echo "=== Check at $(date) ===" | tee -a "$LOG_FILE"

  python3 scripts/devops-monitor.py 2>&1 | tee -a "$LOG_FILE"

  # Adaptive sleep
  active_count=$(gh run list --limit 50 --json status | jq '[.[] | select(.status == "queued" or .status == "in_progress")] | length')

  if [ "$active_count" -gt 0 ]; then
    sleep 60  # Check every minute when active
  else
    sleep 300  # Check every 5 minutes when idle
  fi
done
```

### 3. Create LaunchAgent (macOS) for Auto-Start
```xml
<!-- ~/Library/LaunchAgents/com.aprapipes.devops.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aprapipes.devops</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/akhil/git/ApraPipes/bin/start-devops-monitor.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/akhil/git/ApraPipes/logs/devops-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/akhil/git/ApraPipes/logs/devops-stderr.log</string>
</dict>
</plist>
```

## Phase 2: MCP Server Integration

### Directory Structure
```
mcp-servers/
└── aprapipes-devops/
    ├── package.json
    ├── tsconfig.json
    ├── src/
    │   ├── index.ts          # Main MCP server
    │   ├── workflow.ts       # Workflow monitoring logic
    │   ├── analyzer.ts       # Log analysis
    │   └── state.ts          # Persistent state management
    └── README.md
```

### MCP Configuration
```json
// Add to Claude Code MCP settings
{
  "mcpServers": {
    "aprapipes-devops": {
      "command": "node",
      "args": ["/Users/akhil/git/ApraPipes/mcp-servers/aprapipes-devops/build/index.js"]
    }
  }
}
```

### Tools to Provide
1. `monitor_workflows` - Get current status
2. `analyze_failure(run_id)` - Analyze failure logs
3. `cancel_duplicates` - Auto-cancel duplicates
4. `validate_yaml(file)` - Pre-commit validation
5. `trigger_build(workflow, branch)` - Manual trigger
6. `get_runner_status` - Self-hosted runner health

## Phase 3: Advanced Features (Future)

### Predictive Analysis
- Learn typical build times per platform
- Alert when builds exceed expected duration
- Predict failures based on log patterns

### Auto-Remediation
- Apply known fixes automatically (e.g., restart failed vcpkg download)
- Retry transient failures
- Clear disk space on runners

### Integration with Notifications
- Slack/Discord alerts for failures
- Email reports for daily build status
- Dashboard for visualizing trends

## Getting Started

### Implementation Checklist

**Foundation (Phase 1)**:
- [ ] Create Python monitoring script (`scripts/devops-monitor.py`)
- [ ] Test monitoring script manually
- [ ] Create daemon wrapper (`bin/start-devops-monitor.sh`)
- [ ] Test daemon in foreground

**Deployment**:
- [ ] Set up LaunchAgent for auto-start (macOS)
- [ ] Monitor behavior for 24 hours
- [ ] Fix any issues discovered

**Enhancement (Phase 2)**:
- [ ] Convert to MCP server
- [ ] Integrate with Claude Code
- [ ] Add sophisticated log analysis

## Testing Strategy

### Manual Test
```bash
cd /Users/akhil/git/ApraPipes
python3 scripts/devops-monitor.py
# Should show current workflows and any duplicates
```

### Daemon Test
```bash
./bin/start-devops-monitor.sh
# Let run for 1 hour, check logs/devops-monitor.log
```

### MCP Test
```bash
# From Claude Code:
# > /mcp aprapipes-devops monitor_workflows
# Should return current workflow status
```

## Success Criteria

- ✅ Agent runs 24/7 without crashes
- ✅ Duplicates are canceled within 60 seconds
- ✅ Failures are analyzed and reported
- ✅ No manual intervention needed for routine tasks
- ✅ Can answer "How's CI doing?" instantly

---

**Next Step**: Start with Phase 1 (Python monitoring script).

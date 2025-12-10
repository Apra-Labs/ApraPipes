# ApraPipes 24/7 CI Monitoring & Self-Healing Architecture

## Overview
A Claude-powered monitoring system running on the self-hosted builder (192.168.1.102) that continuously monitors all ApraPipes workflows, automatically diagnoses failures, and applies fixes without human intervention.

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Hosted Builder                          │
│                   (192.168.1.102 / RTX 4060)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         Claude CI Monitor (Systemd Service)               │ │
│  │                                                           │ │
│  │  • GitHub Webhook Listener (Port 9876)                   │ │
│  │  • Workflow Status Poller                                │ │
│  │  • Log Analysis Engine                                   │ │
│  │  • Auto-Remediation Engine                               │ │
│  │  • Notification System                                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ↕                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Knowledge Base / Cache                       │ │
│  │                                                           │ │
│  │  • Known failure patterns & fixes                        │ │
│  │  • vcpkg binary cache state tracker                      │ │
│  │  • Runner health metrics history                         │ │
│  │  • Build success/failure statistics                      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ↕                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │           GitHub Actions Runner Service                   │ │
│  │                                                           │ │
│  │  • Executes CI-Linux-CUDA builds                         │ │
│  │  • Workspace: ~/actions-runner/_work/ApraPipes/          │ │
│  │  • vcpkg cache: ~/actions-runner/_work/ApraPipes/.cache/ │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↕
                   GitHub Actions API
                            ↕
┌─────────────────────────────────────────────────────────────────┐
│                  GitHub Cloud Infrastructure                    │
│                                                                 │
│  • CI-Linux-CUDA (self-hosted)                                 │
│  • CI-Linux-NoCUDA (cloud)                                     │
│  • CI-Linux-ARM64 (self-hosted AGX)                            │
│  • CI-Linux-CUDA-Docker (cloud)                                │
│  • CI-Windows-CUDA/NoCUDA (cloud)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Monitoring Modes

### Mode 1: Event-Driven (Webhook) - PRIMARY
- GitHub webhook sends notifications on workflow events
- Near real-time response (< 5 seconds from failure to detection)
- Low CPU/network overhead
- Requires port forwarding or ngrok tunnel

### Mode 2: Polling Fallback
- Poll GitHub API every 60 seconds for workflow status
- Backup if webhook delivery fails
- Higher latency but guaranteed delivery

### Mode 3: Hybrid (Recommended)
- Webhooks for real-time notifications
- Periodic polling every 5 minutes as health check
- Best of both worlds

## Auto-Remediation Decision Tree

```
Workflow Failed
    │
    ├─> Download & Analyze Logs
    │
    ├─> Pattern Match Against Known Issues:
    │
    ├─> Issue: "vcpkg install failed" + "Ninja not found"
    │   ├─> Check: VCPKG_DEFAULT_BINARY_CACHE path valid?
    │   │   ├─> Invalid → Fix workflow env variable → Commit & Push
    │   │   └─> Valid → Check cache contents
    │   │       ├─> Corrupt packages → Clear cache → Retry build
    │   │       └─> Cache OK → Check Ninja installation
    │
    ├─> Issue: "ffmpeg:x64-linux failed" + "mathops.h:125"
    │   ├─> Check: GCC version mismatch
    │   │   ├─> Cache has GCC-13 packages but GCC-12 is default
    │   │   │   └─> Clear vcpkg cache → Retry build
    │   │   ├─> GCC-13 is still default
    │   │   │   └─> Run setup script → Set GCC-12 → Clear cache → Retry
    │   │   └─> Other GCC issue → Escalate to human
    │
    ├─> Issue: Disk space full
    │   ├─> Clean old build artifacts (> 7 days)
    │   ├─> Clean Docker images
    │   ├─> Clean vcpkg downloads
    │   └─> Retry build
    │
    ├─> Issue: Runner offline/unhealthy
    │   ├─> Restart runner service
    │   └─> If still offline → Notify human
    │
    ├─> Issue: Test failures (not build failures)
    │   ├─> Retry once (flaky test detection)
    │   ├─> If still fails → Create GitHub Issue
    │   └─> Notify human
    │
    └─> Unknown Issue
        ├─> Extract error logs
        ├─> Create GitHub Issue with logs
        ├─> Notify human
        └─> Add to knowledge base for learning
```

## Core Capabilities

### 1. Failure Detection & Classification
- Parse build logs for error patterns
- Categorize by severity: Critical, High, Medium, Low
- Identify root cause vs. symptom
- Track failure frequency (flaky vs. persistent)

### 2. Known Issue Database
Based on learnings from this session:

| Issue Pattern | Root Cause | Auto-Fix | Confidence |
|--------------|------------|----------|------------|
| `VCPKG_DEFAULT_BINARY_CACHE must be a directory (was: ~/.cache/vcpkg)` | Tilde not expanded | Update workflow to use `${{ github.workspace }}/../.cache/vcpkg` | 100% |
| `Environment variable VCPKG_DEFAULT_BINARY_CACHE must be a directory (was: $HOME/.cache/vcpkg)` | Shell var not expanded | Update workflow to use `${{ github.workspace }}/../.cache/vcpkg` | 100% |
| `ffmpeg:x64-linux failed` + `mathops.h:125: Error: operand type mismatch for 'shr'` | GCC-13/ffmpeg incompatibility OR stale vcpkg cache | 1. Check if GCC-12 is default<br>2. Clear vcpkg cache<br>3. Retry | 95% |
| `CMake was unable to find a build program corresponding to "Ninja"` | Usually symptom of earlier vcpkg failure | Check vcpkg logs for real error | 90% |
| Disk usage > 90% on `/` | Build artifacts accumulation | Clean old builds, Docker images, vcpkg downloads | 100% |
| Runner shows offline | Runner service crashed | Restart systemd service | 80% |

### 3. Health Monitoring
Monitor runner health metrics:
- Disk space (critical when > 90%)
- Memory usage
- CPU temperature (important for CUDA builds)
- vcpkg cache size
- Number of failed builds in last 24h
- Build duration trends (detect performance degradation)

### 4. Proactive Maintenance
Scheduled tasks:
- **Daily**: Clean build artifacts > 7 days old
- **Daily**: Check for vcpkg updates
- **Weekly**: Validate all runners are healthy
- **Weekly**: Report build statistics
- **Monthly**: Clean Docker images not used in 30 days

## Implementation Files

### File Structure
```
.claude/
├── monitoring/
│   ├── ci-monitor.sh              # Main monitoring daemon script
│   ├── webhook-listener.js        # GitHub webhook receiver (Node.js)
│   ├── auto-fix.sh               # Auto-remediation logic
│   ├── known-issues.json         # Knowledge base of patterns & fixes
│   └── health-check.sh           # Runner health monitoring
├── systemd/
│   └── claude-ci-monitor.service # Systemd service definition
└── docs/
    └── ci-monitoring-architecture.md  # This file
```

### Systemd Service Configuration
Location: `/etc/systemd/system/claude-ci-monitor.service`

```ini
[Unit]
Description=Claude AI CI Monitor for ApraPipes
After=network.target actions.runner.Apra-Labs-ApraPipes.akhil-30ShattuckRoad01810.service
Wants=actions.runner.Apra-Labs-ApraPipes.akhil-30ShattuckRoad01810.service

[Service]
Type=simple
User=akhil
WorkingDirectory=/home/akhil/git/ApraPipes
Environment="GH_TOKEN=<YOUR_GITHUB_TOKEN_HERE>"
Environment="CLAUDE_API_KEY=<your-claude-api-key>"
ExecStart=/usr/local/bin/claude-code --mode=monitor --config=/home/akhil/git/ApraPipes/.claude/monitoring/config.json
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits (don't interfere with builds)
CPUQuota=10%
MemoryLimit=512M

[Install]
WantedBy=multi-user.target
```

## Notification Strategy

### Notification Levels

#### Level 1: Auto-Fixed (Log Only)
- Issue detected and automatically resolved
- Log to journal: `journalctl -u claude-ci-monitor -f`
- No human notification needed
- Example: Cleared stale vcpkg cache, retried build, succeeded

#### Level 2: FYI (Passive Notification)
- Issue fixed but worth knowing about
- Create GitHub Issue (label: `ci-auto-fixed`)
- Example: Disk space was 92%, cleaned old artifacts

#### Level 3: Action Needed (Active Notification)
- Could not auto-fix, needs human intervention
- Create GitHub Issue (label: `ci-needs-attention`)
- Optional: Slack/Email notification
- Example: New failure pattern not in knowledge base

#### Level 4: Critical (Urgent)
- Critical infrastructure issue
- Slack/Email/SMS notification
- Example: Runner offline and can't restart

## Monitoring Dashboard

### CLI Dashboard (Terminal UI)
Real-time dashboard showing:
```
╔════════════════════════════════════════════════════════════════╗
║         ApraPipes CI Monitor - Self-Hosted Runner              ║
╠════════════════════════════════════════════════════════════════╣
║ Status: Running                    Uptime: 3d 14h 23m          ║
║ Last Check: 2 seconds ago          Next Check: 58s             ║
╠════════════════════════════════════════════════════════════════╣
║                      WORKFLOW STATUS                           ║
╠════════════════════════════════════════════════════════════════╣
║ CI-Linux-CUDA        ✓ Passing    #20087128404  5m ago        ║
║ CI-Linux-NoCUDA      - Disabled                                ║
║ CI-Linux-ARM64       - Disabled                                ║
║ CI-Linux-CUDA-Docker ✓ Passing    #20087001234  2h ago        ║
║ CI-Windows-CUDA      ✗ Failed     #20086999888  4h ago        ║
║   └─> Auto-fix attempted: Cleared cache, retrying...          ║
╠════════════════════════════════════════════════════════════════╣
║                      RUNNER HEALTH                             ║
╠════════════════════════════════════════════════════════════════╣
║ Runner: linux-cuda (192.168.1.102)                            ║
║   Status: Online         Load: 12.4 (60% - building)          ║
║   Disk:   234GB / 500GB (47%)     ✓ OK                        ║
║   Memory: 12GB / 32GB (38%)       ✓ OK                        ║
║   Temp:   67°C                    ✓ OK                        ║
║   vcpkg cache: 4.2GB              ✓ OK                        ║
╠════════════════════════════════════════════════════════════════╣
║                    RECENT ACTIONS                              ║
╠════════════════════════════════════════════════════════════════╣
║ 04:16:19 - Detected: ffmpeg build failure (GCC cache issue)   ║
║ 04:16:22 - Auto-fix: Cleared vcpkg cache (4.1GB freed)        ║
║ 04:16:25 - Triggered: Retry build #20087128404                ║
║ 04:21:30 - Success: Build passed, ffmpeg compiled with GCC-12 ║
║ 04:21:31 - Updated: Knowledge base with successful pattern    ║
╠════════════════════════════════════════════════════════════════╣
║                      STATISTICS (24h)                          ║
╠════════════════════════════════════════════════════════════════╣
║ Total Builds: 12      Passed: 10 (83%)    Failed: 2 (17%)    ║
║ Auto-Fixed:   2       Escalated: 0                            ║
║ Avg Duration: 18m 34s (↓ 12% from last week)                  ║
╚════════════════════════════════════════════════════════════════╝

[Q]uit [R]efresh [L]ogs [H]ealth [F]ix-Now
```

### Web Dashboard (Optional Future Enhancement)
- Simple Express.js server on port 3000
- Real-time updates via WebSocket
- Accessible at http://192.168.1.102:3000
- Same information as CLI but prettier

## Security Considerations

### 1. GitHub Token Security
- Store in environment variable, not in code
- Use fine-grained token with minimal scopes needed
- Rotate token every 90 days

### 2. Webhook Security
- Validate webhook signature using GitHub secret
- Only accept webhooks from GitHub IPs
- Rate limiting to prevent abuse

### 3. File System Access
- Monitor script runs as `akhil` user (same as runner)
- Can only modify files in ~/actions-runner/_work/
- Cannot modify system files (no sudo)

### 4. Auto-Fix Safety
- Never auto-fix on `main` branch builds (only on feature branches)
- Never auto-commit changes without review
- Never delete data without backup
- Maximum 3 retry attempts before escalation

## Integration with Existing Skills

### aprapipes-devops Skill Extension
Add monitoring capabilities to the existing skill:

```javascript
// .claude/skills/aprapipes-devops/monitor-mode.js

async function monitorWorkflows() {
  while (true) {
    const workflows = await getAllWorkflows();

    for (const workflow of workflows) {
      const latestRun = await getLatestRun(workflow);

      if (latestRun.conclusion === 'failure') {
        const diagnosis = await diagnoseFailure(latestRun);
        const fix = await applyAutoFix(diagnosis);

        if (fix.success) {
          await logSuccess(workflow, diagnosis, fix);
        } else {
          await escalateToHuman(workflow, diagnosis);
        }
      }
    }

    await sleep(60000); // Check every minute
  }
}
```

## Deployment Steps

### Initial Setup (One-time)
1. Clone monitoring scripts to runner
2. Install dependencies (Node.js for webhook listener)
3. Configure systemd service
4. Set up GitHub webhook
5. Initialize knowledge base
6. Test with manual workflow trigger

### Maintenance
- Monitor logs: `journalctl -u claude-ci-monitor -f`
- Check status: `systemctl status claude-ci-monitor`
- Restart: `systemctl restart claude-ci-monitor`
- Update knowledge base: Edit `.claude/monitoring/known-issues.json`

## Success Metrics

### Week 1 Goals
- Monitor all workflows 24/7 with 99% uptime
- Auto-detect 100% of build failures within 1 minute
- Auto-fix at least 50% of failures (based on known patterns)

### Month 1 Goals
- Auto-fix rate increases to 80%+ as knowledge base grows
- Zero escalations for known issues
- Average time-to-fix < 5 minutes for auto-fixable issues
- Build success rate improves by 10%+ due to proactive maintenance

### Future Enhancements
- Machine learning to detect new failure patterns
- Predictive maintenance (fix issues before they cause failures)
- Multi-repository monitoring (extend to other projects)
- Integration with Slack for team notifications
- Performance optimization recommendations based on build metrics

## Cost/Benefit Analysis

### Costs
- ~512MB RAM (minimal impact on 32GB system)
- ~10% CPU (only during log analysis, idle otherwise)
- ~100MB disk for logs and knowledge base
- Initial setup time: ~4 hours

### Benefits
- Reduces human intervention by 80%+ for common issues
- Faster feedback loop (minutes vs hours)
- Builds knowledge base automatically
- Frees you to focus on actual development
- Prevents small issues from becoming big problems
- 24/7 monitoring even when you're asleep

**ROI**: If it saves just 30 minutes per day of debugging CI issues, that's 15 hours per month = significant time savings!

## Conclusion

This monitoring system transforms your self-hosted runner from a passive build executor into an intelligent, self-healing CI infrastructure. By leveraging Claude's ability to analyze logs, understand context, and apply fixes automatically, you get a system that learns and improves over time while requiring minimal human oversight.

The architecture is designed to be:
- **Reliable**: Multiple fallback mechanisms, graceful degradation
- **Safe**: Conservative auto-fix policies, human escalation for unknowns
- **Efficient**: Low resource overhead, event-driven architecture
- **Smart**: Learns from every build, improves over time
- **Transparent**: Clear logging, notifications, and dashboards

Ready to implement! 🚀

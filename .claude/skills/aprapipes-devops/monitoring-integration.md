# CI Monitoring Integration Guide

## Overview

The ApraPipes DevOps skill now includes integration with the 24/7 CI monitoring system running on the self-hosted builder. This document explains how to interact with the monitoring system and leverage its capabilities.

## Monitoring System Architecture

The monitoring system consists of:
- **Monitor Daemon**: Runs as systemd service on 192.168.1.102
- **Knowledge Base**: `.claude/monitoring/known-issues.json` - learned failure patterns
- **State Tracking**: `.claude/monitoring/monitor-state.json` - runtime statistics
- **Auto-Fix Engine**: Automatically applies fixes for known issues

See `.claude/docs/ci-monitoring-architecture.md` for full architecture details.

## When to Use Monitoring Data

### Use Cases

1. **Check if issue is already known**
   - Before diving into debugging, check if the monitor has seen this before
   - Look at success rates for known patterns

2. **Learn from auto-fix history**
   - See what fixes have worked in the past
   - Understand frequency of different issues

3. **Add new patterns to knowledge base**
   - After fixing a new issue, document it for future automation

4. **Verify monitoring health**
   - Check if monitor is running and detecting issues
   - See if auto-fixes are working

## Querying the Knowledge Base

### Check for Known Pattern

```bash
# SSH to the runner
ssh akhil@192.168.1.102

# Search for a specific error in knowledge base
jq '.patterns[] | select(.pattern.logMatch | test("ffmpeg.*failed"))' \
   /home/akhil/git/ApraPipes/.claude/monitoring/known-issues.json

# Get all auto-fixable patterns
jq '.patterns[] | select(.autoFix.enabled == true) | {id, name, successRate}' \
   /home/akhil/git/ApraPipes/.claude/monitoring/known-issues.json
```

### View Statistics

```bash
# Get overall stats
jq .statistics /home/akhil/git/ApraPipes/.claude/monitoring/known-issues.json

# Get pattern occurrence counts
jq '.patterns[] | {id, occurrences, lastSeen, successRate}' \
   /home/akhil/git/ApraPipes/.claude/monitoring/known-issues.json | jq -s 'sort_by(.occurrences) | reverse'
```

## Checking Monitor Status

### Is the Monitor Running?

```bash
# Check service status
ssh akhil@192.168.1.102 'systemctl is-active claude-ci-monitor'

# View recent activity
ssh akhil@192.168.1.102 'sudo journalctl -u claude-ci-monitor -n 50'
```

### View Monitor State

```bash
# Get current state
ssh akhil@192.168.1.102 'cat /home/akhil/git/ApraPipes/.claude/monitoring/monitor-state.json' | jq .

# Get key metrics
ssh akhil@192.168.1.102 'jq "{checksPerformed, issuesDetected, issuesAutoFixed, issuesEscalated}" \
  /home/akhil/git/ApraPipes/.claude/monitoring/monitor-state.json'
```

## Adding New Patterns

When you fix a new issue that isn't in the knowledge base, add it for future automation:

### Pattern Template

```json
{
  "id": "descriptive-id-kebab-case",
  "name": "Human readable name",
  "pattern": {
    "logMatch": "regex pattern to match in logs",
    "files": ["list", "of", "relevant", "files"]
  },
  "diagnosis": {
    "issue": "What went wrong",
    "rootCause": "Why it happened",
    "severity": "high|medium|low",
    "confidence": 95
  },
  "autoFix": {
    "enabled": true,
    "type": "workflow-edit|multi-step|cleanup|service-restart",
    "actions": [
      {
        "step": 1,
        "name": "Action name",
        "command": "command to execute",
        "description": "What this does"
      }
    ],
    "retryAfterFix": true,
    "maxRetries": 1
  },
  "successRate": null,
  "occurrences": 0,
  "lastSeen": null,
  "notes": "Additional context"
}
```

### Example: Adding a New Pattern

```bash
# SSH to runner
ssh akhil@192.168.1.102

# Edit knowledge base
cd /home/akhil/git/ApraPipes
nano .claude/monitoring/known-issues.json

# Add your new pattern to the "patterns" array
# Validate JSON syntax
jq . .claude/monitoring/known-issues.json

# Commit the update
git add .claude/monitoring/known-issues.json
git commit -m "ci-monitor: Add pattern for <issue-name>"
git push
```

The monitor will automatically reload the knowledge base on the next check cycle (within 60 seconds).

## Known Patterns Reference

### Current Patterns (as of 2025-12-10)

1. **vcpkg-cache-tilde-expansion**
   - Issue: `VCPKG_DEFAULT_BINARY_CACHE` uses `~` which vcpkg doesn't expand
   - Auto-fix: Replace with `${{ github.workspace }}/../`
   - Success rate: 100%

2. **vcpkg-cache-home-expansion**
   - Issue: `VCPKG_DEFAULT_BINARY_CACHE` uses `$HOME` which GitHub Actions doesn't expand
   - Auto-fix: Replace with `${{ github.workspace }}/../`
   - Success rate: 100%

3. **ffmpeg-gcc13-incompatibility**
   - Issue: ffmpeg 4.4.3 fails with GCC 13.3 (assembly error)
   - Auto-fix: Check GCC version, set GCC-12 as default, clear vcpkg cache, retry
   - Success rate: 100%
   - **Most common issue**: Occurred 3 times during initial debugging

4. **ninja-not-found-symptom**
   - Issue: CMake can't find Ninja (usually a symptom, not root cause)
   - Auto-fix: Search logs for actual vcpkg error, delegate to real pattern
   - Success rate: N/A (always delegates)

5. **disk-space-critical**
   - Issue: Disk usage > 90% on root partition
   - Auto-fix: Clean old builds, vcpkg downloads, Docker images
   - Success rate: Not yet encountered

6. **runner-offline**
   - Issue: Self-hosted runner not responding
   - Auto-fix: Restart runner service
   - Success rate: Not yet encountered

7. **test-flaky-failure**
   - Issue: Tests fail but pass on retry
   - Auto-fix: Retry once, track flaky test
   - Success rate: Not yet encountered

## Integration with Troubleshooting Workflow

### Enhanced Workflow with Monitoring

```
1. Build fails
   ↓
2. Check if monitor already detected it
   - ssh akhil@192.168.1.102 'sudo journalctl -u claude-ci-monitor -n 20'
   - Look for recent "Detected:" messages
   ↓
3. Check if auto-fix was attempted
   - Look for "Auto-fix:" messages in monitor logs
   - If yes: Check why it failed
   - If no: Pattern not known yet
   ↓
4. Query knowledge base for similar patterns
   - Search by error text
   - Check success rates
   ↓
5. Debug using normal troubleshooting guides
   - Use platform-specific guide
   - Apply fix manually
   ↓
6. Add pattern to knowledge base
   - Document for future automation
   - Test auto-fix logic
```

## Monitoring Best Practices

### For DevOps Agent

1. **Check monitor first**: Before diving into debugging, see if monitor already knows about this issue
2. **Trust high-confidence patterns**: If success rate > 90%, use that approach
3. **Document new patterns**: Every fix you make should become a pattern
4. **Validate auto-fixes**: Test your pattern on a feature branch before enabling auto-fix

### For Monitor Maintenance

1. **Review escalations weekly**: Check what issues couldn't be auto-fixed
2. **Update success rates**: Track which patterns work reliably
3. **Prune unused patterns**: Remove patterns that never occur
4. **Improve low-success patterns**: If success rate < 50%, pattern needs refinement

## Disabling Auto-Fix

If a pattern is causing issues, disable it temporarily:

```bash
# SSH to runner
ssh akhil@192.168.1.102

# Edit knowledge base
cd /home/akhil/git/ApraPipes
nano .claude/monitoring/known-issues.json

# Find the pattern and change:
"autoFix": {
  "enabled": false,  # Changed from true
  ...
}

# No need to restart monitor - it reloads automatically
```

## Monitoring API Reference

### Key Files

- **Knowledge Base**: `.claude/monitoring/known-issues.json`
  - Failure patterns and auto-fix logic
  - Updated by both monitor and DevOps agents

- **State File**: `.claude/monitoring/monitor-state.json`
  - Runtime statistics
  - Last check timestamps
  - Issue counters

- **Logs**: `.claude/monitoring/logs/monitor.log`
  - Detailed monitor activity
  - Timestamped actions and decisions

- **Failure Logs**: `.claude/monitoring/logs/failure-*.log`
  - Downloaded workflow logs for failed builds
  - Preserved for analysis

### Monitor Control

```bash
# Start monitoring
sudo systemctl start claude-ci-monitor

# Stop monitoring
sudo systemctl stop claude-ci-monitor

# Restart monitoring
sudo systemctl restart claude-ci-monitor

# View live logs
sudo journalctl -u claude-ci-monitor -f

# Check status
systemctl status claude-ci-monitor
```

## Escalation Protocol

The monitor escalates to humans when:
1. Unknown failure pattern (not in knowledge base)
2. Auto-fix attempted but failed
3. Critical infrastructure issue (runner offline, can't restart)
4. Maximum retry attempts exceeded

Escalations are handled via:
- GitHub Issue creation (label: `ci-needs-attention`)
- Optional Slack notification (if webhook configured)
- Logged to monitor.log

## Future Enhancements

Planned improvements to monitoring integration:

1. **Machine Learning**: Detect new patterns automatically
2. **Predictive Maintenance**: Fix issues before they cause failures
3. **Performance Optimization**: Suggest workflow improvements based on metrics
4. **Multi-Repo**: Extend monitoring to other repositories
5. **Web Dashboard**: Visual interface for monitoring status

## Support

If you encounter issues with monitoring:

1. Check monitor logs: `sudo journalctl -u claude-ci-monitor -f`
2. Verify monitor is running: `systemctl status claude-ci-monitor`
3. Validate knowledge base JSON: `jq . .claude/monitoring/known-issues.json`
4. Review monitor state: `cat .claude/monitoring/monitor-state.json`
5. Check GitHub token validity: `gh auth status`

For questions or to report monitoring bugs, create a GitHub issue with label `ci-monitor`.

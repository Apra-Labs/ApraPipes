# CI Monitor Setup Guide

## Quick Start (5 Minutes)

### Prerequisites
- Self-hosted runner at 192.168.1.102 is set up and running
- GitHub token with `repo` and `workflow` scopes
- `gh` CLI installed on the runner
- `jq` installed on the runner

### Installation Steps

1. **Copy files to the self-hosted runner:**

```bash
# From your local machine
cd /Users/akhil/git/ApraPipes2
scp -r .claude akhil@192.168.1.102:/home/akhil/git/ApraPipes/
```

2. **SSH to the runner and set up the service:**

```bash
ssh akhil@192.168.1.102
cd /home/akhil/git/ApraPipes
```

3. **Install dependencies (if not already installed):**

```bash
# GitHub CLI
type -p curl >/dev/null || sudo apt install curl -y
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh jq -y

# Authenticate gh CLI with the token
echo "<YOUR_GITHUB_TOKEN_HERE>" | gh auth login --with-token
```

4. **Install the systemd service:**

```bash
# Copy service file to systemd directory
sudo cp .claude/systemd/claude-ci-monitor.service /etc/systemd/system/

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable claude-ci-monitor

# Start the service
sudo systemctl start claude-ci-monitor
```

5. **Verify it's running:**

```bash
# Check service status
sudo systemctl status claude-ci-monitor

# Watch live logs
sudo journalctl -u claude-ci-monitor -f
```

## Usage

### View Live Monitoring

```bash
# Follow the logs in real-time
sudo journalctl -u claude-ci-monitor -f

# Show last 100 lines
sudo journalctl -u claude-ci-monitor -n 100

# Show logs since yesterday
sudo journalctl -u claude-ci-monitor --since yesterday
```

### Check Monitor State

```bash
# View current state
cat /home/akhil/git/ApraPipes/.claude/monitoring/monitor-state.json | jq .

# View statistics
jq '.checksPerformed, .issuesDetected, .issuesAutoFixed, .issuesEscalated' \
   /home/akhil/git/ApraPipes/.claude/monitoring/monitor-state.json
```

### Control the Service

```bash
# Stop monitoring
sudo systemctl stop claude-ci-monitor

# Start monitoring
sudo systemctl start claude-ci-monitor

# Restart monitoring
sudo systemctl restart claude-ci-monitor

# Disable (won't start on boot)
sudo systemctl disable claude-ci-monitor

# Re-enable
sudo systemctl enable claude-ci-monitor
```

## Manual Testing

Before setting up the systemd service, you can test the monitor manually:

```bash
cd /home/akhil/git/ApraPipes

# Set the GitHub token
export GH_TOKEN="<YOUR_GITHUB_TOKEN_HERE>"

# Run the monitor (will check every 60 seconds)
./.claude/monitoring/ci-monitor.sh

# Or with custom interval (check every 30 seconds)
MONITOR_INTERVAL=30 ./.claude/monitoring/ci-monitor.sh
```

Press Ctrl+C to stop.

## Troubleshooting

### Service won't start

```bash
# Check for errors
sudo systemctl status claude-ci-monitor
sudo journalctl -u claude-ci-monitor -n 50

# Common issues:
# 1. GitHub token not set or invalid
# 2. Script not executable: chmod +x .claude/monitoring/ci-monitor.sh
# 3. Dependencies missing: install gh, jq
```

### Monitor not detecting failures

```bash
# Verify GitHub token works
gh auth status

# Manually check a workflow
gh run list --workflow=CI-Linux-CUDA.yml --limit 1

# Check if knowledge base is valid JSON
jq . .claude/monitoring/known-issues.json
```

### Logs filling up disk

```bash
# Limit journal size
sudo journalctl --vacuum-size=100M

# Set permanent limit in /etc/systemd/journald.conf:
# SystemMaxUse=100M
```

## Customization

### Change Check Interval

Edit `/etc/systemd/system/claude-ci-monitor.service`:

```ini
Environment="MONITOR_INTERVAL=30"  # Check every 30 seconds instead of 60
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart claude-ci-monitor
```

### Add Slack Notifications

Edit the service file and add:

```ini
Environment="SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### Update Knowledge Base

Edit `.claude/monitoring/known-issues.json` to add new patterns or modify fixes.

The monitor will automatically reload the knowledge base on next check (no restart needed).

### Disable Auto-Fix for a Pattern

Edit `.claude/monitoring/known-issues.json` and set:

```json
{
  "id": "pattern-name",
  "autoFix": {
    "enabled": false,  // Changed from true
    ...
  }
}
```

## Monitoring the Monitor

### Health Checks

```bash
# Is the service running?
systemctl is-active claude-ci-monitor

# How long has it been running?
systemctl show claude-ci-monitor --property=ActiveEnterTimestamp

# How many times has it restarted?
systemctl show claude-ci-monitor --property=NRestarts

# CPU and memory usage
systemctl status claude-ci-monitor | grep -E '(CPU|Memory)'
```

### Performance Metrics

```bash
# View statistics
jq '{
  uptime: (.lastCheck // .startTime),
  checksPerformed,
  issuesDetected,
  issuesAutoFixed,
  issuesEscalated,
  successRate: ((.issuesAutoFixed / .issuesDetected * 100) // 0)
}' /home/akhil/git/ApraPipes/.claude/monitoring/monitor-state.json
```

## Uninstalling

```bash
# Stop and disable service
sudo systemctl stop claude-ci-monitor
sudo systemctl disable claude-ci-monitor

# Remove service file
sudo rm /etc/systemd/system/claude-ci-monitor.service

# Reload systemd
sudo systemctl daemon-reload

# Remove monitoring files (optional)
rm -rf /home/akhil/git/ApraPipes/.claude/monitoring
```

## Next Steps

Once the basic monitoring is working:

1. **Set up webhook listener** for real-time notifications (faster than polling)
2. **Add Slack integration** for critical alerts
3. **Extend knowledge base** as new failure patterns are discovered
4. **Monitor other repositories** by duplicating the service with different configs
5. **Build a web dashboard** for visualization (optional)

## Support

If you encounter issues or want to add new auto-fix patterns:

1. Check the logs: `sudo journalctl -u claude-ci-monitor -f`
2. Look at recent workflow runs: `gh run list --limit 5`
3. Create a GitHub issue with the `ci-monitor` label
4. The monitor itself will escalate unknown issues automatically

---

**Happy Monitoring!** 🚀

The system will learn and improve over time as it encounters and fixes more issues.

#!/bin/bash

################################################################################
# ApraPipes CI Monitor - 24/7 Workflow Monitoring & Auto-Remediation
#
# This script continuously monitors GitHub Actions workflows and automatically
# fixes common issues without human intervention.
#
# Usage:
#   ./ci-monitor.sh [--mode=polling|webhook|hybrid] [--interval=60]
#
# Environment Variables Required:
#   GH_TOKEN - GitHub personal access token with repo and workflow scopes
#
# Optional:
#   SLACK_WEBHOOK - Slack webhook URL for notifications
#   MONITOR_INTERVAL - Seconds between health checks (default: 60)
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
KNOWLEDGE_BASE="$SCRIPT_DIR/known-issues.json"
STATE_FILE="$SCRIPT_DIR/monitor-state.json"
LOG_DIR="$SCRIPT_DIR/logs"
MONITOR_MODE="${1:---mode=hybrid}"
INTERVAL="${MONITOR_INTERVAL:-60}"

# GitHub Configuration
REPO_OWNER="Apra-Labs"
REPO_NAME="ApraPipes"
RUNNER_NAME="akhil/30ShattuckRoad01810"
RUNNER_IP="192.168.1.102"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure log directory exists
mkdir -p "$LOG_DIR"

################################################################################
# Logging Functions
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        INFO)  echo -e "${BLUE}[$timestamp] [INFO]${NC} $message" ;;
        WARN)  echo -e "${YELLOW}[$timestamp] [WARN]${NC} $message" ;;
        ERROR) echo -e "${RED}[$timestamp] [ERROR]${NC} $message" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] [SUCCESS]${NC} $message" ;;
        *) echo "[$timestamp] [$level] $message" ;;
    esac

    echo "[$timestamp] [$level] $message" >> "$LOG_DIR/monitor.log"
}

################################################################################
# State Management
################################################################################

init_state() {
    if [[ ! -f "$STATE_FILE" ]]; then
        cat > "$STATE_FILE" <<EOF
{
  "startTime": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "lastCheck": null,
  "checksPerformed": 0,
  "issuesDetected": 0,
  "issuesAutoFixed": 0,
  "issuesEscalated": 0,
  "knownWorkflows": [
    "CI-Linux-CUDA",
    "CI-Linux-NoCUDA",
    "CI-Linux-ARM64",
    "CI-Linux-CUDA-Docker",
    "CI-Windows-CUDA",
    "CI-Windows-NoCUDA"
  ],
  "lastWorkflowStatus": {}
}
EOF
        log INFO "Initialized monitor state"
    fi
}

update_state() {
    local key="$1"
    local value="$2"

    jq --arg k "$key" --arg v "$value" '.[$k] = $v' "$STATE_FILE" > "$STATE_FILE.tmp"
    mv "$STATE_FILE.tmp" "$STATE_FILE"
}

increment_counter() {
    local counter="$1"
    jq --arg k "$counter" '.[$k] = (.[$k] // 0) + 1' "$STATE_FILE" > "$STATE_FILE.tmp"
    mv "$STATE_FILE.tmp" "$STATE_FILE"
}

################################################################################
# Workflow Monitoring
################################################################################

get_latest_run() {
    local workflow="$1"
    gh api "/repos/$REPO_OWNER/$REPO_NAME/actions/workflows/$workflow.yml/runs" \
        --jq '.workflow_runs[0] | {id: .id, status: .status, conclusion: .conclusion, created_at: .created_at, head_branch: .head_branch}'
}

check_all_workflows() {
    log INFO "Checking all workflows..."

    local workflows=(
        "CI-Linux-CUDA"
        "CI-Linux-NoCUDA"
        "CI-Linux-ARM64"
        "CI-Linux-CUDA-Docker"
        "CI-Windows-CUDA"
        "CI-Windows-NoCUDA"
    )

    for workflow in "${workflows[@]}"; do
        local run_info=$(get_latest_run "$workflow" 2>/dev/null || echo '{}')

        if [[ "$run_info" == "{}" ]]; then
            log WARN "Could not fetch status for $workflow"
            continue
        fi

        local status=$(echo "$run_info" | jq -r '.status')
        local conclusion=$(echo "$run_info" | jq -r '.conclusion')
        local run_id=$(echo "$run_info" | jq -r '.id')
        local branch=$(echo "$run_info" | jq -r '.head_branch')

        if [[ "$status" == "completed" && "$conclusion" == "failure" ]]; then
            log ERROR "Workflow $workflow failed (run: $run_id, branch: $branch)"
            increment_counter "issuesDetected"

            # Attempt auto-fix
            diagnose_and_fix "$workflow" "$run_id" "$branch"
        elif [[ "$status" == "completed" && "$conclusion" == "success" ]]; then
            log SUCCESS "Workflow $workflow passed (run: $run_id)"
        elif [[ "$status" == "in_progress" ]]; then
            log INFO "Workflow $workflow in progress (run: $run_id)"
        fi
    done

    increment_counter "checksPerformed"
    update_state "lastCheck" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

################################################################################
# Diagnosis & Auto-Fix Engine
################################################################################

diagnose_and_fix() {
    local workflow="$1"
    local run_id="$2"
    local branch="$3"

    log INFO "Diagnosing failure for $workflow (run: $run_id)..."

    # Download logs
    local log_file="$LOG_DIR/failure-${run_id}.log"
    gh run view "$run_id" --log > "$log_file" 2>&1

    log INFO "Logs saved to $log_file"

    # Check against known patterns
    local patterns=$(jq -r '.patterns[] | @base64' "$KNOWLEDGE_BASE")

    for pattern_b64 in $patterns; do
        local pattern=$(echo "$pattern_b64" | base64 --decode)
        local pattern_id=$(echo "$pattern" | jq -r '.id')
        local log_match=$(echo "$pattern" | jq -r '.pattern.logMatch')

        # Check if pattern matches
        if grep -qP "$log_match" "$log_file"; then
            log WARN "Matched known pattern: $pattern_id"

            local auto_fix_enabled=$(echo "$pattern" | jq -r '.autoFix.enabled')

            if [[ "$auto_fix_enabled" == "true" ]]; then
                log INFO "Auto-fix enabled for $pattern_id, attempting fix..."
                apply_fix "$pattern" "$branch" "$log_file"
                return 0
            else
                log WARN "Auto-fix disabled for $pattern_id, escalating..."
                escalate_to_human "$workflow" "$run_id" "$pattern_id" "$log_file"
                return 1
            fi
        fi
    done

    # Unknown pattern
    log ERROR "Unknown failure pattern, escalating to human..."
    escalate_to_human "$workflow" "$run_id" "unknown" "$log_file"
}

apply_fix() {
    local pattern="$1"
    local branch="$2"
    local log_file="$3"

    local pattern_id=$(echo "$pattern" | jq -r '.id')
    local fix_type=$(echo "$pattern" | jq -r '.autoFix.type')

    case "$fix_type" in
        "workflow-edit")
            apply_workflow_edit_fix "$pattern" "$branch"
            ;;
        "multi-step")
            apply_multi_step_fix "$pattern" "$branch" "$log_file"
            ;;
        "cleanup")
            apply_cleanup_fix "$pattern"
            ;;
        "service-restart")
            apply_service_restart_fix "$pattern"
            ;;
        *)
            log ERROR "Unknown fix type: $fix_type"
            return 1
            ;;
    esac

    local retry=$(echo "$pattern" | jq -r '.autoFix.retryAfterFix')
    if [[ "$retry" == "true" ]]; then
        log INFO "Triggering retry build on branch $branch..."
        gh workflow run CI-Linux-CUDA.yml --ref "$branch"
        log SUCCESS "Retry triggered"
    fi

    increment_counter "issuesAutoFixed"
}

apply_workflow_edit_fix() {
    local pattern="$1"
    local branch="$2"

    local file=$(echo "$pattern" | jq -r '.autoFix.actions[0].file')
    local search=$(echo "$pattern" | jq -r '.autoFix.actions[0].search')
    local replace=$(echo "$pattern" | jq -r '.autoFix.actions[0].replace')
    local commit_msg=$(echo "$pattern" | jq -r '.autoFix.actions[0].commitMessage')

    log INFO "Editing $file..."
    log INFO "  Search: $search"
    log INFO "  Replace: $replace"

    cd "$REPO_DIR"

    # Make the edit
    sed -i "s|$search|$replace|g" "$file"

    # Commit and push
    git add "$file"
    git commit -m "$commit_msg"
    git push origin "$branch"

    log SUCCESS "Fix committed and pushed to $branch"
}

apply_multi_step_fix() {
    local pattern="$1"
    local branch="$2"
    local log_file="$3"

    local pattern_id=$(echo "$pattern" | jq -r '.id')

    # Special handling for ffmpeg-gcc13-incompatibility
    if [[ "$pattern_id" == "ffmpeg-gcc13-incompatibility" ]]; then
        log INFO "Applying ffmpeg/GCC fix..."

        # Step 1: Check GCC version
        log INFO "Step 1: Checking GCC version on runner..."
        local gcc_version=$(ssh akhil@$RUNNER_IP 'gcc --version' | head -1)
        log INFO "  Current GCC: $gcc_version"

        if echo "$gcc_version" | grep -q "13\.3"; then
            log WARN "  GCC 13.3 detected, need to install GCC-12"

            # Step 2: Install and set GCC-12
            log INFO "Step 2: Installing GCC-12 and setting as default..."
            ssh akhil@$RUNNER_IP 'sudo /home/akhil/git/ApraPipes/build_scripts/setup-linux-cuda-runner.sh'
            log SUCCESS "  GCC-12 installed and set as default"
        else
            log INFO "  GCC-12 already default, skipping installation"
        fi

        # Step 3: Clear vcpkg cache
        log INFO "Step 3: Clearing vcpkg binary cache..."
        local cache_size=$(ssh akhil@$RUNNER_IP 'du -sh /home/akhil/actions-runner/_work/ApraPipes/.cache/vcpkg 2>/dev/null | cut -f1' || echo "0")
        log INFO "  Current cache size: $cache_size"

        ssh akhil@$RUNNER_IP 'rm -rf /home/akhil/actions-runner/_work/ApraPipes/.cache/vcpkg/*'
        log SUCCESS "  Cache cleared (freed: $cache_size)"

        # Step 4: Trigger rebuild (handled by retry flag)
        log INFO "Step 4: Will trigger rebuild after this function returns"
    fi
}

apply_cleanup_fix() {
    local pattern="$1"

    log INFO "Applying cleanup actions..."

    local actions=$(echo "$pattern" | jq -c '.autoFix.actions[]')

    while IFS= read -r action; do
        local name=$(echo "$action" | jq -r '.name')
        local command=$(echo "$action" | jq -r '.command')
        local description=$(echo "$action" | jq -r '.description')

        log INFO "$name..."
        log INFO "  $description"

        eval "$command"

        log SUCCESS "  Done"
    done <<< "$actions"
}

apply_service_restart_fix() {
    local pattern="$1"

    log INFO "Restarting runner service..."

    # Check if machine is reachable
    if ! ping -c 3 "$RUNNER_IP" > /dev/null 2>&1; then
        log ERROR "Runner machine ($RUNNER_IP) is not reachable"
        escalate_to_human "runner-health" "N/A" "runner-unreachable" "Machine not responding to ping"
        return 1
    fi

    # Restart service
    ssh akhil@$RUNNER_IP 'sudo systemctl restart actions.runner.Apra-Labs-ApraPipes.akhil-30ShattuckRoad01810.service'

    log INFO "Waiting 30 seconds for runner to come online..."
    sleep 30

    # Verify runner is online
    local runner_status=$(gh api "/repos/$REPO_OWNER/$REPO_NAME/actions/runners" | jq -r ".runners[] | select(.name==\"$RUNNER_NAME\") | .status")

    if [[ "$runner_status" == "online" ]]; then
        log SUCCESS "Runner is back online"
    else
        log ERROR "Runner failed to come online, status: $runner_status"
        escalate_to_human "runner-health" "N/A" "runner-restart-failed" "Runner still offline after restart"
        return 1
    fi
}

################################################################################
# Escalation to Human
################################################################################

escalate_to_human() {
    local workflow="$1"
    local run_id="$2"
    local pattern_id="$3"
    local details="$4"

    log WARN "Escalating to human: workflow=$workflow, pattern=$pattern_id"

    increment_counter "issuesEscalated"

    # Create GitHub issue
    local title="[CI Monitor] Auto-fix failed: $workflow ($pattern_id)"
    local body="## CI Monitor Alert

**Workflow:** $workflow
**Run ID:** $run_id
**Pattern:** $pattern_id
**Time:** $(date -u +%Y-%m-%dT%H:%M:%SZ)

### Details
\`\`\`
$details
\`\`\`

### Logs
Run: https://github.com/$REPO_OWNER/$REPO_NAME/actions/runs/$run_id

### Action Required
Manual investigation needed. Auto-fix was unable to resolve this issue.

---
*This issue was created automatically by the CI monitoring system.*
"

    gh issue create \
        --title "$title" \
        --body "$body" \
        --label "ci-needs-attention" \
        --repo "$REPO_OWNER/$REPO_NAME"

    log SUCCESS "GitHub issue created for escalation"

    # Send Slack notification if webhook is configured
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        send_slack_notification "$title" "$body"
    fi
}

send_slack_notification() {
    local title="$1"
    local body="$2"

    curl -X POST "$SLACK_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"$title\",\"blocks\":[{\"type\":\"section\",\"text\":{\"type\":\"mrkdwn\",\"text\":\"$body\"}}]}"
}

################################################################################
# Health Monitoring
################################################################################

check_runner_health() {
    log INFO "Checking runner health..."

    # Check runner online status
    local runner_status=$(gh api "/repos/$REPO_OWNER/$REPO_NAME/actions/runners" | jq -r ".runners[] | select(.name==\"$RUNNER_NAME\") | .status")

    if [[ "$runner_status" != "online" ]]; then
        log ERROR "Runner is $runner_status"
        diagnose_and_fix "runner-health" "N/A" "main"
        return
    fi

    log SUCCESS "Runner status: $runner_status"

    # Check disk space
    local disk_usage=$(ssh akhil@$RUNNER_IP 'df / | tail -1 | awk "{print \$5}" | sed "s/%//"')

    if [[ $disk_usage -gt 90 ]]; then
        log ERROR "Disk usage critical: ${disk_usage}%"
        # Trigger cleanup
        local cleanup_pattern=$(jq -r '.patterns[] | select(.id=="disk-space-critical") | @json' "$KNOWLEDGE_BASE")
        apply_cleanup_fix "$cleanup_pattern"
    elif [[ $disk_usage -gt 80 ]]; then
        log WARN "Disk usage high: ${disk_usage}%"
    else
        log SUCCESS "Disk usage OK: ${disk_usage}%"
    fi

    # Check memory
    local mem_usage=$(ssh akhil@$RUNNER_IP 'free | grep Mem | awk "{print (\$3/\$2) * 100.0}"' | cut -d. -f1)
    log INFO "Memory usage: ${mem_usage}%"

    # Check vcpkg cache size
    local cache_size=$(ssh akhil@$RUNNER_IP 'du -sh /home/akhil/actions-runner/_work/ApraPipes/.cache/vcpkg 2>/dev/null | cut -f1' || echo "0")
    log INFO "vcpkg cache size: $cache_size"
}

################################################################################
# Main Monitor Loop
################################################################################

monitor_loop() {
    log INFO "Starting CI monitor in $MONITOR_MODE mode..."
    log INFO "Interval: ${INTERVAL}s"

    while true; do
        log INFO "===== Monitor Check $(increment_counter checksPerformed) ====="

        check_all_workflows
        check_runner_health

        log INFO "Sleeping for ${INTERVAL}s..."
        sleep "$INTERVAL"
    done
}

################################################################################
# Startup
################################################################################

main() {
    log INFO "ApraPipes CI Monitor starting..."
    log INFO "Repository: $REPO_OWNER/$REPO_NAME"
    log INFO "Knowledge Base: $KNOWLEDGE_BASE"

    # Validate dependencies
    command -v gh >/dev/null 2>&1 || { log ERROR "gh CLI not found"; exit 1; }
    command -v jq >/dev/null 2>&1 || { log ERROR "jq not found"; exit 1; }

    # Validate GitHub token
    if [[ -z "${GH_TOKEN:-}" ]]; then
        log ERROR "GH_TOKEN environment variable not set"
        exit 1
    fi

    # Initialize state
    init_state

    # Start monitoring
    monitor_loop
}

# Trap signals for graceful shutdown
trap 'log INFO "Shutting down CI monitor..."; exit 0' SIGTERM SIGINT

main "$@"

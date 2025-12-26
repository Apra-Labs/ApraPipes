#!/bin/bash
# GitHub Actions Workflow Watcher
# Polls workflow status every 60 seconds with minimal, parseable output
# Usage: gh-watch.sh <run_id>

set -euo pipefail

RUN_ID="${1:-}"

# Validate input
if [[ -z "$RUN_ID" ]]; then
    echo "Error: Run ID required"
    echo "Usage: $0 <run_id>"
    exit 1
fi

# Start time for elapsed calculation
START_TIME=$(date +%s)

# Function to get elapsed time
get_elapsed() {
    local current=$(date +%s)
    local elapsed=$((current - START_TIME))
    local minutes=$((elapsed / 60))
    echo "${minutes}m"
}

# Function to get current timestamp
get_timestamp() {
    date +"%H:%M"
}

# Poll loop - check every 60 seconds
while true; do
    # Get run status and job info
    RUN_DATA=$(gh run view "$RUN_ID" --json status,conclusion,createdAt 2>&1 || echo "ERROR")

    # Handle invalid run ID or other errors
    if [[ "$RUN_DATA" == "ERROR" ]] || [[ "$RUN_DATA" == *"could not resolve"* ]]; then
        echo "[$(get_timestamp)] Run $RUN_ID | status=error | Invalid run ID or API error"
        exit 1
    fi

    # Parse status and conclusion
    STATUS=$(echo "$RUN_DATA" | jq -r '.status')
    CONCLUSION=$(echo "$RUN_DATA" | jq -r '.conclusion // "null"')

    # Get job information for step tracking
    JOBS_DATA=$(gh run view "$RUN_ID" --json jobs 2>/dev/null || echo '{"jobs":[]}')

    # Find current step from first in-progress job
    CURRENT_STEP="unknown"
    STEP_PROGRESS=""

    # Try to extract current step info
    if [[ "$STATUS" == "in_progress" ]]; then
        # Get the first in-progress job's current step
        STEP_INFO=$(echo "$JOBS_DATA" | jq -r '
            .jobs[] |
            select(.status == "in_progress") |
            .steps |
            map(select(.status == "in_progress" or .status == "completed")) |
            "step=\(map(select(.status == "in_progress"))[0].name // "Starting") | progress=\((map(select(.status == "completed")) | length) + 1)/\(length)"
        ' 2>/dev/null | head -1)

        if [[ -n "$STEP_INFO" ]]; then
            CURRENT_STEP=$(echo "$STEP_INFO" | sed -n 's/.*step=\([^|]*\).*/\1/p' | xargs)
            STEP_PROGRESS=$(echo "$STEP_INFO" | sed -n 's/.*progress=\([^|]*\).*/\1/p' | xargs)
            STEP_DISPLAY="$CURRENT_STEP ($STEP_PROGRESS)"
        else
            STEP_DISPLAY="Starting"
        fi
    fi

    # Output single informative line
    TIMESTAMP=$(get_timestamp)
    ELAPSED=$(get_elapsed)

    if [[ "$STATUS" == "completed" ]]; then
        # Final status output
        echo "[${TIMESTAMP}] Run $RUN_ID | status=completed | conclusion=${CONCLUSION} | elapsed=${ELAPSED}"

        if [[ "$CONCLUSION" == "success" ]]; then
            echo "✓ PASSED"
            exit 0
        elif [[ "$CONCLUSION" == "cancelled" ]]; then
            echo "⊘ CANCELLED"
            exit 2
        else
            echo "✗ FAILED"
            echo ""
            echo "Failed job logs (last 200 lines):"
            gh run view "$RUN_ID" --log-failed | head -200
            exit 1
        fi
    elif [[ "$STATUS" == "in_progress" ]]; then
        echo "[${TIMESTAMP}] Run $RUN_ID | status=in_progress | step=${STEP_DISPLAY} | elapsed=${ELAPSED}"
    elif [[ "$STATUS" == "queued" ]] || [[ "$STATUS" == "waiting" ]] || [[ "$STATUS" == "pending" ]]; then
        echo "[${TIMESTAMP}] Run $RUN_ID | status=queued | elapsed=${ELAPSED}"
    else
        echo "[${TIMESTAMP}] Run $RUN_ID | status=${STATUS} | elapsed=${ELAPSED}"
    fi

    # Wait 60 seconds before next poll
    sleep 60
done

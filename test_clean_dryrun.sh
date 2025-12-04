#!/bin/bash

# Clean dry run for workflow run 19912556305
# Shows EXACTLY what would be sent to LLM with new GitHub context code

echo "================================================================================"
echo "CLEAN DRY RUN: Simulating Workflow Run 19912556305"
echo "================================================================================"
echo ""
echo "Job: https://github.com/Apra-Labs/ApraPipes/actions/runs/19912556305/job/57084352851"
echo "Failure: Win-nocuda prep step with intentional 'exit 1'"
echo ""
echo "This dry run will show:"
echo "  1. GitHub annotations fetched from job logs"
echo "  2. Workflow files (CI-Win-NoCUDA.yml, build-test-win.yml)"
echo "  3. Platform: Win-nocuda"
echo "  4. Recent git history"
echo "  5. COMPLETE prompt that would be sent to LLM"
echo ""
echo "================================================================================"
echo ""

# Set environment variables exactly as they would be in CI
export ANTHROPIC_API_KEY="mock-key-for-testing"
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"  # Use real token if available
export BUILD_FLAVOR="Win-nocuda"
export GITHUB_REPOSITORY="Apra-Labs/ApraPipes"
export GITHUB_RUN_ID="19912556305"
export GITHUB_WORKFLOW="CI-Win-NoCUDA"

# Create a mock failure log matching the actual error
mkdir -p build-logs
cat > build-logs/mock-failure.log << 'EOF'
Run python --version ; cmake --version ; ninja --version ; git --version; pwsh --version ; exit 1
Python 3.12.8
cmake version 3.31.2

CMake suite maintained and supported by Kitware (kitware.com/cmake).
ninja version 1.12.1
git version 2.47.1.windows.2
PowerShell 7.4.6

##[error]Process completed with exit code 1.
EOF

echo "Created mock failure log"
echo ""
echo "================================================================================"
echo "RUNNING AGENT IN MOCK MODE"
echo "================================================================================"
echo ""

# Capture all output
OUTPUT_FILE="/tmp/autonomous-agent-dryrun-output.log"

python3 .github/autonomous-agent/agent/autonomous_agent.py \
  --branch "feature/autonomous-devops-agent" \
  --build-status "failure" \
  --failure-log "build-logs/mock-failure.log" \
  --build-flavor "Win-nocuda" \
  --run-id "19912556305" \
  --workflow-name "CI-Win-NoCUDA" \
  --mock-llm \
  --output "agent-result-clean.json" 2>&1 | tee "$OUTPUT_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
echo "DRY RUN COMPLETE"
echo "================================================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Agent executed successfully"
    echo ""
    echo "Output saved to: $OUTPUT_FILE"
    echo "Result saved to: agent-result-clean.json"
    echo ""

    # Extract and display key information
    echo "================================================================================"
    echo "KEY INFORMATION EXTRACTED"
    echo "================================================================================"
    echo ""

    echo "1. Platform Detection:"
    grep -i "platform" "$OUTPUT_FILE" | head -5
    echo ""

    echo "2. GitHub Context Fetched:"
    grep -i "github context\|workflow files\|job logs" "$OUTPUT_FILE" | head -10
    echo ""

    echo "3. Investigation Turns:"
    grep -i "investigation turn\|action=" "$OUTPUT_FILE" | head -10
    echo ""

    # Check if prompt was logged
    if grep -q "PROMPT TO LLM" "$OUTPUT_FILE"; then
        echo "4. LLM Prompt Preview:"
        echo "   (Searching for 'PROMPT TO LLM' section...)"
        grep -A 20 "PROMPT TO LLM" "$OUTPUT_FILE" | head -25
    else
        echo "4. LLM Prompt:"
        echo "   Note: In mock mode, full prompt may not be logged."
        echo "   Add logging to llm_client.py if needed to see complete prompt."
    fi

    echo ""
    echo "================================================================================"
    echo "REVIEW INSTRUCTIONS"
    echo "================================================================================"
    echo ""
    echo "To see the COMPLETE output:"
    echo "  cat $OUTPUT_FILE"
    echo ""
    echo "To see what would be sent to LLM:"
    echo "  grep -A 500 'PROMPT TO LLM' $OUTPUT_FILE"
    echo ""
    echo "To search for specific sections:"
    echo "  grep -i 'github annotations' $OUTPUT_FILE"
    echo "  grep -i 'workflow files' $OUTPUT_FILE"
    echo "  grep -i 'platform' $OUTPUT_FILE"
    echo ""
else
    echo "❌ Agent failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the output above for errors"
    echo "Full output saved to: $OUTPUT_FILE"
fi

echo ""
echo "================================================================================"

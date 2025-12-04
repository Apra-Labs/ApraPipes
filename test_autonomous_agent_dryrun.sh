#!/bin/bash

# Dry run test for autonomous agent with GitHub context
# Uses mock mode to show what would be sent to LLM without making API calls

echo "==================================================================="
echo "AUTONOMOUS AGENT DRY RUN TEST"
echo "==================================================================="
echo ""
echo "This test simulates what would be sent to the LLM for the failed"
echo "build 19912556305 (Win-nocuda prep step intentional failure)."
echo ""
echo "Mode: MOCK (no actual API calls)"
echo "==================================================================="
echo ""

# Set environment variables
export ANTHROPIC_API_KEY="mock-key-for-testing"
export GITHUB_TOKEN="${GITHUB_TOKEN}"  # Use real token for GitHub API
export BUILD_FLAVOR="Win-nocuda"
export GITHUB_REPOSITORY="Apra-Labs/ApraPipes"
export GITHUB_RUN_ID="19912556305"
export GITHUB_WORKFLOW="CI-Win-NoCUDA"

# Create a mock failure log (since we don't have the actual artifact)
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

echo "Created mock failure log at build-logs/mock-failure.log"
echo ""

# Run the agent in mock mode
echo "Running autonomous agent with --mock-llm flag..."
echo ""

python3 .github/autonomous-agent/agent/autonomous_agent.py \
  --branch "ak/autonomous-devops-agent" \
  --build-status "failure" \
  --failure-log "build-logs/mock-failure.log" \
  --build-flavor "Win-nocuda" \
  --run-id "19912556305" \
  --workflow-name "CI-Win-NoCUDA" \
  --mock-llm \
  --output "agent-result-dryrun.json"

echo ""
echo "==================================================================="
echo "DRY RUN COMPLETE"
echo "==================================================================="
echo ""
echo "Check the logs above to see what context was fetched:"
echo "  - GitHub job logs with error annotations"
echo "  - Workflow files (CI-Win-NoCUDA.yml, build-test-win.yml)"
echo "  - Platform: Win-nocuda"
echo "  - Recent git history"
echo ""
echo "The prompt that would be sent to LLM is shown in the logs."
echo ""
echo "Result saved to: agent-result-dryrun.json"
echo ""

#!/bin/bash
# Test gh CLI installation and cache deletion logic in docker

set -e

echo "=== Testing gh CLI installation and cache deletion logic ==="
docker run --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  nvidia/cuda:11.8.0-devel-ubuntu22.04 \
  bash -c '
    echo "=== Installing gh CLI (same way as prep-cmd will) ==="
    apt-get update -qq
    apt-get install -y curl

    # Install gh CLI (Ubuntu 22.04)
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update -qq
    apt-get install -y gh

    echo ""
    echo "=== Verifying gh CLI installation ==="
    gh --version

    echo ""
    echo "=== Testing cache deletion logic (dry run) ==="
    CURRENT_BRANCH="fix/ci-additional-workflows"
    FLAV="Linux-Cuda"

    echo "Current branch: $CURRENT_BRANCH"
    echo "Flavor: $FLAV"

    # Test the grep pattern
    echo ""
    echo "Testing grep pattern for cache key matching:"
    echo "Linux-Cuda-10-abc123" | grep "${FLAV}-10-" && echo "  ✓ Match found" || echo "  ✗ No match"
    echo "Linux-NoCuda-10-abc123" | grep "${FLAV}-10-" && echo "  ✓ Match found" || echo "  ✗ No match (expected)"
    echo "Windows-10-abc123" | grep "${FLAV}-10-" && echo "  ✓ Match found" || echo "  ✗ No match (expected)"

    echo ""
    echo "=== Testing awk field extraction ==="
    echo "Sample cache list output:"
    echo "1234567890	Linux-Cuda-10-abc123	refs/heads/fix/branch	1000000000	2024-12-04T00:00:00Z"
    echo ""
    echo "Extracting cache ID with awk:"
    echo "1234567890	Linux-Cuda-10-abc123	refs/heads/fix/branch	1000000000	2024-12-04T00:00:00Z" | awk "{print \$1}"

    echo ""
    echo "=== SUCCESS: gh CLI installation and logic verified ==="
  '

echo ""
echo "=== Test completed successfully ==="

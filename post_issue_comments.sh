#!/bin/bash
# Script to post issue comments using gh CLI
# Usage: ./post_issue_comments.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "  ApraPipes Issue Comment Posting Script"
echo "============================================"
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: gh CLI is not installed${NC}"
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}Not authenticated with GitHub${NC}"
    echo "Please run: gh auth login"
    exit 1
fi

# Repository
REPO="Apra-Labs/ApraPipes"

# Directory containing comment files
COMMENT_DIR="./issue_comments"

if [ ! -d "$COMMENT_DIR" ]; then
    echo -e "${RED}Error: Comment directory not found: $COMMENT_DIR${NC}"
    exit 1
fi

# Array of issues with detailed comments
ISSUES_WITH_COMMENTS=(
    496 441 440 438 421 419 418 410 408 402 401 400 396 394
)

# Function to post comment to an issue
post_comment() {
    local issue_num=$1
    local comment_file="${COMMENT_DIR}/${issue_num}.md"
    
    if [ ! -f "$comment_file" ]; then
        echo -e "${YELLOW}  ⚠ Comment file not found for issue #${issue_num}${NC}"
        return 1
    fi
    
    echo -n "  Posting comment to issue #${issue_num}... "
    
    if gh issue comment "$issue_num" --repo "$REPO" --body-file "$comment_file" 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Main execution
echo "Found $(ls -1 $COMMENT_DIR/*.md 2>/dev/null | wc -l) comment files"
echo ""

# Statistics
total=0
success=0
failed=0
skipped=0

# Option to post all or specific issues
if [ "$1" == "--all" ]; then
    # Post all comment files found
    echo "Posting comments to all issues..."
    echo ""
    
    for comment_file in "$COMMENT_DIR"/*.md; do
        if [ -f "$comment_file" ]; then
            issue_num=$(basename "$comment_file" .md)
            total=$((total + 1))
            
            if post_comment "$issue_num"; then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi
            
            # Rate limiting: wait 2 seconds between posts
            sleep 2
        fi
    done
    
elif [ "$1" == "--dry-run" ]; then
    # Dry run: just show what would be posted
    echo "DRY RUN: No comments will be posted"
    echo ""
    
    for issue_num in "${ISSUES_WITH_COMMENTS[@]}"; do
        comment_file="${COMMENT_DIR}/${issue_num}.md"
        if [ -f "$comment_file" ]; then
            echo "  Would post to issue #${issue_num}"
            total=$((total + 1))
        fi
    done
    
elif [ -n "$1" ]; then
    # Post to specific issue
    issue_num=$1
    total=1
    
    if post_comment "$issue_num"; then
        success=1
    else
        failed=1
    fi
    
else
    # Default: post to issues with detailed comments
    echo "Posting comments to ${#ISSUES_WITH_COMMENTS[@]} issues with detailed analysis..."
    echo ""
    echo "Issues: ${ISSUES_WITH_COMMENTS[*]}"
    echo ""
    
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
    
    for issue_num in "${ISSUES_WITH_COMMENTS[@]}"; do
        total=$((total + 1))
        
        if post_comment "$issue_num"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
        
        # Rate limiting: wait 2 seconds between posts
        sleep 2
    done
fi

# Summary
echo ""
echo "============================================"
echo "  Summary"
echo "============================================"
echo "  Total:     $total"
echo -e "  Success:   ${GREEN}$success${NC}"
if [ $failed -gt 0 ]; then
    echo -e "  Failed:    ${RED}$failed${NC}"
fi
if [ $skipped -gt 0 ]; then
    echo -e "  Skipped:   ${YELLOW}$skipped${NC}"
fi
echo "============================================"

if [ $failed -eq 0 ] && [ $total -gt 0 ]; then
    echo -e "${GREEN}All comments posted successfully!${NC}"
    exit 0
else
    exit 1
fi

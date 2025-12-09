# Autonomous DevOps Agent Design for ApraPipes CI/CD

**Purpose**: Design a permanent, autonomous agent that continuously monitors GitHub Actions workflows and proactively fixes build issues to reduce burden on software engineers.

**Status**: Proposal/Design Document
**Created**: 2025-12-01
**Applies To**: All ApraPipes CI/CD workflows

---

## Executive Summary

The autonomous DevOps agent is a continuous background process that:
1. **Monitors** all GitHub Actions workflows in real-time
2. **Analyzes** failures using historical patterns and troubleshooting guides
3. **Fixes** issues automatically when confident (low-risk changes)
4. **Escalates** complex issues to engineers with detailed analysis

**Goal**: Achieve 90%+ green CI builds without manual intervention, freeing engineers to focus on feature development.

---

## Architecture

### Component 1: Continuous Monitor
**Technology**: GitHub Actions webhook + long-running process

```python
# Pseudo-code for monitoring loop
while True:
    workflows = gh_api.get_workflow_runs(status=['completed', 'in_progress'])

    for run in workflows:
        if run.conclusion == 'failure':
            agent.analyze_and_fix(run)
        elif run.status == 'queued' and run.wait_time > threshold:
            agent.check_runner_availability(run)

    time.sleep(60)  # Check every minute
```

**Key Features**:
- Real-time webhook notifications from GitHub
- Polling fallback for missed events
- Tracks workflow run history and patterns
- Maintains state database of known issues

### Component 2: Intelligent Analyzer
**Technology**: LLM-powered analysis + pattern matching

**Analysis Pipeline**:
1. **Fetch logs**: Download failed workflow logs
2. **Pattern match**: Check against known issue database
3. **LLM analysis**: Use Claude/GPT to understand novel failures
4. **Root cause**: Identify exact failure point and category
5. **Solution lookup**: Find fix from troubleshooting guides

**Known Issue Database**:
```json
{
  "issue_id": "GLIBC_NODE20_COMPAT",
  "pattern": "GLIBC_2.28' not found.*node20",
  "category": "docker_compatibility",
  "severity": "high",
  "fix_type": "automated",
  "solution": "downgrade_upload_artifact_v3",
  "confidence": 0.95
}
```

### Component 3: Automated Fixer
**Technology**: GitHub API + Git operations

**Fix Categories by Confidence Level**:

**HIGH Confidence (Auto-Fix)**:
- Cache key version mismatches
- Known dependency version pins
- Disabled test markers
- Action version downgrades (GLIBC issues)
- Known vcpkg version conflicts

**MEDIUM Confidence (Create PR)**:
- New dependency version pins
- Workflow configuration changes
- Build flag modifications

**LOW Confidence (Create Issue)**:
- Novel build failures
- Platform-specific crashes
- Test failures requiring code changes

**Auto-Fix Workflow**:
```
1. Create fix branch: auto-fix/<issue-id>-<timestamp>
2. Apply fix (edit workflow/config files)
3. Commit with detailed message
4. Push branch
5. Create PR with:
   - Root cause analysis
   - Fix explanation
   - Test evidence
   - Related logs/artifacts
6. Auto-merge if tests pass (for HIGH confidence)
```

### Component 4: Escalation Manager
**Technology**: GitHub Issues + Notifications

**Escalation Triggers**:
- Novel failure patterns (not in database)
- Multiple consecutive fix attempts failed
- Critical workflows blocked > 4 hours
- Self-hosted runner unavailability

**Escalation Actions**:
1. Create GitHub issue with:
   - Detailed failure analysis
   - Logs and artifacts
   - Attempted fixes
   - Recommended actions
2. Tag relevant engineers
3. Slack/email notification
4. Update agent knowledge base

---

## Implementation Phases

### Phase 1: Monitoring & Analysis (Weeks 1-2)
- [ ] Set up GitHub webhook listener
- [ ] Implement workflow run fetching
- [ ] Create failure log parser
- [ ] Build known issue database (seed with current guides)
- [ ] Implement pattern matching engine

**Deliverable**: Agent can detect and categorize all failures

### Phase 2: Auto-Fix for Known Issues (Weeks 3-4)
- [ ] Implement Git operations (branch, commit, push)
- [ ] Create PR automation
- [ ] Build fix templates for HIGH confidence issues:
  - Cache key updates
  - Action version downgrades
  - Dependency pins
  - Test disabling
- [ ] Test auto-fix on historical failures

**Deliverable**: Agent can fix 5 most common issue types

### Phase 3: LLM-Powered Analysis (Weeks 5-6)
- [ ] Integrate Claude API for novel failure analysis
- [ ] Train on ApraPipes troubleshooting guides
- [ ] Implement multi-step reasoning for complex issues
- [ ] Build confidence scoring system
- [ ] Create fix verification tests

**Deliverable**: Agent can analyze and propose fixes for new issues

### Phase 4: Full Autonomy & Learning (Weeks 7-8)
- [ ] Implement auto-merge for HIGH confidence fixes
- [ ] Build feedback loop (fix success → confidence boost)
- [ ] Create escalation manager
- [ ] Add Slack/email integration
- [ ] Implement agent performance metrics dashboard

**Deliverable**: Fully autonomous agent with human oversight

---

## Safety Mechanisms

### 1. Fix Validation
- All fixes create PRs (not direct commits to main)
- CI must pass before auto-merge
- Human approval required for MEDIUM/LOW confidence
- Rollback capability for failed auto-merges

### 2. Rate Limiting
- Max 10 auto-fixes per day (prevent runaway)
- Max 3 attempts per issue (prevent fix loops)
- Cool-down period after failed fixes

### 3. Human Oversight
- Weekly review of auto-fixes
- Agent performance metrics tracking
- Manual override capability
- Emergency stop button

### 4. Blast Radius Control
- Test auto-fixes on feature branches first
- Gradual rollout (start with NoCUDA workflows)
- Never modify production deployment workflows
- Restrict to CI/CD workflow changes only

---

## Success Metrics

### Primary KPIs
1. **Green Build Rate**: Target 90%+ (currently ~60%)
2. **Time to Fix**: Median time from failure to fix < 1 hour
3. **Auto-Fix Success Rate**: >85% of auto-fixes pass CI
4. **Engineer Time Saved**: Reduce CI fixing time by 70%

### Secondary KPIs
1. **Novel Issue Detection**: Identify new patterns within 24h
2. **Escalation Accuracy**: <5% false positive escalations
3. **Agent Uptime**: 99.5% availability
4. **Knowledge Base Growth**: Add 2+ new patterns per week

---

## Technology Stack

### Core Components
- **Language**: Python 3.11+
- **LLM**: Claude 3.5 Sonnet (via Anthropic API)
- **Database**: SQLite (lightweight, file-based)
- **Git**: PyGitHub library
- **CI Integration**: GitHub REST API + Webhooks

### Infrastructure
- **Hosting**: GitHub Actions (self-hosted runner) or AWS Lambda
- **Storage**: Git repository for knowledge base (version controlled)
- **Monitoring**: Prometheus + Grafana
- **Notifications**: Slack API, GitHub Issues

### AI/ML
- **Pattern Matching**: Regex + fuzzy matching
- **Root Cause Analysis**: Claude 3.5 with RAG (troubleshooting guides)
- **Confidence Scoring**: Logistic regression on historical fix success
- **Learning**: Feedback loop updating confidence scores

---

## Cost Analysis

### Monthly Costs
- **Claude API**: ~$50/month (est. 100 analyses @ $0.50 each)
- **Infrastructure**: $20/month (if using cloud hosting)
- **GitHub API**: Free (within rate limits)
- **Total**: ~$70/month

### ROI Calculation
- **Engineer Time Saved**: 20 hours/month × $100/hour = $2,000/month
- **Faster CI Fixes**: Reduced development delays = $5,000/month (estimated)
- **Net Benefit**: $6,930/month
- **ROI**: 99x return

---

## Knowledge Base Structure

### Troubleshooting Guides (Existing)
- troubleshooting.windows.md
- troubleshooting.linux.md
- troubleshooting.cuda.md
- troubleshooting.docker.md
- troubleshooting.jetson.md

### Agent-Specific Data
```
agent-knowledge/
├── known-issues.json          # Pattern database
├── fix-templates/              # Reusable fix scripts
│   ├── cache-key-update.py
│   ├── action-downgrade.py
│   ├── dependency-pin.py
│   └── test-disable.py
├── fix-history.db             # SQLite database
├── confidence-models/          # ML models
│   └── fix-success-predictor.pkl
└── escalation-rules.yaml      # When to escalate
```

---

## Example Scenarios

### Scenario 1: GLIBC Compatibility Issue (AUTO-FIX)
```
1. Webhook: CI-Linux-CUDA-Docker failed
2. Download logs: "GLIBC_2.28 not found"
3. Pattern match: Known issue (GLIBC_NODE20_COMPAT)
4. Confidence: 0.95 (HIGH)
5. Apply fix: Downgrade upload-artifact v4 → v3
6. Create PR: "fix: Docker GLIBC compatibility..."
7. CI passes
8. Auto-merge
9. Update knowledge base: Success count++
```

### Scenario 2: Novel Test Failure (ESCALATE)
```
1. Webhook: CI-Win-NoCUDA failed
2. Download logs: "Test X failed: assertion error"
3. Pattern match: No match in database
4. LLM Analysis: "Appears to be race condition..."
5. Confidence: 0.30 (LOW)
6. Create Issue: "Investigate test_X race condition"
7. Tag @engineer
8. Slack notification
9. Add pattern to database for future learning
```

### Scenario 3: Self-Hosted Runner Unavailable (MONITOR)
```
1. Monitor: 5 workflows queued for 'windows-cuda' > 2 hours
2. Check runner status via API: Offline
3. Create alert: "Runner windows-cuda offline"
4. Slack notification to DevOps
5. Log incident for trend analysis
```

---

## Future Enhancements

### Phase 5+ (Long-term)
1. **Predictive Failures**: Detect issues before they break CI
2. **Multi-Repo Support**: Extend to other Apra-Labs projects
3. **Cost Optimization**: Suggest cheaper build configurations
4. **Performance Optimization**: Identify slow build steps
5. **Security Scanning**: Auto-fix security vulnerabilities
6. **Dependency Updates**: Automated dependency upgrades with testing

---

## Getting Started (Prototype)

### Quick Start Script
```python
# agent.py - Minimal viable agent
import os
from github import Github
from anthropic import Anthropic

g = Github(os.getenv('GITHUB_TOKEN'))
repo = g.get_repo("Apra-Labs/ApraPipes")

def monitor_workflows():
    runs = repo.get_workflow_runs(status='completed')

    for run in runs[:10]:  # Last 10 runs
        if run.conclusion == 'failure':
            analyze_failure(run)

def analyze_failure(run):
    logs = run.logs_url  # Fetch logs
    # Pattern match against known issues
    # If novel, use Claude for analysis
    # Propose fix or escalate
    pass

if __name__ == "__main__":
    monitor_workflows()
```

### Required Secrets
- `GITHUB_TOKEN`: Personal access token with repo/workflow permissions
- `ANTHROPIC_API_KEY`: Claude API key
- `SLACK_WEBHOOK`: For notifications (optional)

---

## Open Questions

1. **Auto-merge approval**: Should we require 1 human approval even for HIGH confidence?
2. **Runner management**: Should agent restart/provision self-hosted runners?
3. **Test disabling**: What's the threshold for auto-disabling flaky tests?
4. **Knowledge sharing**: How to share learnings across Apra-Labs repos?
5. **Compliance**: Any regulatory concerns with automated code changes?

---

## References

- **GitHub Actions API**: https://docs.github.com/en/rest/actions
- **Claude API**: https://docs.anthropic.com/claude/reference
- **Similar Tools**: Renovate, Dependabot, GitHub Copilot Workspace
- **ApraPipes Guides**: See troubleshooting.*.md files

---

**Next Steps**:
1. Review and approve this design
2. Set up Phase 1 prototype (monitoring only)
3. Test on historical failures
4. Iterate based on feedback

**Owner**: DevOps/Infrastructure Team
**Reviewers**: @engineers, @SWE leads

---

**Status**: DRAFT - Pending Review
**Last Updated**: 2025-12-01

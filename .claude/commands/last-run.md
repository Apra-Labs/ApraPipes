Summarize the most recent GitHub Actions run:

1. Get latest run: `gh run list -L1 --json databaseId,status,conclusion,name`
2. If completed, fetch logs:
   - If failed: `gh run view <id> --log-failed`
   - If passed: `gh run view <id> --log | tail -100`
3. Summarize:
   - Workflow name
   - PASS/FAIL
   - Duration
   - Key output (errors if failed, confirmation if passed)
   - What we learned (if anything)
4. State next action
5. Continue working
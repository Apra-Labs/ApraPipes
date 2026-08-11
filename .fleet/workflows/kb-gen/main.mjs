/**
 * kb-gen -- Deep-scan a repo and generate Knowledge Bank entries.
 *
 * Usage:
 *   apra-fleet workflow kb-gen --member <name> [--depth shallow|deep] [--budget <usd>]
 *
 * The workflow dispatches agents to the specified member to:
 *   1. Map the repo structure (directories, key files, stack detection)
 *   2. Scan each module/directory for architecture, patterns, conventions
 *   3. Capture KB entries (context-cache, knowledge, runbook)
 *   4. Export the canonical bible
 */
import { parseArgs } from 'node:util';
import { FleetWorkflow } from '@apralabs/apra-fleet-workflow';
import { WorkflowEngine } from '@apralabs/apra-fleet-workflow/engine';
import { connectFleet } from '@apralabs/apra-fleet-client/server-resolution';

export const selfExecuting = true;

async function main() {
    const args = process.argv.slice(2);
    const { values } = parseArgs({
        args,
        options: {
            member:  { type: 'string', short: 'm' },
            depth:   { type: 'string', short: 'd', default: 'deep' },
            budget:  { type: 'string', short: 'b' },
            'no-promote': { type: 'boolean', default: false },
        },
        strict: false,
    });

    if (!values.member) {
        console.error('Usage: apra-fleet workflow kb-gen --member <name> [--depth shallow|deep] [--budget <usd>] [--no-promote]');
        process.exit(1);
    }

    const memberName = values.member;
    const depth = values.depth || 'deep';
    const budgetUsd = values.budget ? parseFloat(values.budget) : null;
    const autoPromote = !values['no-promote'];

    console.log(`[kb-gen] Starting KB generation for member "${memberName}" (depth=${depth}, promote=${autoPromote})`);
    if (budgetUsd) console.log(`[kb-gen] Budget cap: $${budgetUsd}`);

    // Connect to fleet
    const { mcpClient, fleetApi, mode } = await connectFleet({ env: process.env });
    console.log(`[kb-gen] Connected to fleet server (mode=${mode})`);

    // Build the workflow
    const workflow = new FleetWorkflow(fleetApi);
    const engine = new WorkflowEngine(workflow);

    // Write the runner script to a temp file and execute it
    const runnerPath = new URL('./runner.mjs', import.meta.url).pathname
        // On Windows, pathToFileURL produces /C:/... but import.meta.url
        // gives file:///C:/..., so strip leading slash if drive letter follows
        .replace(/^\/([A-Za-z]:)/, '$1');

    await engine.executeFile(runnerPath, {
        memberName,
        depth,
        budget: budgetUsd,
        autoPromote,
    });

    console.log('[kb-gen] Done.');
    mcpClient.transport?.stop?.();
    process.exit(0);
}

main().catch(err => {
    console.error('[kb-gen] Fatal:', err);
    process.exit(1);
});

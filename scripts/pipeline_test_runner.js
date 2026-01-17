#!/usr/bin/env node
// ==============================================================================
// Pipeline Test Runner - Node.js based pipeline testing
// ==============================================================================
// Usage: node pipeline_test_runner.js <command> <pipeline.json> [options]
//
// Commands:
//   validate <file>         Validate a pipeline configuration
//   run <file> [duration]   Run a pipeline for specified duration (default: 2s)
//
// Exit codes:
//   0 - Success
//   1 - Validation/Runtime error
//   2 - Script error
// ==============================================================================

const path = require('path');
const fs = require('fs');

// Find the addon
const projectRoot = path.resolve(__dirname, '..');
const addonPath = path.join(projectRoot, 'build', 'aprapipes.node');

if (!fs.existsSync(addonPath)) {
    console.error(`ERROR: Node addon not found at ${addonPath}`);
    console.error('Please build the project with -DBUILD_NODE_ADDON=ON');
    process.exit(2);
}

const addon = require(addonPath);

// Parse command line arguments
const args = process.argv.slice(2);
const command = args[0];
const pipelineFile = args[1];

if (!command || !pipelineFile) {
    console.error('Usage: node pipeline_test_runner.js <validate|run> <pipeline.json> [duration]');
    process.exit(2);
}

// Read pipeline file
function readPipelineConfig(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(content);
    } catch (err) {
        console.error(`Failed to read pipeline file: ${err.message}`);
        process.exit(1);
    }
}

// Validate command
async function validatePipeline(configPath) {
    const config = readPipelineConfig(configPath);

    try {
        const result = addon.validatePipeline(config);

        if (result.valid) {
            console.log('Validation: PASS');
            return true;
        } else {
            console.log('Validation: FAIL');
            for (const issue of result.issues) {
                console.log(`  [${issue.level.toUpperCase()}] ${issue.code}: ${issue.message}`);
                if (issue.suggestion) {
                    console.log(`    Suggestion: ${issue.suggestion}`);
                }
            }
            return false;
        }
    } catch (err) {
        console.error(`Validation error: ${err.message}`);
        return false;
    }
}

// Run command
async function runPipeline(configPath, durationSeconds) {
    const config = readPipelineConfig(configPath);

    let pipeline;
    try {
        // Create pipeline
        pipeline = addon.createPipeline(config);
        console.log('Pipeline created');

        // Initialize
        await pipeline.init();
        console.log('Pipeline initialized');

        // Start running (non-blocking)
        const runPromise = pipeline.run();
        console.log('Pipeline running...');

        // Wait for specified duration
        await new Promise(resolve => setTimeout(resolve, durationSeconds * 1000));

        // Get status during run
        const status = pipeline.getStatus();
        console.log(`Status: ${status}`);

        // Stop pipeline
        await pipeline.stop();
        console.log('Pipeline stopped');

        // Terminate
        await pipeline.terminate();
        console.log('Pipeline terminated');

        return true;
    } catch (err) {
        console.error(`Runtime error: ${err.message}`);

        // Try to clean up
        if (pipeline) {
            try {
                await pipeline.stop();
                await pipeline.terminate();
            } catch (cleanupErr) {
                // Ignore cleanup errors
            }
        }

        return false;
    }
}

// Main
async function main() {
    let success = false;

    switch (command) {
        case 'validate':
            success = await validatePipeline(pipelineFile);
            break;

        case 'run':
            const duration = parseFloat(args[2]) || 2;
            success = await runPipeline(pipelineFile, duration);
            break;

        default:
            console.error(`Unknown command: ${command}`);
            process.exit(2);
    }

    process.exit(success ? 0 : 1);
}

main().catch(err => {
    console.error(`Unexpected error: ${err.message}`);
    process.exit(2);
});

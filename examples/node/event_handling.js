/**
 * Event Handling Example
 *
 * Demonstrates how to use the event system to monitor pipeline health,
 * handle errors, and react to pipeline state changes.
 *
 * Usage: node examples/node/event_handling.js
 *
 * Output: Creates event_????.jpg files in ./output/
 */

const path = require('path');
const fs = require('fs');

// Load the addon
const addonPath = path.join(__dirname, '../../aprapipes.node');
let ap;
try {
    ap = require(addonPath);
} catch (e) {
    console.error('Failed to load addon:', e.message);
    process.exit(1);
}

// Create output directory
const outputDir = path.join(__dirname, 'output');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Pipeline configuration
const config = {
    name: "EventDemo",
    modules: {
        source: {
            type: "TestSignalGenerator",
            props: { width: 640, height: 480 }
        },
        transform: {
            type: "ColorConversion",
            props: { conversionType: "YUV420PLANAR_TO_RGB" }
        },
        encoder: {
            type: "ImageEncoderCV"
        },
        writer: {
            type: "FileWriterModule",
            props: {
                strFullFileNameWithPattern: path.join(outputDir, "event_????.jpg")
            }
        }
    },
    connections: [
        { from: "source", to: "transform" },
        { from: "transform", to: "encoder" },
        { from: "encoder", to: "writer" }
    ]
};

async function main() {
    console.log('=== Event Handling Example ===\n');

    // Create the pipeline
    const pipeline = ap.createPipeline(config);

    // Event counters for demonstration
    const stats = {
        healthEvents: 0,
        errorEvents: 0
    };

    // Register health event handler
    const healthHandler = (event) => {
        stats.healthEvents++;
        console.log(`[HEALTH] Module: ${event.moduleId}`);
        console.log(`         Message: ${event.message}`);
        console.log(`         Timestamp: ${new Date(event.timestamp).toISOString()}`);
        console.log('');
    };
    pipeline.on('health', healthHandler);

    // Register error event handler
    pipeline.on('error', (event) => {
        stats.errorEvents++;
        console.error(`[ERROR] Module: ${event.moduleId}`);
        console.error(`        Code: ${event.errorCode}`);
        console.error(`        Message: ${event.message}`);
        console.error(`        Timestamp: ${new Date(event.timestamp).toISOString()}`);
        console.error('');
    });

    console.log('Event handlers registered.\n');
    console.log('Supported events:');
    console.log('  - "health": Module health status updates');
    console.log('  - "error": Module error notifications\n');

    // Demonstrate event API
    console.log('--- Event API Demo ---\n');

    // Chaining example
    console.log('1. Method chaining:');
    console.log('   pipeline.on("health", fn1).on("error", fn2)');

    // Multiple handlers
    console.log('\n2. Multiple handlers per event:');
    const handler1 = (e) => { /* handler 1 */ };
    const handler2 = (e) => { /* handler 2 */ };
    pipeline.on('health', handler1);
    pipeline.on('health', handler2);
    console.log('   Two additional handlers registered for "health" event');

    // Remove specific handler
    console.log('\n3. Remove specific handler:');
    pipeline.off('health', handler1);
    console.log('   Handler 1 removed');

    // Show module information
    console.log('\n--- Module Information ---\n');

    const modules = ['source', 'transform', 'encoder', 'writer'];
    modules.forEach(name => {
        const mod = pipeline.getModule(name);
        console.log(`${name}:`);
        console.log(`  Type: ${mod.type}`);
        console.log(`  ID: ${mod.id}`);
        console.log(`  Has dynamic props: ${mod.hasDynamicProperties()}`);
        console.log('');
    });

    // Initialize and run the pipeline
    console.log('--- Running Pipeline ---\n');
    console.log('Initializing pipeline...');
    await pipeline.init();

    console.log('Starting pipeline...');
    pipeline.run();

    // Let it run for 2 seconds
    console.log('Running for 2 seconds (watch for health events)...\n');
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Stop the pipeline
    console.log('Stopping pipeline...');
    await pipeline.stop();
    console.log('Pipeline stopped.\n');

    // Summary
    console.log('--- Event Summary ---');
    console.log(`  Health events received: ${stats.healthEvents}`);
    console.log(`  Error events received: ${stats.errorEvents}`);

    // Check output files
    const files = fs.readdirSync(outputDir).filter(f => f.startsWith('event_') && f.endsWith('.jpg'));
    console.log(`\nGenerated ${files.length} JPEG files in ${outputDir}/`);

    // Clean up listeners
    pipeline.removeAllListeners('health');
    pipeline.removeAllListeners('error');
    console.log('\nEvent listeners cleaned up.');

    console.log('\n=== Example Complete ===\n');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});

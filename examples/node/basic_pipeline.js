/**
 * Basic Pipeline Example
 *
 * Demonstrates how to create and run a simple pipeline using the
 * ApraPipes Node.js addon. This example creates a test signal generator
 * that outputs to a statistics sink.
 *
 * Usage: node examples/node/basic_pipeline.js
 */

const path = require('path');

// Load the addon from the project root
const addonPath = path.join(__dirname, '../../aprapipes.node');
let ap;
try {
    ap = require(addonPath);
    console.log('ApraPipes addon loaded successfully');
} catch (e) {
    console.error('Failed to load addon:', e.message);
    console.error('Make sure you have built the project first.');
    process.exit(1);
}

// Define a simple pipeline configuration
const pipelineConfig = {
    name: "BasicPipeline",
    modules: {
        // Test signal generator - creates synthetic video frames
        source: {
            type: "TestSignalGenerator",
            props: {
                width: 640,
                height: 480
            }
        },
        // Statistics sink - collects frame statistics
        sink: {
            type: "StatSink"
        }
    },
    connections: [
        { from: "source", to: "sink" }
    ]
};

console.log('\n=== Basic Pipeline Example ===\n');
console.log('Pipeline configuration:');
console.log(JSON.stringify(pipelineConfig, null, 2));

// Create the pipeline
console.log('\nCreating pipeline...');
const pipeline = ap.createPipeline(pipelineConfig);

// Get module handles
const source = pipeline.getModule('source');
const sink = pipeline.getModule('sink');

console.log('\nModules created:');
console.log(`  - source: ${source.type} (${source.id})`);
console.log(`  - sink: ${sink.type} (${sink.id})`);

// Check initial properties
console.log('\nSource properties:');
const props = source.getProps();
console.log(`  - fps: ${props.fps}`);
console.log(`  - qlen: ${props.qlen}`);

// Set up event handlers
pipeline
    .on('health', (event) => {
        console.log(`[Health] ${event.moduleId}: ${event.message}`);
    })
    .on('error', (event) => {
        console.error(`[Error] ${event.moduleId}: ${event.message}`);
    });

console.log('\nEvent handlers registered.');
console.log('\nPipeline is ready. To run it, you would call:');
console.log('  pipeline.start()');
console.log('  // ... let it process frames ...');
console.log('  pipeline.stop()');

console.log('\n=== Example Complete ===\n');

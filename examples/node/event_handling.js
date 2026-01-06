/**
 * Event Handling Example
 *
 * Demonstrates how to use the event system to monitor pipeline health,
 * handle errors, and react to pipeline state changes.
 *
 * Usage: node examples/node/event_handling.js
 */

const path = require('path');

// Load the addon
const addonPath = path.join(__dirname, '../../aprapipes.node');
let ap;
try {
    ap = require(addonPath);
} catch (e) {
    console.error('Failed to load addon:', e.message);
    process.exit(1);
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
        sink: {
            type: "StatSink"
        }
    },
    connections: [
        { from: "source", to: "transform" },
        { from: "transform", to: "sink" }
    ]
};

console.log('=== Event Handling Example ===\n');

// Create the pipeline
const pipeline = ap.createPipeline(config);

// Event counters for demonstration
const stats = {
    healthEvents: 0,
    errorEvents: 0
};

// Register health event handler
pipeline.on('health', (event) => {
    stats.healthEvents++;
    console.log(`[HEALTH] Module: ${event.moduleId}`);
    console.log(`         Message: ${event.message}`);
    console.log(`         Timestamp: ${new Date(event.timestamp).toISOString()}`);
    console.log('');
});

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
const handler1 = (e) => console.log('Handler 1 called');
const handler2 = (e) => console.log('Handler 2 called');
pipeline.on('health', handler1);
pipeline.on('health', handler2);
console.log('   Two handlers registered for "health" event');

// Remove specific handler
console.log('\n3. Remove specific handler:');
pipeline.off('health', handler1);
console.log('   Handler 1 removed, Handler 2 still active');

// Remove all handlers
console.log('\n4. Remove all handlers for an event:');
pipeline.removeAllListeners('health');
console.log('   All "health" handlers removed');

// Re-register for final demo
pipeline.on('health', (e) => {
    console.log(`   Health update: ${e.moduleId}`);
});

console.log('\n5. Re-registered health handler');

// Show module information
console.log('\n--- Module Information ---\n');

const modules = ['source', 'transform', 'sink'];
modules.forEach(name => {
    const mod = pipeline.getModule(name);
    console.log(`${name}:`);
    console.log(`  Type: ${mod.type}`);
    console.log(`  ID: ${mod.id}`);
    console.log(`  Running: ${mod.isRunning()}`);
    console.log(`  Has dynamic props: ${mod.hasDynamicProperties()}`);
    console.log('');
});

// Final summary
console.log('--- Usage Notes ---\n');
console.log('In a real application:');
console.log('1. Register event handlers before calling pipeline.start()');
console.log('2. Use "health" events to monitor frame rates and throughput');
console.log('3. Use "error" events to handle failures gracefully');
console.log('4. Clean up handlers when done with pipeline.removeAllListeners()');

console.log('\n=== Example Complete ===\n');

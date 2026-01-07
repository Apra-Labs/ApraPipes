/**
 * Event System Tests for Node.js Addon
 * Phase 4: Event System
 */

const assert = require('assert');
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

console.log('=== Event System Tests ===\n');

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`✓ ${name}`);
        passed++;
    } catch (e) {
        console.log(`✗ ${name}`);
        console.log(`  Error: ${e.message}`);
        failed++;
    }
}

// Test 1: Pipeline has on method
test('Pipeline has on() method', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    assert.strictEqual(typeof pipeline.on, 'function', 'on should be a function');
});

// Test 2: Pipeline has off method
test('Pipeline has off() method', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    assert.strictEqual(typeof pipeline.off, 'function', 'off should be a function');
});

// Test 3: Pipeline has removeAllListeners method
test('Pipeline has removeAllListeners() method', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    assert.strictEqual(typeof pipeline.removeAllListeners, 'function', 'removeAllListeners should be a function');
});

// Test 4: on() returns this for chaining
test('on() returns this for chaining', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    const result = pipeline.on('error', () => {});
    assert.strictEqual(result, pipeline, 'on() should return pipeline for chaining');
});

// Test 5: off() returns this for chaining
test('off() returns this for chaining', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    const handler = () => {};
    pipeline.on('error', handler);
    const result = pipeline.off('error', handler);
    assert.strictEqual(result, pipeline, 'off() should return pipeline for chaining');
});

// Test 6: removeAllListeners() returns this for chaining
test('removeAllListeners() returns this for chaining', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    pipeline.on('error', () => {});
    const result = pipeline.removeAllListeners('error');
    assert.strictEqual(result, pipeline, 'removeAllListeners() should return pipeline for chaining');
});

// Test 7: Can chain multiple on() calls
test('Can chain multiple on() calls', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    // This should not throw
    pipeline
        .on('error', () => {})
        .on('health', () => {})
        .on('started', () => {});
});

// Test 8: on() requires string event name
test('on() throws on invalid event name', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    assert.throws(() => {
        pipeline.on(123, () => {});
    }, /string.*function/i, 'on() should throw on non-string event');
});

// Test 9: on() requires function callback
test('on() throws on invalid callback', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    assert.throws(() => {
        pipeline.on('error', 'not a function');
    }, /string.*function/i, 'on() should throw on non-function callback');
});

// Test 10: Module access returns actual module pointers
test('getModule() returns module with valid properties', () => {
    const pipeline = ap.createPipeline({
        modules: {
            source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
            sink: { type: 'StatSink' }
        },
        connections: [{ from: 'source', to: 'sink' }]
    });

    const mod = pipeline.getModule('source');
    assert.ok(mod, 'getModule should return a module');
    assert.strictEqual(mod.id, 'source', 'module id should match');
    assert.strictEqual(mod.type, 'TestSignalGenerator', 'module type should match');
});

// Summary
console.log('\n=== Summary ===');
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Total:  ${passed + failed}`);

process.exit(failed > 0 ? 1 : 0);

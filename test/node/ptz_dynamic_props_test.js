/**
 * PTZ Dynamic Properties Test
 * Demonstrates changing VirtualPTZ properties at runtime
 *
 * This test shows how to:
 * 1. Create a pipeline with VirtualPTZ module
 * 2. Read dynamic property values (roiX, roiY, roiWidth, roiHeight)
 * 3. Change properties at runtime to simulate PTZ movement
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

console.log('=== PTZ Dynamic Properties Test ===\n');

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`OK ${name}`);
        passed++;
    } catch (e) {
        console.log(`FAIL ${name}`);
        console.log(`  Error: ${e.message}`);
        failed++;
    }
}

async function asyncTest(name, fn) {
    try {
        await fn();
        console.log(`OK ${name}`);
        passed++;
    } catch (e) {
        console.log(`FAIL ${name}`);
        console.log(`  Error: ${e.message}`);
        failed++;
    }
}

// Pipeline config with PTZ
// Note: Use explicit floats (0.0, 1.0) not ints (0, 1) for VirtualPTZ props
const ptzConfig = {
    modules: {
        source: {
            type: 'TestSignalGenerator',
            props: { width: 1920, height: 1080 }
        },
        convert: {
            type: 'ColorConversion',
            props: { conversionType: 'YUV420PLANAR_TO_RGB' }
        },
        ptz: {
            type: 'VirtualPTZ'
            // Use default props - they're already floats
        },
        sink: {
            type: 'StatSink'
        }
    },
    connections: [
        { from: 'source', to: 'convert' },
        { from: 'convert', to: 'ptz' },
        { from: 'ptz', to: 'sink' }
    ]
};

// Test 1: VirtualPTZ module has dynamic properties
test('VirtualPTZ module has hasDynamicProperties method', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');
    assert.ok(ptz, 'ptz module should exist');
    assert.strictEqual(typeof ptz.hasDynamicProperties, 'function', 'hasDynamicProperties should be a function');
});

// Test 2: VirtualPTZ reports it has dynamic properties
test('VirtualPTZ reports it has dynamic properties', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');
    const hasDynProps = ptz.hasDynamicProperties();
    assert.strictEqual(hasDynProps, true, 'VirtualPTZ should have dynamic properties');
});

// Test 3: Can get dynamic property names
test('VirtualPTZ has roiX, roiY, roiWidth, roiHeight dynamic properties', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');
    const names = ptz.getDynamicPropertyNames();

    assert.ok(Array.isArray(names), 'should return array');
    assert.ok(names.includes('roiX'), 'should include roiX');
    assert.ok(names.includes('roiY'), 'should include roiY');
    assert.ok(names.includes('roiWidth'), 'should include roiWidth');
    assert.ok(names.includes('roiHeight'), 'should include roiHeight');
});

// Test 4: Can read initial property values
test('Can read initial property values', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');

    // Initial values - VirtualPTZProps defaults are roiX=0, roiY=0, roiWidth=1, roiHeight=1
    const roiX = ptz.getProperty('roiX');
    const roiY = ptz.getProperty('roiY');
    const roiWidth = ptz.getProperty('roiWidth');
    const roiHeight = ptz.getProperty('roiHeight');

    // Values should be numbers (could be 0 or 0.0)
    assert.strictEqual(typeof roiX, 'number', 'roiX should be a number');
    assert.strictEqual(typeof roiY, 'number', 'roiY should be a number');
    assert.strictEqual(typeof roiWidth, 'number', 'roiWidth should be a number');
    assert.strictEqual(typeof roiHeight, 'number', 'roiHeight should be a number');

    // Default values
    assert.ok(roiX === 0 || roiX === 0.0, 'roiX should be 0');
    assert.ok(roiY === 0 || roiY === 0.0, 'roiY should be 0');
    assert.ok(roiWidth === 1 || roiWidth === 1.0, 'roiWidth should be 1');
    assert.ok(roiHeight === 1 || roiHeight === 1.0, 'roiHeight should be 1');
});

// Test 5: Can set property values
test('Can set property values', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');

    // Set new values
    const result = ptz.setProperty('roiX', 0.25);
    assert.strictEqual(result, true, 'setProperty should return true');

    // Verify the change
    const roiX = ptz.getProperty('roiX');
    assert.strictEqual(roiX, 0.25, 'roiX should be updated to 0.25');
});

// Test 6: StatSink does NOT have dynamic properties
test('StatSink does NOT have dynamic properties', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const sink = pipeline.getModule('sink');
    const hasDynProps = sink.hasDynamicProperties();
    assert.strictEqual(hasDynProps, false, 'StatSink should NOT have dynamic properties');
});

// Test 7: getDynamicPropertyNames returns empty array for modules without dynamic props
test('getDynamicPropertyNames returns empty array for StatSink', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const sink = pipeline.getModule('sink');
    const names = sink.getDynamicPropertyNames();
    assert.ok(Array.isArray(names), 'should return array');
    assert.strictEqual(names.length, 0, 'should be empty');
});

// Test 8: Simulate PTZ pan (change roiX)
test('Simulate PTZ pan by changing roiX', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');

    // Start at left edge
    ptz.setProperty('roiX', 0.0);
    ptz.setProperty('roiWidth', 0.5);
    assert.strictEqual(ptz.getProperty('roiX'), 0.0);

    // Pan to center
    ptz.setProperty('roiX', 0.25);
    assert.strictEqual(ptz.getProperty('roiX'), 0.25);

    // Pan to right
    ptz.setProperty('roiX', 0.5);
    assert.strictEqual(ptz.getProperty('roiX'), 0.5);
});

// Test 9: Simulate PTZ tilt (change roiY)
test('Simulate PTZ tilt by changing roiY', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');

    // Start at top
    ptz.setProperty('roiY', 0.0);
    ptz.setProperty('roiHeight', 0.5);
    assert.strictEqual(ptz.getProperty('roiY'), 0.0);

    // Tilt to center
    ptz.setProperty('roiY', 0.25);
    assert.strictEqual(ptz.getProperty('roiY'), 0.25);

    // Tilt to bottom
    ptz.setProperty('roiY', 0.5);
    assert.strictEqual(ptz.getProperty('roiY'), 0.5);
});

// Test 10: Simulate PTZ zoom (change roiWidth and roiHeight)
test('Simulate PTZ zoom by changing roiWidth and roiHeight', () => {
    const pipeline = ap.createPipeline(ptzConfig);
    const ptz = pipeline.getModule('ptz');

    // Full view (zoom out)
    ptz.setProperty('roiWidth', 1.0);
    ptz.setProperty('roiHeight', 1.0);
    assert.strictEqual(ptz.getProperty('roiWidth'), 1.0);
    assert.strictEqual(ptz.getProperty('roiHeight'), 1.0);

    // 2x zoom (50% of image)
    ptz.setProperty('roiWidth', 0.5);
    ptz.setProperty('roiHeight', 0.5);
    ptz.setProperty('roiX', 0.25);  // Center the crop
    ptz.setProperty('roiY', 0.25);
    assert.strictEqual(ptz.getProperty('roiWidth'), 0.5);
    assert.strictEqual(ptz.getProperty('roiHeight'), 0.5);

    // 4x zoom (25% of image)
    ptz.setProperty('roiWidth', 0.25);
    ptz.setProperty('roiHeight', 0.25);
    ptz.setProperty('roiX', 0.375);  // Center the crop
    ptz.setProperty('roiY', 0.375);
    assert.strictEqual(ptz.getProperty('roiWidth'), 0.25);
    assert.strictEqual(ptz.getProperty('roiHeight'), 0.25);
});

// Summary
console.log('\n=== Summary ===');
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Total:  ${passed + failed}`);

process.exit(failed > 0 ? 1 : 0);

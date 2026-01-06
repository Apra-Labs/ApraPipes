/**
 * PTZ Control Example
 *
 * Demonstrates how to use dynamic properties to control VirtualPTZ
 * module at runtime. This simulates pan/tilt/zoom operations by
 * changing the ROI (Region of Interest) parameters.
 *
 * Usage: node examples/node/ptz_control.js
 *
 * Output: Creates ptz_????.jpg files in ./output/ showing different PTZ positions
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

// Pipeline with VirtualPTZ for PTZ simulation
const ptzPipelineConfig = {
    name: "PTZControlDemo",
    modules: {
        // Generate 1080p test signal
        source: {
            type: "TestSignalGenerator",
            props: { width: 1920, height: 1080 }
        },
        // Convert to RGB for PTZ processing
        colorConvert: {
            type: "ColorConversion",
            props: { conversionType: "YUV420PLANAR_TO_RGB" }
        },
        // Virtual PTZ - allows runtime control of pan/tilt/zoom
        ptz: {
            type: "VirtualPTZ"
            // Default props: roiX=0, roiY=0, roiWidth=1, roiHeight=1 (full frame)
        },
        // Encode as JPEG
        encoder: {
            type: "ImageEncoderCV"
        },
        // Write to files
        writer: {
            type: "FileWriterModule",
            props: {
                strFullFileNameWithPattern: path.join(outputDir, "ptz_????.jpg")
            }
        }
    },
    connections: [
        { from: "source", to: "colorConvert" },
        { from: "colorConvert", to: "ptz" },
        { from: "ptz", to: "encoder" },
        { from: "encoder", to: "writer" }
    ]
};

console.log('=== PTZ Control Example ===\n');

// Create pipeline
const pipeline = ap.createPipeline(ptzPipelineConfig);
const ptz = pipeline.getModule('ptz');

// Check dynamic property support
console.log('VirtualPTZ Dynamic Properties:');
console.log(`  Has dynamic properties: ${ptz.hasDynamicProperties()}`);
console.log(`  Available properties: ${ptz.getDynamicPropertyNames().join(', ')}`);

// Read initial values
console.log('\nInitial PTZ Position (full frame):');
console.log(`  roiX: ${ptz.getProperty('roiX')} (left edge)`);
console.log(`  roiY: ${ptz.getProperty('roiY')} (top edge)`);
console.log(`  roiWidth: ${ptz.getProperty('roiWidth')} (100% of frame)`);
console.log(`  roiHeight: ${ptz.getProperty('roiHeight')} (100% of frame)`);

// Simulate PTZ operations
console.log('\n--- Simulating PTZ Operations ---\n');

// Zoom to 2x (50% of frame, centered)
console.log('1. ZOOM IN (2x):');
ptz.setProperty('roiWidth', 0.5);
ptz.setProperty('roiHeight', 0.5);
ptz.setProperty('roiX', 0.25);  // Center horizontally
ptz.setProperty('roiY', 0.25);  // Center vertically
console.log(`   Position: x=${ptz.getProperty('roiX')}, y=${ptz.getProperty('roiY')}`);
console.log(`   Size: ${ptz.getProperty('roiWidth')}x${ptz.getProperty('roiHeight')} (50% of frame)`);

// Pan right
console.log('\n2. PAN RIGHT:');
ptz.setProperty('roiX', 0.5);  // Move to right side
console.log(`   Position: x=${ptz.getProperty('roiX')}, y=${ptz.getProperty('roiY')}`);

// Tilt down
console.log('\n3. TILT DOWN:');
ptz.setProperty('roiY', 0.5);  // Move to bottom
console.log(`   Position: x=${ptz.getProperty('roiX')}, y=${ptz.getProperty('roiY')}`);

// Zoom to 4x (25% of frame)
console.log('\n4. ZOOM IN MORE (4x):');
ptz.setProperty('roiWidth', 0.25);
ptz.setProperty('roiHeight', 0.25);
ptz.setProperty('roiX', 0.375);  // Re-center
ptz.setProperty('roiY', 0.375);
console.log(`   Position: x=${ptz.getProperty('roiX')}, y=${ptz.getProperty('roiY')}`);
console.log(`   Size: ${ptz.getProperty('roiWidth')}x${ptz.getProperty('roiHeight')} (25% of frame)`);

// Reset to full frame
console.log('\n5. RESET (zoom out to full frame):');
ptz.setProperty('roiX', 0);
ptz.setProperty('roiY', 0);
ptz.setProperty('roiWidth', 1);
ptz.setProperty('roiHeight', 1);
console.log(`   Position: x=${ptz.getProperty('roiX')}, y=${ptz.getProperty('roiY')}`);
console.log(`   Size: ${ptz.getProperty('roiWidth')}x${ptz.getProperty('roiHeight')} (full frame)`);

// Show PTZ helper functions
console.log('\n--- PTZ Helper Functions ---\n');

/**
 * Set PTZ position with normalized coordinates
 * @param {number} pan - Pan position (0=left, 1=right)
 * @param {number} tilt - Tilt position (0=top, 1=bottom)
 * @param {number} zoom - Zoom level (1=full frame, 2=2x zoom, etc.)
 */
function setPTZ(pan, tilt, zoom) {
    const size = 1 / zoom;
    const maxOffset = 1 - size;
    const x = pan * maxOffset;
    const y = tilt * maxOffset;

    ptz.setProperty('roiWidth', size);
    ptz.setProperty('roiHeight', size);
    ptz.setProperty('roiX', x);
    ptz.setProperty('roiY', y);

    return { x, y, width: size, height: size };
}

console.log('Example PTZ helper function:');
console.log('  setPTZ(0.5, 0.5, 2)  // Center at 2x zoom');
const result = setPTZ(0.5, 0.5, 2);
console.log(`  Result: ${JSON.stringify(result)}`);

console.log(`\nOutput directory: ${outputDir}/`);
console.log('Run pipeline.start() to generate PTZ frames as JPEG files.');

console.log('\n=== Example Complete ===\n');

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
        // Generate 1080p test signal with GRID pattern for visible PTZ effects
        source: {
            type: "TestSignalGenerator",
            props: { width: 1920, height: 1080, pattern: "GRID" }
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

/**
 * Set PTZ position with normalized coordinates
 * @param {object} ptz - PTZ module handle
 * @param {number} pan - Pan position (0=left, 1=right)
 * @param {number} tilt - Tilt position (0=top, 1=bottom)
 * @param {number} zoom - Zoom level (1=full frame, 2=2x zoom, etc.)
 */
function setPTZ(ptz, pan, tilt, zoom) {
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

async function main() {
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

    // Initialize and run the pipeline
    console.log('\nInitializing pipeline...');
    await pipeline.init();
    console.log('Starting pipeline...');
    pipeline.run();

    // PTZ Operations with delays to capture frames at each position
    console.log('\n--- Live PTZ Operations ---\n');

    // Full frame for 500ms
    console.log('1. FULL FRAME (initial position)');
    await new Promise(resolve => setTimeout(resolve, 500));

    // Zoom to 2x (50% of frame, centered)
    console.log('2. ZOOM IN (2x):');
    let pos = setPTZ(ptz, 0.5, 0.5, 2);
    console.log(`   Position: x=${pos.x.toFixed(2)}, y=${pos.y.toFixed(2)}`);
    console.log(`   Size: ${pos.width}x${pos.height} (50% of frame)`);
    await new Promise(resolve => setTimeout(resolve, 500));

    // Pan right
    console.log('3. PAN RIGHT:');
    ptz.setProperty('roiX', 0.5);
    console.log(`   Position: x=${ptz.getProperty('roiX')}, y=${ptz.getProperty('roiY')}`);
    await new Promise(resolve => setTimeout(resolve, 500));

    // Tilt down
    console.log('4. TILT DOWN:');
    ptz.setProperty('roiY', 0.5);
    console.log(`   Position: x=${ptz.getProperty('roiX')}, y=${ptz.getProperty('roiY')}`);
    await new Promise(resolve => setTimeout(resolve, 500));

    // Zoom to 4x (25% of frame)
    console.log('5. ZOOM IN MORE (4x):');
    pos = setPTZ(ptz, 0.5, 0.5, 4);
    console.log(`   Position: x=${pos.x.toFixed(3)}, y=${pos.y.toFixed(3)}`);
    console.log(`   Size: ${pos.width}x${pos.height} (25% of frame)`);
    await new Promise(resolve => setTimeout(resolve, 500));

    // Reset to full frame
    console.log('6. RESET (zoom out to full frame):');
    pos = setPTZ(ptz, 0, 0, 1);
    console.log(`   Position: x=${pos.x}, y=${pos.y}`);
    console.log(`   Size: ${pos.width}x${pos.height} (full frame)`);
    await new Promise(resolve => setTimeout(resolve, 500));

    // Stop the pipeline
    console.log('\nStopping pipeline...');
    await pipeline.stop();
    console.log('Pipeline stopped.');

    // Check output files
    const files = fs.readdirSync(outputDir).filter(f => f.startsWith('ptz_') && f.endsWith('.jpg'));
    console.log(`\nGenerated ${files.length} JPEG files in ${outputDir}/`);
    console.log('Each file shows the frame at different PTZ positions.');

    console.log('\n=== Example Complete ===\n');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});

/**
 * Image Processing Pipeline Example
 *
 * Demonstrates a pipeline with multiple processing stages and
 * dynamic property support using VirtualPTZ module.
 * Outputs actual JPEG images to show the processing results.
 *
 * Usage: node examples/node/image_processing.js
 *
 * Output: Creates processed_????.jpg files in ./output/
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

// Image processing pipeline with PTZ control and file output
const config = {
    name: "ImageProcessingPipeline",
    modules: {
        // Source: generates 1080p test frames
        source: {
            type: "TestSignalGenerator",
            props: {
                width: 1920,
                height: 1080
            }
        },

        // Convert to RGB for processing
        colorConvert: {
            type: "ColorConversion",
            props: {
                conversionType: "YUV420PLANAR_TO_RGB"
            }
        },

        // Virtual PTZ - crop and zoom into regions of interest
        // Supports dynamic property changes for pan/tilt/zoom control
        ptz: {
            type: "VirtualPTZ"
            // Default: full frame (roiX=0, roiY=0, roiWidth=1, roiHeight=1)
        },

        // Encode as JPEG
        encoder: {
            type: "ImageEncoderCV"
        },

        // Write to files
        writer: {
            type: "FileWriterModule",
            props: {
                strFullFileNameWithPattern: path.join(outputDir, "processed_????.jpg")
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

console.log('=== Image Processing Pipeline Example ===\n');

// Create pipeline
const pipeline = ap.createPipeline(config);

// Register error handler
pipeline.on('error', (e) => {
    console.error(`ERROR in ${e.moduleId}: ${e.message}`);
});

// Display pipeline structure
console.log('Pipeline Structure:');
console.log('');
console.log('  [TestSignalGenerator] 1920x1080');
console.log('          |');
console.log('          v');
console.log('  [ColorConversion] YUV420 -> RGB');
console.log('          |');
console.log('          v');
console.log('  [VirtualPTZ] <- Has dynamic properties!');
console.log('          |');
console.log('          v');
console.log('  [ImageEncoderCV] RGB -> JPEG');
console.log('          |');
console.log('          v');
console.log('  [FileWriterModule] -> processed_????.jpg');
console.log('');

// Get module handles
const modules = {
    source: pipeline.getModule('source'),
    colorConvert: pipeline.getModule('colorConvert'),
    ptz: pipeline.getModule('ptz'),
    encoder: pipeline.getModule('encoder'),
    writer: pipeline.getModule('writer')
};

// Display module info
console.log('--- Module Info ---\n');
Object.entries(modules).forEach(([name, mod]) => {
    const hasDyn = mod.hasDynamicProperties();
    console.log(`${name}:`);
    console.log(`  Type: ${mod.type}`);
    console.log(`  ID: ${mod.id}`);
    console.log(`  Dynamic props: ${hasDyn ? mod.getDynamicPropertyNames().join(', ') : 'none'}`);
    console.log('');
});

// Demonstrate PTZ dynamic properties
const ptz = modules.ptz;
console.log('--- PTZ Dynamic Properties Demo ---\n');

// Initial state
console.log('Initial PTZ state (full frame):');
console.log(`  roiX: ${ptz.getProperty('roiX')}`);
console.log(`  roiY: ${ptz.getProperty('roiY')}`);
console.log(`  roiWidth: ${ptz.getProperty('roiWidth')}`);
console.log(`  roiHeight: ${ptz.getProperty('roiHeight')}`);

// Simulate image processing operations
console.log('\n--- Simulated Processing Operations ---\n');

// Operation 1: Zoom to center
console.log('1. Focus on center region (2x zoom):');
ptz.setProperty('roiWidth', 0.5);
ptz.setProperty('roiHeight', 0.5);
ptz.setProperty('roiX', 0.25);
ptz.setProperty('roiY', 0.25);
console.log(`   Crop: ${ptz.getProperty('roiWidth')}x${ptz.getProperty('roiHeight')} at (${ptz.getProperty('roiX')}, ${ptz.getProperty('roiY')})`);
console.log('   Effect: Extracts 960x540 region from center of 1920x1080');

// Operation 2: Pan to corner
console.log('\n2. Pan to bottom-right corner:');
ptz.setProperty('roiX', 0.5);
ptz.setProperty('roiY', 0.5);
console.log(`   Position: (${ptz.getProperty('roiX')}, ${ptz.getProperty('roiY')})`);
console.log('   Effect: Now viewing bottom-right quadrant');

// Operation 3: High zoom
console.log('\n3. Zoom in further (4x zoom):');
ptz.setProperty('roiWidth', 0.25);
ptz.setProperty('roiHeight', 0.25);
ptz.setProperty('roiX', 0.375);
ptz.setProperty('roiY', 0.375);
console.log(`   Crop: ${ptz.getProperty('roiWidth')}x${ptz.getProperty('roiHeight')}`);
console.log('   Effect: Extracts 480x270 region, upscaled to output');

// Reset
console.log('\n4. Reset to full frame:');
ptz.setProperty('roiX', 0);
ptz.setProperty('roiY', 0);
ptz.setProperty('roiWidth', 1);
ptz.setProperty('roiHeight', 1);
console.log('   Effect: Full frame, no crop');

// Show output info
console.log('\n--- Output ---\n');
console.log(`JPEG images will be written to: ${outputDir}/`);
console.log('Files: processed_0001.jpg, processed_0002.jpg, ...');

// Show usage pattern
console.log('\n--- Usage Pattern ---\n');
console.log('// Create pipeline');
console.log('const pipeline = ap.createPipeline(config);');
console.log('');
console.log('// Get PTZ module');
console.log('const ptz = pipeline.getModule("ptz");');
console.log('');
console.log('// Start processing');
console.log('pipeline.start();');
console.log('');
console.log('// Adjust PTZ in real-time while pipeline is running');
console.log('ptz.setProperty("roiX", 0.3);');
console.log('ptz.setProperty("roiY", 0.2);');
console.log('ptz.setProperty("roiWidth", 0.4);');
console.log('ptz.setProperty("roiHeight", 0.4);');
console.log('');
console.log('// View output files: processed_????.jpg');
console.log('');
console.log('// Stop when done');
console.log('pipeline.stop();');

console.log('\n=== Example Complete ===\n');

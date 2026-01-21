/**
 * Jetson L4TM Hardware JPEG Demo
 *
 * Demonstrates Jetson hardware-accelerated JPEG encoding/decoding using
 * the L4T Multimedia (L4TM) API. This example uses JPEGDecoderL4TM and
 * JPEGEncoderL4TM modules which leverage NVIDIA's hardware JPEG codec.
 *
 * Requirements:
 *   - Jetson device (Xavier, Orin, etc.)
 *   - JetPack 5.x or later
 *   - Input JPEG file at ./data/frame.jpg (or modify path below)
 *
 * Usage: node examples/node/jetson_l4tm_demo.js
 *
 * Output: Creates re-encoded JPEG files in ./data/testOutput/
 */

const path = require('path');
const fs = require('fs');

// Load the addon
const addonPath = path.join(__dirname, '../../bin/aprapipes.node');
let ap;
try {
    ap = require(addonPath);
    console.log('ApraPipes addon loaded successfully');
} catch (e) {
    console.error('Failed to load addon:', e.message);
    console.error('Make sure you have built the project with -DBUILD_NODE_ADDON=ON');
    process.exit(1);
}

// Check if L4TM modules are available
const modules = ap.listModules();
const hasL4TM = modules.includes('JPEGDecoderL4TM') && modules.includes('JPEGEncoderL4TM');

if (!hasL4TM) {
    console.error('L4TM modules not available. This example requires a Jetson device.');
    console.error('Available modules:', modules.filter(m => m.includes('JPEG')).join(', '));
    process.exit(1);
}

console.log('L4TM modules available: JPEGDecoderL4TM, JPEGEncoderL4TM');

// Create output directory
const outputDir = path.join(__dirname, '../../data/testOutput');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Find input JPEG file
const possibleInputPaths = [
    path.join(__dirname, '../../data/frame.jpg'),
    path.join(__dirname, '../../data/frame_0000.jpg'),
    '/data/actions-runner/_work/ApraPipes/ApraPipes/data/frame.jpg'
];

let inputPath = null;
for (const p of possibleInputPaths) {
    if (fs.existsSync(p)) {
        inputPath = p;
        break;
    }
}

if (!inputPath) {
    console.error('No input JPEG file found. Tried:', possibleInputPaths);
    process.exit(1);
}

console.log(`Using input file: ${inputPath}`);

// Pipeline configuration using L4TM hardware JPEG codec
const pipelineConfig = {
    name: "JetsonL4TMPipeline",
    modules: {
        // Read JPEG files
        reader: {
            type: "FileReaderModule",
            props: {
                strFullFileNameWithPattern: inputPath,
                readLoop: true,
                maxIndex: 20,  // Process 20 frames
                outputFrameType: "EncodedImage"  // Important: tells reader to output as encoded
            }
        },
        // Hardware JPEG decoder (L4TM)
        decoder: {
            type: "JPEGDecoderL4TM"
            // No props needed - uses defaults
        },
        // Hardware JPEG encoder (L4TM)
        encoder: {
            type: "JPEGEncoderL4TM",
            props: {
                quality: 90  // JPEG quality (1-100)
            }
        },
        // Write output files
        writer: {
            type: "FileWriterModule",
            props: {
                strFullFileNameWithPattern: path.join(outputDir, "encoded_????.jpg")
            }
        }
    },
    connections: [
        { from: "reader", to: "decoder" },
        { from: "decoder", to: "encoder" },
        { from: "encoder", to: "writer" }
    ]
};

async function main() {
    console.log('\n=== Jetson L4TM Hardware JPEG Demo ===\n');

    console.log('Pipeline:');
    console.log('  FileReaderModule (JPEG file)');
    console.log('       |');
    console.log('       v');
    console.log('  JPEGDecoderL4TM (HW decode)');
    console.log('       |');
    console.log('       v');
    console.log('  JPEGEncoderL4TM (HW encode @ quality=90)');
    console.log('       |');
    console.log('       v');
    console.log('  FileWriterModule -> ./data/testOutput/');
    console.log('');

    // Create the pipeline
    console.log('Creating pipeline...');
    const pipeline = ap.createPipeline(pipelineConfig);

    // Get module handles
    const reader = pipeline.getModule('reader');
    const decoder = pipeline.getModule('decoder');
    const encoder = pipeline.getModule('encoder');
    const writer = pipeline.getModule('writer');

    console.log('\nModules created:');
    console.log(`  - reader: ${reader.type} (${reader.id})`);
    console.log(`  - decoder: ${decoder.type} (${decoder.id})`);
    console.log(`  - encoder: ${encoder.type} (${encoder.id})`);
    console.log(`  - writer: ${writer.type} (${writer.id})`);

    // Set up event handlers
    pipeline
        .on('health', (event) => {
            console.log(`[Health] ${event.moduleId}: ${event.message}`);
        })
        .on('error', (event) => {
            console.error(`[Error] ${event.moduleId}: ${event.message}`);
        });

    // Initialize the pipeline
    console.log('\nInitializing pipeline...');
    await pipeline.init();
    console.log('Pipeline initialized.');

    // Run the pipeline
    console.log('Starting pipeline...');
    const startTime = Date.now();
    pipeline.run();

    // Let it run for 3 seconds
    console.log('Running for 3 seconds...\n');
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Stop the pipeline
    console.log('Stopping pipeline...');
    await pipeline.stop();
    const elapsed = Date.now() - startTime;
    console.log(`Pipeline stopped after ${elapsed}ms.`);

    // Check output files
    const files = fs.readdirSync(outputDir).filter(f => f.startsWith('encoded_') && f.endsWith('.jpg'));
    console.log(`\nGenerated ${files.length} JPEG files in ${outputDir}/`);

    if (files.length > 0) {
        // Show file sizes
        const firstFile = path.join(outputDir, files[0]);
        const lastFile = path.join(outputDir, files[files.length - 1]);
        const firstSize = fs.statSync(firstFile).size;
        const lastSize = fs.statSync(lastFile).size;

        console.log(`First file: ${files[0]} (${(firstSize / 1024).toFixed(1)} KB)`);
        console.log(`Last file: ${files[files.length - 1]} (${(lastSize / 1024).toFixed(1)} KB)`);

        // Calculate throughput
        const fps = files.length / (elapsed / 1000);
        console.log(`\nThroughput: ${fps.toFixed(1)} frames/sec (hardware accelerated)`);
    }

    console.log('\n=== Demo Complete ===\n');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
